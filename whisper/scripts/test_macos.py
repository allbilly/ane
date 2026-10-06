#!/usr/bin/env python3
"""Transcribe JFK with CPU, Metal and ANEForge; reject fallback and compare words."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/metal/bin/whisper-cli")
    parser.add_argument("--model", type=Path, default=ROOT / "models/ggml-tiny.en.bin")
    parser.add_argument("--audio", type=Path, default=ROOT / "vendor/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/whisper-tiny.en-ane")
    parser.add_argument("--dylib", type=Path, required=True)
    parser.add_argument("--coreml-binary", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this test requires a physical Apple Silicon Mac")
    if args.runs < 1:
        parser.error("--runs must be positive")
    for path in (args.binary, args.model, args.audio, args.dylib):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    for name in ("ports.txt", "pos.f16", "model.mil", "weights.bin"):
        if not (args.bundle / name).is_file():
            parser.error(f"incomplete bundle: {args.bundle / name}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "results" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING", "utc": stamp,
        "macos": subprocess.check_output(["sw_vers"], text=True).strip(),
        "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "binary": str(args.binary.resolve()), "model": str(args.model.resolve()),
        "model_sha256": digest(args.model), "audio_sha256": digest(args.audio),
        "binary_sha256": digest(args.binary), "dylib": str(args.dylib.resolve()),
        "dylib_sha256": digest(args.dylib), "bundle": str(args.bundle.resolve()),
        "mil_sha256": digest(args.bundle / "model.mil"), "backends": {},
        "method": "Separate CLI processes, greedy English, four threads, full audio context; ANE decoder on CPU.",
    }
    expected = words("And so my fellow Americans ask not what your country can do for you ask what you can do for your country")
    backends = ["cpu", "metal", "aneforge"]
    if args.coreml_binary:
        backends.append("coreml")
    try:
        for backend in backends:
            runs = []
            report["backends"][backend] = {"runs": runs}
            for index in range(args.runs):
                prefix = output / f"{backend}-{index + 1}"
                env = os.environ.copy()
                env.pop("ANEFORGE_ENCODER", None)
                env.pop("ANEFORGE_DYLIB", None)
                binary = args.coreml_binary if backend == "coreml" else args.binary
                command = [str(binary.resolve()), "-m", str(args.model.resolve()), "-f", str(args.audio.resolve()),
                           "-l", "en", "-t", "4", "-bs", "1", "-bo", "1", "-tp", "0", "-nf", "-nt", "-otxt", "-of", str(prefix)]
                if backend in ("cpu", "aneforge", "coreml"):
                    command.append("-ng")
                if backend == "aneforge":
                    env["ANEFORGE_ENCODER"] = str(args.bundle.resolve())
                    env["ANEFORGE_DYLIB"] = str(args.dylib.resolve())
                started = time.perf_counter()
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
                elapsed = time.perf_counter() - started
                log = result.stdout + "\n" + result.stderr
                prefix.with_suffix(".log").write_text(log)
                if result.returncode:
                    raise RuntimeError(f"{backend}: exit {result.returncode}; see {prefix}.log")
                if backend == "metal" and not re.search(r"whisper_backend_init_gpu: using (?:MTL\d+|Metal) backend", log):
                    raise RuntimeError("Metal backend readiness evidence missing")
                if backend in ("cpu", "aneforge") and not re.search(r"whisper_init_with_params_no_state: use gpu\s*=\s*0", log):
                    raise RuntimeError(f"CPU decoder configuration evidence missing for {backend}")
                if backend == "aneforge" and "aneforge: encoder ready" not in log:
                    raise RuntimeError("ANEForge readiness evidence missing")
                if backend == "aneforge" and re.search(r"aneforge: (?:compile failed|mel size|dlopen|missing|pos.f16 read failed)", log):
                    raise RuntimeError("ANEForge reported a runtime error")
                if backend == "coreml" and "Core ML model loaded" not in log:
                    raise RuntimeError("Core ML readiness evidence missing")
                text = prefix.with_suffix(".txt").read_text().strip()
                if words(text) != expected:
                    raise RuntimeError(f"unexpected JFK transcript from {backend}: {text!r}")
                timings = {}
                for name in ("load", "encode", "decode", "total"):
                    match = re.search(rf"\b{name} time\s*=\s*([\d.]+) ms", log)
                    if match:
                        timings[name + "_ms"] = float(match.group(1))
                if "encode_ms" not in timings or "total_ms" not in timings:
                    raise RuntimeError(f"timing evidence missing for {backend}")
                runs.append({"command": command, "transcript": text, "wall_ms": elapsed * 1000, **timings})
                print(f"{backend} {index + 1}: PASS, encode={timings['encode_ms']:.2f} ms", flush=True)
            report["backends"][backend]["median_ms"] = {
                name: statistics.median(run[name] for run in runs)
                for name in ("encode_ms", "total_ms", "wall_ms")
            }
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
