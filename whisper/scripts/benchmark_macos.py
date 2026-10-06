#!/usr/bin/env python3
"""Benchmark every upstream CPU/Metal/ANE encoder route with warm contexts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import wave

from test_macos import digest, words

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
    "cpu_cpu": (False, False),
    "metal_metal": (True, False),
    "ane_cpu": (False, True),
    "ane_metal": (True, True),
}
EXPECTED = words("And so my fellow Americans ask not what your country can do for you ask what you can do for your country")


def parse_runs(result, backend, audio_seconds):
    log = result.stderr
    use_gpu, use_ane = BACKENDS[backend]
    if use_gpu and not re.search(r"whisper_backend_init_gpu: using (?:MTL\d+|Metal) backend", log):
        raise RuntimeError(f"{backend}: Metal readiness evidence missing")
    if not use_gpu and not re.search(r"whisper_init_with_params_no_state: use gpu\s*=\s*0", log):
        raise RuntimeError(f"{backend}: CPU configuration evidence missing")
    if use_ane and ("aneforge: encoder ready" not in log or re.search(
        r"aneforge: (?:compile failed|mel size|dlopen|missing|pos.f16 read failed)", log
    )):
        raise RuntimeError(f"{backend}: ANE initialization failed")
    records = {}
    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT\t"):
            _, phase, index, wall_ms, transcript = line.split("\t", 4)
            if words(transcript) != EXPECTED:
                raise RuntimeError(f"{backend}: transcript mismatch: {transcript!r}")
            records[(phase, int(index))] = {
                "phase": phase, "index": int(index), "wall_ms": float(wall_ms),
                "transcript": transcript.strip(),
            }
    for match in re.finditer(r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2", log, re.S):
        run = records[(match[1], int(match[2]))]
        for name in ("mel", "sample", "encode", "decode", "batchd", "prompt"):
            timing = re.search(rf"\b{name} time\s*=\s*([\d.]+) ms(?: /\s*(\d+) runs)?", match[3])
            if not timing:
                raise RuntimeError(f"{backend}: missing {name} timing")
            run[name + "_ms"] = float(timing[1])
            if timing[2]:
                run[name + "_units"] = int(timing[2])
        fallback = re.search(r"fallbacks\s*=\s*(\d+) p /\s*(\d+) h", match[3])
        if not fallback or int(fallback[1]) or int(fallback[2]):
            raise RuntimeError(f"{backend}: decoding fallback detected")
        run["decoder_ms"] = run["decode_ms"] + run["batchd_ms"] + run["prompt_ms"]
        run["decode_ms_per_token"] = run["decode_ms"] / run["decode_units"]
        run["rtf"] = run["wall_ms"] / (audio_seconds * 1000)
    if any("rtf" not in run for run in records.values()):
        raise RuntimeError(f"{backend}: incomplete timing blocks")
    return list(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/metal")
    parser.add_argument("--model", type=Path, default=ROOT / "models/ggml-tiny.en.bin")
    parser.add_argument("--audio", type=Path, default=ROOT / "vendor/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/whisper-tiny.en-ane")
    parser.add_argument("--dylib", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("requires a physical Apple Silicon Mac")
    if min(args.warmups, args.runs, args.rounds) < 1:
        parser.error("warmups, runs and rounds must be positive")
    for path in (args.model, args.audio, args.dylib, args.build / "bin/libwhisper.dylib"):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    for name in ("ports.txt", "pos.f16", "model.mil", "weights.bin"):
        if not (args.bundle / name).is_file():
            parser.error(f"incomplete bundle: {args.bundle / name}")
    import numpy as np
    with wave.open(str(args.audio)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            parser.error("requires 16 kHz mono PCM16 WAV")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("<f4") / 32768
    audio_seconds = len(audio) / 16000
    if audio_seconds > 30:
        parser.error("this benchmark covers one clip of at most 30 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pcm = output / "audio.f32"
    audio.tofile(pcm)
    binary = (args.build / "bin/benchmark-whisper").resolve()
    libdir = (args.build / "bin").resolve()
    command = ["/usr/bin/c++", "-std=c++17", "-O3", "-DNDEBUG", "-arch", "arm64",
               str(ROOT / "scripts/benchmark_whisper.cpp"),
               "-I" + str(ROOT / "vendor/whisper.cpp/include"),
               "-I" + str(ROOT / "vendor/whisper.cpp/ggml/include"),
               "-L" + str(libdir), "-Wl,-rpath," + str(libdir), "-lwhisper", "-lggml",
               "-lggml-cpu", "-lggml-blas", "-lggml-metal", "-lggml-base", "-o", str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True, check=True)
    (output / "build.log").write_text(compiled.stdout + compiled.stderr)
    report = {
        "status": "RUNNING", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "macos": subprocess.check_output(["sw_vers"], text=True).strip(),
        "audio_seconds": audio_seconds, "model": str(args.model.resolve()),
        "model_sha256": digest(args.model), "audio_sha256": digest(args.audio),
        "binary_sha256": digest(binary), "library_sha256": digest(libdir / "libwhisper.dylib"),
        "dylib_sha256": digest(args.dylib), "mil_sha256": digest(args.bundle / "model.mil"),
        "warmups_per_context": args.warmups, "runs_per_context": args.runs, "rounds": args.rounds,
        "method": "One persistent context per backend/round, warmup excluded, reverse backend order in alternate rounds, serial execution, English greedy, four threads, no timestamps/fallback, full 30-second audio context, reset text history between calls.",
        "metric_notes": {
            "encode_ms": "whisper.cpp encode timer, including cross-attention K/V preparation",
            "decoder_ms": "sum of decode + batchd + prompt evaluation timers; excludes sampling",
            "decode_ms_per_token": "single-token evaluation timer divided by its call count; excludes sampling",
            "wall_ms": "wall time of whisper_full only; includes mel, model execution, sampling and orchestration; excludes loading, compilation and file I/O",
            "rtf": "wall seconds / actual audio seconds; lower is better",
        },
        "backends": {backend: {"runs": [], "warmups": []} for backend in BACKENDS},
    }
    try:
        for round_index in range(args.rounds):
            order = list(BACKENDS)
            if round_index % 2:
                order.reverse()
            for backend in order:
                use_gpu, use_ane = BACKENDS[backend]
                env = os.environ.copy()
                env.pop("ANEFORGE_ENCODER", None)
                env.pop("ANEFORGE_DYLIB", None)
                if use_ane:
                    env["ANEFORGE_ENCODER"] = str(args.bundle.resolve())
                    env["ANEFORGE_DYLIB"] = str(args.dylib.resolve())
                command = [str(binary), str(args.model.resolve()), str(pcm), str(int(use_gpu)),
                           str(args.warmups), str(args.runs)]
                print(f"Round {round_index + 1}: {backend}, warming up then measuring", flush=True)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
                (output / f"{backend}-{round_index + 1}.log").write_text(result.stdout + "\n" + result.stderr)
                result.check_returncode()
                runs = parse_runs(result, backend, audio_seconds)
                if len(runs) != args.warmups + args.runs:
                    raise RuntimeError(f"{backend}: unexpected run count")
                for run in runs:
                    run["round"] = round_index + 1
                    report["backends"][backend]["warmups" if run["phase"] == "warmup" else "runs"].append(run)
                median = statistics.median(run["wall_ms"] for run in runs if run["phase"] == "measure")
                print(f"{backend}: PASS, warm transcription median {median:.2f} ms", flush=True)
        for data in report["backends"].values():
            keys = ("encode_ms", "decoder_ms", "decode_ms", "batchd_ms", "prompt_ms",
                    "decode_ms_per_token", "mel_ms", "sample_ms", "wall_ms", "rtf")
            data["median"] = {key: statistics.median(run[key] for run in data["runs"]) for key in keys}
            data["wall_range_ms"] = [min(run["wall_ms"] for run in data["runs"]), max(run["wall_ms"] for run in data["runs"])]
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        pcm.unlink(missing_ok=True)
        print(f"Evidence: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
