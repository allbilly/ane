"""Capture real-audio ANE Whisper fixtures and exercise additional native routes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "Desktop/ANEForge"))
from capture_macos_program import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coreml", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("ANEFORGE_NO_AUTOBUILD", "1")
    whisper = ROOT / "whisper"
    with wave.open(str(whisper / "vendor/whisper.cpp/samples/jfk.wav")) as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
    clips = {"jfk": pcm, "jfk-first-5s": pcm[:80000],
             "jfk-repeat": np.concatenate([pcm, np.zeros(16000, "<i2"), pcm])}
    report = dict(audio_note="Original JFK, truncated five-second speech, and two JFK copies separated by one second of silence. Constructed coverage, not a diverse speech corpus or a WER benchmark.",
                  hf_model=str(a.model),
                  checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.iterdir() if p.suffix in (".bin", ".safetensors")},
                  ggml_model_sha256=hashlib.sha256((whisper / "models/ggml-tiny.en.bin").read_bytes()).hexdigest(),
                  cli_sha256={route:hashlib.sha256((whisper / path).read_bytes()).hexdigest()
                              for route, path in (("metal", "build/metal/bin/whisper-cli"), ("coreml", "build/coreml/bin/whisper-cli"))
                              if (whisper / path).is_file()},
                  fixtures=[], transcriptions=[], timing_note="Separate CLI calls include cold model setup. Timers are retained as diagnostics, not warm-context throughput.",
                  linux_replay="pending native Asahi hardware")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from aneforge._runtime import E5RT, _find_dylib
    report["runtime_sha256"] = hashlib.sha256(_find_dylib().read_bytes()).hexdigest()
    torch.set_num_threads(4)
    hf = WhisperForConditionalGeneration.from_pretrained(a.model, attn_implementation="eager").eval()
    processor = WhisperProcessor.from_pretrained(a.model)
    import shutil
    source_bundle = whisper / "models/whisper-tiny.en-ane"
    bundle = a.output / "bundle"
    bundle.mkdir()
    for filename in ("model.mil", "weights.bin", "pos.f16", "ports.txt"):
        shutil.copy2(source_bundle / filename, bundle / filename)
    inputs = {"t1": (1, 80, 1, 3000), "t0": (1, 384, 1, 1500)}
    outputs = {"t1383": (1500, 384)}
    program = None
    try:
        program = E5RT.compile(bundle / "model.mil", cache_dir=a.output / "cache", inputs=inputs, outputs=outputs, device_mask=4)
        if program._device_mask != 4: raise RuntimeError("ANE-only device mask required")
        report["hardware_execution"] = "ANE-only E5RT"
    except Exception as error:
        report.update(hardware_execution="runtime_compile_failed", runtime_error=str(error))
    report["reference_kind"] = "macOS ANE FP16 outputs" if program else "HF CPU FP32 on exact FP16-rounded mel input"
    position = np.fromfile(bundle / "pos.f16", "<f2").reshape(inputs["t0"])
    try:
        for label, samples in clips.items():
            audio = a.output / (label + ".wav")
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed")); wav.writeframes(samples.tobytes())
            mel = processor(samples.astype(np.float32) / 32768, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad(): reference = hf.model.encoder(mel).last_hidden_state.numpy()[0]
            # Reference sees the exact rounded input fed to ANE.
            rounded = mel.numpy().astype(np.float16)
            with torch.no_grad(): exact_input_reference = hf.model.encoder(torch.from_numpy(rounded.astype(np.float32))).last_hidden_state.numpy()[0]
            if program:
                program.set_input("t1", rounded.reshape(inputs["t1"]))
                program.set_input("t0", position)
                program.execute()
                actual = program.read_output("t1383").copy()
            else:
                actual = exact_input_reference
            av, ev = actual.astype(np.float64).ravel(), exact_input_reference.astype(np.float64).ravel()
            cosine = float(av @ ev / (np.linalg.norm(av) * np.linalg.norm(ev)))
            fixture = a.output / (label + ".npz")
            np.savez_compressed(fixture, input00=rounded.reshape(inputs["t1"]), input01=position,
                                output=actual, hf_output=exact_input_reference, hf_unrounded=reference)
            item = dict(name=label, fixture=fixture.name, audio_seconds=len(samples) / 16000,
                        audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), cosine=cosine if program else None,
                        normalized_rmse=float(np.linalg.norm(av - ev) / np.linalg.norm(ev)) if program else None,
                        passes_encoder_cosine=cosine > .999 if program else None,
                        reference_kind=report["reference_kind"])
            report["fixtures"].append(item)
            print(json.dumps(item), flush=True)
    finally:
        if program: program.release()
    del hf, processor
    report["ports"] = dict(inputs=[dict(name=name, shape=shape) for name, shape in inputs.items()],
                            outputs=[dict(name=name, shape=shape) for name, shape in outputs.items()])
    report["export"] = export(bundle, a.output / "hwx", ROOT / "gpt2/training/build/dump_hwx")
    for label in clips:
        baseline = None
        for route in ("cpu", "metal", "ane_cpu", "ane_metal", *(["coreml"] if a.coreml else [])):
            prefix = a.output / f"{label}-{route}"
            env = dict(os.environ)
            env.pop("ANEFORGE_ENCODER", None); env.pop("ANEFORGE_DYLIB", None)
            model = a.coreml / "ggml-tiny.en.bin" if route == "coreml" else whisper / "models/ggml-tiny.en.bin"
            binary = whisper / ("build/coreml/bin/whisper-cli" if route == "coreml" else "build/metal/bin/whisper-cli")
            command = [str(binary), "-m", str(model), "-f", str(a.output / (label + ".wav")),
                       "-l", "en", "-t", "4", "-bs", "1", "-bo", "1", "-tp", "0", "-nf", "-nt", "-otxt", "-of", str(prefix)]
            if route in ("cpu", "ane_cpu", "coreml"): command.append("-ng")
            if route.startswith("ane_"):
                env.update(ANEFORGE_ENCODER=str(bundle), ANEFORGE_DYLIB=str(_find_dylib()))
            started = time.perf_counter()
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
            log = result.stdout + "\n" + result.stderr
            prefix.with_suffix(".log").write_text(log)
            transcript = prefix.with_suffix(".txt").read_text().strip() if prefix.with_suffix(".txt").exists() else ""
            words = re.findall(r"[a-z0-9']+", transcript.lower())
            if route == "cpu": baseline = words
            ready = result.returncode == 0 and bool(transcript)
            if label == "jfk": ready = ready and words == re.findall(r"[a-z0-9']+", "And so my fellow Americans ask not what your country can do for you ask what you can do for your country".lower())
            if route.startswith("ane_"): ready = ready and "aneforge: encoder ready" in log and not re.search(r"aneforge: (?:compile failed|mel size|dlopen|missing|pos.f16 read failed)", log)
            if route == "coreml": ready = ready and "Core ML model loaded" in log and "failed to load Core ML" not in log
            if route in ("metal", "ane_metal"): ready = ready and bool(re.search(r"whisper_backend_init_gpu: using (?:MTL\d+|Metal) backend", log))
            row = dict(audio=label, route=route, command=command, returncode=result.returncode,
                       transcript=transcript, words_match_cpu=words == baseline, ready=ready,
                       wall_ms=(time.perf_counter() - started) * 1000)
            for name in ("encode", "decode", "total"):
                match = re.search(rf"\b{name} time\s*=\s*([\d.]+) ms", log)
                if match: row[name + "_ms"] = float(match[1])
            report["transcriptions"].append(row)
            (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    report["status"] = "pass" if program and all(r["passes_encoder_cosine"] for r in report["fixtures"]) and all(r["ready"] and r["words_match_cpu"] for r in report["transcriptions"]) else "incomplete_runtime_validation"
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__":
    main()
