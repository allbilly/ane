#!/usr/bin/env python3
"""Validate real Asahi ANE encoder projections and benchmark warm transcriptions."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def write_log(path, result):
    # Strip trailing presentation whitespace; retain every stdout/stderr line.
    text = result.stdout + "\n" + result.stderr
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def compare(reference, actual):
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("invalid numerical comparison arrays")
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    norm = np.linalg.norm(a)
    if norm == 0 or np.linalg.norm(b) == 0:
        raise ValueError("zero numerical comparison norm")
    return dict(nrmse=float(np.linalg.norm(a-b)/norm),
                cosine=float(np.dot(a, b)/(norm*np.linalg.norm(b))),
                max_abs=float(np.max(np.abs(a-b))))


def logits_records(path):
    data, offset, records = path.read_bytes(), 0, []
    while offset < len(data):
        count, vocabulary = struct.unpack_from("<2i", data, offset)
        offset += 8
        if not 1 <= count <= 448 or vocabulary != 51864:
            raise ValueError("unexpected tiny.en logit record")
        tokens = np.frombuffer(data, "<i4", count, offset).copy()
        offset += count*4
        values = np.frombuffer(data, "<f4", vocabulary, offset).copy()
        offset += vocabulary*4
        records.append((tokens, values))
    if offset != len(data) or not records:
        raise ValueError("incomplete logit capture")
    return records


def check_runtime(log, mode, expected_encodes):
    if not re.search(r"use gpu\s*=\s*0", log):
        raise ValueError("CPU host/decoder configuration evidence missing")
    fallbacks = re.findall(r"fallbacks\s*=\s*(\d+) p /\s*(\d+) h", log)
    if len(fallbacks) != expected_encodes or any(p != "0" or h != "0" for p, h in fallbacks):
        raise ValueError("unexpected decoding fallback")
    dispatches = re.findall(r"ASAHI_ANE encoder: projections=(\d+) submissions=(\d+) plans=(\d+) replicas=(\d+)", log)
    if mode:
        if "ASAHI_ANE ready:" not in log or len(dispatches) != expected_encodes:
            raise ValueError("missing actual ANE encoder evidence")
        if any(tuple(map(int, row)) != (24, 1128, 24, 1) for row in dispatches):
            raise ValueError("incomplete tiny.en ANE encoder execution")
    elif dispatches or "ASAHI_ANE ready:" in log:
        raise ValueError("CPU baseline unexpectedly used ANE")


def parse_warm_runs(result, mode, expected_words):
    records = {}
    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT\t"):
            _, phase, index, wall_ms, text = line.split("\t", 4)
            if words(text) != expected_words:
                raise ValueError("warm transcript mismatch")
            key = (phase, int(index))
            if key in records:
                raise ValueError("duplicate benchmark record")
            records[key] = dict(phase=phase, index=int(index), wall_ms=float(wall_ms), transcript=text.strip())
    check_runtime(result.stderr, mode, len(records))
    for block in re.finditer(r"BENCH_BEGIN\t(\w+)\t(\d+)\n(.*?)BENCH_END\t\1\t\2", result.stderr, re.S):
        record = records[(block[1], int(block[2]))]
        for name in ("mel", "sample", "encode", "decode", "batchd", "prompt"):
            match = re.search(rf"\b{name} time\s*=\s*([\d.]+) ms(?: /\s*(\d+) runs)?", block[3])
            if not match:
                raise ValueError("missing stage timing: " + name)
            record[name + "_ms"] = float(match[1])
            if match[2]:
                record[name + "_units"] = int(match[2])
        record["decoder_ms"] = record["decode_ms"] + record["batchd_ms"] + record["prompt_ms"]
        record["decode_ms_per_token"] = record["decode_ms"]/record["decode_units"]
        record["rtf"] = record["wall_ms"]/11000
        if mode and len(re.findall(r"ASAHI_ANE encoder:", block[3])) != 1:
            raise ValueError("timed transcription did not execute the ANE encoder")
    if not records or any("rtf" not in row for row in records.values()):
        raise ValueError("incomplete benchmark timings")
    return list(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/asahi-ane")
    parser.add_argument("--model", type=Path, default=ROOT / "models/hf-ggml/ggml-model.bin")
    parser.add_argument("--hf-model", type=Path, default=ROOT / "models/hf-tiny.en")
    parser.add_argument("--audio", type=Path, default=ROOT / "vendor/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("requires native M1 Asahi Linux")
    if min(args.warmups, args.runs, args.rounds) < 1:
        parser.error("warmups, runs and rounds must be positive")
    if digest(args.model) != "776f39cd70d01a7df3c6098f87f121b62d1fb634d1272b5f36bb6f369fc34372":
        parser.error("requires the pinned HF-to-ggml tiny.en conversion used on macOS")
    if digest(args.hf_model / "model.safetensors") != "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad":
        parser.error("requires the pinned HF tiny.en checkpoint")
    with wave.open(str(args.audio)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (16000, 1, 2, 176000):
            parser.error("requires the 11-second mono PCM16 JFK fixture")
        pcm16 = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("ANEFORGE_ENCODER", None)
    env.pop("ANEFORGE_DYLIB", None)
    env.pop("WHISPER_ASAHI_TRACE", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE", HF_HUB_OFFLINE="1")
    report = dict(status="RUNNING", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  kernel=platform.release(), model_sha256=digest(args.model),
                  hf_checkpoint_sha256=digest(args.hf_model / "model.safetensors"),
                  audio_sha256=digest(args.audio), whisper_cpp_revision="60c0be6ac8fa71b1a2ae2dd938a31a34a508e774",
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  scope="ANE encoder dense projections, 32 audio positions per submission, one rounding grid; CPU convolution, attention, normalizations, exact encoder GELU, cross-K/V and decoder; widened FP32 NEON dot accumulation and real-length encoder K/V views in both modes",
                  feature_and_logit_gate_nrmse=.005, hf_encoder_gate_cosine=.999,
                  warmups_per_context=args.warmups, runs_per_context=args.runs, rounds=args.rounds,
                  method="Persistent contexts, warmup excluded, reversed backend order in round two; four workers, greedy English, no timestamps/fallback, full 30-second encoder context. Decoder total includes prompt plus token evaluation; whole timer includes host work and sampling.",
                  limitations="Active desktop, clocks not fixed; native ggml CPU backend without BLAS. macOS uses different encoder implementation and Accelerate; this is not an isolated OS comparison.",
                  correctness=[], backends={name:dict(runs=[], warmups=[]) for name in ("cpu_cpu", "ane_projections_cpu")})
    try:
        import torch
        from transformers import WhisperForConditionalGeneration
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        hf = WhisperForConditionalGeneration.from_pretrained(args.hf_model, local_files_only=True).eval()
        # Exact mel tensors from the native run are used for the independent HF
        # reference. Feature-extractor differences cannot conceal kernel errors.
        for name, audio in (("jfk", pcm16), ("jfk-first-5s", pcm16[:80000]),
                            ("jfk-repeat", np.concatenate((pcm16, np.zeros(16000, dtype="<i2"), pcm16)))):
            wav_path = output / (name + ".wav")
            with wave.open(str(wav_path), "wb") as wav:
                wav.setparams((1, 2, 16000, len(audio), "NONE", "not compressed"))
                wav.writeframes(audio.tobytes())
            transcripts, directories = [], []
            for mode in (0, 1):
                directory = output / f"{name}-{'ane' if mode else 'cpu'}"
                directory.mkdir()
                directories.append(directory)
                run_env = dict(env, WHISPER_ASAHI_ANE=str(mode), WHISPER_ASAHI_TRACE=str(directory))
                prefix = directory / "transcript"
                command = [str((args.build / "bin/whisper-cli").resolve()), "-m", str(args.model.resolve()),
                           "-f", str(wav_path), "-l", "en", "-t", "4", "-bs", "1", "-bo", "1",
                           "-tp", "0", "-nf", "-nt", "-ng", "-otxt", "-of", str(prefix)]
                result = subprocess.run(command, env=run_env, capture_output=True, text=True, timeout=120)
                write_log(directory / "run.log", result)
                result.check_returncode()
                check_runtime(result.stderr, mode, 1)
                transcripts.append(prefix.with_suffix(".txt").read_text().strip())
            if words(transcripts[0]) != words(transcripts[1]) or (name == "jfk" and words(transcripts[0]) != words(EXPECTED)):
                raise ValueError("CPU/ANE transcript mismatch: " + name)
            cpu, ane = directories
            mel = np.fromfile(cpu / "mel.f32", "<f4").reshape(1, 80, 3000)
            np.testing.assert_array_equal(mel.ravel(), np.fromfile(ane / "mel.f32", "<f4"))
            native_features = np.fromfile(cpu / "encoder.f32", "<f4").reshape(1, 1500, 384)
            ane_features = np.fromfile(ane / "encoder.f32", "<f4").reshape(1, 1500, 384)
            feature_error = compare(native_features, ane_features)
            with torch.no_grad():
                reference = hf.model.encoder(torch.from_numpy(mel)).last_hidden_state.numpy()
            np.save(output / (name + "-hf-encoder.npy"), reference)
            hf_error = compare(reference, ane_features)
            cpu_hf_error = compare(reference, native_features)
            a, b = logits_records(cpu / "logits.bin"), logits_records(ane / "logits.bin")
            if len(a) != len(b):
                raise ValueError("decoder call count changed")
            checks = []
            for (cpu_tokens, cpu_logits), (ane_tokens, ane_logits) in zip(a, b):
                np.testing.assert_array_equal(cpu_tokens, ane_tokens)
                error = compare(cpu_logits, ane_logits)
                error["argmax_match"] = int(np.argmax(cpu_logits)) == int(np.argmax(ane_logits))
                checks.append(error)
            report["correctness"].append(dict(audio=name, audio_seconds=len(audio)/16000,
                audio_sha256=digest(wav_path), cpu_transcript=transcripts[0], ane_transcript=transcripts[1],
                encoder_vs_cpu=feature_error, encoder_vs_hf=hf_error, cpu_encoder_vs_hf=cpu_hf_error,
                decoder_calls=len(checks), all_histories_match=True,
                raw_argmax_matches=sum(x["argmax_match"] for x in checks),
                maximum_logit_nrmse=max(x["nrmse"] for x in checks), logit_checks=checks))
            if feature_error["nrmse"] >= .005 or hf_error["cosine"] < .999 or cpu_hf_error["cosine"] < .999:
                raise ValueError("encoder numerical gate failed: " + name)
            if any(x["nrmse"] >= .005 or not x["argmax_match"] for x in checks):
                raise ValueError("full decoder logit gate failed: " + name)
            print(f"{name}: PASS, 1128 ANE submissions, {len(checks)} decoder vectors; HF cosine {hf_error['cosine']:.8f}", flush=True)
        del hf
        driver = (args.build / "bin/benchmark-whisper").resolve()
        libdir = (args.build / "bin").resolve()
        command = ["c++", "-std=c++17", "-O3", str(ROOT / "scripts/benchmark_whisper.cpp"),
                   "-I" + str(ROOT / "vendor/whisper.cpp/include"),
                   "-I" + str(ROOT / "vendor/whisper.cpp/ggml/include"),
                   "-L" + str(libdir), "-Wl,-rpath," + str(libdir),
                   "-lwhisper", "-lggml", "-lggml-cpu", "-lggml-base", "-o", str(driver)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        write_log(output / "driver-build.log", compiled)
        compiled.check_returncode()
        pcm = output / "audio.f32"
        (pcm16.astype("<f4")/32768).tofile(pcm)
        for round_index in range(args.rounds):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for mode in order:
                name = "ane_projections_cpu" if mode else "cpu_cpu"
                run_env = dict(env, WHISPER_ASAHI_ANE=str(mode))
                result = subprocess.run([str(driver), str(args.model.resolve()), str(pcm), "0",
                    str(args.warmups), str(args.runs)], env=run_env, capture_output=True, text=True, timeout=180)
                write_log(output / f"{name}-{round_index + 1}.log", result)
                result.check_returncode()
                records = parse_warm_runs(result, mode, words(EXPECTED))
                if len(records) != args.warmups + args.runs:
                    raise ValueError("warm benchmark run count changed")
                for row in records:
                    row["round"] = round_index + 1
                    report["backends"][name]["warmups" if row["phase"] == "warmup" else "runs"].append(row)
                print(f"round {round_index + 1} {name}: PASS", flush=True)
        keys = ("encode_ms", "decoder_ms", "decode_ms", "batchd_ms", "prompt_ms", "decode_ms_per_token", "wall_ms", "rtf")
        for name, data in report["backends"].items():
            data["median"] = {key:statistics.median(row[key] for row in data["runs"]) for key in keys}
            data["wall_range_ms"] = [min(row["wall_ms"] for row in data["runs"]), max(row["wall_ms"] for row in data["runs"])]
            print(name, json.dumps(data["median"]), flush=True)
        report["binary_sha256"] = digest(args.build / "bin/whisper-cli")
        report["driver_sha256"] = digest(driver)
        report["library_sha256"] = {name:digest(libdir / name) for name in ("libwhisper.so", "libggml.so", "libggml-cpu.so", "libggml-base.so")}
        report["source_sha256"] = {str(path.relative_to(ROOT.parent)):digest(path) for path in
            (ROOT / "asahi_encoder.h", ROOT / "asahi_encoder.cpp", ROOT / "scripts/prepare_asahi.py",
             ROOT / "scripts/benchmark_asahi.py", ROOT / "scripts/benchmark_whisper.cpp",
             ROOT / "scripts/verify_asahi_matrices.py", ROOT.parent / "qwen35/ane_matmul.c",
             ROOT.parent / "qwen35/ane_matmul.h", ROOT.parent / "qwen35/linear_template.h",
             ROOT / "vendor/whisper-asahi/src/whisper.cpp",
             ROOT / "vendor/whisper-asahi/ggml/src/ggml-cpu/simd-mappings.h",
             ROOT / "vendor/whisper-asahi/src/CMakeLists.txt")}
        report["artifacts"] = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Evidence:", output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
