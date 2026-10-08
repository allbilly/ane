#!/usr/bin/env python3
"""Validate and benchmark the same native CPU/ANE pipeline on macOS and Asahi."""
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
import sys
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from whisper.validation import compare, digest, logits_records, parse_runs, words, hardware_locks, matrix_profiles
from whisper.attention import add_attention_profiles, attention_profiles

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"


def write_log(path, result):
    # Strip trailing presentation whitespace; retain every stdout/stderr line.
    text = result.stdout + "\n" + result.stderr
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def check_runtime(log, mode, expected_encodes, task_count=1779):
    if not re.search(r"use gpu\s*=\s*0", log):
        raise ValueError("CPU host/decoder configuration evidence missing")
    fallbacks = re.findall(r"fallbacks\s*=\s*(\d+) p /\s*(\d+) h", log)
    if len(fallbacks) != expected_encodes or any(p != "0" or h != "0" for p, h in fallbacks):
        raise ValueError("unexpected decoding fallback")
    dispatches = re.findall(r"ASAHI_ANE encoder: projections=(\d+) submissions=(\d+) plans=(\d+) replicas=(\d+)", log)
    complete = re.findall(r"ASAHI_FULL_ANE encoder: tasks=(\d+) submissions=(\d+)", log)
    if mode == 2:
        if "ASAHI_FULL_ANE ready:" not in log or len(complete) != expected_encodes or dispatches:
            raise ValueError("missing complete ANE encoder evidence")
        if any(tuple(map(int, row)) != (task_count, 1) for row in complete):
            raise ValueError("incomplete full encoder execution")
    elif mode:
        if complete:
            raise ValueError("projection run unexpectedly used complete encoder")
        if "ASAHI_ANE ready:" not in log or len(dispatches) != expected_encodes:
            raise ValueError("missing actual ANE encoder evidence")
        if any(tuple(map(int, row)) != (24, 1128, 24, 1) for row in dispatches):
            raise ValueError("incomplete tiny.en ANE encoder execution")
    elif dispatches or complete or "ASAHI_ANE ready:" in log or "ASAHI_FULL_ANE ready:" in log:
        raise ValueError("CPU baseline unexpectedly used ANE")


def parse_warm_runs(result, mode, expected_words, task_count=1779, audio_seconds=11,
                    matrix_layout="separate"):
    marker = "ASAHI_FULL_ANE encoder:" if mode == 2 else "ASAHI_ANE encoder:" if mode else None
    return parse_runs(result, audio_seconds, expected_words,
        check_runtime=lambda log, count: check_runtime(log, mode, count, task_count),
        encoder_marker=marker, matrix_layout=matrix_layout)


def parse_audio_runs(result, mode, correctness, task_count, backend="asahi", matrix_layout="separate"):
    stdout, stderr = result.stdout.split("BENCH_AUDIO\t"), result.stderr.split("BENCH_AUDIO\t")
    if len(stdout) != len(correctness) + 1 or len(stderr) != len(stdout):
        raise ValueError("missing warm audio boundaries")
    records = {}
    for index, expected in enumerate(correctness):
        out_id, out = stdout[index + 1].split("\n", 1)
        err_id, err = stderr[index + 1].split("\n", 1)
        if out_id != str(index) or err_id != str(index):
            raise ValueError("warm audio order changed")
        clip = subprocess.CompletedProcess(result.args, result.returncode, out, stderr[0] + err)
        if backend == "macos":
            from whisper.scripts.benchmark_macos import parse_runs as parse_macos_runs
            records[expected["audio"]] = parse_macos_runs(clip, "precision_cpu" if mode == 3 else "ane_cpu" if mode else "cpu_cpu",
                expected["audio_seconds"], words(expected["cpu_transcript"]), require_dispatch=True,
                matrix_layout=matrix_layout)
        else:
            records[expected["audio"]] = parse_warm_runs(clip, mode, words(expected["cpu_transcript"]),
                task_count, expected["audio_seconds"], matrix_layout=matrix_layout)
    return records


def run(default_backend="auto", default_encoder="complete"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "asahi", "macos"), default=default_backend)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--source", type=Path, help="Prepared whisper.cpp worktree; default read from CMakeCache.txt")
    parser.add_argument("--dylib", type=Path, help="ANEForge E5RT library for the native macOS encoder")
    parser.add_argument("--model", type=Path, default=ROOT / "models/hf-ggml/ggml-model.bin")
    parser.add_argument("--hf-model", type=Path, default=ROOT / "models/hf-tiny.en")
    parser.add_argument("--audio", type=Path, default=ROOT / "vendor/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--encoder", choices=("projections", "complete", "paired"), default=default_encoder)
    parser.add_argument("--precision-programs", type=Path, help="Checkpoint-verified paired Mac projection programs")
    parser.add_argument("--payloads", type=Path, help="SHA-256 checked whisper.encoder_kernel output for complete replay")
    parser.add_argument("--kernels", type=Path, default=ROOT / "kernels/tiny-en-encoder-fast")
    parser.add_argument("--diagnostic-timings", action="store_true",
                        help="collect numerical failures and unvalidated warm timings; failed gates still exit with FAIL")
    parser.add_argument("--profile-stages", action="store_true", help="Log encoder host stages and CPU cross-K/V on both hosts")
    parser.add_argument("--profile-matmul", action="store_true", help="Profile the requested cross-K/V BLAS products and retain one activation input per context")
    parser.add_argument("--fused-cross-kv", action="store_true",
                        help="Use the opt-in cached FP32 cross-K/V fusion; requires prepare_cross_kv")
    parser.add_argument("--encoder-attention", choices=("flash", "blas"), default="flash",
                        help="Opt-in contiguous batched encoder attention; decoder attention remains unchanged")
    args = parser.parse_args()
    if args.fused_cross_kv and not args.profile_matmul:
        parser.error("--fused-cross-kv requires --profile-matmul to verify actual fused execution")
    if args.encoder_attention == "blas" and (not args.profile_matmul or args.encoder == "complete"):
        parser.error("batched encoder attention requires --profile-matmul and a projection encoder")
    if args.backend == "auto":
        args.backend = "macos" if platform.system() == "Darwin" else "asahi"
    if args.backend == "asahi":
        if platform.system() != "Linux" or platform.machine() != "aarch64":
            parser.error("requires native M1 Asahi Linux")
    elif platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("requires native Apple Silicon macOS")
    args.build = args.build or ROOT / ("build/macos-matched" if args.backend == "macos" else "build/asahi-ane")
    cache = (args.build / "CMakeCache.txt").read_text()
    source_match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache, re.M)
    if args.source is None:
        if not source_match:
            parser.error("cannot resolve prepared worktree; pass --source")
        args.source = Path(source_match[1])
    # Shared production reference settings must be explicit on both hosts.
    source_text = (args.source / "src/whisper.cpp").read_text()
    if ('ggml_type itype = ggml_type::GGML_TYPE_F32;' not in source_text
            or 'ggml_gelu(ctx0, cur)' in source_text
            or 'n_state_head, n_audio_ctx_pad, n_head' in source_text):
        parser.error("prepare this worktree with whisper.scripts.prepare_native --precision fp32")
    if args.fused_cross_kv and 'WhisperCrossKV::view(ctx0, fused' not in source_text:
        parser.error("prepare this worktree with whisper.scripts.prepare_cross_kv first")
    if args.encoder_attention == "blas" and 'whisper.encoder_attention.%d.qk' not in source_text:
        parser.error("prepare this worktree with whisper.scripts.prepare_encoder_attention first")
    matrix_layout = "fused" if args.fused_cross_kv else "separate"
    if args.backend == "macos" and (args.encoder not in ("complete", "paired") or not args.dylib or not args.dylib.is_file()):
        parser.error("macOS requires --encoder complete/paired and --dylib")
    if args.encoder == "paired":
        if args.backend != "macos" or not args.precision_programs or 'WhisperMacPrecision' not in source_text:
            parser.error("paired encoder requires the prepared Mac precision worktree and --precision-programs")
        from whisper.scripts.prepare_macos_precision import validate_programs
        precision_manifest = validate_programs(args.precision_programs, args.hf_model / "model.safetensors")
    args.kernels = args.kernels.resolve()
    ane_mode = 3 if args.encoder == "paired" else 2 if args.encoder == "complete" else 1
    ane_label = "ane_precision_cpu" if ane_mode == 3 else "ane_complete_cpu" if ane_mode == 2 else "ane_projections_cpu"
    if ane_mode == 2:
        if not args.payloads:
            parser.error("--encoder complete requires --payloads")
        sys.path.insert(0, str(ROOT.parent))
        from whisper.encoder_kernel import reconstruct, native_layout
        meta, payloads = reconstruct(args.hf_model / "model.safetensors", args.kernels)
        if (args.payloads / "layout.txt").read_text() != native_layout(meta):
            parser.error("complete encoder layout mismatch")
        for name in ("commands", "coefficients", "constants", "positions", "source-mil", "source-weights"):
            filename = {"positions":"pos.f16", "source-mil":"model.mil", "source-weights":"weights.bin"}.get(name, name+".bin")
            if (args.payloads / filename).read_bytes() != payloads[name]:
                parser.error("complete encoder payload mismatch: " + filename)
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
    env.pop("WHISPER_ASAHI_PROFILE", None)
    env.pop("WHISPER_ASAHI_ENCODER", None)
    env.pop("WHISPER_ASAHI_ANE", None)
    env.pop("WHISPER_TRACE", None)
    env.pop("WHISPER_PROFILE", None)
    env.pop("WHISPER_PROFILE_MATMUL", None)
    env.pop("WHISPER_PROFILE_INPUT", None)
    env.pop("WHISPER_FUSED_CROSS_KV", None)
    env.pop("WHISPER_MACOS_PRECISION", None)
    env.pop("WHISPER_ENCODER_BLAS_ATTENTION", None)
    env.pop("WHISPER_PROFILE_ENCODER_ATTENTION", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE", HF_HUB_OFFLINE="1")
    if args.profile_stages:
        env["WHISPER_PROFILE"] = "1"
        if args.backend == "asahi":
            env["WHISPER_ASAHI_PROFILE"] = "1"
    if args.profile_matmul:
        env["WHISPER_PROFILE_MATMUL"] = "1"
    if args.fused_cross_kv:
        env["WHISPER_FUSED_CROSS_KV"] = "1"
    if args.encoder_attention == "blas":
        env.update(WHISPER_ENCODER_BLAS_ATTENTION="1", WHISPER_PROFILE_ENCODER_ATTENTION="1")
    report = dict(status="RUNNING", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  kernel=platform.release(), host_backend=args.backend, cpu_precision="fp32", model_sha256=digest(args.model),
                  hf_checkpoint_sha256=digest(args.hf_model / "model.safetensors"),
                  audio_sha256=digest(args.audio), whisper_cpp_revision="60c0be6ac8fa71b1a2ae2dd938a31a34a508e774",
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  encoder_backend=args.encoder,
                  scope="ANE encoder dense projections, 32 audio positions per submission, one rounding grid; vectorized host scaling and four-worker uncached output reads; CPU convolution, attention, normalizations, exact GELU, cross-K/V and decoder; FP32 K/V caches and activations with F16 weights, widened FP32 NEON dot and tiled matrix accumulation including mixed-precision GEMV, GCC NEON attention kernel selection fix, and real-length encoder K/V views in both modes",
                  encoder_cpu_gate_nrmse=.005 if ane_mode in (1, 3) else None,
                  full_logit_gate_nrmse=.005, hf_encoder_gate_cosine=.999,
                  warmups_per_context=args.warmups, runs_per_context=args.runs, rounds=args.rounds,
                  method="Persistent contexts, warmup excluded, reversed backend order in round two; four workers, greedy English, no timestamps/fallback, full 30-second encoder context. Decoder total includes prompt plus token evaluation; whole timer includes host work and sampling.",
                  limitations="Active desktop, clocks not fixed; native ggml CPU backend without BLAS. Both hosts use the shared FP32 CPU precision patches; BLAS, compiler and driver behavior can still differ.",
                  correctness=[], backends={name:dict(runs=[], warmups=[]) for name in ("cpu_cpu", ane_label)})
    report["profile_matmul_requested"] = args.profile_matmul
    report["cross_kv_layout"] = matrix_layout
    report["encoder_attention"] = args.encoder_attention
    if args.fused_cross_kv:
        report["cross_kv_cache"] = dict(weight_type="f32", dimensions=[384, 3072],
            weight_bytes=384*3072*4, result_bytes=1500*3072*4,
            conversion_scope="once per persistent decoder state during graph preparation")
    report.update(diagnostic_timings_requested=args.diagnostic_timings, numerical_failures=[],
                  timings_accepted=False)
    report["blas_enabled"] = "GGML_BLAS:BOOL=ON" in cache
    if args.profile_matmul and not report["blas_enabled"]:
        parser.error("--profile-matmul requires a BLAS-enabled build")
    if report["blas_enabled"]:
        report["limitations"] = "Active desktop, clocks not fixed; CPU BLAS enabled. Both hosts use the shared FP32 CPU precision patches; BLAS, compiler and driver behavior can still differ."
    if ane_mode == 2:
        report["scope"] = f"Complete {meta['td_count']:,}-task tiny.en encoder in one ANE submission; FP16 mel/encoder output, CPU cross-K/V and decoder with F16 weights and FP32 activations/KV."
        report["payload_sha256"] = {name:hashlib.sha256(payloads[name]).hexdigest()
                                   for name in ("commands", "coefficients", "constants", "positions")}
    elif ane_mode == 3:
        report["scope"] = "24 native paired ANE projections; FP32 CPU convolution, attention, normalization, exact GELU, residuals, cross-K/V and decoder. Two input partitions and two high/residual rounding grids per projection, one submission per projection."
        report["precision_programs"] = precision_manifest
        report["precision_manifest_sha256"] = digest(args.precision_programs / "manifest.json")
    def backend_env(mode):
        run_env = dict(env)
        if args.backend == "asahi":
            run_env["WHISPER_ASAHI_ANE"] = "1" if mode == 1 else "0"
            if mode == 2:
                run_env["WHISPER_ASAHI_ENCODER"] = str(args.payloads.resolve())
        elif mode == 3:
            run_env.update(WHISPER_MACOS_PRECISION=str(args.precision_programs.resolve()), ANEFORGE_DYLIB=str(args.dylib.resolve()))
        elif mode:
            run_env.update(ANEFORGE_ENCODER=str(args.payloads.resolve()), ANEFORGE_DYLIB=str(args.dylib.resolve()))
        return run_env
    try:
        import torch
        from transformers import WhisperForConditionalGeneration
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        hf = WhisperForConditionalGeneration.from_pretrained(args.hf_model, local_files_only=True, attn_implementation="eager").eval()
        # Exact mel tensors from the native run are used for the independent HF
        # reference. Feature-extractor differences cannot conceal kernel errors.
        for name, audio in (("jfk", pcm16), ("jfk-first-5s", pcm16[:80000]),
                            ("jfk-repeat", np.concatenate((pcm16, np.zeros(16000, dtype="<i2"), pcm16)))):
            wav_path = output / (name + ".wav")
            with wave.open(str(wav_path), "wb") as wav:
                wav.setparams((1, 2, 16000, len(audio), "NONE", "not compressed"))
                wav.writeframes(audio.tobytes())
            transcripts, directories = [], []
            for mode in (0, ane_mode):
                directory = output / f"{name}-{'ane' if mode else 'cpu'}"
                directory.mkdir()
                directories.append(directory)
                run_env = backend_env(mode)
                run_env["WHISPER_TRACE"] = str(directory)
                if args.profile_matmul:
                    run_env["WHISPER_PROFILE_INPUT"] = str(directory / "cross-kv-input.f32")
                prefix = directory / "transcript"
                command = [str((args.build / "bin/whisper-cli").resolve()), "-m", str(args.model.resolve()),
                           "-f", str(wav_path), "-l", "en", "-t", "4", "-bs", "1", "-bo", "1",
                           "-tp", "0", "-nf", "-nt", "-ng", "-otxt", "-of", str(prefix)]
                result = subprocess.run(command, env=run_env, capture_output=True, text=True, timeout=120)
                write_log(directory / "run.log", result)
                result.check_returncode()
                attention_profiles(result.stderr, args.encoder_attention == "blas")
                if args.profile_matmul:
                    matrix_profiles(result.stderr, required=True, layout=matrix_layout)
                    captured = np.fromfile(directory / "cross-kv-input.f32", "<f4")
                    np.testing.assert_array_equal(captured, np.fromfile(directory / "encoder.f32", "<f4"))
                if args.backend == "macos":
                    from whisper.scripts.benchmark_macos import check_runtime as check_macos_runtime
                    check_macos_runtime(result.stderr, "precision_cpu" if mode == 3 else "ane_cpu" if mode else "cpu_cpu", 1, require_dispatch=True)
                else:
                    check_runtime(result.stderr, mode, 1, meta["td_count"] if ane_mode == 2 else 1779)
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
            if len(a) != {"jfk":25, "jfk-first-5s":8, "jfk-repeat":47}[name]:
                raise ValueError("the three-clip gate requires all 80 full decoder vectors")
            checks, hf_checks, history = [], [], []
            for (cpu_tokens, cpu_logits), (ane_tokens, ane_logits) in zip(a, b):
                np.testing.assert_array_equal(cpu_tokens, ane_tokens)
                error = compare(cpu_logits, ane_logits)
                error["argmax_match"] = int(np.argmax(cpu_logits)) == int(np.argmax(ane_logits))
                checks.append(error)
                # Independently evaluate every captured native prefix in HF,
                # using the HF encoder (not either native feature array).
                history.extend(cpu_tokens.tolist())
                with torch.no_grad():
                    hidden = hf.model.decoder(input_ids=torch.tensor([history]),
                        encoder_hidden_states=torch.from_numpy(reference), use_cache=False).last_hidden_state
                    expected_logits = hf.proj_out(hidden[:, -1, :]).numpy().ravel()
                hf_checks.append(dict(cpu=compare(expected_logits, cpu_logits),
                    ane=compare(expected_logits, ane_logits),
                    cpu_argmax_match=int(np.argmax(expected_logits)) == int(np.argmax(cpu_logits)),
                    ane_argmax_match=int(np.argmax(expected_logits)) == int(np.argmax(ane_logits))))
            report["correctness"].append(dict(audio=name, audio_seconds=len(audio)/16000,
                audio_sha256=digest(wav_path), cpu_transcript=transcripts[0], ane_transcript=transcripts[1],
                encoder_vs_cpu=feature_error, encoder_vs_hf=hf_error, cpu_encoder_vs_hf=cpu_hf_error,
                decoder_calls=len(checks), all_histories_match=True,
                raw_argmax_matches=sum(x["argmax_match"] for x in checks),
                maximum_logit_nrmse=max(x["nrmse"] for x in checks), logit_checks=checks,
                maximum_cpu_logit_nrmse_vs_hf=max(x["cpu"]["nrmse"] for x in hf_checks),
                maximum_ane_logit_nrmse_vs_hf=max(x["ane"]["nrmse"] for x in hf_checks),
                cpu_raw_argmax_matches_hf=sum(x["cpu_argmax_match"] for x in hf_checks),
                ane_raw_argmax_matches_hf=sum(x["ane_argmax_match"] for x in hf_checks),
                hf_logit_checks=hf_checks))
            # The complete FP16 export has its own captured-output relative-L2
            # gate in replay_encoder.py. Its task-defined independent HF gate
            # is cosine >= .999; the FP32 CPU features remain diagnostic here.
            failures = []
            if (ane_mode in (1, 3) and feature_error["nrmse"] >= .005) or hf_error["cosine"] < .999 or cpu_hf_error["cosine"] < .999:
                failures.append("encoder numerical gate failed: " + name)
            if any(x["nrmse"] >= .005 or not x["argmax_match"] for x in checks):
                failures.append("full decoder logit gate failed: " + name)
            if any(x["cpu"]["nrmse"] >= .005 or x["ane"]["nrmse"] >= .005 or
                   not x["cpu_argmax_match"] or not x["ane_argmax_match"] for x in hf_checks):
                failures.append("independent HF full decoder logit gate failed: " + name)
            report["numerical_failures"].extend(failures)
            if failures:
                if not args.diagnostic_timings:
                    raise ValueError(failures[0])
                print(f"{name}: FAIL; collecting diagnostic timings, accuracy gates unchanged", flush=True)
                continue
            count = 24 if ane_mode == 3 else 1 if ane_mode == 2 else 1128
            print(f"{name}: PASS, {count} ANE submissions, {len(checks)} decoder vectors; HF cosine {hf_error['cosine']:.8f}", flush=True)
        del hf
        driver = (args.build / "bin/benchmark-whisper").resolve()
        libdir = (args.build / "bin").resolve()
        command = ["c++", "-std=c++17", "-O3", str(ROOT / "scripts/benchmark_whisper.cpp"),
                   "-I" + str(args.source / "include"),
                   "-I" + str(args.source / "ggml/include"),
                   "-L" + str(libdir), "-Wl,-rpath," + str(libdir),
                   "-lwhisper", "-lggml", "-lggml-cpu", "-lggml-base", "-o", str(driver)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        write_log(output / "driver-build.log", compiled)
        compiled.check_returncode()
        pcm_paths = []
        for name, samples in (("jfk", pcm16), ("jfk-first-5s", pcm16[:80000]),
                             ("jfk-repeat", np.concatenate((pcm16, np.zeros(16000, "<i2"), pcm16)))):
            path = output / (name + ".f32")
            (samples.astype("<f4") / 32768).tofile(path)
            pcm_paths.append(path)
        for data in report["backends"].values():
            data["clips"] = {row["audio"]:dict(runs=[], warmups=[]) for row in report["correctness"]}
        for round_index in range(args.rounds):
            order = (0, ane_mode) if round_index % 2 == 0 else (ane_mode, 0)
            for mode in order:
                name = ane_label if mode else "cpu_cpu"
                run_env = backend_env(mode)
                if args.profile_matmul:
                    run_env["WHISPER_PROFILE_INPUT"] = str(output / f"{name}-{round_index+1}-cross-kv-input.f32")
                result = subprocess.run([str(driver), str(args.model.resolve()), str(pcm_paths[0]), "0",
                    str(args.warmups), str(args.runs), *map(str, pcm_paths[1:])], env=run_env, capture_output=True, text=True, timeout=180)
                write_log(output / f"{name}-{round_index + 1}.log", result)
                result.check_returncode()
                clips = parse_audio_runs(result, mode, report["correctness"], meta["td_count"] if ane_mode == 2 else 1779,
                                         args.backend, matrix_layout=matrix_layout)
                add_attention_profiles(clips, result.stderr, args.encoder_attention == "blas")
                for audio, records in clips.items():
                    if len(records) != args.warmups + args.runs:
                        raise ValueError("warm benchmark run count changed")
                    for row in records:
                        if args.profile_matmul and len(row.get("cross_kv_matrices", [])) != (1 if args.fused_cross_kv else 8):
                            raise ValueError("timed run is missing its requested cross-K/V matrix profiles")
                        row["round"] = round_index + 1
                        phase = "warmups" if row["phase"] == "warmup" else "runs"
                        report["backends"][name]["clips"][audio][phase].append(row)
                        if audio == "jfk":
                            report["backends"][name][phase].append(row)
                label = "diagnostic timing capture" if report["numerical_failures"] else "PASS"
                print(f"round {round_index + 1} {name}: {label}", flush=True)
        keys = ("encode_ms", "decoder_ms", "decode_ms", "batchd_ms", "prompt_ms", "decode_ms_per_token", "wall_ms", "rtf")
        for name, data in report["backends"].items():
            for clip in data["clips"].values():
                clip["median"] = {key:statistics.median(row[key] for row in clip["runs"]) for key in keys}
                if args.profile_matmul:
                    stage_keys = ("allocate_us", "convert_us", "thread_setup_us", "gemm_us", "total_us")
                    clip["matrix_profile_median_ms"] = {
                        key.removesuffix("_us")+"_ms": statistics.median(
                            sum(m[key] for m in row["cross_kv_matrices"])/1000 for row in clip["runs"])
                        for key in stage_keys}
                    clip["matrix_profile_by_operation"] = {
                        matrix: {key.removesuffix("_us")+"_ms": statistics.median(
                            next(m[key] for m in row["cross_kv_matrices"] if m["name"] == matrix)/1000
                            for row in clip["runs"]) for key in stage_keys}
                        for matrix in sorted(m["name"] for m in clip["runs"][0]["cross_kv_matrices"])}
            data["median"] = {key:statistics.median(row[key] for row in data["runs"]) for key in keys}
            data["wall_range_ms"] = [min(row["wall_ms"] for row in data["runs"]), max(row["wall_ms"] for row in data["runs"])]
            print(name, json.dumps(data["median"]), flush=True)
        report["binary_sha256"] = digest(args.build / "bin/whisper-cli")
        report["driver_sha256"] = digest(driver)
        suffix = ".dylib" if args.backend == "macos" else ".so"
        report["library_sha256"] = {name+suffix:digest(libdir / (name+suffix))
            for name in ("libwhisper", "libggml", "libggml-cpu", "libggml-base")}
        if args.dylib:
            report["runtime_sha256"] = digest(args.dylib)
        if report["blas_enabled"]:
            report["library_sha256"]["libggml-blas"+suffix] = digest(libdir / ("libggml-blas"+suffix))
            report["blas_backend_observed"] = all("using BLAS backend" in (output / f"{name}-1.log").read_text()
                for name in report["backends"])
        common_sources = [ROOT / name for name in (
            "encoder_kernel.py", "encoder_runtime.py", "validation.py", "validation.h", "native.py",
            "scripts/prepare_native.py", "scripts/benchmark_native.py", "scripts/benchmark_whisper.cpp")]
        if args.fused_cross_kv:
            common_sources.extend(ROOT / name for name in ("cross_kv.h", "scripts/prepare_cross_kv.py"))
        if ane_mode == 3:
            common_sources.extend(ROOT / name for name in ("macos_precision.cpp", "macos_precision.h", "scripts/prepare_macos_precision.py"))
        if args.encoder_attention == "blas":
            common_sources.extend(ROOT / name for name in ("attention.py", "scripts/prepare_encoder_attention.py"))
        native_sources = [args.source / name for name in (
            "src/whisper.cpp", "src/CMakeLists.txt", "ggml/src/ggml-cpu/simd-mappings.h",
            "ggml/src/ggml-cpu/llamafile/sgemm.cpp", "ggml/src/ggml-blas/ggml-blas.cpp")]
        if args.backend == "asahi":
            common_sources.extend(ROOT / name for name in ("asahi_encoder.h", "asahi_encoder.cpp", "asahi_full_encoder.cpp", "scripts/prepare_asahi.py"))
        report["source_sha256"] = {str(path):digest(path) for path in common_sources + native_sources + [args.kernels / "meta.json"]}
        report["artifacts"] = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}
        if report["numerical_failures"]:
            raise ValueError("numerical validation failed; warm timings are diagnostic and are not accepted")
        report.update(status="PASS", timings_accepted=True)
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Evidence:", output / "summary.json", flush=True)


def main(default_backend="auto", default_encoder="complete", acquire_locks=True):
    if acquire_locks:
        with hardware_locks():
            return run(default_backend, default_encoder)
    return run(default_backend, default_encoder)


if __name__ == "__main__":
    main()
