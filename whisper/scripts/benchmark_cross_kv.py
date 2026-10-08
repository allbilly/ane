"""Compare warm separate/fused cross-K/V graphs after full native validation."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from whisper.scripts.benchmark_native import parse_audio_runs, write_log
from whisper.scripts.summarize_cpu_ablations import add_stages, medians
from whisper.validation import compare, digest, hardware_locks, logits_records

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"jfk": 25, "jfk-first-5s": 8, "jfk-repeat": 47}


def compact_receipt(report):
    """Intern exact matrix metadata, retaining every address and stage timing."""
    report = json.loads(json.dumps(report))
    columns = [key + "_us" for key in ("allocate", "convert", "thread_setup", "gemm", "total")]
    metadata, indices = [], {}
    for kinds in report["configurations"].values():
        for clips in kinds.values():
            for clip in clips.values():
                for row in clip["runs"] + clip["warmups"]:
                    for field in ("cross_kv_matrices", "attention_matrices"):
                        if field not in row:
                            continue
                        encoded = []
                        for matrix in row.pop(field):
                            meta = {key:value for key,value in matrix.items() if key not in columns}
                            key = json.dumps(meta, sort_keys=True)
                            if key not in indices:
                                indices[key] = len(metadata)
                                metadata.append(meta)
                            encoded.append(dict(metadata=indices[key], timing_us=[matrix[key] for key in columns]))
                        row[field] = encoded
    report.update(matrix_metadata=metadata, matrix_timing_columns=columns)
    return report


def validate_pair(reference, validation):
    reports = [json.loads((p / "summary.json").read_text()) for p in (reference, validation)]
    for report in reports:
        if (report.get("status") not in ("PASS", "FAIL") or "binary_sha256" not in report or
                report["full_logit_gate_nrmse"] != .005 or report["encoder_backend"] != "complete" or
                {r["audio"]: r["decoder_calls"] for r in report["correctness"]} != EXPECTED):
            raise ValueError("requires completed, unchanged full 80-vector native validations")
        for clip in report["correctness"]:
            if len(clip["hf_logit_checks"]) != EXPECTED[clip["audio"]]:
                raise ValueError("independent HF validation lost full logit vectors")
            if any(not math.isfinite(c["cpu"]["nrmse"]) or c["cpu"]["nrmse"] >= .005 or not c["cpu_argmax_match"]
                   for c in clip["hf_logit_checks"]):
                raise ValueError("CPU reference failed the independent HF gate")
    if reports[0].get("cross_kv_layout", "separate") != "separate" or reports[1].get("cross_kv_layout") != "fused":
        raise ValueError("requires separate reference and fused validation")
    for key in ("host_backend", "cpu_precision", "model_sha256", "hf_checkpoint_sha256", "payload_sha256", "runtime_sha256"):
        if reports[0].get(key) != reports[1].get(key):
            raise ValueError("validation identity changed: " + key)
    checks = []
    for audio in EXPECTED:
        for backend in ("cpu", "ane"):
            a, b = (p / f"{audio}-{backend}" for p in (reference, validation))
            boundaries = {}
            for name in ("mel.f32", "encoder.f32"):
                boundaries[name] = digest(a / name)
                if boundaries[name] != digest(b / name):
                    raise ValueError("cross-K/V changed encoder boundary bytes: " + str(b / name))
            left, right = logits_records(a / "logits.bin"), logits_records(b / "logits.bin")
            if len(left) != EXPECTED[audio] or len(right) != len(left):
                raise ValueError("cross-layout logit comparison lost vectors")
            vectors = []
            for (tokens, original), (actual_tokens, actual) in zip(left, right):
                np.testing.assert_array_equal(tokens, actual_tokens)
                error = compare(original, actual)
                error["argmax_match"] = int(original.argmax()) == int(actual.argmax())
                if error["nrmse"] >= .005 or not error["argmax_match"]:
                    raise ValueError("fusion changed full native decoder logits")
                vectors.append(error)
            checks.append(dict(audio=audio, backend=backend, boundaries=boundaries,
                               histories_match=True, logit_checks=vectors))
    return reports[1], checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/ggml-tiny.en.bin")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--dylib", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-receipt", type=Path, help="Also write a compact receipt retaining all repetitions")
    args = parser.parse_args()
    validated, checks = validate_pair(args.reference, args.validation)
    backend = "macos" if platform.system() == "Darwin" else "asahi"
    if backend != validated["host_backend"]:
        parser.error("run on the native validation host")
    build, output = args.build.resolve(), args.output.resolve()
    driver = build / "bin/benchmark-whisper"
    if digest(driver) != validated["driver_sha256"] or digest(args.model) != validated["model_sha256"]:
        parser.error("driver or model changed since validation")
    for name, checksum in validated["library_sha256"].items():
        if digest(build / "bin" / name) != checksum:
            parser.error("native library changed since validation: " + name)
    for name, checksum in validated["payload_sha256"].items():
        path = args.payloads / ("pos.f16" if name == "positions" else name + ".bin")
        if digest(path) != checksum:
            parser.error("encoder payload changed since validation")
    if backend == "macos" and (not args.dylib or digest(args.dylib) != validated["runtime_sha256"]):
        parser.error("requires the validated E5RT runtime")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", timings_accepted=False, host_backend=backend,
        validation_utc=validated["utc"], model_sha256=validated["model_sha256"],
        hf_checkpoint_sha256=validated["hf_checkpoint_sha256"],
        payload_sha256=validated["payload_sha256"], runtime_sha256=validated.get("runtime_sha256"),
        full_logit_gate_nrmse=.005, numerical_failures=validated["numerical_failures"],
        reference_summary_sha256=digest(args.reference / "summary.json"),
        validation_summary_sha256=digest(args.validation / "summary.json"),
        correctness=validated["correctness"], cross_layout_checks=checks,
        library_sha256=validated["library_sha256"], driver_sha256=digest(driver),
        method="Same native binary/libraries; two persistent contexts per layout/backend, two excluded warmups and ten measures per clip per round. Layout and CPU/ANE order both reversed in round two.",
        limitations="Active desktop, unfixed clocks; macOS/OpenBLAS is a portable-kernel proxy, not an Asahi run. Original ANE accuracy failures remain explicit.",
        configurations={layout: {kind: {} for kind in ("cpu_cpu", "ane_complete_cpu")}
                        for layout in ("separate", "fused")})
    env = os.environ.copy()
    for key in ("ANEFORGE_ENCODER", "ANEFORGE_DYLIB", "WHISPER_ASAHI_ANE", "WHISPER_ASAHI_ENCODER",
                "WHISPER_TRACE", "WHISPER_ASAHI_TRACE", "WHISPER_PROFILE_INPUT", "WHISPER_FUSED_CROSS_KV"):
        env.pop(key, None)
    env.update(WHISPER_PROFILE="1", WHISPER_PROFILE_MATMUL="1", OPENBLAS_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE")
    if backend == "asahi":
        env["WHISPER_ASAHI_PROFILE"] = "1"
    pcm = [str((args.validation / (name + ".f32")).resolve()) for name in EXPECTED]
    try:
        with hardware_locks():
            for round_index in range(2):
                layouts = ("separate", "fused") if round_index == 0 else ("fused", "separate")
                modes = (0, 2) if round_index == 0 else (2, 0)
                for layout in layouts:
                    for mode in modes:
                        kind = "ane_complete_cpu" if mode else "cpu_cpu"
                        run_env = dict(env)
                        if layout == "fused":
                            run_env["WHISPER_FUSED_CROSS_KV"] = "1"
                        if mode and backend == "macos":
                            run_env.update(ANEFORGE_ENCODER=str(args.payloads.resolve()), ANEFORGE_DYLIB=str(args.dylib.resolve()))
                        elif mode:
                            run_env["WHISPER_ASAHI_ENCODER"] = str(args.payloads.resolve())
                        log = output / f"{layout}-{kind}-{round_index+1}.log"
                        result = subprocess.run([str(driver), str(args.model.resolve()), pcm[0], "0", "2", "10", *pcm[1:]],
                                                env=run_env, capture_output=True, text=True, timeout=180)
                        write_log(log, result)
                        result.check_returncode()
                        clips = parse_audio_runs(result, mode, validated["correctness"], 1779, backend, matrix_layout=layout)
                        add_stages(clips, result.stderr, bool(mode))
                        for audio, rows in clips.items():
                            if len(rows) != 12 or sum(r["phase"] == "measure" for r in rows) != 10:
                                raise ValueError("paired warm run counts changed")
                            for row in rows:
                                row["round"] = round_index + 1
                                matrices = row["cross_kv_matrices"]
                                if len(matrices) != (1 if layout == "fused" else 8):
                                    raise ValueError("matrix execution count changed")
                                if any(m["requested_threads"] != 4 or
                                       (m["backend"] == "OpenBLAS" and m["blas_threads"] != 4) for m in matrices):
                                    raise ValueError("matrix thread count changed")
                            report["configurations"][layout][kind].setdefault(audio, []).extend(rows)
                        print(f"round {round_index+1}: {layout} {kind} captured", flush=True)
        for configuration in report["configurations"].values():
            for kind, clips in configuration.items():
                for audio, rows in list(clips.items()):
                    measured = [r for r in rows if r["phase"] == "measure"]
                    clips[audio] = dict(runs=measured, warmups=[r for r in rows if r["phase"] == "warmup"],
                        median=medians(measured), round_medians={str(i): medians([r for r in measured if r["round"] == i]) for i in (1, 2)})
        report["source_sha256"] = {str(p):digest(p) for p in (
            Path(__file__), ROOT / "cross_kv.h", ROOT / "scripts/prepare_cross_kv.py")}
        report["artifacts"] = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
        report["status"] = "measured_diagnostic" if report["numerical_failures"] else "PASS"
        report["timings_accepted"] = not report["numerical_failures"]
    except Exception as error:
        report.update(status="ERROR", error=str(error))
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.compact_receipt and report["status"] in ("PASS", "measured_diagnostic"):
            with args.compact_receipt.open("x") as stream:
                stream.write(json.dumps(compact_receipt(report), indent=2) + "\n")
        print("Evidence:", output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
