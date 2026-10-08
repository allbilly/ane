"""Measure the native PR encoder with the same zero-mel harness as macOS."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

import numpy as np

from whisper.encoder_kernel import require
from whisper.pr_encoder_kernel import reconstruct
from whisper.pr_encoder_replay import Encoder, digest as bytes_digest, native_descriptor, prepare
from whisper.validation import digest, hardware_locks

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "whisper/results/pr3905-m1-20261008"


def encode_samples(stdout, warmups=2, runs=5):
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("BENCH_ENCODE\t"):
            continue
        _, phase, index, elapsed = line.split("\t")
        elapsed = float(elapsed)
        require(int(index) == len(rows) and phase == ("warmup" if len(rows) < warmups else "measure")
                and math.isfinite(elapsed) and elapsed > 0, "invalid encode timing record")
        rows.append(dict(phase=phase, index=int(index), encode_ms=elapsed))
    require(len(rows) == warmups + runs, "incomplete encode timing records")
    return rows


def native_stages(stderr, count):
    pattern = (r"ASAHI_PR_PROFILE encoder: convert=([\d.]+) pack=([\d.]+) clear=([\d.]+) "
               r"dispatch=([\d.]+) read=([\d.]+) widen=([\d.]+) total=([\d.]+) "
               r"ms submissions=1 read_workers=(\d+)")
    records = re.findall(pattern, stderr)
    require(len(records) == count, "missing native ANE execution/stage evidence")
    stages = []
    for record in records:
        require(record[-1] == "4", "native benchmark requires four readback workers")
        values = list(map(float, record[:-1]))
        require(all(math.isfinite(x) and x >= 0 for x in values), "invalid native stage timing")
        stages.append(dict(zip(("convert_ms", "pack_ms", "clear_ms", "dispatch_ms", "read_ms",
                                "widen_ms", "ane_total_ms"), values), read_workers=4))
    return stages


def driver_profile(tasks):
    path = Path("/sys/module/ane/parameters")
    enabled = path / "profile_submit"
    if not enabled.exists() or enabled.read_text().strip() != "Y":
        return None
    values = {key:int(value) for key,value in re.findall(
        r"(\w+)=(-?\d+)", (path / "profile_last").read_text())}
    require(set(values) == {"tasks", "poll_sleep_us", "enqueue_ns", "push_ns", "wait_ns",
                            "irq_ns", "release_ns", "irq0", "irq1", "result"},
            "incomplete driver submission profile")
    require(values["tasks"] == tasks and values["result"] == 0
            and values["irq0"] + values["irq1"] > 0,
            "driver task/result/completion event mismatch")
    return values


def host_state():
    paths = [path / name for path in Path("/sys/devices/system/cpu/cpufreq").glob("policy*")
             for name in ("affected_cpus", "scaling_governor", "scaling_cur_freq",
                          "scaling_min_freq", "scaling_max_freq")]
    paths += [path / name for path in Path("/sys/class/thermal").glob("thermal_zone*")
              for name in ("type", "temp")]
    return {str(path):path.read_text().strip() for path in paths if path.is_file()}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  model=args.model, method="Two persistent contexts, two excluded warmups and five measurements each; four CPU and readback workers, full-context zero mel.",
                  scope="Native ANE encoder plus CPU cross-K/V; loading, compilation, Python replay checks and decoding excluded from encode_ms.",
                  accuracy_qualification="Strict full-decoder accuracy remains unverified; this is a diagnostic speed reference.",
                  contexts=[], timings_accepted=False)
    try:
        report["host_state_before"] = host_state()
        report["unavailable_counters"] = ["ANE clock", "ANE hardware execution duration", "package energy"]
        if Path("/sys/module/ane/parameters/profile_submit").exists():
            report["driver_profiling"] = dict(
                enabled=Path("/sys/module/ane/parameters/profile_submit").read_text().strip() == "Y",
                module_file_sha256=digest(ROOT / "kmod/ane.ko") if (ROOT / "kmod/ane.ko").is_file() else None,
                profiling_source_sha256=digest(ROOT / "kmod/ane_tm.c"),
                scope="Driver wall times sampled after submission, outside encoder timers; wait includes completion polling.",
                native_last_submissions=[])
        require(args.checkpoint.is_file(), "missing multilingual checkpoint: " + str(args.checkpoint))
        meta, hwx, positions = reconstruct(args.checkpoint, ROOT / "whisper/kernels/pr3905" / args.model)
        report.update(checkpoint_sha256=digest(args.checkpoint), hwx_sha256=meta["hwx_sha256"],
                      position_sha256=meta["position_sha256"], task_count=meta["layout"]["thread"]["td_count"])
        plan, payloads = prepare(meta, hwx, positions)
        payload_dir = output / "payloads"
        payload_dir.mkdir()
        for name, data in payloads.items():
            (payload_dir / (name + ".bin")).write_bytes(data)
        (payload_dir / "native-layout.txt").write_text(native_descriptor(plan, payloads))
        report["payload_sha256"] = plan["payloads"]
        cache = (args.build / "CMakeCache.txt").read_text()
        require("GGML_BLAS:BOOL=ON" in cache, "benchmark requires a BLAS-enabled build")
        require("GGML_BLAS_VENDOR:STRING=OpenBLAS" in cache, "benchmark requires OpenBLAS")
        report["build_cache_sha256"] = digest(args.build / "CMakeCache.txt")
        source = REFERENCE / "benchmark_encode.cpp"
        libraries = (args.build / "bin").resolve()
        driver = libraries / "benchmark-pr-encode"
        command = ["c++", "-std=c++17", "-O3", str(source),
                   "-I" + str(ROOT / "whisper/vendor/whisper.cpp/include"),
                   "-I" + str(ROOT / "whisper/vendor/whisper.cpp/ggml/include"),
                   "-L" + str(libraries), "-Wl,-rpath," + str(libraries),
                   "-lwhisper", "-lggml", "-lggml-cpu", "-lggml-base", "-o", str(driver)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        (output / "driver-build.log").write_text(compiled.stdout + compiled.stderr)
        compiled.check_returncode()
        report.update(driver_sha256=digest(driver), driver_source_sha256=digest(source),
                      library_sha256={p.name:digest(p) for p in libraries.glob("*.so")},
                      adapter_source_sha256=digest(ROOT / "whisper/asahi_full_encoder.cpp"),
                      cpu_affinity=sorted(os.sched_getaffinity(0)))
        linkage = subprocess.run(["ldd", str(libraries / "libggml-blas.so")],
                                 capture_output=True, text=True, check=True)
        (output / "blas-linkage.txt").write_text(linkage.stdout + linkage.stderr)
        report["resolved_blas_dependencies"] = {}
        for line in linkage.stdout.splitlines():
            if "=>" not in line:
                continue
            path = Path(line.split("=>", 1)[1].split(" (", 1)[0].strip())
            if path.is_file():
                report["resolved_blas_dependencies"][str(path)] = digest(path)
        require(any("libopenblas" in path for path in report["resolved_blas_dependencies"]),
                "OpenBLAS linkage evidence missing")
        env = os.environ.copy()
        for name in ("ANEFORGE_ENCODER", "ANEFORGE_DYLIB", "WHISPER_MACOS_PRECISION", "WHISPER_ASAHI_ANE"):
            env.pop(name, None)
        env.update(WHISPER_ASAHI_ENCODER=str(payload_dir), WHISPER_PROFILE="1",
                   OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
        report["thread_environment"] = {k:env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_WAIT_POLICY")}
        fixture_manifest = json.loads((REFERENCE / "replay-fixtures.json").read_text())
        reference = next(c for c in fixture_manifest["models"][args.model]["cases"] if c["name"] == "zero-mel")
        mel = np.zeros((80, 3000), "<f2")
        require(bytes_digest(mel.tobytes()) == reference["arrays"]["mel"]["sha256"], "zero input identity mismatch")
        require(meta["position_sha256"] == reference["arrays"]["positions"]["sha256"],
                "zero-mel reference positions mismatch")
        with hardware_locks():
            # Exact zero-mel arrays are reconstructible without transferring NPZ files.
            encoder = Encoder(args.checkpoint, ROOT / "whisper/kernels/pr3905" / args.model)
            check = dict(expected_output_sha256=reference["arrays"]["output"]["sha256"],
                         warmups=3, measured_runs=20, samples=[],
                         scope="Python complete encoder only; CPU cross-K/V and decoder excluded.")
            report["zero_mel_check"] = check
            try:
                for index in range(check["warmups"] + check["measured_runs"]):
                    before = encoder.submissions
                    output_hash = bytes_digest(encoder(mel).tobytes())
                    require(encoder.submissions == before + 1, "zero-mel requires one complete submission")
                    check["samples"].append(dict(index=index, phase="warmup" if index < 3 else "measure",
                                                 output_sha256=output_hash, **encoder.last_timing_ms))
                    profile = driver_profile(plan["td_count"])
                    if profile is not None:
                        check["samples"][-1]["driver_profile"] = profile
                    require(output_hash == check["expected_output_sha256"],
                            "zero-mel replay differs from Mac; timings not measured")
            finally:
                encoder.close()
            check["repeat_bitwise_equal"] = True
            check["status"] = "PASS_EXACT_MAC_ZERO_MEL"
            measured_zero = [s for s in check["samples"] if s["phase"] == "measure"]
            check["median_ms"] = {name:statistics.median(s[name] for s in measured_zero)
                                  for name in ("prepare_ms", "dispatch_ms", "readback_ms", "total_ms")}
            for context in range(2):
                result = subprocess.run([str(driver), str(args.checkpoint.resolve()), "2", "5"],
                                        env=env, capture_output=True, text=True, timeout=180)
                (output / f"context-{context + 1}.log").write_text(result.stdout + "\n" + result.stderr)
                result.check_returncode()
                rows = encode_samples(result.stdout)
                stages = native_stages(result.stderr, len(rows))
                require(f"ASAHI_PR_ANE ready: state={plan['dimensions']['state']} layers={plan['dimensions']['layers']} tasks={plan['td_count']}" in result.stderr,
                        "native runtime identity missing")
                report["contexts"].append([dict(**r, **s) for r, s in zip(rows, stages)])
                profile = driver_profile(plan["td_count"])
                if profile is not None:
                    report["driver_profiling"]["native_last_submissions"].append(
                        dict(context=context + 1, scope="Last of seven native calls only", **profile))
        measured = [r for context in report["contexts"] for r in context if r["phase"] == "measure"]
        mac = json.loads((REFERENCE / "host-encode-only.json").read_text())
        report["mac_reference_receipt_sha256"] = digest(REFERENCE / "host-encode-only.json")
        report["mac_reference"] = mac["backends"][args.model]
        report.update(status="PASS_ZERO_MEL_DIAGNOSTIC_TIMING", warmups_per_context=2,
                      measured_runs_per_context=5,
                      median_ms={name:statistics.median(r[name] for r in measured) for name in measured[0] if name.endswith("_ms")},
                      encode_range_ms=[min(r["encode_ms"] for r in measured), max(r["encode_ms"] for r in measured)])
        if all("driver_profile" in sample for sample in measured_zero):
            report["driver_profiling"]["python_replay_median_ms"] = {
                name.removesuffix("_ns") + "_ms":statistics.median(
                    sample["driver_profile"][name] / 1e6 for sample in measured_zero)
                for name in ("enqueue_ns", "push_ns", "wait_ns", "irq_ns", "release_ns")}
        report["host_state_after"] = host_state()
    except Exception as error:
        report.update(status="FAIL", error=str(error))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k in ("status", "model", "median_ms", "error", "accuracy_qualification")}))
    return 0 if report["status"] == "PASS_ZERO_MEL_DIAGNOSTIC_TIMING" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tiny", "base", "small"), default="tiny")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--build", type=Path, default=ROOT / "whisper/build/pr3905-asahi-native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or ROOT / f"whisper/models/ggml-{args.model}.bin"
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
