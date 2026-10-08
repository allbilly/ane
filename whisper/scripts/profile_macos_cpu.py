"""Sample the warmed native Mac transcription process and retain host identities."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from whisper.validation import digest, hardware_locks


def command_output(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return dict(command=command, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("WHISPER_TRACE", "WHISPER_ASAHI_TRACE", "WHISPER_PROFILE_INPUT",
                "WHISPER_PROFILE_MATMUL", "ANEFORGE_ENCODER", "ANEFORGE_DYLIB",
                "WHISPER_MACOS_PRECISION", "WHISPER_FUSED_CROSS_KV",
                "WHISPER_ENCODER_BLAS_ATTENTION", "WHISPER_PROFILE_ENCODER_ATTENTION",
                "WHISPER_PROFILE_VOCABULARY"):
        env.pop(key, None)
    env.update(ANEFORGE_ENCODER=str(args.payloads.resolve()), ANEFORGE_DYLIB=str(args.dylib.resolve()),
        WHISPER_PROFILE="1", OPENBLAS_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE")
    if getattr(args, "vocabulary", False):
        env["WHISPER_PROFILE_VOCABULARY"] = "1"
    cache = (args.build / "CMakeCache.txt").read_text()
    compilers = {language:re.search(r"^CMAKE_"+language+r"_COMPILER:FILEPATH=(.+)$",cache,re.M)[1]
                 for language in ("C","CXX")}
    command = [str((args.build / "bin/benchmark-whisper").resolve()), str(args.model.resolve()),
        str(args.pcm.resolve()), "0", "2", str(args.runs)]
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(), command=command, requested_workers=4,
        scope="Separate CPU wall-clock stack sampling of the warmed ANE + CPU path; sample shares are not end-to-end wall-time or CPU-cycle shares. The original ANE graph still fails the full-logit gate.",
        identities={name:command_output(cmd) for name, cmd in (
            ("os", ["sw_vers"]), ("compiler", [compilers["CXX"], "--version"]),
            ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("thermal_before", ["pmset", "-g", "therm"]),
            ("power_counters", ["/usr/bin/powermetrics", "-n", "1", "-i", "100", "--samplers", "cpu_power,ane_power,thermal"]),
        )},
        unavailable=dict(cpu_ane_clocks="powermetrics requires administrator access",
            power="powermetrics requires administrator access", hardware_device_timestamps="not exposed by the E5RT adapter",
            exact_core_placement="sample identifies threads, not their physical performance/efficiency core placement",
            actual_blas_threads="Accelerate exposes no thread query used by this runner"),
        hashes=dict(model=digest(args.model), runtime=digest(args.dylib),
            driver=digest(args.build / "bin/benchmark-whisper")))
    report["build"] = dict(compilers=compilers, cmake_cache_sha256=digest(args.build / "CMakeCache.txt"),
        flags={str(p.relative_to(args.build)):p.read_text() for p in args.build.rglob("flags.make")
               if any(name in str(p) for name in ("ggml-cpu.dir", "ggml-blas.dir", "whisper.dir"))})
    report["environment"] = {key:env[key] for key in ("OPENBLAS_NUM_THREADS","OMP_WAIT_POLICY","WHISPER_PROFILE")}
    with hardware_locks(), (args.output / "driver.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            # Observe completion of both warmups rather than assuming a delay suffices.
            deadline = time.monotonic()+30
            while "BENCH_END\twarmup\t2" not in (args.output / "driver.log").read_text():
                if process.poll() is not None:
                    raise RuntimeError("native workload terminated before sampling; inspect driver.log")
                if time.monotonic() >= deadline:
                    raise RuntimeError("native warmups did not complete within 30 seconds")
                time.sleep(.05)
            report["sampling_after_excluded_warmups"] = 2
            if process.poll() is not None:
                raise RuntimeError("native workload ended before sampling")
            sampling = command_output(["/usr/bin/sample", str(process.pid), str(args.seconds), "1",
                "-mayDie", "-fullPaths", "-file", str((args.output / "sample.txt").resolve())])
            report["sampling"] = sampling
            if sampling["returncode"]:
                raise RuntimeError("CPU sampling failed: " + sampling["stderr"])
            report["driver_returncode"] = process.wait(timeout=120)
            if report["driver_returncode"]:
                raise RuntimeError("native profiling workload failed")
            report["identities"]["thermal_after"] = command_output(["pmset", "-g", "therm"])
            report["status"] = "sampled"
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            (args.output / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CPU sampling evidence:", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pcm", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--dylib", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", action="store_true", help="Record vocabulary shapes in a decoder-profile preparation")
    args = parser.parse_args()
    if platform.system() != "Darwin" or min(args.seconds, args.runs) < 1 or args.seconds > 15:
        parser.error("requires macOS, positive counts, and at most 15 sampling seconds")
    run(args)


if __name__ == "__main__":
    main()
