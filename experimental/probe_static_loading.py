"""Historical HWX rejection controls beside private MIL execution on macOS."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_HWX = Path("/System/Library/PrivateFrameworks/VideoProcessing.framework/Versions/A/Resources/cnn_frame_enhancer_320p.H13.espresso.hwx")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mil-dir", type=Path, required=True)
    p.add_argument("--hwx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lock-timeout", type=float, default=600)
    a = p.parse_args()
    if platform.system() != "Darwin":
        p.error("requires macOS")
    if a.lock_timeout <= 0:
        p.error("lock-timeout must be positive")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), cases=[], source_sha256={},
                  scope="64-value MUL: exact constant and signed fractional checks; system control loads only",
                  compiler_route="_ANEClient compileModel with kANEFModelMIL, then loadModel and evaluate")
    locks = []
    try:
        source = a.output / "mil"
        source.mkdir()
        for path in [a.mil_dir / "model.mil", *a.mil_dir.glob("*.bin")]:
            expected = digest(path)
            shutil.copy2(path, source / path.name)
            if digest(source / path.name) != expected:
                raise ValueError("source changed during capture")
            report["source_sha256"][path.name] = expected
        hwx = a.output / "offline.hwx"
        expected = digest(a.hwx)
        shutil.copy2(a.hwx, hwx)
        if digest(hwx) != expected:
            raise ValueError("HWX changed during capture")
        report["offline_hwx_sha256"] = expected
        report["system_hwx_sha256"] = digest(SYSTEM_HWX)
        helper = a.output / "test_static_hwx"
        report["checker_source_sha256"] = digest(ROOT / "experimental/test_static_hwx.m")
        with (a.output / "build.log").open("w") as log:
            subprocess.run(["/usr/bin/clang", "-fobjc-arc", "-O2", str(ROOT / "experimental/test_static_hwx.m"),
                            "-framework", "Foundation", "-framework", "IOSurface", "-o", str(helper)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        for name in ("ane.lock", "gpu.lock"):
            path = Path.home() / name
            lock = path.open("r" if path.exists() else "a")
            locks.append(lock)
            deadline = time.monotonic() + a.lock_timeout
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("shared hardware lock remained busy: " + name)
                    time.sleep(2)
        for route, path, model_type in (("offline-hwx", hwx, "kANEFModelPreCompiled"),
                                        ("client-mil", source, "kANEFModelMIL"),
                                        ("system-control", SYSTEM_HWX, "kANEFModelPreCompiled")):
            options = a.output / (route + "-options.json")
            options.write_text(json.dumps({"kANEFModelType": model_type}) + "\n")
            for mode in (("load-only",) if route == "system-control" else ("constant", "pattern")):
                env = {k:v for k,v in os.environ.items() if not k.startswith("ANE_STATIC_")}
                env.update(ANE_STATIC_LOAD_OPTIONS=str(options), ANE_STATIC_MODEL_KEY="net", ANE_STATIC_QOS="25")
                if route == "client-mil":
                    env["ANE_STATIC_COMPILE_TYPE"] = model_type
                directory = a.output / (route + "-" + mode)
                command = [str(helper), str(path), str(directory)]
                if mode != "constant":
                    command.append("--load-only" if mode == "load-only" else mode)
                with directory.with_suffix(".log").open("w") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                row = json.loads((directory / "report.json").read_text())
                case = dict(route=route, mode=mode, returncode=result.returncode,
                            status=row["status"], loaded=row["loaded"], executed=row.get("executed", False),
                            values_checked=len(row.get("output_values", [])), load_error=row.get("load_error_details"),
                            report=str((directory / "report.json").relative_to(a.output)))
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raw = [c for c in report["cases"] if c["route"] == "offline-hwx"]
        mil = [c for c in report["cases"] if c["route"] == "client-mil"]
        system = [c for c in report["cases"] if c["route"] == "system-control"]
        report["mil_numerical_pass"] = all(c["status"] == "pass" and c["returncode"] == 0 for c in mil)
        report["raw_hwx_numerical_pass"] = all(c["status"] == "pass" and c["returncode"] == 0 for c in raw)
        report["system_load_pass"] = all(c["status"] == "load_pass" and c["returncode"] == 0 for c in system)
        report["status"] = "comparison_completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        for lock in reversed(locks):
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        report["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["mil_numerical_pass"] or not report["system_load_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
