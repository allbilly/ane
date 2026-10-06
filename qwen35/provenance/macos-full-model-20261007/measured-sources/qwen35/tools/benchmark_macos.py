"""Three matched full-model CPU/ANE repeats with complete host observations."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

from qwen35.tools.quiet_benchmark import observe
from qwen35.weights import MODEL_SHA256, REVISION, CONFIG_SHA256, TOKENIZER_SHA256, sha256

ROOT = Path(__file__).resolve().parents[2]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--lock-timeout", type=float, default=600)
    a = p.parse_args()
    if platform.system() != "Darwin" or a.rounds < 3 or a.lock_timeout <= 0:
        p.error("requires macOS, at least three rounds and a positive lock timeout")
    a.model, a.traces, a.output = a.model.resolve(), a.traces.resolve(), a.output.resolve()
    checkpoint = {}
    for name, expected in (("model.safetensors", MODEL_SHA256), ("config.json", CONFIG_SHA256), ("tokenizer.json", TOKENIZER_SHA256)):
        actual = sha256(a.model / name)
        if actual != expected:
            raise ValueError("pinned checkpoint mismatch: " + name)
        checkpoint[name] = actual
    metadata = json.loads((a.traces / "tokens.json").read_text())
    if (metadata["revision"] != REVISION or metadata["model_sha256"] != MODEL_SHA256
            or metadata["steps"] != 64 or metadata["lengths"] != [1024, 2048]
            or [len(r["tokens"]) for r in metadata["records"]] != [23, 19, 21, 18, 1024, 2048]):
        raise ValueError("requires the matched Asahi six-prompt/64-decode protocol")
    a.output.mkdir(parents=True, exist_ok=False)
    sources = ["qwen35/model.py", "qwen35/cpu.c", "qwen35/native.py", "qwen35/weights.py",
               "qwen35/ane.py", "qwen35/macos_ane.py", "qwen35/tools/benchmark.py",
               "qwen35/tools/benchmark_macos.py", "qwen35/tools/quiet_benchmark.py"]
    snapshot = a.output / "sources"
    source_hashes = {}
    for name in sources:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
        source_hashes[name] = sha256(destination)
    report = dict(status="running", started_utc=utc(), platform=platform.platform(),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  source_sha256=source_hashes, checkpoint_sha256=checkpoint,
                  protocol="qwen35-prefill-decode/v2", revision=REVISION,
                  traces=str(a.traces), trace_sha256=sha256(a.traces / "tokens.json"),
                  rounds=a.rounds, cpu_workers=4, context=2112, steps=64,
                  full_logit_nrmse_limit=.005, tasks=[],
                  compiler=subprocess.check_output([os.environ.get("CC", "cc"), "--version"], text=True).splitlines()[0],
                  limitations=["Active desktop; shared locks are voluntary and do not isolate other applications.",
                               "E5RT execution counts describe logical projections; internal ANE task dispatches are not measured.",
                               "CPU SDOT uses packed W4; ANE uses FP16 body matrices and CPU recurrence, attention and SDOT vocabulary head."])
    locks = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE", VECLIB_MAXIMUM_THREADS="1",
               QWEN35_MACOS_ANE_CACHE_DIR=str(a.output.parent / "ane-cache"))
    def save():
        (a.output / "session.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        for name in ("ane.lock", "gpu.lock"):
            lock = (Path.home() / name).open("a")
            locks.append(lock)
            deadline = time.monotonic() + a.lock_timeout
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("shared lock remained busy: " + name)
                    time.sleep(2)
        report["locks"] = [str(Path.home() / name) for name in ("ane.lock", "gpu.lock")]
        save()
        for round_index in range(a.rounds):
            order = ("ane", "cpu") if round_index % 2 else ("cpu", "ane")
            for backend in order:
                label = f"{backend}-r{round_index + 1}"
                command = [sys.executable, "-m", "qwen35.tools.benchmark", "--model", str(a.model),
                           "--traces", str(a.traces), "--backend", backend, "--kernels", "dot",
                           "--threads", "4", "--steps", "64", "--lengths", "1024", "2048",
                           "--output", str(a.output / (label + ".json"))]
                task = dict(name=label, backend=backend, command=command, started_utc=utc(), host_samples=[observe()])
                report["tasks"].append(task)
                stop = threading.Event()
                with (a.output / (label + ".log")).open("w") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    def sample():
                        while not stop.is_set():
                            try:
                                task["host_samples"].append(observe((child.pid,)))
                            except Exception as error:
                                task["host_samples"].append(dict(observation_errors=dict(sampler=str(error))))
                            stop.wait(5)
                    thread = threading.Thread(target=sample, daemon=True)
                    thread.start()
                    try:
                        task["returncode"] = child.wait()
                    finally:
                        stop.set()
                        thread.join()
                task["host_samples"].append(observe((child.pid,)))
                thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
                task.update(ended_utc=utc(), thermal=thermal.stdout + thermal.stderr, thermal_returncode=thermal.returncode)
                first, last = task["host_samples"][0], task["host_samples"][-1]
                counters = ("Swapouts", "Pageouts")
                for name in counters:
                    task["observed_" + name.lower()] = max(0, last["vm"].get(name, 0) - first["vm"].get(name, 0))
                task["affected"] = bool(task["observed_swapouts"] or task["observed_pageouts"] or thermal.returncode
                                        or any(s.get("observation_errors") or (s.get("external_cpu_percent") or 0) > 150
                                               for s in task["host_samples"]))
                result_path = a.output / (label + ".json")
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    task["result_sha256"] = sha256(result_path)
                    task["numerical_pass"] = len(result["results"]) == 6 and all(
                        c["max_normalized_rmse"] <= .005 and c["argmax_matches"] == c["predictions"]
                        for r in result["results"] for c in (r["prefill_accuracy"], r["decode_accuracy"]))
                else:
                    task["numerical_pass"] = False
                save()
                print(json.dumps({k:task[k] for k in ("name", "returncode", "numerical_pass", "affected")}), flush=True)
                if task["returncode"] or not task["numerical_pass"]:
                    raise RuntimeError("benchmark failed: " + label)
        report.update(status="pass", ended_utc=utc())
    except BaseException as error:
        report.update(status="failed", ended_utc=utc(), error=str(error))
        raise
    finally:
        save()
        for lock in reversed(locks):
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    main()
