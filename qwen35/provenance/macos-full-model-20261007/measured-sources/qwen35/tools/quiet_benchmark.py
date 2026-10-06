"""Repeat the existing CPU protocol with host observations and quiet preflights."""
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


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def observe(excluded=()):
    errors = {}
    try:
        text = subprocess.check_output(["ps", "-Ao", "pid,ppid,pcpu,rss,comm", "-r"], text=True, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        errors["processes"] = str(error)
        text = ""
    processes = []
    for line in text.splitlines()[1:]:
        values = line.strip().split(None, 4)
        if len(values) == 5:
            processes.append(dict(pid=int(values[0]), ppid=int(values[1]), cpu=float(values[2]),
                                  rss_kib=int(values[3]), executable=values[4]))
    own = {os.getpid(), *excluded}
    for _ in range(8):
        own.update(p["pid"] for p in processes if p["ppid"] in own)
    external = [p for p in processes if p["pid"] not in own]
    try:
        vm = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        errors["vm"] = str(error)
        vm = ""
    counters = {}
    for line in vm.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if value.strip().rstrip(".").isdigit(): counters[key] = int(value.strip().rstrip("."))
    try: load = list(os.getloadavg())
    except OSError as error:
        errors["loadavg"] = str(error)
        load = None
    try: swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        errors["swap"] = str(error)
        swap = None
    return dict(utc=utc(), loadavg=load, external_cpu_percent=sum(p["cpu"] for p in external) if text else None,
                top_external=external[:8], vm=counters,
                swap=swap, observation_errors=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--preflight-seconds", type=int, default=600)
    p.add_argument("--allow-missing-host-observations", action="store_true",
                   help="Run numerical repeats despite unavailable host queries; flag all affected timings")
    a = p.parse_args()
    if a.rounds < 1 or a.preflight_seconds < 1:
        p.error("rounds and preflight-seconds must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(started_utc=utc(), platform=platform.platform(), protocol="qwen35-prefill-decode/v2",
                  allow_missing_host_observations=a.allow_missing_host_observations,
                  predeclared_host_gates=dict(preflight_load_1m=4., preflight_external_cpu_percent=100.,
                                             affected_external_cpu_percent=150., affected_swapout_pages=0),
                  tasks=[], limitations=["Ordinary desktop applications remain active; shared lock participation is voluntary.",
                                         "Host observations cover process setup and numerical checks as well as engine timing.",
                                         "Two phases on one boot are not independent days or a controlled OS comparison."])
    env = dict(os.environ, CC="/opt/homebrew/bin/gcc-16", OPENBLAS_NUM_THREADS="1",
               OMP_WAIT_POLICY="PASSIVE", VECLIB_MAXIMUM_THREADS="1")
    lock_path = Path.home() / "gpu.lock"
    lock = lock_path.open("r" if lock_path.exists() else "a")
    configs = [(workers, kernel) for workers in (1, 2, 4) for kernel in ("native", "dot")]
    try:
        # Keep lock timeouts inside the receipt-writing failure path.
        deadline = time.monotonic() + a.preflight_seconds
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline: raise RuntimeError("shared gpu.lock remained busy")
                time.sleep(5)
        report["shared_lock"] = str(lock_path)
        for round_index in range(a.rounds):
            offset = (round_index * 2) % len(configs)
            order = configs[offset:] + configs[:offset]
            if round_index % 2: order.reverse()
            for workers, kernel in order:
                deadline = time.monotonic() + a.preflight_seconds
                preflights = []
                consecutive = 0
                while consecutive < 3:
                    sample = observe()
                    preflights.append(sample)
                    if sample["observation_errors"] and not a.allow_missing_host_observations:
                        raise RuntimeError("host observations unavailable; explicit affected-repeat flag required")
                    load_ok = sample["loadavg"] is None or sample["loadavg"][0] <= 4.
                    cpu_ok = sample["external_cpu_percent"] is None or sample["external_cpu_percent"] <= 100.
                    consecutive = consecutive + 1 if load_ok and cpu_ok else 0
                    if time.monotonic() >= deadline:
                        (a.output / "preflight-timeout.json").write_text(json.dumps(preflights, indent=2) + "\n")
                        raise RuntimeError("quiet preflight not achieved; observations retained")
                    time.sleep(2)
                label = f"{kernel}-t{workers}-r{round_index + 1}"
                command = [sys.executable, "-m", "qwen35.tools.benchmark", "--model", str(a.model),
                           "--traces", str(a.traces), "--kernels", kernel, "--threads", str(workers),
                           "--output", str(a.output / (label + ".json"))]
                task = dict(name=label, command=command, started_utc=utc(), preflights=preflights)
                task["preflight_observations_complete"] = not any(s["observation_errors"] for s in preflights)
                # Bracket the entire child lifetime, including the tail between
                # the last periodic sample and process exit.
                samples = [observe()]
                stop = threading.Event()
                with (a.output / (label + ".log")).open("w") as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    def sample_host():
                        while not stop.is_set():
                            samples.append(observe((child.pid,)))
                            stop.wait(5)
                    thread = threading.Thread(target=sample_host, daemon=True)
                    thread.start()
                    returncode = child.wait()
                    stop.set()
                    thread.join()
                samples.append(observe((child.pid,)))
                thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
                task.update(ended_utc=utc(), returncode=returncode, host_samples=samples,
                            thermal=thermal.stdout + thermal.stderr, thermal_returncode=thermal.returncode)
                swapouts = max(0, samples[-1]["vm"].get("Swapouts", 0) - samples[0]["vm"].get("Swapouts", 0)) if samples else None
                task["observed_swapout_pages"] = swapouts
                task["affected"] = bool(swapouts or not task["preflight_observations_complete"] or thermal.returncode
                                        or any(s["observation_errors"] or (s["external_cpu_percent"] is not None and s["external_cpu_percent"] > 150.) for s in samples))
                report["tasks"].append(task)
                (a.output / "session.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(dict(name=label, returncode=returncode, affected=task["affected"],
                                      external_cpu_max=max((s["external_cpu_percent"] for s in samples if s["external_cpu_percent"] is not None), default=None))), flush=True)
                if returncode: raise RuntimeError(f"numerical benchmark failed: {label}")
        # Retain every run in the aggregate, including any marked affected.
        from qwen35.tools import summarize_benchmark
        summaries = {}
        for workers, kernel in configs:
            paths = [a.output / f"{kernel}-t{workers}-r{r + 1}.json" for r in range(a.rounds)]
            reports = [json.loads(path.read_text()) for path in paths]
            records = [record for receipt in reports for record in receipt["results"]]
            summaries[f"{kernel}-t{workers}"] = dict(rounds=a.rounds, overall=summarize_benchmark.summarize(records))
        report.update(ended_utc=utc(), summary=summaries, affected_runs=sum(t["affected"] for t in report["tasks"]), status="pass")
    except Exception as error:
        report.update(ended_utc=utc(), status="incomplete", error=str(error))
        raise
    finally:
        (a.output / "session.json").write_text(json.dumps(report, indent=2) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
