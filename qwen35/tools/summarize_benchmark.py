"""Seal matched prefill/decode receipts and print a throughput table."""
import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path

from qwen35.tools.benchmark import PROTOCOL
from qwen35.weights import sha256


def summarize(records):
    prompt_tokens = sum(r["prompt_tokens"] for r in records)
    prefill_seconds = sum(r["prefill_seconds"] for r in records)
    ttft = [r["warm_ttft_seconds"] for r in records]
    latencies = [t for r in records for t in r["decode_seconds"]]
    components = {}
    for phase, denominator in (("prefill", prompt_tokens), ("decode", len(latencies))):
        names = set().union(*(r[f"{phase}_components_seconds"] for r in records))
        components[phase] = {name: 1000 * sum(r[f"{phase}_components_seconds"].get(name, 0)
                                            for r in records) / denominator for name in sorted(names)}
    accuracy = {}
    for phase in ("prefill", "decode"):
        checks = [r[f"{phase}_accuracy"] for r in records]
        accuracy[phase] = dict(predictions=sum(c["predictions"] for c in checks),
                               argmax_matches=sum(c["argmax_matches"] for c in checks),
                               max_normalized_rmse=max(c["max_normalized_rmse"] for c in checks))
    return dict(requests=len(records), prefill_tokens=prompt_tokens, prefill_seconds=prefill_seconds,
                prefill_tokens_per_second=prompt_tokens / prefill_seconds,
                mean_warm_ttft_seconds=statistics.mean(ttft), warm_ttft_range_seconds=[min(ttft), max(ttft)],
                prefill_rate_range=[min(r["prefill_tokens_per_second"] for r in records),
                                    max(r["prefill_tokens_per_second"] for r in records)],
                decode_steps=len(latencies), decode_seconds=sum(latencies),
                decode_steps_per_second=len(latencies) / sum(latencies),
                decode_rate_range=[min(r["decode_steps_per_second"] for r in records),
                                   max(r["decode_steps_per_second"] for r in records)],
                median_decode_ms=1000 * statistics.median(latencies),
                components_ms_per_token=components, accuracy=accuracy,
                prefill_ane_submissions=sum(r["prefill_ane_submissions"] for r in records),
                decode_ane_submissions=sum(r["decode_ane_submissions"] for r in records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--capture", type=Path, nargs="+", help="Coreglass captures, including any retried attempts")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    runs = {path.name: json.loads(path.read_text()) for path in a.runs}
    if len(runs) != len(a.runs):
        raise ValueError("receipt filenames must be unique")
    reference = next(iter(runs.values()))
    fields = ("protocol", "revision", "model_sha256", "trace_sha256", "threads", "context")
    if reference["protocol"] != PROTOCOL:
        raise ValueError("unsupported timing protocol")
    traces = a.traces / "tokens.json"
    if sha256(traces) != reference["trace_sha256"]:
        raise ValueError("trace metadata checksum differs from measured receipts")
    trace_metadata = json.loads(traces.read_text())
    expected_ids = [r["id"] for r in trace_metadata["records"]]
    grouped = {}
    for run in runs.values():
        if any(run[field] != reference[field] for field in fields):
            raise ValueError("receipts do not share the same model, traces and thread settings")
        if [r["id"] for r in run["results"]] != expected_ids:
            raise ValueError("receipt workload differs from saved trace metadata")
        for r, saved in zip(run["results"], trace_metadata["records"]):
            if (r["prompt_tokens"] != len(saved["tokens"])
                    or r["decode_steps"] != trace_metadata["steps"]
                    or len(r["decode_seconds"]) != r["decode_steps"]
                    or r["prefill_accuracy"]["predictions"] != 1
                    or r["decode_accuracy"]["predictions"] != r["decode_steps"]):
                raise ValueError("receipt token counts differ from the timing contract")
        if run["timed_steps"] != sum(r["decode_steps"] for r in run["results"]):
            raise ValueError("receipt timed step count differs from its records")
        if run["backend"] == "cpu" or run.get("ane_mode") == "accurate":
            if any(check["max_normalized_rmse"] > .005
                   or check["argmax_matches"] != check["predictions"]
                   for r in run["results"]
                   for check in (r["prefill_accuracy"], r["decode_accuracy"])):
                raise ValueError("a measured receipt failed its numerical gate")
        if run["backend"] == "ane":
            for r in run["results"]:
                if (r["prefill_ane_submissions"] != 96 * r["prompt_tokens"]
                        or r["decode_ane_submissions"] != 96 * r["decode_steps"]):
                    raise ValueError("ANE receipt did not execute all 96 body projections per token")
        path = f'{run["backend"]}-{run["kernels"]}'
        if run.get("ane_mode"):
            path += f'-{run["ane_mode"]}'
        grouped.setdefault(path, []).append(run)
    counts = {len(items) for items in grouped.values()}
    if len(counts) != 1:
        raise ValueError("paths must have an equal number of repeated runs")
    summary = {}
    for path, items in grouped.items():
        records = [r for run in items for r in run["results"]]
        groups = {"short-chat": [r for r in records if r["id"].startswith("chat-")]}
        groups.update({name: [r for r in records if r["id"] == name]
                       for name in expected_ids if name.startswith("context-")})
        summary[path] = dict(rounds=len(items), overall=summarize(records),
                             groups={name: summarize(rows) for name, rows in groups.items()})
    root = Path(__file__).resolve().parents[2]
    sources = ["qwen35/tools/benchmark.py", "qwen35/tools/summarize_benchmark.py",
               "qwen35/model.py", "qwen35/native.py", "qwen35/cpu.c", "qwen35/weights.py",
               "qwen35/ane.py", "qwen35/ane_matmul.c", "qwen35/linear_template.h"]
    result = dict(protocol=PROTOCOL, hardware="base M1 MacBook Air, 8 GB, Asahi Linux",
                  kernel=platform.release(), runtime_source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  source_sha256={path: sha256(root / path) for path in sources},
                  model_revision=reference["revision"], model_sha256=reference["model_sha256"],
                  policy=dict(cpu_workers=reference["threads"], openblas_threads=1,
                              omp_wait_policy="PASSIVE", order=list(runs)),
                  timing_scope="Full warm prefill through vocabulary head; warm TTFT also includes first argmax. "
                               "Decode calls ingest saved generated tokens, never prompt tokens. "
                               "Exclude loading, preparation, tokenization, reset and logit comparison; continue beyond EOS.",
                  limitations=["Active desktop; shared hardware locks serialize workloads. Setup from queued jobs may overlap.",
                               "CPU timing varied considerably; paging was observed. No isolated sustained throughput claim.",
                               "Sequential prompt ingestion on all paths; no batched prefill kernel.",
                               "ANE FP16 body is experimental; numerical drift is measured separately.",
                               "GPU/ANE busy counters and measured DRAM bandwidth unavailable."],
                  traces=trace_metadata, trace_sha256=sha256(traces), summary=summary, runs=runs,
                  receipt_sha256={path.name: sha256(path) for path in a.runs})
    if a.capture:
        captures, successful, failed = [], set(), []
        for capture in a.capture:
            manifest = capture.with_suffix(".run.json")
            receipt = json.loads(manifest.read_text())
            captures.append(dict(capture_file=capture.name, capture_sha256=sha256(capture),
                                 manifest_sha256=sha256(manifest), commit=receipt.get("coreglass_commit"),
                                 samples=receipt["samples"], seconds=receipt["seconds"],
                                 load_gate_overridden=bool(receipt.get("forced_over"))))
            for step in receipt["steps"]:
                if step["rc"] == 0:
                    successful.add(step["label"])
                else:
                    failed.append(dict(capture_file=capture.name, label=step["label"], rc=step["rc"]))
        if any(path.stem not in successful for path in a.runs):
            raise ValueError("a measured receipt has no successful Coreglass workload step")
        result["coreglass"] = dict(captures=captures, failed_attempts=failed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("| Path | Prompt | Prefill tok/s | Warm TTFT s | Decode tok/s |")
    print("| --- | --- | ---: | ---: | ---: |")
    for path, entry in summary.items():
        for group, metrics in entry["groups"].items():
            print(f'| {path} | {group} | {metrics["prefill_tokens_per_second"]:.2f} | '
                  f'{metrics["mean_warm_ttft_seconds"]:.3f} | {metrics["decode_steps_per_second"]:.2f} |')


if __name__ == "__main__":
    main()
