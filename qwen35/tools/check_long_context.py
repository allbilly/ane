"""Prepare portable long traces and compare independent greedy CPU streams."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from qwen35.__main__ import prompt_tokens
from qwen35.model import Model
from qwen35.tools.benchmark import compare


GREEDY_PROMPTS = ["What is 17 times 23? Explain briefly.",
                  "Write a Python function that reverses a list without changing the input.",
                  "用中文简单解释月亮为什么会发光。",
                  "Give two practical ways to reduce water use at home."]


def check_greedy(model, output, threads=4, steps=64):
    """Compare independently evolving histories; never feed reference tokens to SDOT."""
    floating = Model(model, kernels="native", threads=threads, context=512)
    integer = Model(model, kernels="dot", threads=threads, context=512)
    records = []
    for index, prompt in enumerate(GREEDY_PROMPTS):
        tokenizer, tokens, rendered = prompt_tokens(model, floating.config, prompt)
        if len(tokens) + steps > floating.context:
            raise ValueError("greedy prompt and decode steps exceed context")
        floating.reset()
        integer.reset()
        for token in tokens[:-1]:
            floating.step(token, logits=False)
            integer.step(token, logits=False)
        expected, actual = floating.step(tokens[-1]), integer.step(tokens[-1])
        float_tokens, dot_tokens, checks = [], [], []
        first_mismatch = None
        for step in range(steps):
            ft, dt = int(expected.argmax()), int(actual.argmax())
            float_tokens.append(ft)
            dot_tokens.append(dt)
            if first_mismatch is None:
                check = compare(actual[None], expected[None])
                checks.append(check)
                if ft != dt:
                    first_mismatch = step
                    np.savez_compressed(output / f"first-greedy-mismatch-{index}.npz", expected=expected, actual=actual)
            if step + 1 < steps:
                expected, actual = floating.step(ft), integer.step(dt)
        record = dict(prompt=prompt, rendered=rendered, prompt_ids=tokens,
                      floating_tokens=float_tokens, dot_tokens=dot_tokens,
                      floating_text=tokenizer.decode(float_tokens), dot_text=tokenizer.decode(dot_tokens),
                      first_mismatch=first_mismatch, tokens_match=float_tokens == dot_tokens,
                      max_comparable_nrmse=max(c["max_normalized_rmse"] for c in checks),
                      compared_before_histories_diverge=len(checks),
                      continuation="Fixed steps beyond EOS for numerical coverage; not application throughput")
        records.append(record)
        print(json.dumps({k: record[k] for k in ("prompt", "tokens_match", "first_mismatch", "max_comparable_nrmse")}), flush=True)
    report = dict(steps=steps, records=records, require_all_argmax_matches=True, full_logit_nrmse_limit=.005,
                  status="pass" if all(r["tokens_match"] and r["max_comparable_nrmse"] <= .005 for r in records) else "fail")
    (output / "greedy.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--greedy-only", action="store_true", help="Run independent histories without preparing long traces")
    a = p.parse_args()
    if a.threads < 1 or a.steps < 1:
        p.error("threads and steps must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    if a.greedy_only:
        report = check_greedy(a.model, a.output, a.threads, a.steps)
        if report["status"] != "pass": raise RuntimeError("free greedy numerical gate failed; captures retained")
        return
    traces = a.output / "traces"
    base = [sys.executable, "-m", "qwen35.tools.benchmark", "--model", str(a.model),
            "--traces", str(traces), "--lengths", "1024", "2048", "--steps", str(a.steps),
            "--threads", str(a.threads)]
    tasks = []
    for name, extra in [("prepare", ["--prepare-traces"]),
                        ("dot", ["--kernels", "dot", "--output", str(a.output / "dot.json")])]:
        command = base + extra
        with (a.output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        tasks.append(dict(name=name, command=command, returncode=result.returncode))
        (a.output / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
        print(json.dumps(tasks[-1]), flush=True)
        if result.returncode and name == "prepare": raise RuntimeError("long-context reference preparation failed")
    report = check_greedy(a.model, a.output, a.threads, a.steps)
    # A Linux hardware invocation is packaged, not reported as executed on Mac.
    command = ["qwen35/.venv/bin/python", "-m", "qwen35.tools.benchmark", "--model", "MODEL_DIR",
               "--traces", "TRACE_DIR", "--lengths", "1024", "2048", "--steps", str(a.steps),
               "--threads", "4", "--kernels", "dot", "--backend", "ane", "--ane-mode", "accurate",
               "--output", "NEW_OUTPUT.json"]
    (a.output / "asahi-replay.json").write_text(json.dumps(dict(command=command, status="requires native Asahi ANE",
                                                               trace_directory=str(traces), full_logit_nrmse_limit=.005), indent=2) + "\n")
    combined = dict(tasks=tasks, greedy_status=report["status"],
                    status="pass" if report["status"] == "pass" and all(t["returncode"] == 0 for t in tasks) else "fail")
    (a.output / "result.json").write_text(json.dumps(combined, indent=2) + "\n")
    if combined["status"] != "pass": raise RuntimeError("long-context or free greedy numerical gate failed; captures retained")


if __name__ == "__main__":
    main()
