"""Matched teacher-forced CPU/ANE timings on four actual chat prompts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from qwen35.__main__ import prompt_tokens
from qwen35.model import Model
from qwen35.weights import DEFAULT_MODEL, REVISION

PROMPTS = ["What is 2 + 2? Answer briefly.", "Write one sentence about the Moon.",
           "Write a Python function that adds two numbers.", "Name the capital of France."]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--kernels", choices=("native", "dot"), default="native")
    p.add_argument("--backend", choices=("cpu", "ane"), default="cpu")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model = Model(a.model, kernels=a.kernels, context=512)
    if a.backend == "ane":
        from qwen35.ane import Ane
        model.backend = Ane()
        for layer in model.layers:
            for name in ("proj", "out", "up", "down"):
                model.backend.prepare(layer[name])
    tokenizer, _, _ = prompt_tokens(a.model, model.config, PROMPTS[0])
    a.traces.mkdir(parents=True, exist_ok=True)
    traces = a.traces / "tokens.json"
    if not traces.exists():
        if a.kernels != "native" or a.backend != "cpu":
            raise ValueError("create the baseline traces with CPU native first")
        records = []
        for i, prompt in enumerate(PROMPTS):
            _, tokens, _ = prompt_tokens(a.model, model.config, prompt)
            model.reset()
            for token in tokens[:-1]:
                model.step(token, logits=False)
            inputs, outputs, logits = [tokens[-1]], [], []
            for step in range(a.steps):
                y = model.step(inputs[-1])
                logits.append(y)
                outputs.append(int(y.argmax()))
                if step + 1 < a.steps:
                    inputs.append(outputs[-1])
            np.save(a.traces / f"logits-{i}.npy", np.array(logits))
            records.append(dict(prompt=prompt, tokens=tokens, inputs=inputs, predictions=outputs,
                                text=tokenizer.decode(outputs)))
        traces.write_text(json.dumps(records, indent=2) + "\n")
    records = json.loads(traces.read_text())
    model.step(records[0]["tokens"][0])
    results = []
    submissions = model.backend.submissions if model.backend else 0
    for i, record in enumerate(records):
        model.reset()
        model.timings.clear()
        start = time.perf_counter()
        for token in record["tokens"][:-1]:
            model.step(token, logits=False)
        prefill = time.perf_counter() - start
        model.timings.clear()
        expected = np.load(a.traces / f"logits-{i}.npy")
        if len(record["inputs"]) != a.steps:
            raise ValueError("saved trace length differs from --steps")
        durations, errors, matches = [], [], []
        for step, token in enumerate(record["inputs"]):
            start = time.perf_counter()
            y = model.step(token)
            durations.append(time.perf_counter() - start)
            ref = expected[step]
            error = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-20))
            errors.append(error)
            matches.append(int(y.argmax()) == int(ref.argmax()))
        result = dict(prompt=record["prompt"], prompt_tokens=len(record["tokens"]),
                      prefill_seconds=prefill, decode_seconds=durations,
                      decode_steps_per_second=len(durations) / sum(durations),
                      max_normalized_rmse=max(errors), argmax_matches=sum(matches),
                      components_seconds=dict(model.timings))
        results.append(result)
        print(json.dumps(result), flush=True)
    result = dict(revision=REVISION, backend=a.backend, kernels=a.kernels,
                  timed_steps=a.steps * len(results), results=results,
                  ane_submissions=model.backend.submissions - submissions if model.backend else 0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    if model.backend:
        model.backend.close()
    if a.backend == "cpu" and any(r["max_normalized_rmse"] > .005 or r["argmax_matches"] != a.steps for r in results):
        raise RuntimeError("CPU numerical gate failed")


if __name__ == "__main__":
    main()
