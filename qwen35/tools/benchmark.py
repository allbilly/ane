"""Matched full-prompt prefill and generated-token decode on CPU/ANE."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from qwen35.__main__ import prompt_tokens
from qwen35.model import Model
from qwen35.weights import DEFAULT_MODEL, REVISION, sha256

PROTOCOL = "qwen35-prefill-decode/v2"
PROMPTS = ["What is 2 + 2? Answer briefly.", "Write one sentence about the Moon.",
           "Write a Python function that adds two numbers.", "Name the capital of France."]


def cases(directory, config, lengths):
    records = []
    for i, prompt in enumerate(PROMPTS):
        _, tokens, rendered = prompt_tokens(directory, config, prompt)
        records.append(dict(id=f"chat-{i}", prompt=prompt, tokens=tokens, rendered=rendered))
    for length in lengths:
        text = ("Read these notes.\n" +
                "The Moon orbits Earth and reflects sunlight. Its surface has craters and plains.\n" * length +
                "Summarize the notes in one sentence.")
        tokenizer, tokens, _ = prompt_tokens(directory, config, text)
        # Crop the user-content interior while preserving the chat prefix and
        # final user instruction / assistant generation suffix. Time actual IDs.
        tokens = tokens[:length - 32] + tokens[-32:]
        records.append(dict(id=f"context-{length}", prompt="Repeated Moon notes; summarize in one sentence.",
                            tokens=tokens, rendered=tokenizer.decode(tokens, skip_special_tokens=False),
                            construction="Templated chat; crop user-content interior to the exact token count."))
    return records


def prefill(model, tokens):
    for token in tokens[:-1]:
        model.step(token, logits=False)
    return model.step(tokens[-1])


def compare(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("invalid logits or incompatible reference shape")
    # Accumulate norms in float64; comparisons run outside all engine timers.
    difference = actual.astype(np.float64) - expected
    reference = expected.astype(np.float64)
    errors = np.sqrt(np.sum(difference * difference, axis=-1) /
                     np.maximum(np.sum(reference * reference, axis=-1), 1e-40))
    matches = actual.argmax(axis=-1) == expected.argmax(axis=-1)
    return dict(max_normalized_rmse=float(errors.max()), argmax_matches=int(matches.sum()),
                predictions=len(matches))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--kernels", choices=("native", "dot"), default="native")
    p.add_argument("--backend", choices=("cpu", "ane"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, default=32, help="Generated-token decode calls, after full prefill")
    p.add_argument("--lengths", type=int, nargs="*", default=[128, 512],
                   help="Additional exact prompt lengths, alongside four short chats")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--prepare-traces", action="store_true", help="Create CPU floating references, without timing")
    a = p.parse_args()
    if a.threads < 1 or not 1 <= a.steps <= 262144 - 23 or any(n < 64 or n + a.steps > 262144 for n in a.lengths):
        p.error("threads/steps must be positive; additional lengths must be >=64 and fit the context")
    if not a.prepare_traces and a.output is None:
        p.error("timed runs require --output")
    if a.prepare_traces and (a.kernels != "native" or a.backend != "cpu"):
        p.error("prepare references with CPU native")
    traces = a.traces / "tokens.json"
    if not a.prepare_traces and not traces.exists():
        p.error("create references with --prepare-traces first")
    if a.prepare_traces and traces.exists():
        p.error("references already exist; choose a new trace directory")
    checkpoint_hash = sha256(a.model / "model.safetensors")
    config = json.loads((a.model / "config.json").read_text())
    records = cases(a.model, config, a.lengths)
    context = max(len(r["tokens"]) for r in records) + a.steps
    model = Model(a.model, kernels=a.kernels, threads=a.threads, context=context)
    if a.backend == "ane":
        from qwen35.ane import Ane
        model.backend = Ane()
        for layer in model.layers:
            for name in ("proj", "out", "up", "down"):
                model.backend.prepare(layer[name])
    try:
        if a.prepare_traces:
            a.traces.mkdir(parents=True, exist_ok=True)
            for i, record in enumerate(records):
                model.reset()
                y = prefill(model, record["tokens"])
                inputs, outputs, logits = [], [int(y.argmax())], [y]
                for _ in range(a.steps):
                    inputs.append(outputs[-1])
                    y = model.step(inputs[-1])
                    logits.append(y)
                    outputs.append(int(y.argmax()))
                np.save(a.traces / f"logits-{i}.npy", np.array(logits))
                record.update(inputs=inputs, predictions=outputs)
                print(json.dumps(dict(reference=record["id"], prompt_tokens=len(record["tokens"]),
                                      decode_steps=len(inputs))), flush=True)
            metadata = dict(protocol=PROTOCOL, revision=REVISION, model_sha256=checkpoint_hash,
                            steps=a.steps, lengths=a.lengths, records=records,
                            logit_sha256=[sha256(a.traces / f"logits-{i}.npy") for i in range(len(records))])
            traces.write_text(json.dumps(metadata, indent=2) + "\n")
            return
        metadata = json.loads(traces.read_text())
        if (metadata["protocol"] != PROTOCOL or metadata["revision"] != REVISION
                or metadata["model_sha256"] != checkpoint_hash or metadata["steps"] != a.steps
                or metadata["lengths"] != a.lengths
                or [r["tokens"] for r in metadata["records"]] != [r["tokens"] for r in records]):
            raise ValueError("saved references do not match this benchmark configuration")
        records = metadata["records"]
        references = []
        for i, record in enumerate(records):
            path = a.traces / f"logits-{i}.npy"
            if sha256(path) != metadata["logit_sha256"][i]:
                raise ValueError(f"reference checksum mismatch: {path.name}")
            expected = np.load(path)
            if len(record["inputs"]) != a.steps or expected.shape != (a.steps + 1, 248320):
                raise ValueError("saved reference length differs from --steps")
            references.append(expected)

        # Exercise prefill, the head and every decode kernel before measuring.
        prefill(model, records[0]["tokens"])
        for token in records[0]["inputs"]:
            model.step(token)
        results = []
        submissions = model.backend.submissions if model.backend else 0
        for record, expected in zip(records, references):
            model.reset()
            model.timings.clear()
            before = model.backend.submissions if model.backend else 0
            start = time.perf_counter()
            first_logits = prefill(model, record["tokens"])
            prefill_seconds = time.perf_counter() - start
            first_prediction = int(first_logits.argmax())
            warm_ttft_seconds = time.perf_counter() - start
            prefill_components = dict(model.timings)
            prefill_submissions = (model.backend.submissions - before) if model.backend else 0
            model.timings.clear()
            durations, actual = [], [first_logits]
            before = model.backend.submissions if model.backend else 0
            for token in record["inputs"]:
                start = time.perf_counter()
                y = model.step(token)
                durations.append(time.perf_counter() - start)
                actual.append(y)
            decode_components = dict(model.timings)
            decode_submissions = (model.backend.submissions - before) if model.backend else 0
            if model.position != len(record["tokens"]) + a.steps:
                raise RuntimeError("prefill/decode token boundary check failed")
            if model.backend and (prefill_submissions != 96 * len(record["tokens"])
                                  or decode_submissions != 96 * a.steps):
                raise RuntimeError("ANE submission count differs from all 96 projections per token")
            actual = np.array(actual)
            result = dict(id=record["id"], prompt=record["prompt"], prompt_tokens=len(record["tokens"]),
                          prefill_seconds=prefill_seconds,
                          prefill_tokens_per_second=len(record["tokens"]) / prefill_seconds,
                          warm_ttft_seconds=warm_ttft_seconds, first_prediction=first_prediction,
                          prefill_accuracy=compare(actual[:1], expected[:1]),
                          prefill_components_seconds=prefill_components, prefill_ane_submissions=prefill_submissions,
                          decode_steps=a.steps, decode_seconds=durations,
                          decode_steps_per_second=len(durations) / sum(durations),
                          decode_accuracy=compare(actual[1:], expected[1:]),
                          decode_components_seconds=decode_components, decode_ane_submissions=decode_submissions)
            results.append(result)
            print(json.dumps(result), flush=True)
        result = dict(protocol=PROTOCOL, revision=REVISION, model_sha256=checkpoint_hash,
                      trace_sha256=sha256(traces), backend=a.backend, kernels=a.kernels, threads=a.threads,
                      context=context, timed_steps=a.steps * len(results), results=results,
                      ane_submissions=model.backend.submissions - submissions if model.backend else 0)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        if a.backend == "cpu" and any(check["max_normalized_rmse"] > .005
                                      or check["argmax_matches"] != check["predictions"]
                                      for r in results for check in (r["prefill_accuracy"], r["decode_accuracy"])):
            raise RuntimeError("CPU numerical gate failed")
    finally:
        if model.backend:
            model.backend.close()


if __name__ == "__main__":
    main()
