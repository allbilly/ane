#!/usr/bin/env python3
"""Measure warmed Asahi decode using the saved macOS Orion token traces."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--backend", choices=("ane", "cpu"), default="ane")
    cli.add_argument("--trials", type=int, default=4)
    args = cli.parse_args()
    if args.output.exists() or args.trials < 1:
        cli.error("output must be a new path and trials must be positive")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import numpy as np
    from bpe import Tokenizer
    from checks import ane_parity, cpu_parity, digest, hybrid_parity, integrity
    from external_weights import find_weights, load_weights, verify_weights
    from hwx import require
    from model import ANEKernels, CPUKernels, GPT2
    from packing import PackedAssets
    from replay import Device

    integrity(ROOT)
    reference_path = ROOT / "provenance/orion-performance/m1-contexts.json"
    reference = json.loads(reference_path.read_text())["reference"]
    source = find_weights()
    require(source is not None, "download the checkpoint with first-run.sh first")
    verify_weights(source, ROOT)
    weights = load_weights(source)
    tokenizer = Tokenizer(ROOT / "tokenizer")
    report = dict(
        format_version=1, date=time.strftime("%Y-%m-%d"), backend=args.backend,
        host=dict(platform=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, openblas_num_threads=os.environ["OPENBLAS_NUM_THREADS"]),
        reference=str(reference_path.relative_to(ROOT)), reference_sha256=digest(reference_path),
        checkpoint_revision=reference["checkpoint_revision"], checkpoint_sha256=reference["checkpoint_sha256"],
        configuration=dict(batch_size=1, decode_steps=64, trials=args.trials,
                           warmups_per_trial=2, warmup_decode_steps=16,
                           engine_timer="Synchronous model.step returning full-vocabulary logits; includes embeddings, CPU attention, ANE dispatch/transfers and internal finite checks. Excludes prefill, allocation, input lookup, argmax, printing and diagnostic replays.",
                           prompt_processing="Sequential decode kernels; differs from native macOS Orion bucketed prefill"),
        runtime_source_sha256={name: digest(ROOT / name) for name in
                               ("gpt2.py", "model.py", "replay.py", "packing.py", "tools/bench_asahi.py")},
        limitations=["Active desktop; no resource isolation or cross-OS paired session.",
                     "Token-choice and greedy checks cover these traces; HF KL and full-logit parity were not measured.",
                     "Graph/frontend and prompt-processing implementations differ from native macOS Orion."],
        cases=[],
    )
    device = Device(ROOT) if args.backend == "ane" else None
    with device if device else nullcontext():
        if device:
            device.assets = PackedAssets(ROOT, weights)
        model = GPT2(weights, ANEKernels(device) if device else CPUKernels(weights))
        if device:
            report["device"] = str(device.path)
            report["decode_kernel_checks_passed"] = len(ane_parity(ROOT, device))
            hybrid_parity(ROOT, model, tokenizer)
            report["full_generation_parity"] = "PASS"
        else:
            report["cpu_reference_prompts_passed"] = len(cpu_parity(ROOT, model, tokenizer))

        def ingest(case):
            model.reset()
            for token in case["prompt_ids"]:
                logits = model.step(token)
            return logits

        for case in reference["cases"]:
            require(tokenizer.encode(case["prompt"]) == case["prompt_ids"], "reference tokenizer mismatch")
            require(len(case["next_tokens"]) == 65, "expected 65 reference predictions")
            trials = []
            for trial in range(args.trials):
                for _ in range(2):
                    ingest(case)
                    for token in case["next_tokens"][:16]:
                        model.step(token)
                ingest(case)
                samples = []
                for token in case["next_tokens"][:64]:
                    started = time.perf_counter_ns()
                    model.step(token)
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                trials.append(dict(trial=trial + 1, raw_decode_ms=samples,
                                   decode_engine_steps_per_second=64000 / sum(samples)))
            # Numerical diagnostics are kept outside timed trials.
            logits = ingest(case)
            predictions = [int(logits.argmax())]
            for token in case["next_tokens"][:64]:
                predictions.append(int(model.step(token).argmax()))
            mismatches = sum(actual != expected for actual, expected in zip(predictions, case["next_tokens"]))
            greedy = list(model.generate(case["prompt_ids"], 16))
            require(mismatches == 0 and greedy == case["next_tokens"][:16], "HF token-choice/greedy check failed")
            samples = [ms for trial in trials for ms in trial["raw_decode_ms"]]
            result = dict(prompt=case["prompt"], prompt_tokens=len(case["prompt_ids"]),
                          prompt_ids=case["prompt_ids"], token_trace=case["next_tokens"][:64],
                          trials=trials, decode_engine_steps_per_second=len(samples) * 1000 / sum(samples),
                          decode_mean_ms=float(np.mean(samples)), decode_p50_ms=float(np.percentile(samples, 50)),
                          decode_p90_ms=float(np.percentile(samples, 90)), hf_top1_mismatches=mismatches,
                          hf_greedy_16_token_match=True)
            report["cases"].append(result)
            print(f"{result['prompt_tokens']} prompt tokens: {result['decode_engine_steps_per_second']:.2f} decode steps/s; HF choices and greedy PASS", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
