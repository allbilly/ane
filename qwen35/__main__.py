import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from .model import Model
from .weights import DEFAULT_MODEL, MODEL_ID, REVISION, MODEL_SHA256, TOKENIZER_SHA256, CONFIG_SHA256, sha256, download


def prompt_tokens(directory, config, text, raw=False, thinking=False):
    tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    if not raw:
        template = config["token_codec_config"]["prompt_template"]
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def reject(message):
            raise ValueError(message)
        environment.globals["raise_exception"] = reject
        text = environment.from_string(template).render(
            messages=[dict(role="user", content=text)], add_generation_prompt=True,
            enable_thinking=thinking, tools=None)
    return tokenizer, tokenizer.encode(text, add_special_tokens=False).ids, text


def main():
    parser = argparse.ArgumentParser(description="Exact Mirai Qwen3.5-0.8B-M checkpoint")
    parser.add_argument("command", choices=("setup", "inspect", "verify", "generate", "bench"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--kernels", choices=("auto", "native", "numpy", "dot"), default="auto")
    parser.add_argument("--backend", choices=("cpu", "ane"), default="cpu",
                        help="ANE uses experimental FP16 body projections; see README for accuracy results")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-hashes", action="store_true")
    args = parser.parse_args()
    if args.command == "setup":
        download(args.model)
        print(f"Pinned checkpoint ready: {args.model}")
        return
    if args.command == "verify":
        from .verify import verify
        result = verify(args.model)
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.check_hashes:
        for file, expected in (("config.json", CONFIG_SHA256), ("model.safetensors", MODEL_SHA256), ("tokenizer.json", TOKENIZER_SHA256)):
            if sha256(args.model / file) != expected:
                raise ValueError(f"{file}: checksum mismatch for pinned revision")
    start = time.perf_counter()
    model = Model(args.model, args.precision, args.kernels, args.threads, context=args.context)
    load = time.perf_counter() - start
    if args.command == "inspect":
        print(json.dumps(dict(model=MODEL_ID, revision=REVISION, tensors=len(model.tensors.header) - 1,
                              layers=len(model.layers), load_seconds=load), indent=2))
        return
    if args.backend == "ane":
        from .ane import Ane
        model.backend = Ane()
        for layer in model.layers:
            for name in ("proj", "out", "up", "down"):
                model.backend.prepare(layer[name])
        load = time.perf_counter() - start
    tokenizer, tokens, rendered = prompt_tokens(args.model, model.config, args.prompt, args.raw, args.thinking)
    if not tokens or len(tokens) + args.max_tokens > args.context:
        parser.error("prompt is empty or exceeds the context capacity")
    if args.command == "bench":
        model.step(tokens[0])
        model.reset()
        model.timings.clear()
    submissions_start = model.backend.submissions if model.backend else 0
    start = time.perf_counter()
    for token in tokens[:-1]:
        model.step(token, logits=False)
    logits = model.step(tokens[-1])
    prefill = time.perf_counter() - start
    first_token = int(np.argmax(logits))
    warm_ttft = time.perf_counter() - start
    prefill_components = dict(model.timings)
    prefill_submissions = (model.backend.submissions - submissions_start) if model.backend else 0
    model.timings.clear()
    generated, latencies = [], []
    stop = set(model.config["generation_config"]["stop_token_ids"])
    for i in range(args.max_tokens):
        token = first_token if i == 0 else int(np.argmax(logits))
        generated.append(token)
        if token in stop and args.command == "generate":
            break
        if i + 1 < args.max_tokens:
            start = time.perf_counter()
            logits = model.step(token)
            latencies.append(time.perf_counter() - start)
    decode_components = dict(model.timings)
    decode_submissions = (model.backend.submissions - submissions_start - prefill_submissions) if model.backend else 0
    components = {name: prefill_components.get(name, 0) + decode_components.get(name, 0)
                  for name in prefill_components.keys() | decode_components.keys()}
    result = dict(model=MODEL_ID, revision=REVISION, precision=args.precision, kernels=model.kernels,
                  backend=args.backend, ane_submissions=prefill_submissions + decode_submissions,
                  load_seconds=load, prompt_tokens=tokens, generated_tokens=generated,
                  generated_text=tokenizer.decode(generated, skip_special_tokens=True),
                  prefill_seconds=prefill, prefill_tokens_per_second=len(tokens) / prefill,
                  warm_ttft_seconds=warm_ttft, prefill_components_seconds=prefill_components,
                  prefill_ane_submissions=prefill_submissions, decode_seconds=latencies, decode_steps=len(latencies),
                  decode_steps_per_second=len(latencies) / sum(latencies) if latencies else None,
                  decode_components_seconds=decode_components, decode_ane_submissions=decode_submissions,
                  components_seconds=components)
    if args.command == "generate":
        print(result["generated_text"])
    print(json.dumps(result, indent=2), file=sys.stderr if args.command == "generate" else sys.stdout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if model.backend:
        model.backend.close()


if __name__ == "__main__":
    main()
