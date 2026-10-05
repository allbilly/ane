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
    parser.add_argument("command", choices=("setup", "inspect", "generate", "bench"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--kernels", choices=("native", "numpy", "dot"), default="native")
    parser.add_argument("--backend", choices=("cpu", "ane"), default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-hashes", action="store_true")
    args = parser.parse_args()
    if args.command == "setup":
        download(args.model)
        print(f"Pinned checkpoint ready: {args.model}")
        return
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.check_hashes:
        for file, expected in (("config.json", CONFIG_SHA256), ("model.safetensors", MODEL_SHA256), ("tokenizer.json", TOKENIZER_SHA256)):
            if sha256(args.model / file) != expected:
                raise ValueError(f"{file}: checksum mismatch for pinned revision")
    start = time.perf_counter()
    model = Model(args.model, args.precision, args.kernels, args.threads, context=args.context)
    if args.backend == "ane":
        from .ane import Ane
        model.backend = Ane()
        for layer in model.layers:
            for name in ("proj", "out", "up", "down"):
                model.backend.prepare(layer[name])
    load = time.perf_counter() - start
    if args.command == "inspect":
        print(json.dumps(dict(model=MODEL_ID, revision=REVISION, tensors=len(model.tensors.header) - 1,
                              layers=len(model.layers), load_seconds=load), indent=2))
        return
    tokenizer, tokens, rendered = prompt_tokens(args.model, model.config, args.prompt, args.raw, args.thinking)
    if not tokens or len(tokens) + args.max_tokens > args.context:
        parser.error("prompt is empty or exceeds the context capacity")
    if args.command == "bench":
        model.step(tokens[0])
        model.reset()
        model.timings.clear()
    start = time.perf_counter()
    for token in tokens[:-1]:
        model.step(token, logits=False)
    logits = model.step(tokens[-1])
    prefill = time.perf_counter() - start
    generated, latencies = [], []
    stop = set(model.config["generation_config"]["stop_token_ids"])
    for i in range(args.max_tokens):
        token = int(np.argmax(logits))
        generated.append(token)
        if args.command == "generate":
            print(tokenizer.decode(generated, skip_special_tokens=True)[len(tokenizer.decode(generated[:-1], skip_special_tokens=True)):], end="", flush=True)
        if token in stop and args.command == "generate":
            break
        if i + 1 < args.max_tokens:
            start = time.perf_counter()
            logits = model.step(token)
            latencies.append(time.perf_counter() - start)
    result = dict(model=MODEL_ID, revision=REVISION, precision=args.precision, kernels=args.kernels,
                  backend=args.backend, ane_submissions=model.backend.submissions if model.backend else 0,
                  load_seconds=load, prompt_tokens=tokens, generated_tokens=generated,
                  generated_text=tokenizer.decode(generated, skip_special_tokens=True),
                  prefill_seconds=prefill, decode_seconds=latencies,
                  decode_steps_per_second=len(latencies) / sum(latencies) if latencies else None,
                  components_seconds=dict(model.timings))
    if args.command == "generate":
        print()
    print(json.dumps(result, indent=2), file=sys.stderr if args.command == "generate" else sys.stdout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if model.backend:
        model.backend.close()


if __name__ == "__main__":
    main()
