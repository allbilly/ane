#!/usr/bin/env python3
"""Portable Orion GPT-2 generation and first-run verification."""
import argparse
import codecs
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("generate", "verify", "doctor", "setup", "pack"))
    cli.add_argument("--backend", choices=("ane", "cpu"), default="ane")
    cli.add_argument("--device", help="ANE /dev/accel node (otherwise discovered by driver name)")
    cli.add_argument("--prompt", default="Hello world")
    cli.add_argument("--max-tokens", "--max_tokens", type=int, default=32)
    cli.add_argument("--temperature", type=float, default=0)
    cli.add_argument("--top-k", type=int, default=40)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--weights", type=Path, help="cached HF model.safetensors or external Orion BLOBFILE directory")
    cli.add_argument("--checkpoint", type=Path, help="local HF model.safetensors for setup")
    cli.add_argument("--all-kernels", action="store_true", help="also verify 25 reference prefill kernels")
    args = cli.parse_args()
    if not 0 <= args.temperature < float("inf") or not 1 <= args.top_k <= 50257 or args.max_tokens < 0:
        cli.error("invalid temperature, top-k, or token count")
    try:
        from bpe import Tokenizer
        from model import ANEKernels, CPUKernels, GPT2
        from checks import integrity, cpu_parity, ane_parity, hybrid_parity
        from external_weights import find_weights, verify_weights, setup, load_weights
        files = integrity(ROOT)
        print(f"Package integrity: {files} files PASS", file=sys.stderr)
        tokenizer = Tokenizer(ROOT / "tokenizer")
        if args.command in ("setup", "pack"):
            from packing import PackedAssets
            selected = find_weights(args.weights)
            if args.command == "setup" and (selected is None or args.checkpoint is not None):
                selected = setup(ROOT, checkpoint=args.checkpoint, progress=lambda s: print(s, file=sys.stderr, flush=True))
            if selected is None:
                raise ValueError("no external weights found; run setup first or specify --weights")
            verify_weights(selected, ROOT)
            assets = PackedAssets(ROOT, load_weights(selected))
            names = json.loads((ROOT / "package.json").read_text())["kernels"]
            if not args.all_kernels:
                names = [name for name in names if name.startswith("decode_")]
            count = assets.verify(names, lambda name: print(f"Packing/checking {name}...", file=sys.stderr, flush=True))
            print(f"Byte-exact ANE packing: {len(names)} kernels, {count} unique payloads PASS; cache: {assets.cache}")
            return 0
        device = None
        if args.backend == "ane":
            from replay import Device
            device = Device(ROOT, args.device)
            print(f"ANE device: {device.path}", file=sys.stderr)
        with device if device else nullcontext():
            selected = find_weights(args.weights)
            if args.command == "doctor":
                if selected:
                    verify_weights(selected, ROOT)
                    print(f"External model: {selected}", file=sys.stderr)
                else:
                    print("External model missing; generate/setup will download HF weights into the external cache.", file=sys.stderr)
                print(f"Ready for {args.backend} verification; hardware numerical tests have not run.")
                return 0
            if selected is None:
                selected = setup(ROOT, checkpoint=args.checkpoint, progress=lambda s: print(s, file=sys.stderr, flush=True))
            verify_weights(selected, ROOT)
            weights = load_weights(selected)
            if device:
                from packing import PackedAssets
                device.assets = PackedAssets(ROOT, weights)
            kernels = ANEKernels(device) if device else CPUKernels(weights)
            model = GPT2(weights, kernels)
            if device:
                progress = lambda name: print(f"Checking {name}...", file=sys.stderr, flush=True)
                results = ane_parity(ROOT, device, args.all_kernels, progress)
                print(f"ANE parity: {len(results)} kernels PASS", file=sys.stderr)
                results["hybrid_generation"] = hybrid_parity(ROOT, model, tokenizer)
                print("ANE full generation parity: PASS", file=sys.stderr)
            else:
                results = cpu_parity(ROOT, model, tokenizer)
                print(f"Orion CPU parity: {len(results)} prompts PASS", file=sys.stderr)
            if args.command == "verify":
                print(json.dumps(dict(backend=args.backend, status="PASS", results=results), indent=2))
                return 0
            tokens = tokenizer.encode(args.prompt)
            if not tokens:
                cli.error("prompt is empty")
            if len(tokens) + max(args.max_tokens - 1, 0) > 1024:
                cli.error("prompt plus generation exceeds 1024-token context")
            print(args.prompt, end="", flush=True)
            started = time.monotonic()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            produced = 0
            for token in model.generate(tokens, args.max_tokens, args.temperature, args.top_k, args.seed):
                if token == tokenizer.eos:
                    break
                print(decoder.decode(tokenizer.decode_bytes([token])), end="", flush=True)
                produced += 1
            print(decoder.decode(b"", final=True), flush=True)
            print(f"Backend: {args.backend}; prompt={len(tokens)}; generated={produced}; elapsed={time.monotonic() - started:.2f}s", file=sys.stderr)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"GPT-2: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
