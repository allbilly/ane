#!/usr/bin/env python3
"""Capture end-to-end golden logits/tokens with Orion's macOS ANE backend."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bpe import Tokenizer
from model import GPT2
from external_weights import find_weights, verify_weights, load_weights


class MacKernels:
    def __init__(self, library, dump):
        self.lib, self.dump, self.cache = ctypes.CDLL(str(library)), dump, {}
        self.lib.gpt2_mac_open.argtypes = [ctypes.c_char_p]
        self.lib.gpt2_mac_open.restype = ctypes.c_void_p
        self.lib.gpt2_mac_eval.argtypes = [ctypes.c_void_p] * 3
        self.lib.gpt2_mac_eval.restype = ctypes.c_int
        self.lib.gpt2_mac_close.argtypes = [ctypes.c_void_p]

    def run(self, name, x, count):
        if name not in self.cache:
            handle = self.lib.gpt2_mac_open(str(self.dump / "bundles" / (name + "_loaded")).encode())
            if not handle:
                raise RuntimeError(f"macOS compile failed: {name}")
            self.cache[name] = handle
        data = np.zeros((768, 32), dtype="<f2")
        data[:, 0] = x
        output = np.empty((count, 768, 32), dtype="<f2")
        if self.lib.gpt2_mac_eval(self.cache[name], data.ctypes.data, output.ctypes.data):
            raise RuntimeError(f"macOS eval failed: {name}")
        return output[:, :, 0].astype(np.float32)

    def project(self, layer, x):
        k, q, v = self.run(f"decode_proj_L{layer}", x, 3)
        return q, k, v

    def ffn(self, layer, x):
        return self.run(f"decode_ffn_L{layer}", x, 1)[0]

    def close(self):
        for handle in self.cache.values():
            self.lib.gpt2_mac_close(handle)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--library", type=Path, required=True)
    cli.add_argument("--dump", type=Path, required=True)
    args = cli.parse_args()
    kernels = MacKernels(args.library, args.dump.resolve())
    try:
        tokenizer = Tokenizer(ROOT / "tokenizer")
        source = find_weights()
        if source is None:
            raise RuntimeError("external GPT-2 weights required")
        verify_weights(source, ROOT)
        model = GPT2(load_weights(source), kernels)
        prompt = "Hello world"
        tokens = tokenizer.encode(prompt)
        for token in tokens:
            logits = model.step(token)
        path = ROOT / "fixtures/hybrid-logits.bin"
        logits.astype("<f4").tofile(path)
        generated = []
        for _ in range(4):
            token = int(logits.argmax())
            generated.append(token)
            logits = model.step(token)
        report = dict(prompt=prompt, tokens=tokens, generated=generated,
                      text=tokenizer.decode(tokens + generated), logits="fixtures/hybrid-logits.bin",
                      backend="Python GPT-2 flow with Orion macOS ANE projection/FFN; sequential prompt ingest",
                      linux_hardware_verified=False)
        (ROOT / "fixtures/hybrid.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        kernels.close()


if __name__ == "__main__":
    main()
