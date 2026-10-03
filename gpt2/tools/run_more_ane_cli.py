#!/usr/bin/env python3
"""Smoke-run upstream generate.py using coremltools 9's public compiled API.

The 2023 CLI's private _MLModelProxy constructor no longer works with v9.
This adapter changes its Python bridge and cached-tokenizer location only;
it does not edit the external checkout or model. CLI timings are not the
prewarmed native benchmark reported by bench_coreml_macos.py.
"""
import argparse
from pathlib import Path
import os
import runpy
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "more-ane-transformers")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path.home() / ".cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    import coremltools as ct
    from transformers import AutoTokenizer
    repo, model = args.repo.expanduser().resolve(), args.model.expanduser().resolve()
    compiled = Path(str(model).removesuffix(".mlpackage") + ".mlmodelc")
    if not compiled.is_dir():
        raise ValueError("Compile the model with bench_coreml_macos.py first")
    tokenizer = AutoTokenizer.from_pretrained(str(args.weights.expanduser()), local_files_only=True)
    sys.path.insert(0, str(repo))
    from src.utils import model_proxy

    class ModernProxy:
        supports_input_output_cache = False

        def __init__(self, model_path, compute_unit):
            self.model = ct.models.CompiledMLModel(model_path, compute_units=compute_unit)

        def predict(self, data, input_output_mapping):
            return self.model.predict(data)

    old_args, old_cwd = sys.argv, Path.cwd()
    with tempfile.TemporaryDirectory(prefix="more-ane-cli-") as temporary:
        directory = Path(temporary)
        (directory / "gpt2.mlpackage").symlink_to(model, target_is_directory=True)
        (directory / "gpt2.mlmodelc").symlink_to(compiled, target_is_directory=True)
        try:
            os.chdir(directory)
            sys.argv = ["generate.py", "--model_path", "gpt2.mlpackage", "--input_prompt", args.prompt,
                        "--length", str(args.tokens), "--compute_unit", "CPUAndANE", "--argmax", "--timingstats"]
            print("Using the public CompiledMLModel compatibility adapter; this CLI copies KV arrays through Python.", flush=True)
            with patch.object(model_proxy, "MLModelProxy", ModernProxy), patch.object(
                    AutoTokenizer, "from_pretrained", return_value=tokenizer):
                runpy.run_path(str(repo / "generate.py"), run_name="__main__")
        finally:
            sys.argv = old_args
            os.chdir(old_cwd)


if __name__ == "__main__":
    main()
