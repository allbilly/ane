"""Capture a new GPT-2 sequence shape without changing the validated 32-token kit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpt2/training"))
from backends import Backend, Primitives
from train import Adam, GPT2, TEXT, Tokenizer, load_weights, metrics, torch_oracle
from capture_macos_program import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence", type=int, default=64)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--offline", action="store_true", help="emit graphs and CPU reference fixtures; no ANE execution")
    a = p.parse_args()
    if a.sequence < 1 or a.steps < 1: p.error("sequence and steps must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ["ANEFORGE_CACHE_DIR"] = str(a.output / "cache")
    os.environ.setdefault("ANEFORGE_NO_AUTOBUILD", "1")
    tokenizer = Tokenizer(ROOT / "gpt2/tokenizer")
    tokens = np.array(tokenizer.encode((TEXT + " ") * 4)[:a.sequence + 1], np.int64)
    if len(tokens) != a.sequence + 1: raise ValueError("insufficient tokens")
    weight_directory = Path.home() / "Desktop/Orion/model/blobs/gpt2_124m"
    weight_hashes = {str(p.relative_to(weight_directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(weight_directory.rglob("*.bin"))}
    expected_hashes = json.loads((ROOT / "gpt2/model-checksums.json").read_text())
    if len(expected_hashes) != 196 or any(weight_hashes.get(name) != digest for name, digest in expected_hashes.items()):
        raise RuntimeError("weights differ from the verified pretrained GPT-2 checkpoint")
    weights = load_weights(weight_directory)
    oracle = torch_oracle(weights, tokens)
    oracle_gradients = oracle.pop("gradient_samples")
    if a.offline:
        from offline_backend import OfflineBackend
        backend = OfflineBackend(a.output)
    else:
        backend = Backend("aneforge", a.output)
    original = backend.program
    captured = set()
    def program(label, shapes, builder):
        native = original(label, shapes, builder)
        def run(*arrays):
            value = native(*arrays)
            if label not in captured:
                np.savez_compressed(a.output / "kernels" / label / "fixture.npz",
                                    **{f"input{i:02d}": np.asarray(array, np.float16) for i, array in enumerate(arrays)},
                                    output=np.asarray(value, np.float16))
                captured.add(label)
            return value
        return run
    backend.program = program
    report = dict(sequence_length=a.sequence, batch_size=1, tokens=tokens.tolist(), steps=[],
                  initial_weights=str(weight_directory), weight_sha256=weight_hashes,
                  verified_weight_files=len(expected_hashes),
                  oracle=oracle, backend="CPU reference with ANEForge offline graphs" if a.offline else "ANEForge macOS",
                  reference_kind="CPU FP32 with FP16 program boundaries" if a.offline else "macOS ANE FP16 outputs",
                  hardware_execution="not attempted (--offline)" if a.offline else "ANE-only E5RT",
                  linux_replay="pending native Asahi validation")
    try:
        model = GPT2(weights, Primitives(backend), tokens)
        loss, caches = model.forward()
        grads = model.backward(caches)
        report.update(initial_loss=loss, loss_abs_error=abs(loss - oracle["loss"]),
                      gradient_checks={name: metrics(grads[name], reference) for name, reference in oracle_gradients.items()})
        if not np.isfinite(loss) or report["loss_abs_error"] > .1 or any(not np.isfinite(check["cosine"]) or check["cosine"] < .98 for check in report["gradient_checks"].values()):
            raise RuntimeError("unchanged training oracle gate failed")
        del grads, caches, oracle_gradients
        optimizer = Adam(1e-4)
        for step in range(a.steps):
            before = backend.dispatches
            started = time.perf_counter()
            loss, caches = model.forward()
            grads = model.backward(caches)
            norm = optimizer.update(weights, grads)
            row = dict(step=step + 1, loss=loss, gradient_norm=norm,
                       **{("cpu_program_calls" if a.offline else "dispatches"): backend.dispatches - before},
                       seconds=time.perf_counter() - started)
            report["steps"].append(row)
            print(json.dumps(row), flush=True)
            del grads, caches
            if backend.dispatches - before != 580 or not np.isfinite(norm): raise RuntimeError("training program-count/finite-gradient contract failed")
        final, caches = model.forward()
        del caches
        report.update(final_loss=final, loss_decreased=final < report["initial_loss"], fixtures=len(captured))
        if len(captured) != 27: raise RuntimeError("expected 27 captured training templates")
        if not report["loss_decreased"]: raise RuntimeError("loss did not decrease")
        report["exports"] = []
        compiler = ROOT / "gpt2/training/build/dump_hwx"
        for record in backend.receipts:
            kernel = a.output / record["mil"]
            receipt = export(kernel.parent, kernel.parent / "hwx", compiler)
            report["exports"].append(dict(name=record["name"], **receipt))
        if any(r["status"] != "exported" for r in report["exports"]): raise RuntimeError("template export incomplete")
        report["status"] = "cpu_reference_and_export_pass" if a.offline else "pass"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        backend.close()
        (a.output / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
