"""Offline integrity checks and numerical parity checks for first run."""
import hashlib
import json
from pathlib import Path
import numpy as np
from hwx import parse_tasks, require


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def integrity(root):
    manifest = json.loads((root / "checksums.json").read_text())
    for name, expected in manifest.items():
        path = root / name
        require(path.is_file(), f"missing package file: {name}")
        require(digest(path) == expected, f"checksum mismatch: {name}")
    package = json.loads((root / "package.json").read_text())
    require(package["architecture"] == "H13G/M1" and package["format_version"] == 1, "unsupported package")
    for name in package["kernels"]:
        meta = json.loads((root / "kernels" / name / "meta.json").read_text())
        require(set(meta.get("packing", {})) == {"program", "constants", "weights"},
                f"missing external packing recipes: {name}")
        for field, recipe in meta["packing"].items():
            require(recipe["size"] > 0 and len(recipe["sha256"]) == 64, "invalid packing recipe")
            if field != "program":
                require(Path(meta[field]).stem == recipe["sha256"], "reference hash mismatch")
        data = (root / meta["program"]).read_bytes()
        tasks = parse_tasks(data, meta["td_size"], meta["td_count"])
        banks = {0, 1, 2} | {b["bank"] for b in meta["buffers"]}
        for task in tasks:
            for word in task["header"][8:10]:
                for shift in (0, 6, 12, 18):
                    if word & (1 << (shift + 5)):
                        require((word >> shift) & 31 in banks, f"unbound BAR in {name}")
        require(len(data) == meta["tsk_size"] and len(data) % 16 == 0, "invalid kernel boundary")
    return len(manifest)


def compare(actual, expected, label, rtol=0.01, atol=0.03, min_close_fraction=1.0):
    require(actual.shape == expected.shape, f"shape mismatch: {label}")
    require(bool(np.isfinite(actual).all() and np.isfinite(expected).all()), f"nonfinite: {label}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    scale = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-3)
    close = np.abs(difference) <= atol + rtol * np.abs(expected)
    fraction = float(np.mean(close))
    require(0 < min_close_fraction <= 1 and fraction >= min_close_fraction and rmse / scale < 0.005,
            f"numerical mismatch: {label}; max_abs={np.max(np.abs(difference)):.5g}, normalized_rmse={rmse / scale:.5g}, close_fraction={fraction:.6g}")
    result = dict(max_abs=float(np.max(np.abs(difference))), normalized_rmse=rmse / scale)
    if min_close_fraction < 1:
        result["close_fraction"] = fraction
    return result


def cpu_parity(root, model, tokenizer):
    fixtures = json.loads((root / "fixtures/cpu.json").read_text())
    results = []
    for item in fixtures:
        require(tokenizer.encode(item["prompt"]) == item["tokens"], "tokenizer disagrees with fixture")
        model.reset()
        for token in item["tokens"]:
            actual = model.step(token)
        expected = np.fromfile(root / item["logits"], dtype="<f4")
        results.append(compare(actual, expected, item["prompt"], rtol=0.0002, atol=0.0005))
        require(int(actual.argmax()) == int(expected.argmax()), "CPU greedy token mismatch")
    return results


def ane_parity(root, device, all_kernels=False, progress=None):
    names = json.loads((root / "package.json").read_text())["kernels"]
    if not all_kernels:
        names = [name for name in names if name.startswith("decode_")]
    results = {}
    for name in names:
        if progress:
            progress(name)
        fixture = root / "fixtures" / name
        require((fixture / "input.bin").is_file(), f"missing macOS fixture: {name}")
        x = np.fromfile(fixture / "input.bin", dtype="<f2").reshape(768, 32)
        kernel = device.kernel(name)
        actual = kernel.run(x)
        results[name] = {output: compare(value, np.fromfile(fixture / (output + ".bin"), dtype="<f2").reshape(768, 32), f"{name}/{output}")
                         for output, value in actual.items()}
        # Retained kernels are reused by generation; reference-only prefill
        # kernels do not need to stay allocated after verification.
        if not name.startswith("decode_"):
            kernel.close()
            del device.kernels[name]
    return results


def hybrid_parity(root, model, tokenizer):
    fixture = json.loads((root / "fixtures/hybrid.json").read_text())
    require(tokenizer.encode(fixture["prompt"]) == fixture["tokens"], "hybrid tokenizer mismatch")
    model.reset()
    for token in fixture["tokens"]:
        actual = model.step(token)
    result = compare(actual, np.fromfile(root / fixture["logits"], dtype="<f4"), "hybrid prompt logits")
    generated = []
    for index in range(len(fixture["generated"])):
        token = int(actual.argmax())
        generated.append(token)
        if index + 1 < len(fixture["generated"]):
            actual = model.step(token)
    require(generated == fixture["generated"], f"hybrid greedy token mismatch: {generated}")
    result["generated"] = generated
    return result


def cpu_checkpoint_check(root, model, tokenizer):
    """Smoke checks for a selected checkpoint without pinned-model fixtures."""
    results = []
    for item in json.loads((root / "fixtures/cpu.json").read_text()):
        model.reset()
        for token in tokenizer.encode(item["prompt"]):
            logits = model.step(token)
        require(logits.shape == (50257,) and bool(np.isfinite(logits).all()), "invalid checkpoint logits")
        results.append(dict(prompt=item["prompt"], greedy_token=int(logits.argmax())))
    return results


def cpu_kernel_outputs(weights, name, x):
    """Independent NumPy computations for the fixed GPT-2 kernel shapes."""
    from model import CPUKernels, layernorm
    cpu = CPUKernels(weights)
    if name == "prefill_final_ln_L-1":
        return {"hidden": layernorm(x.T, weights.get("ln_f_g", (768,)), weights.get("ln_f_b", (768,))).T}
    layer = int(name.rsplit("_L", 1)[1])
    if "ffn" in name:
        return {"hidden": np.column_stack([cpu.ffn(layer, column) for column in x.T])}
    q, k, v = (np.column_stack(values) for values in zip(*(cpu.project(layer, column) for column in x.T)))
    if name.startswith("decode_proj"):
        return dict(q16=q, k16=k, v16=v)
    require(name.startswith("prefill_attn"), f"no CPU reference for kernel: {name}")
    qh, kh, vh = (value.T.reshape(32, 12, 64).transpose(1, 0, 2) for value in (q, k, v))
    scores = qh @ kh.transpose(0, 2, 1) * np.float32(0.125)
    scores += np.triu(np.full((32, 32), -1e4, dtype=np.float32), 1)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    attended = (probabilities @ vh).transpose(1, 0, 2).reshape(32, 768).T
    hidden = x + weights.layer(layer, "wo") @ attended + weights.layer(layer, "bo")[:, None]
    return dict(hidden=hidden, k_cache=k, v_cache=v)


def ane_checkpoint_parity(root, device, weights, all_kernels=False, progress=None):
    """Compare ANE to CPU with the selected checkpoint's decoded weights."""
    names = json.loads((root / "package.json").read_text())["kernels"]
    if not all_kernels:
        names = [name for name in names if name.startswith("decode_")]
    results = {}
    for name in names:
        if progress:
            progress(name)
        x = np.fromfile(root / "fixtures" / name / "input.bin", dtype="<f2").reshape(768, 32).astype(np.float32)
        expected = cpu_kernel_outputs(weights, name, x)
        actual = device.kernel(name).run(x)
        # FP16 attention reductions can produce isolated cancellation errors
        # near zero compared with the FP32 oracle. Keep the 0.5% NRMSE gate
        # and require 99.9% of elements to meet the pointwise tolerance.
        results[name] = {output: compare(value, expected[output], f"{name}/{output}", rtol=0.02, atol=0.05,
                                        min_close_fraction=0.999 if name.startswith("prefill_attn") else 1.0)
                         for output, value in actual.items()}
        if not name.startswith("decode_"):
            device.kernels.pop(name).close()
    return results


def checkpoint_hybrid_parity(root, model, tokenizer, weights):
    from model import GPT2, CPUKernels
    cpu = GPT2(weights, CPUKernels(weights))
    fixture = json.loads((root / "fixtures/hybrid.json").read_text())
    tokens = fixture["tokens"]
    require(tokenizer.encode(fixture["prompt"]) == tokens, "hybrid tokenizer mismatch")
    cpu.reset()
    model.reset()
    results = []
    # Use CPU-selected tokens for both models; rounding near a tied argmax
    # should not cause the two verification paths to consume different input.
    for token in tokens:
        expected = cpu.step(token)
        actual = model.step(token)
    for index in range(4):
        results.append(compare(actual, expected, f"checkpoint logits/{index}", rtol=0.02, atol=0.1))
        if index < 3:
            token = int(expected.argmax())
            expected = cpu.step(token)
            actual = model.step(token)
    return results
