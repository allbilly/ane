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


def compare(actual, expected, label, rtol=0.01, atol=0.03):
    require(actual.shape == expected.shape, f"shape mismatch: {label}")
    require(bool(np.isfinite(actual).all() and np.isfinite(expected).all()), f"nonfinite: {label}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    scale = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-3)
    close = np.abs(difference) <= atol + rtol * np.abs(expected)
    require(bool(close.all()) and rmse / scale < 0.005,
            f"numerical mismatch: {label}; max_abs={np.max(np.abs(difference)):.5g}, normalized_rmse={rmse / scale:.5g}")
    return dict(max_abs=float(np.max(np.abs(difference))), normalized_rmse=rmse / scale)


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
