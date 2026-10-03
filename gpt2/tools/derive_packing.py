#!/usr/bin/env python3
"""Derive compact recipes and compare reconstruction to every original HWX.

Only preparation requires the dump. Runtime packing requires cached HF weights
and the portable recipes, and has no dependency on this macOS reference tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_weights import find_weights, verify_weights, load_weights
from hwx import parse_container, parse_tasks, relocate, require
from packing import operation_bytes, reconstruct
from prepare import object_file


def sha(data):
    return hashlib.sha256(data).hexdigest()


def references(dump, name, meta):
    hwx = (dump / "hwx" / name / "model.hwx").read_bytes()
    require(sha(hwx) == meta["source_hwx_sha256"], "original HWX changed")
    container = parse_container(hwx)
    text = next(s for s in container["segments"] if s["name"] == "__TEXT")
    kern = next(s for s in container["segments"] if s["name"] == "__KERN_0")
    const = next(s for s in container["sections"] if s["name"] == "__const" and s["segment"] == "__TEXT")
    program = hwx[text["fileoff"]:text["fileoff"] + text["filesize"]]
    program = relocate(program, parse_tasks(program, meta["td_size"], meta["td_count"]),
                       {int(k): v for k, v in meta["bank_map"].items()})
    return dict(program=program, weights=hwx[kern["fileoff"]:kern["fileoff"] + kern["filesize"]],
                constants=hwx[const["offset"]:const["offset"] + const["size"]])


def matched_operation(data, weights, operation):
    value = operation_bytes(weights, operation)
    offset = data.find(value)
    require(offset >= 0, f"packing hypothesis failed: {operation}")
    require(data.find(value, offset + 1) < 0, "ambiguous coefficient location")
    return dict(operation, offset=offset, size=len(value))


def derive(root, name, meta, reference, weights):
    layer = int(name.rsplit("_L", 1)[1])
    prefix = f"layer{layer}/"
    ops = {field: [] for field in reference}
    blocks = []
    if name.startswith("decode_proj"):
        blocks = [("wq", "bq", [16] * 3), ("wv", "bv", [16] * 3), ("wk", "bk", [16] * 3)]
    elif "ffn" in name:
        blocks = [("wfc", "bfc", [32] * 6), ("wproj", "bproj", [16] * 3)]
    elif name.startswith("prefill_attn"):
        blocks = [("wv", "bv", [16] * 3), ("wk", "bk", [16] * 3),
                  ("wq", "bq", [32, 16]), ("wo", "bo", [16] * 3)]
    from model import LAYER_SHAPES
    for matrix, bias, tiles in blocks:
        operation = dict(kind="matrix", matrix=prefix + matrix, bias=prefix + bias,
                         shape=LAYER_SHAPES[matrix], tiles=tiles)
        ops["weights"].append(matched_operation(reference["weights"], weights, operation))
    ln = "ln_f" if layer == -1 else "ln2" if "ffn" in name else "ln1"
    prefix = "" if layer == -1 else prefix
    operation = dict(kind="affine", gamma=prefix + ln + "_g", beta=prefix + ln + "_b", layout="linear")
    value = operation_bytes(weights, operation)
    if reference["constants"].find(value) >= 0:
        for field in ("constants", "program"):
            ops[field].append(matched_operation(reference[field], weights, operation))
    else:
        # The captured compiler moved LN2 affine into the coefficient bank for
        # layers 6 and 10, scaling its bias by a fixed power of two.
        require("ffn" in name and layer in (6, 10), "unexpected compiler affine strategy")
        operation.update(layout="engine_pairs", scale=32 if layer == 6 else 2)
        ops["weights"].append(matched_operation(reference["weights"], weights, operation))
    recipes = {}
    for field, data in reference.items():
        masked = bytearray(data)
        covered = np.zeros(len(data), dtype=bool)
        for operation in ops[field]:
            offset, size = operation["offset"], operation["size"]
            require(not covered[offset:offset + size].any(), "overlapping packing operations")
            covered[offset:offset + size] = True
            masked[offset:offset + size] = bytes(size)
        recipe = dict(size=len(data), sha256=sha(data), operations=ops[field])
        if field == "weights":
            # Residue must consist exclusively of nonlearned activation LUTs
            # and zero alignment. Never store unexplained model data as literals.
            allowed = np.zeros(len(data), dtype=bool)
            allowed[:256] = True  # pow/exp lookup tables shared by all kernels
            if "ffn" in name:
                fc = ops[field][0]
                gelu = fc["offset"] + fc["size"]
                allowed[gelu:gelu + 128] = True  # tanh approximation table
            if name.startswith("prefill_attn"):
                q = ops[field][2]
                softmax = q["offset"] + q["size"]
                require(masked[softmax:softmax + 128] == masked[128:256], "unexpected softmax lookup table")
                allowed[softmax:softmax + 128] = True
            require(not np.frombuffer(masked, dtype="u1")[~allowed].any(),
                    f"unexplained coefficient bytes: {name}")
            literals = [[0, bytes(masked[:256]).hex()]]
            if "ffn" in name:
                literals.append([gelu, bytes(masked[gelu:gelu + 128]).hex()])
            if name.startswith("prefill_attn"):
                literals.append([softmax, bytes(masked[softmax:softmax + 128]).hex()])
            recipe["literals"] = literals
        else:
            recipe["template"] = object_file(root, bytes(masked))
        require(reconstruct(root, weights, recipe) == data, f"byte comparison failed: {name}/{field}")
        recipes[field] = recipe
    meta["packing"] = recipes
    # Programs are stored as templates with learned constants zeroed. The
    # historical weights/constants fields identify reference hashes, not files.
    meta["program"] = recipes["program"]["template"]
    return meta


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dump", type=Path, required=True)
    cli.add_argument("--root", type=Path, default=ROOT, help="portable package destination")
    cli.add_argument("--weights", type=Path)
    cli.add_argument("--prune", action="store_true", help="remove reference objects only after every reconstruction passes")
    args = cli.parse_args()
    root = args.root.resolve()
    source = find_weights(args.weights)
    require(source is not None, "cached GPT-2 weights required")
    verify_weights(source, root)
    weights = load_weights(source)
    names = json.loads((root / "package.json").read_text())["kernels"]
    pending, fixed_tables = [], set()
    for name in names:
        path = root / "kernels" / name / "meta.json"
        meta = json.loads(path.read_text())
        reference = references(args.dump, name, meta)
        meta = derive(root, name, meta, reference, weights)
        pending.append((path, meta))
        fixed_tables.add(meta["packing"]["weights"]["literals"][0][1])
    require(len(fixed_tables) == 1, "unexpected model-specific activation lookup tables")
    gelu_tables = {m["packing"]["weights"]["literals"][1][1]
                   for _, m in pending if "ffn" in m["kernel"]}
    require(len(gelu_tables) == 1, "unexpected model-specific tanh lookup tables")
    retained = set()
    for path, meta in pending:
        for recipe in meta["packing"].values():
            if "template" in recipe:
                retained.add(recipe["template"])
        path.write_text(json.dumps(meta, indent=2) + "\n")
    if args.prune:
        for path in (root / "objects").glob("*.bin"):
            if str(path.relative_to(root)) not in retained:
                path.unlink()
    report = dict(kernels=len(names), payloads=len(names) * 3, byte_exact=True,
                  checkpoint_sha256=sha(source.read_bytes()) if source.is_file() else "verified Orion blobs",
                  learned_weights_packaged=False, linux_hardware_verified=False,
                  fixed_activation_lut_bytes=256 + 128,
                  description="HF fp16 -> 16-engine bias/matrix tiles and compiler affine folding; all HWX bytes matched")
    (root / "packing-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Byte-exact reconstruction: {len(names)} kernels / {len(names) * 3} payloads PASS")


if __name__ == "__main__":
    main()
