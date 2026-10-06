"""Repack the captured M1 tiny.en encoder from an external HF checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from qwen35.weights import SafeTensors, sha256

ROOT = Path(__file__).resolve().parent / "kernels/tiny-en-encoder"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpack(root, name, metadata):
    data = zlib.decompress((root / name).read_bytes())
    require(len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"],
            "kernel asset checksum mismatch: " + name)
    return data


def reconstruct(checkpoint, root=ROOT):
    """Require the pinned checkpoint and reproduce every captured payload byte."""
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    require(sha256(checkpoint) == meta["checkpoint"]["sha256"], "Whisper checkpoint checksum mismatch")
    packing = json.loads(unpack(root, meta["packing"]["file"], meta["packing"])) if "packing" in meta else meta["payloads"]
    tensors = SafeTensors(checkpoint)
    arrays = {}
    def tensor(name):
        if name not in arrays:
            arrays[name] = tensors.tensor(name).astype("<f2")
            require(bool(np.isfinite(arrays[name]).all()), "nonfinite checkpoint tensor: " + name)
        return arrays[name]
    result = {}
    for name, recipe in meta["payloads"].items():
        recipe = dict(recipe, **packing[name]) if "packing" in meta else recipe
        data = bytearray(unpack(root, recipe["template"], recipe["template_metadata"]))
        occupied = np.zeros(len(data), dtype=bool)
        def write(offset, value):
            require(0 <= offset <= len(data) - len(value), "packing operation out of bounds")
            require(not occupied[offset:offset + len(value)].any() and not any(data[offset:offset + len(value)]),
                    "overlapping packing operations or learned bytes in template")
            data[offset:offset + len(value)] = value
            occupied[offset:offset + len(value)] = True
        for operation in recipe["operations"]:
            source = tensor(operation["tensor"])
            kind = operation["kind"]
            if kind == "tile":
                first, count = operation["first"], operation["count"]
                value = source[first:first + count]
                if source.ndim > 1:
                    value = value.reshape(count, -1).T
            elif kind == "ratio":
                gamma = tensor(operation["gamma"]).astype(np.float32)
                require(bool((gamma != 0).all()), "zero layer-norm gamma")
                value = (source.astype(np.float32) / gamma).astype("<f2")
            else:
                require(kind == "tensor", "unknown coefficient recipe")
                value = source
            require(value.nbytes == operation["bytes"] and bool(np.isfinite(value).all()), "invalid packing tensor")
            write(operation["offset"], value.tobytes())
        if "scatter" in recipe:
            scatter = recipe["scatter"]
            require(scatter["encoding"] == "delta-i32", "unknown convolution map encoding")
            positions = np.cumsum(np.frombuffer(unpack(root, scatter["file"], scatter), dtype="<i4"), dtype=np.int64).reshape(-1, 2)
            source = tensor(scatter["tensor"]).reshape(96, 4, 384, 3).transpose(0, 2, 3, 1).copy().reshape(-1, 4)
            require(positions.shape == (len(source), 2), "invalid convolution scatter shape")
            target = np.frombuffer(data, dtype=np.uint8)
            for pair in range(2):
                offsets = positions[:, pair]
                require(bool((offsets >= -1).all()), "invalid convolution map sentinel")
                valid = offsets >= 0
                value = source[:, pair * 2:pair * 2 + 2].copy().view(np.uint8).reshape(-1, 4)[valid]
                addresses = offsets[valid, None].astype(np.int64) + np.arange(4)
                require(addresses.size and addresses.min() >= 0 and addresses.max() < len(data)
                        and len(np.unique(addresses)) == addresses.size
                        and not occupied[addresses].any() and not target[addresses].any(), "invalid/overlapping convolution scatter")
                target[addresses] = value
                occupied[addresses] = True
            for index, channel, offset in scatter["extra"]:
                write(offset, source[index, channel].tobytes())
        require(len(data) == recipe["bytes"] and hashlib.sha256(data).hexdigest() == recipe["sha256"],
                "repacked encoder differs from captured " + name)
        result[name] = bytes(data)
    positions = tensor("model.encoder.embed_positions.weight").T.copy()
    require(positions.shape == (384, 1500) and hashlib.sha256(positions.tobytes()).hexdigest() == meta["position_sha256"],
            "position input differs from captured checkpoint")
    result["positions"] = positions.tobytes()
    result["source-mil"] = unpack(root, meta["mil"]["file"], meta["mil"])
    return meta, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    meta, payloads = reconstruct(a.checkpoint)
    if a.output:
        a.output.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            filename = {"source-mil":"model.mil", "source-weights":"weights.bin", "positions":"pos.f16"}.get(name, name + ".bin")
            (a.output / filename).write_bytes(data)
    print(json.dumps(dict(status="pass", task_count=meta["td_count"],
                         payloads={k:dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k, v in payloads.items()})))


if __name__ == "__main__":
    main()
