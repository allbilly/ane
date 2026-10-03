"""Reconstruct the captured H13G coefficient layout from external GPT-2 tensors.

Recipes describe compiler decisions for this pinned model, not a general ANE
compiler. No learned coefficient payloads are stored in the portable package.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from hwx import require


def matrix_bytes(weights, operation):
    shape = tuple(operation["shape"])
    matrix = weights.get(operation["matrix"], shape).astype("<f2")
    bias = weights.get(operation["bias"], (shape[0],)).astype("<f2")
    engines, tiles = 16, operation["tiles"]
    require(matrix.ndim == 2 and all(t in (16, 32) for t in tiles)
            and sum(tiles) * engines == shape[0], "invalid H13 matrix schedule")
    output = bytearray()
    for engine in range(engines):
        part, base = bytearray(), 0
        for tile in tiles:
            start = base + engine * tile
            part.extend(bias[start:start + tile].tobytes())
            part.extend(matrix[start:start + tile].T.tobytes(order="C"))
            base += tile * engines
        part.extend(bytes(-len(part) % 64))
        output.extend(part)
    return bytes(output)


def affine_bytes(weights, operation):
    gamma = weights.get(operation["gamma"], (768,))
    beta = weights.get(operation["beta"], (768,))
    require(bool(np.all(gamma != 0)), "zero layer-norm gamma unsupported by captured recipe")
    # Input tensors have already been rounded to fp16. Division/scaling happen
    # in float32 before the compiler's final fp16 rounding (including subnormals).
    ratio = beta / gamma * np.float32(operation.get("scale", 1))
    if operation["layout"] == "linear":
        return np.concatenate((ratio, gamma)).astype("<f2").tobytes()
    require(operation["layout"] == "engine_pairs", "unknown affine layout")
    pairs = np.column_stack((gamma, ratio)).astype("<f2")
    return pairs.reshape(48, 16, 2).transpose(1, 0, 2).tobytes()


def operation_bytes(weights, operation):
    if operation["kind"] == "matrix":
        return matrix_bytes(weights, operation)
    require(operation["kind"] == "affine", "unknown packing operation")
    return affine_bytes(weights, operation)


def reconstruct(root, weights, recipe):
    data = bytearray(recipe["size"])
    if "template" in recipe:
        data[:] = (root / recipe["template"]).read_bytes()
        require(len(data) == recipe["size"], "packing template size mismatch")
    for offset, value in recipe.get("literals", []):
        value = bytes.fromhex(value)
        require(0 <= offset <= len(data) - len(value), "packing literal out of bounds")
        data[offset:offset + len(value)] = value
    for operation in recipe["operations"]:
        value = operation_bytes(weights, operation)
        offset = operation["offset"]
        require(len(value) == operation["size"] and 0 <= offset <= len(data) - len(value),
                "packing operation size/bounds mismatch")
        require(not any(data[offset:offset + len(value)]), "learned data remains in packing template")
        data[offset:offset + len(value)] = value
    require(hashlib.sha256(data).hexdigest() == recipe["sha256"],
            "reconstructed ANE payload differs from captured reference")
    return bytes(data)


class PackedAssets:
    """Generated payloads live in external cache, validated against dump hashes."""
    def __init__(self, root, weights, cache=None):
        if cache is None:
            from external_weights import cache_root
            cache = cache_root() / "h13g-packed-v1"
        self.root, self.weights, self.cache = root, weights, Path(cache)
        self.validated = set()

    def payload(self, meta, field):
        recipe = meta["packing"][field]
        target = self.cache / (recipe["sha256"] + ".bin")
        if target.is_file():
            data = target.read_bytes()
            if len(data) == recipe["size"] and hashlib.sha256(data).hexdigest() == recipe["sha256"]:
                self.validated.add(recipe["sha256"])
                return data
        data = reconstruct(self.root, self.weights, recipe)
        self.cache.mkdir(parents=True, exist_ok=True)
        # Unique temporary files plus atomic rename support concurrent runners.
        fd, temporary = tempfile.mkstemp(prefix=".packing-", dir=self.cache)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.validated.add(recipe["sha256"])
        return data

    def verify(self, names, progress=None):
        for name in names:
            if progress:
                progress(name)
            meta = json.loads((self.root / "kernels" / name / "meta.json").read_text())
            for field in ("program", "constants", "weights"):
                self.payload(meta, field)
        return len(self.validated)
