"""Rebuild PR 3905 H13G encoder dumps from external whisper.cpp F16 models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from whisper.encoder_checkpoints import read_checkpoint
from whisper.encoder_kernel import require, unpack


def scatter_vectors(weight):
    outputs, inputs, width = weight.shape
    require(outputs % 4 == 0 and width == 3, "unsupported conv2 shape")
    return weight.reshape(outputs // 4, 4, inputs, width).transpose(0, 2, 3, 1).copy().reshape(-1, 4)


def goc_affine(gamma, beta, engines):
    """Two small-model affine/GELU blocks pack [gamma, 2*beta/gamma] per engine."""
    require(gamma.ndim == 1 and beta.shape == gamma.shape and len(gamma) % engines == 0,
            "invalid GOC affine shape")
    ratio = (beta.astype(np.float32) / gamma.astype(np.float32) * 2).astype("<f2")
    return np.stack((gamma, ratio), axis=1).reshape(-1, engines, 2).transpose(1, 0, 2).copy()


def rebuild_hwx(template, packing, tensors):
    """Reject overlaps or retained learned bytes before filling compiled packets."""
    data = bytearray(template)
    occupied = np.zeros(len(data), bool)
    def write(offset, raw):
        require(type(offset) is int and 0 <= offset <= len(data) - len(raw), "packing offset out of bounds")
        require(not occupied[offset:offset + len(raw)].any() and not any(data[offset:offset + len(raw)]),
                "overlapping packing or retained learned bytes")
        data[offset:offset + len(raw)] = raw
        occupied[offset:offset + len(raw)] = True
    for op in packing["operations"]:
        value = tensors[op["tensor"]]
        if op["kind"] == "tile":
            first, count = op["first"], op["count"]
            require(type(first) is int and type(count) is int and 0 <= first < first + count <= len(value),
                    "invalid matrix tile")
            value = value[first:first + count].reshape(count, -1).T
        elif op["kind"] == "ratio":
            value = (value.astype(np.float32) / tensors[op["gamma"]].astype(np.float32)).astype("<f2")
        elif op["kind"] == "goc-affine":
            value = goc_affine(value, tensors[op["beta"]], op["engines"])
        else:
            require(op["kind"] == "tensor", "unknown packing operation")
        raw = value.tobytes()
        require(len(raw) == op["bytes"] and bool(np.isfinite(value).all()), "invalid packing value")
        write(op["offset"], raw)
    scatter = packing["scatter"]
    vectors = scatter_vectors(tensors[scatter["tensor"]])
    deltas = np.frombuffer(scatter["offset_bytes"], "<i4")
    require(deltas.size == vectors.shape[0] * 2, "bad convolution offset count")
    offsets = deltas.astype(np.int64).cumsum().reshape(-1, 2)
    view = np.frombuffer(data, np.uint8)
    for pair in range(2):
        valid = offsets[:, pair] >= 0
        addresses = offsets[valid, pair, None] + np.arange(4)
        require(addresses.size > 0 and addresses.min() >= 0 and addresses.max() < len(data)
                and np.unique(addresses).size == addresses.size
                and not occupied[addresses].any() and not view[addresses].any(), "invalid/overlapping convolution offsets")
        values = vectors[:, pair * 2:pair * 2 + 2].copy().view(np.uint8).reshape(-1, 4)[valid]
        view[addresses] = values
        occupied[addresses] = True
    for index, channel, offset in scatter["extra"]:
        require(0 <= index < len(vectors) and 0 <= channel < 4, "invalid sparse convolution scalar")
        write(offset, vectors[index, channel].tobytes())
    # Each coefficient is mapped once or omitted by the compiler as an exact zero.
    mapped = np.zeros(vectors.shape, bool)
    for pair in range(2):
        mapped[offsets[:, pair] >= 0, pair * 2:pair * 2 + 2] = True
    for index, channel, _ in scatter["extra"]:
        require(not mapped[index, channel], "duplicate sparse convolution scalar")
        mapped[index, channel] = True
    require(bool((vectors[~mapped] == 0).all()), "unmapped nonzero convolution coefficient")
    require(int(occupied.sum()) == packing["stripped_bytes"], "learned byte coverage mismatch")
    return data


def reconstruct(checkpoint, root):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    require(meta["format"] == "whisper-pr3905-h13g/v1", "unsupported encoder packing format")
    dims = meta["dimensions"]
    tensors = read_checkpoint(checkpoint, meta)
    packing = json.loads(unpack(root, meta["packing"]["file"], meta["packing"]))
    packing["scatter"]["offset_bytes"] = unpack(root, meta["scatter"]["file"], meta["scatter"])
    template = unpack(root, meta["template"]["file"], meta["template"])
    data = rebuild_hwx(template, packing, tensors)
    require(hashlib.sha256(data).hexdigest() == meta["hwx_sha256"], "rebuilt HWX differs from capture")
    position = tensors["model.encoder.embed_positions.weight"]
    require(position.shape == (dims["context"], dims["state"]), "invalid position shape")
    positions = position.T.copy().tobytes()
    require(hashlib.sha256(positions).hexdigest() == meta["position_sha256"], "position checksum mismatch")
    return meta, data, positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="External matching safetensors, lossless GGUF or pinned GGML F16 model")
    p.add_argument("--kernel", type=Path, required=True, help="Small stripped kernel package")
    p.add_argument("--output", type=Path, required=True, help="New ignored/external directory for large local payloads")
    a = p.parse_args()
    meta, data, positions = reconstruct(a.checkpoint, a.kernel)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model.hwx").write_bytes(data)
    (a.output / "pos.f16").write_bytes(positions)
    (a.output / "layout.json").write_text(json.dumps(meta["layout"], indent=2) + "\n")
    print(json.dumps(dict(status="PASS_BYTE_EXACT_REPACK", model=meta["checkpoint"]["model"],
                         hwx_sha256=meta["hwx_sha256"], bytes=len(data),
                         linux_hardware_validation="pending")), flush=True)


if __name__ == "__main__":
    main()
