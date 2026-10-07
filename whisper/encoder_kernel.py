"""Repack the captured M1 tiny.en encoder from an external HF checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from qwen35.weights import SafeTensors, sha256

ROOT = Path(__file__).resolve().parent / "kernels/tiny-en-encoder-fast"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpack(root, name, metadata):
    data = zlib.decompress((root / name).read_bytes())
    require(len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"],
            "kernel asset checksum mismatch: " + name)
    return data


def port_shape(port):
    return tuple(port["compiler_layout"][key] for key in ("Batches", "Channels", "Height", "Width"))


def port_view(buffer, port):
    """View the compiler's planar FP16 layout, including row/channel padding."""
    layout = port["compiler_layout"]
    shape = port_shape(port)
    require(layout["Type"] == "Float16" and layout["Depth"] == 1 and layout["Interleave"] == 1
            and layout["PlaneCount"] == shape[1], "unsupported encoder port layout")
    require(all(type(n) is int and n > 0 for n in shape), "invalid encoder port dimensions")
    strides = tuple(layout[key] for key in ("BatchStride", "PlaneStride", "RowStride")) + (2,)
    require(all(type(n) is int and n > 0 and n % 2 == 0 for n in strides)
            and strides[2] >= shape[3] * 2 and strides[1] >= shape[2] * strides[2]
            and strides[0] >= shape[1] * strides[1], "overlapping encoder port strides")
    offset = port["byte_offset"]
    end = offset + sum((n - 1) * stride for n, stride in zip(shape, strides)) + 2
    require(type(offset) is int and offset >= 0 and end <= len(buffer), "encoder port exceeds buffer")
    return np.ndarray(shape, dtype="<f2", buffer=buffer, offset=offset, strides=strides)


def pack_port(array, port, size=None):
    """Copy logical values into native padded rows; all padding stays zero."""
    array = np.asarray(array, dtype="<f2")
    require(array.size == int(np.prod(port_shape(port))) and bool(np.isfinite(array).all()), "invalid encoder port input")
    if size is None:
        size = port["byte_offset"] + port["compiler_layout"]["BatchStride"] * port_shape(port)[0]
    buffer = bytearray(size)
    port_view(buffer, port)[...] = array.reshape(port_shape(port))
    return buffer


def native_layout(meta):
    """Small text descriptor for the dependency-free C++ tiny.en runner."""
    buffers = {p["replay_bank"]: p["size"] for p in meta["layout"]["buffers"]}
    require(set(buffers) == {3, 4, 5, 6}, "unsupported native encoder banks")
    strides = []
    for role, bank, channels, width in (("input", 5, 80, 3000), ("input", 4, 384, 1500),
                                        ("output", 6, 1500, 384)):
        matches = [p for p in meta["layout"]["ports"] if p["role"] == role
                   and int(np.prod(port_shape(p))) == channels * width]
        require(len(matches) == 1, "ambiguous native encoder port")
        port = matches[0]
        require(port["replay_bank"] == bank and port["byte_offset"] == 0, "unsupported native encoder port bank/offset")
        view = port_view(bytearray(buffers[bank]), port)
        if view.flags.c_contiguous:
            stride = width * 2
        else:
            require(port_shape(port) == (1, channels, 1, width), "unsupported native encoder shape")
            stride = port["compiler_layout"]["PlaneStride"]
        require(bank != 6 or stride == width * 2, "native readback requires tight output rows")
        strides.append(stride)
    lengths = [meta["payloads"][name]["bytes"] for name in ("commands", "coefficients", "constants")]
    values = [meta["td_count"], meta["td_size"], *lengths, *(buffers[i] for i in range(3, 7)), *strides[:2]]
    return "ANE_WHISPER_V1\n" + " ".join(map(str, values)) + "\n"


def encoder_ports(meta):
    ports = {}
    for role, kind, count in (("mel", "input", 80*3000), ("positions", "input", 384*1500),
                              ("output", "output", 1500*384)):
        matches = [p for p in meta["layout"]["ports"]
                   if p["role"] == kind and int(np.prod(port_shape(p))) == count]
        require(len(matches) == 1, "ambiguous encoder port: " + role)
        ports[role] = matches[0]
    return ports


def write_bundle(output, meta, payloads):
    """Write the same checkpoint-derived bundle for either native backend."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in payloads.items():
        filename = {"source-mil":"model.mil", "source-weights":"weights.bin", "positions":"pos.f16"}.get(name, name + ".bin")
        files[filename] = data
    ports = encoder_ports(meta)
    files["ports.txt"] = ("\n".join(f"{ports[role]['name']} {count}" for role, count in
        (("mel", 80*3000), ("positions", 384*1500), ("output", 1500*384))) + "\n").encode()
    files["layout.txt"] = native_layout(meta).encode()
    for filename, data in files.items():
        path = output / filename
        if path.exists():
            require(path.read_bytes() == data, "refusing to replace a different encoder bundle: " + str(path))
        else:
            path.write_bytes(data)


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
    p.add_argument("--kernels", type=Path, default=ROOT, help="Select original fast kernels or retained dense baseline")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    meta, payloads = reconstruct(a.checkpoint, a.kernels)
    if a.output:
        a.output.mkdir(parents=True, exist_ok=False)
        write_bundle(a.output, meta, payloads)
    print(json.dumps(dict(status="pass", task_count=meta["td_count"],
                         payloads={k:dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k, v in payloads.items()})))


if __name__ == "__main__":
    main()
