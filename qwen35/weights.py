"""Memory-mapped Mirai W4 checkpoints; preserve their RHT basis and metadata."""
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

MODEL_ID = "trymirai/Qwen3.5-0.8B-M"
REVISION = "c12202e4c764e559960827761566aaa1fd15a87a"
MODEL_SHA256 = "fa595349afb112763731f5d548a7af4baff7b6088c99f1f2cada29c89442870c"
TOKENIZER_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
DEFAULT_MODEL = Path.home() / ".cache/ane-qwen35/model"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def bf16_round(x):
    """Round finite FP32 values to BF16, ties to even, retaining FP32 storage."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def hadamard(x):
    """Normalized Sylvester transform, independently on each 32-element block."""
    y = np.array(x, dtype=np.float32, copy=True)
    if y.shape[-1] % 32:
        raise ValueError("RHT requires a multiple of 32 elements")
    original_shape = y.shape
    y = y.reshape(-1, 32)
    for stride in (1, 2, 4, 8, 16):
        v = y.reshape(-1, 2, stride)
        left, right = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = left + right, left - right
    y *= np.float32(1 / math.sqrt(32))
    return y.reshape(original_shape)


def unpack4(packed):
    p = np.asarray(packed, dtype=np.uint8)
    result = np.empty((*p.shape[:-1], p.shape[-1] * 2), dtype=np.uint8)
    result[..., 0::2], result[..., 1::2] = p & 15, p >> 4
    return result


class SafeTensors:
    """Reject invalid spans and unsupported dtypes before exposing mmap views."""
    DTYPES = {"U8": "u1", "I32": "<i4", "F32": "<f4", "BF16": "<u2"}

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if size > 16 << 20 or size + 8 > self.path.stat().st_size:
                raise ValueError("invalid safetensors header size")
            self.header = json.loads(f.read(size))
        self.offset = size + 8
        self.metadata = self.header.get("__metadata__", {})
        spans = []
        for name, item in self.header.items():
            if name == "__metadata__":
                continue
            if item["dtype"] not in self.DTYPES:
                raise ValueError(f"unsupported dtype: {name}: {item['dtype']}")
            start, end = item["data_offsets"]
            shape = item["shape"]
            if any(not isinstance(d, int) or d <= 0 for d in shape):
                raise ValueError(f"invalid shape: {name}")
            expected = math.prod(shape) * np.dtype(self.DTYPES[item["dtype"]]).itemsize
            if start < 0 or end - start != expected or self.offset + end > self.path.stat().st_size:
                raise ValueError(f"invalid tensor span: {name}")
            spans.append((start, end))
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("safetensors spans overlap or have gaps")
            cursor = end
        if self.offset + cursor != self.path.stat().st_size:
            raise ValueError("unexpected trailing safetensors data")
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def tensor(self, name, shape=None, dtype=None):
        item = self.header[name]
        if shape is not None and list(shape) != item["shape"]:
            raise ValueError(f"{name}: expected shape {shape}, found {item['shape']}")
        if dtype is not None and item["dtype"] != dtype:
            raise ValueError(f"{name}: expected {dtype}, found {item['dtype']}")
        start, end = item["data_offsets"]
        view = self.data[self.offset + start:self.offset + end].view(self.DTYPES[item["dtype"]]).reshape(item["shape"])
        if item["dtype"] == "BF16":
            return (view.astype(np.uint32) << 16).view(np.float32)
        return view


class Matrix:
    """Row-major packed W4; normalize only the small quantization parameters."""
    def __init__(self, tensors, prefix, rows, cols, embedding=False):
        self.name, self.rows, self.cols = prefix, rows, cols
        spec = json.loads(tensors.metadata[prefix + ".spec"])
        q = spec.get("quantization_spec", {})
        if (spec.get("type") != "HybridSpec" or spec.get("adapter_spec") is not None
                or spec.get("incoherence_block_size") != 32
                or q.get("type") != "IntSpec" or q.get("bits") != 4
                or q.get("group_size") != 32 or q.get("is_symmetric") is not False
                or q.get("layout") != ("input_output" if embedding else "output_input")):
            raise ValueError(f"unsupported Mirai spec: {prefix}: {spec}")
        if cols % 32 or rows % 32:
            raise ValueError(f"unsupported RHT dimensions: {prefix}")
        self.embedding = embedding
        mode = "output" if embedding else "input_output"
        if spec.get("incoherence_processing_mode") != mode:
            raise ValueError(f"{prefix}: expected {mode} RHT")
        groups = cols // 32
        self.weights = tensors.tensor(prefix + ".quantized.weights", (rows, cols // 2), "U8")
        if embedding:
            self.scales = tensors.tensor(prefix + ".quantized.scales", (rows, groups), "BF16")
            self.zeros = unpack4(tensors.tensor(prefix + ".quantized.zero_points", (rows, groups // 2), "U8"))
        else:
            self.scales = np.ascontiguousarray(tensors.tensor(prefix + ".quantized.scales", (groups, rows), "BF16").T)
            self.zeros = np.ascontiguousarray(unpack4(tensors.tensor(prefix + ".quantized.zero_points", (groups, rows // 2), "U8")).T)
        self.output_signs = tensors.tensor(prefix + ".incoherence_signs.output_signs", (cols if embedding else rows,), "I32")
        self.input_signs = None if embedding else tensors.tensor(prefix + ".incoherence_signs.input_signs", (cols,), "I32")
        for signs in (self.input_signs, self.output_signs):
            if signs is not None and not np.all((signs == 1) | (signs == -1)):
                raise ValueError(f"invalid RHT signs: {prefix}")
        if not np.isfinite(self.scales).all():
            raise ValueError(f"nonfinite scales: {prefix}")

    def decode(self, start=0, end=None):
        end = self.rows if end is None else end
        codes = unpack4(self.weights[start:end]).astype(np.float32).reshape(end - start, -1, 32)
        scale = self.scales[start:end, :, None]
        return (codes * scale - scale * self.zeros[start:end, :, None]).reshape(end - start, self.cols)

    def lookup(self, token, precision="fp32"):
        if not 0 <= token < self.rows or not self.embedding:
            raise ValueError("invalid embedding lookup")
        y = self.decode(token, token + 1)[0]
        if precision == "bf16":
            y = bf16_round(y)
        y = hadamard(y) * self.output_signs
        return bf16_round(y) if precision == "bf16" else y.astype(np.float32)

    def reference(self, x, precision="fp32", chunk=1024):
        """Decoded NumPy oracle; never used by native packed execution."""
        signs = self.output_signs if self.embedding else self.input_signs
        x = hadamard(np.asarray(x, dtype=np.float32) * signs)
        if precision == "bf16":
            x = bf16_round(x)
        y = np.empty((*x.shape[:-1], self.rows), dtype=np.float32)
        for start in range(0, self.rows, chunk):
            end = min(start + chunk, self.rows)
            y[..., start:end] = x @ self.decode(start, end).T
        if precision == "bf16":
            y = bf16_round(y)
        if not self.embedding:
            y = hadamard(y) * self.output_signs
            if precision == "bf16":
                y = bf16_round(y)
        return np.asarray(y, dtype=np.float32)
