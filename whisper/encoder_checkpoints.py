"""Load external encoder tensors, pinned by their shapes and FP16 value hashes."""
import hashlib
from pathlib import Path
import sys

import numpy as np

from qwen35.weights import SafeTensors, sha256
from whisper.encoder_kernel import require
from whisper.ggml_encoder_weights import encoder_name, read_encoder


class EncoderSafeTensors(SafeTensors):
    DTYPES = dict(SafeTensors.DTYPES, F16="<f2")


def canonical_name(name):
    if name.startswith("model.encoder."):
        return name
    if name.startswith("encoder."):
        return encoder_name(name)
    return None


def tensor_manifest(tensors):
    return {name:dict(shape=list(value.shape), sha256=hashlib.sha256(value.astype("<f2").tobytes()).hexdigest())
            for name, value in sorted(tensors.items())}


def read_checkpoint(path, meta):
    """Lossless GGUF/SafeTensors must contain the same encoder values as the capture."""
    path = Path(path)
    tensors = {}
    def add(name, value):
        key = canonical_name(name)
        if key is None:
            return
        value = np.asarray(value).astype("<f2")
        if key.endswith(".bias"):
            value = value.reshape(-1)
        require(key not in tensors and bool(np.isfinite(value).all()), "duplicate/nonfinite encoder tensor: " + key)
        tensors[key] = value
    if path.suffix.lower() == ".safetensors":
        source = EncoderSafeTensors(path)
        for name in source.header:
            if canonical_name(name):
                add(name, source.tensor(name))
    elif path.suffix.lower() == ".gguf":
        from gguf import GGUFReader, GGMLQuantizationType, dequantize
        source = GGUFReader(str(path), "r")
        require(sys.byteorder == "little" and source.byte_order == "I", "expected little-endian GGUF")
        for tensor in source.tensors:
            if canonical_name(tensor.name):
                require(tensor.tensor_type in (GGMLQuantizationType.F16, GGMLQuantizationType.F32),
                        "byte-exact kernels require lossless F16/F32 GGUF: " + tensor.name)
                shape = tuple(int(n) for n in reversed(tensor.shape))
                require(tensor.data_offset + tensor.n_bytes <= path.stat().st_size, "truncated GGUF encoder tensor")
                add(tensor.name, dequantize(tensor.data, tensor.tensor_type).reshape(shape))
    else:
        require(sha256(path) == meta["checkpoint"]["sha256"], "Whisper checkpoint checksum mismatch")
        dims, tensors = read_encoder(path)
        require(dims == meta["dimensions"], "encoder dimensions mismatch")
    require(tensor_manifest(tensors) == meta["encoder_tensors"], "encoder tensor values/shapes differ from captured model")
    return tensors
