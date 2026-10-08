"""Read external, unquantized whisper.cpp encoder tensors without Torch."""
import struct

import numpy as np

from whisper.encoder_kernel import require


def encoder_name(name):
    name = name.removeprefix("encoder.")
    replacements = (
        ("positional_embedding", "embed_positions.weight"), ("ln_post.", "layer_norm."),
        ("blocks.", "layers."), ("attn_ln.", "self_attn_layer_norm."),
        ("mlp_ln.", "final_layer_norm."), ("attn.query.", "self_attn.q_proj."),
        ("attn.key.", "self_attn.k_proj."), ("attn.value.", "self_attn.v_proj."),
        ("attn.out.", "self_attn.out_proj."), ("mlp.0.", "fc1."), ("mlp.2.", "fc2."),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return "model.encoder." + name


def read_encoder(path):
    """Parse the format written by upstream convert-pt-to-ggml.py; skip decoder data."""
    tensors = {}
    with open(path, "rb") as f:
        def integers(count):
            data = f.read(count * 4)
            require(len(data) == count * 4, "truncated GGML integers")
            return struct.unpack("<" + "i" * count, data)
        header = integers(12)
        require(header[0] == 0x67676d6c and header[11] == 1, "expected unquantized F16 Whisper GGML")
        _, vocab, context, state, heads, layers, _, _, _, _, mels, _ = header
        require(vocab > 0 and context > 0 and state > 0 and heads > 0 and layers > 0 and mels > 0,
                "invalid GGML encoder dimensions")
        a, b = integers(2)
        require(a > 0 and b > 0, "invalid GGML mel filters")
        f.seek(a * b * 4, 1)
        count, = integers(1)
        # The converter writes len(tokens), excluding special tokens in n_vocab.
        require(0 < count <= vocab, "GGML vocabulary mismatch")
        for _ in range(count):
            length, = integers(1)
            require(length >= 0, "invalid GGML token length")
            f.seek(length, 1)
        while first := f.read(12):
            require(len(first) == 12, "truncated GGML tensor header")
            rank, length, kind = struct.unpack("<3i", first)
            require(1 <= rank <= 4 and 0 < length < 1024 and kind in (0, 1), "unsupported GGML tensor")
            shape = integers(rank)[::-1]
            require(all(n > 0 for n in shape), "invalid GGML tensor shape")
            name = f.read(length).decode("utf-8")
            require(len(name.encode()) == length, "truncated GGML tensor name")
            dtype = "<f4" if kind == 0 else "<f2"
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if name.startswith("encoder."):
                raw = f.read(size)
                require(len(raw) == size, "truncated GGML encoder tensor")
                value = np.frombuffer(raw, dtype).reshape(shape).astype("<f2")
                if name.endswith(".bias"):
                    value = value.reshape(-1)
                require(bool(np.isfinite(value).all()), "nonfinite GGML encoder tensor: " + name)
                key = encoder_name(name)
                require(key not in tensors, "duplicate GGML encoder tensor")
                tensors[key] = value
            else:
                f.seek(size, 1)
        require(len(tensors) == 7 + 15 * layers, "incomplete GGML encoder tensors")
    dims = dict(state=state, heads=heads, layers=layers, context=context, mels=mels, frames=context * 2)
    return dims, tensors
