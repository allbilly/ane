"""Read external, unquantized whisper.cpp tensors into HF model namespaces."""
import struct

import numpy as np

from whisper.encoder_kernel import require


def encoder_name(name):
    return model_name("encoder." + name.removeprefix("encoder."))


def model_name(name):
    axis, name = name.split(".", 1)
    require(axis in ("encoder", "decoder"), "unknown Whisper tensor namespace")
    if axis == "decoder" and name.startswith("ln."):
        name = "layer_norm." + name[3:]
    replacements = (
        ("positional_embedding", "embed_positions.weight"), ("ln_post.", "layer_norm."),
        ("token_embedding.", "embed_tokens."),
        ("cross_attn_ln.", "encoder_attn_layer_norm."),
        ("cross_attn.query.", "encoder_attn.q_proj."), ("cross_attn.key.", "encoder_attn.k_proj."),
        ("cross_attn.value.", "encoder_attn.v_proj."), ("cross_attn.out.", "encoder_attn.out_proj."),
        ("blocks.", "layers."), ("attn_ln.", "self_attn_layer_norm."),
        ("mlp_ln.", "final_layer_norm."), ("attn.query.", "self_attn.q_proj."),
        ("attn.key.", "self_attn.k_proj."), ("attn.value.", "self_attn.v_proj."),
        ("attn.out.", "self_attn.out_proj."), ("mlp.0.", "fc1."), ("mlp.2.", "fc2."),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return "model." + axis + "." + name


def read_encoder(path):
    """Parse the format written by upstream convert-pt-to-ggml.py; skip decoder data."""
    dimensions, tensors = read_model(path, encoder_only=True)
    dims = dict(state=dimensions["audio_state"], heads=dimensions["audio_heads"],
                layers=dimensions["audio_layers"], context=dimensions["audio_context"],
                mels=dimensions["mels"], frames=dimensions["audio_context"]*2)
    return dims, tensors


def read_model(path, encoder_only=False):
    """Read the same lossless GGML tensors into encoder or full HF namespaces."""
    tensors = {}
    with open(path, "rb") as f:
        def integers(count):
            data = f.read(count * 4)
            require(len(data) == count * 4, "truncated GGML integers")
            return struct.unpack("<" + "i" * count, data)
        header = integers(12)
        require(header[0] == 0x67676d6c and header[11] == 1, "expected unquantized F16 Whisper GGML")
        _, vocab, context, state, heads, layers, text_context, text_state, text_heads, text_layers, mels, _ = header
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
            if name.startswith("encoder.") or (not encoder_only and name.startswith("decoder.")):
                raw = f.read(size)
                require(len(raw) == size, "truncated GGML encoder tensor")
                value = np.frombuffer(raw, dtype).reshape(shape)
                if encoder_only:
                    value = value.astype("<f2")
                if name.endswith(".bias"):
                    value = value.reshape(-1)
                require(bool(np.isfinite(value).all()), "nonfinite GGML encoder tensor: " + name)
                key = model_name(name)
                require(key not in tensors, "duplicate GGML encoder tensor")
                tensors[key] = value
            else:
                f.seek(size, 1)
                if not encoder_only:
                    require(name.startswith("decoder."), "unexpected GGML tensor namespace")
        encoder_count = sum(name.startswith("model.encoder.") for name in tensors)
        require(encoder_count == 7 + 15 * layers, "incomplete GGML encoder tensors")
        if not encoder_only:
            require(sum(name.startswith("model.decoder.") for name in tensors) == 4 + 24*text_layers,
                    "incomplete GGML decoder tensors")
    dims = dict(vocabulary=vocab, audio_context=context, audio_state=state, audio_heads=heads,
                audio_layers=layers, text_context=text_context, text_state=text_state,
                text_heads=text_heads, text_layers=text_layers, mels=mels)
    return dims, tensors


def reference_model(path):
    """Independent HF FP32 arithmetic, using every stored GGML value unchanged."""
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    dims, tensors = read_model(path)
    require(dims["audio_state"] == dims["text_state"], "HF Whisper requires equal encoder/decoder widths")
    multilingual = dims["vocabulary"] == 51865
    require(multilingual or dims["vocabulary"] == 51864, "unsupported Whisper vocabulary")
    config = WhisperConfig(vocab_size=dims["vocabulary"], num_mel_bins=dims["mels"],
        d_model=dims["audio_state"], encoder_layers=dims["audio_layers"], decoder_layers=dims["text_layers"],
        encoder_attention_heads=dims["audio_heads"], decoder_attention_heads=dims["text_heads"],
        encoder_ffn_dim=4*dims["audio_state"], decoder_ffn_dim=4*dims["text_state"],
        max_source_positions=dims["audio_context"], max_target_positions=dims["text_context"],
        activation_function="gelu", pad_token_id=50257 if multilingual else 50256,
        bos_token_id=50257 if multilingual else 50256, eos_token_id=50257 if multilingual else 50256,
        decoder_start_token_id=50258 if multilingual else 50257)
    config._attn_implementation = "eager"
    # Meta construction avoids initializing another full model's random weights.
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    state = {name:torch.from_numpy(value.astype(np.float32)) for name,value in tensors.items()}
    state["proj_out.weight"] = state["model.decoder.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model.eval(), dims
