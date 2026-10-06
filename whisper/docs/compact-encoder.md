# Complete tiny.en encoder kernels

The [kernel package](../kernels/tiny-en-encoder/meta.json) contains the complete
1,783-task H13G encoder for base M1 / T8103, in approximately **147 KiB**.
Learned weights and position embeddings come from the external pinned
`openai/whisper-tiny.en` safetensors checkpoint. Full HWX files, model weights,
audio and numerical arrays stay outside Git.

The recovered export was already validated through macOS ANE-only E5RT
execution on three real-audio inputs. Every result exactly matched the
captured ANE output and passed the independent HF encoder cosine gate of
0.999. The new repacker reproduces the relocated command stream, constants,
compiled coefficients, source MIL weight blob and position input **byte for
byte**. [Proof and fixture hashes](../kernels/tiny-en-encoder/proof.json)
retain that evidence. Linux hardware execution has not yet been performed.

## Repack on either host

Run from the parent `ane` repository with an existing NumPy environment:

```sh
qwen35/.venv/bin/python -m whisper.encoder_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors

# Optional: materialize the runtime payloads into a new ignored directory.
qwen35/.venv/bin/python -m whisper.encoder_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --output whisper/.cache/complete-encoder
```

Checkpoint revision: `87c7102498dcde7456f24cfd30239ca606ed9063`.
SHA-256: `db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`.
This repacker currently supports that HF safetensors checkpoint. GGUF/ggml
input support is separate work. No Apple compiler or full export is needed
to reconstruct the executable payloads on Linux.

The committed templates contain compiler instructions, tables, sparse packet
masks and padding. Matrix tiles and biases, all nine layer-norm affine pairs,
and every source MIL weight tensor are stripped. Convolution two uses a
delta-encoded byte-offset map; the compiler omitted one zero coefficient.
Reconstruction rejects overlapping writes, remaining learned template bytes,
corrupt assets, the wrong checkpoint and any final payload hash mismatch.

## Native Asahi replay

`whisper.replay_encoder.Encoder` accepts the normal frontend's finite FP16 mel
array `[80,3000]`, supplies checkpoint position embeddings, and returns encoder
features `[1500,384]`. One DRM submission executes the complete task chain,
including convolutions, attention, normalization, GELU and all projections.
The loader requires native Linux, base M1 device-tree identity and the ANE
accel driver before allocating or submitting anything.

```python
from whisper.replay_encoder import Encoder

encoder = Encoder("whisper/models/hf-tiny.en/model.safetensors")
try:
    features = encoder(mel)
finally:
    encoder.close()
```

Hold the existing ANE/GPU locks around hardware work. To compare with the
three local captured fixtures, use the recovered kit outside Git:

```sh
flock "$HOME/ane.lock" flock "$HOME/gpu.lock" flock /tmp/m1-gpu.lock \
  env OPENBLAS_NUM_THREADS=1 \
  qwen35/.venv/bin/python -m whisper.replay_encoder \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --fixtures LOCAL_RECOVERED_KIT \
  --output whisper/.cache/complete-encoder-linux.json
```

The fixture command retains the original relative-L2 < 0.005,
`allclose(rtol=0.01, atol=0.03)` and HF cosine >= 0.999 gates. It checks all
three fixtures and rejects nonfinite or unwritten output. Fixture arrays are
optional validation data; ordinary encoder execution only needs kernels,
checkpoint and frontend mel input.

The Python replay entry point is prepared for Linux validation. Integration
with the whisper.cpp decoder, transcript checks and matching CPU/ANE encode,
prompt, decode and whole-transcription benchmarks remain pending. The current
[projection path](asahi-native.md) still performs 1,128 separate submissions
and is independently validated; its performance claims remain separate.

## Layout and provenance

| Payload / bank | Bytes | Purpose |
| --- | ---: | --- |
| Commands / 0 | 1,277,952 | Complete relocated stream, first descriptor 504 bytes |
| Coefficients / implicit 1 | 15,450,112 | Repacked weights appended after commands |
| Constants / 2 | 26,112 | Repacked layer-norm affines and static constants |
| Scratch / 3 | 10,371,072 | All intermediate tensors |
| Position input / 4 | 1,163,264 | 1,152,000 payload bytes, physical `[1,1,4500,128]` |
| Mel input / 5 | 491,520 | 480,000 payload bytes, physical `[1,1,1875,128]` |
| Encoder output / 6 | 1,163,264 | 1,152,000 payload bytes, physical `[1,1,1500,384]` |

The original export used constants on BAR 1 and coefficients on BAR 7.
Only active task-header BAR selectors are relocated to Linux constants BAR 2
and implicit coefficient BAR 1; task order, dependencies and NextPtr stay
intact. Compiler strings are `ANEC v1` and `zin_ane_compiler v10.26.6`.
The macOS runtime dylib hash is recorded in the metadata and proof.

The recovery tool is `experimental.package_whisper_encoder`; the reproducible
stripping/recipe derivation tool is `experimental.derive_whisper_kernels`.
Its input is the ignored recovered kit. The final package does not depend on
the exploratory offset-map NPZ or a full model dump.
