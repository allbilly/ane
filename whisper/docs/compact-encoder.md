# Complete tiny.en encoder kernels

The default [fast kernel package](../kernels/tiny-en-encoder-fast/meta.json) contains
the original **1,779-task** H13G encoder for base M1 / T8103. Its source MIL matches
the earlier fast macOS benchmark, without extra input reshape nodes. The
[1,783-task dense wrapper](../kernels/tiny-en-encoder/meta.json) remains available
as a baseline. The original package is approximately **149 KiB**; the baseline
is approximately **147 KiB**.
Learned weights and position embeddings come from the external pinned
`openai/whisper-tiny.en` safetensors checkpoint. Full HWX files, model weights,
audio and numerical arrays stay outside Git.

The original graph was freshly exported and validated through macOS ANE-only E5RT
execution on three real-audio inputs. Every result exactly matched the
captured ANE output and passed the independent HF encoder cosine gate of
0.999. The new repacker reproduces the relocated command stream, constants,
compiled coefficients, source MIL weight blob and position input **byte for
byte**. [Proof and fixture hashes](../kernels/tiny-en-encoder-fast/proof.json)
retain that evidence. Native Linux execution of the new original package remains
pending. The existing dense wrapper has been run on Linux according to `todo.md`.

**The fast graph fails the full-logit NRMSE < 0.005 gate on macOS too.**
All 80 raw argmaxes match HF, but the same HF CPU decoder fed the original ANE
features reaches 3.08%, 6.64% and 2.78% NRMSE on the 11/5/23-second cases.
Encoder cosine and transcript checks alone did not establish this stricter
accuracy requirement. See [dump identity and performance](fast-dump-review.md).

## Repack on either host

Run from the parent `ane` repository with an existing NumPy environment:

```sh
qwen35/.venv/bin/python -m whisper.encoder_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors

# Optional: materialize the runtime payloads into a new ignored directory.
qwen35/.venv/bin/python -m whisper.encoder_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --output whisper/.cache/complete-encoder

# Select the retained dense-wrapper baseline explicitly.
qwen35/.venv/bin/python -m whisper.encoder_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --kernels whisper/kernels/tiny-en-encoder
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

The Python entry point verifies packing and execution; it does not claim the
optimized native readback performance reported from Linux. Integrating the new
native-stride package into that Linux decoder and repeating its complete gates
and matching stage benchmarks remain pending. The current
[projection path](asahi-native.md) still performs 1,128 separate submissions
and is independently validated; its performance claims remain separate.

## Layout and provenance

| Payload / bank | Bytes | Purpose |
| --- | ---: | --- |
| Commands / 0 | 1,277,952 | Complete relocated stream, first descriptor 504 bytes |
| Coefficients / implicit 1 | 15,450,112 | Repacked weights appended after commands |
| Constants / 2 | 26,112 | Repacked layer-norm affines and static constants |
| Scratch / 3 | 8,093,696 | All intermediate tensors (dense wrapper: 10,371,072) |
| Position input / 4 | 1,163,264 | Logical `[1,384,1,1500]`, 3,008-byte row/channel stride |
| Mel input / 5 | 491,520 | Logical `[1,80,1,3000]`, 6,016-byte row/channel stride |
| Encoder output / 6 | 1,163,264 | 1,152,000 payload bytes, physical `[1,1,1500,384]` |

The original export used constants on BAR 1 and coefficients on BAR 7.
Only active task-header BAR selectors are relocated to Linux constants BAR 2
and implicit coefficient BAR 1; task order, dependencies and NextPtr stay
intact. Compiler strings are `ANEC v1` and `zin_ane_compiler v10.26.6`.
The macOS runtime dylib hash is recorded in the metadata and proof.

`pack_port` writes mel channel `c` at byte `c * 6016`, leaving 16 zero bytes
after its 3,000 FP16 values. Position input is the checkpoint's `[1500,384]`
embedding transposed to `[384,1500]`: channel `c` starts at `c * 3008`, leaving
eight zero bytes after 1,500 values. Batches occupy 481,280 and 1,155,072 bytes
respectively. Output rows are tight, at 768 bytes. These strides come from the
fresh compiler status, rather than from the logical tensor dimensions.
The old dense wrapper uses tight width-128 input rows; selecting its package
makes the same loader use those layouts instead.

All checkpoint values round to FP16 before packing. Linear and conv1 weights
use 16- or 8-output-row tiles, flatten input dimensions, then transpose to
input-major/output-minor order; associated biases occupy the preceding FP16
slots. Conv2 groups four output channels, interleaves them for each input
channel/kernel position, and scatters two pairs into compiler sparse packets.
Its bias groups contain ten or four values. The delta offset map retains
packet placement; the one omitted zero stays absent. Layer-norm gamma is FP16;
its companion affine stores `FP16(FP32(beta_fp16) / FP32(gamma_fp16))` in both
the constants bank and the embedded command constants. Every source MIL weight
tensor is restored in its original logical order. Templates retain sparse
masks and zero padding, and final captured hashes guard every byte.

The recovery tool is `experimental.package_whisper_encoder`; the reproducible
stripping/recipe derivation tool is `experimental.derive_whisper_kernels`.
Its input is the ignored recovered kit. The final package does not depend on
the exploratory offset-map NPZ or a full model dump.

Fresh source capture and the independent decoder probe can be repeated with
`experimental.recapture_whisper_fast` on macOS, using local fixtures and the
pinned HF model directory. It acquires ANE then GPU locks. Source MIL execution
uses the private E5RT API; the exported HWX is reserved for Linux replay.
