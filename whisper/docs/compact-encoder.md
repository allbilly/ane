# Complete tiny.en encoder kernels

For the newer multilingual tiny/base/small PR 3905 exports, see
[weight-free packages and safetensors/GGUF repacking](pr3905-packing.md).
The sections below describe the older pinned tiny.en packages and loader.

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
retain that evidence. The recorded native Linux result matches the earlier Linux dense
wrapper bit for bit on all three clips, with one submission and four-worker
readback. Exact cross-host output verification remains open because the Linux
frontend regenerates different FP16 mel hashes. The named Linux receipt,
`whisper/results/asahi-fast-20261007.json`, is absent from this Mac checkout;
its return and fresh Linux verification remain pending.

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
  --output whisper/build/complete-encoder

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
  --output whisper/build/complete-encoder-linux.json
```

The fixture command retains the original relative-L2 < 0.005,
`allclose(rtol=0.01, atol=0.03)` and HF cosine >= 0.999 gates. It checks all
three fixtures and rejects nonfinite or unwritten output. Fixture arrays are
optional validation data; ordinary encoder execution only needs kernels,
checkpoint and frontend mel input.

To profile either compact package without transferring any captured arrays,
run from the repository root on either host:

```sh
env OPENBLAS_NUM_THREADS=1 \
  whisper/.venv/bin/python -m whisper.scripts.benchmark_encoder \
  --hf-model whisper/models/hf-tiny.en \
  --backend auto --compare-baseline \
  --output whisper/build/encoder-replay-profile.json
```

This command acquires ANE, GPU and `/tmp/m1-gpu.lock` itself. It recreates the
three clips from the vendored JFK WAV and checks exact Mac FP16 input/output
hashes in `results/fast-recapture-20261007/encoder-reference.json`. WAV header
metadata can differ; the PCM samples must match. `--prepare-only` checks the
inputs on either host without allocating or submitting hardware. Add
`--fixtures EXISTING_CAPTURE_DIRECTORY` to use the original three `jfk*.npz`
arrays, avoiding frontend drift; mel, position and output hashes are all checked.
On this Asahi host the strict input check fails. `--diagnostic-inputs` explicitly
allows local profiling and fast/wrapper comparisons, records that the Mac
comparison is invalid, and exits nonzero. It does not turn the strict check into
a pass. Exact cross-host verification needs the existing three raw Mac inputs.

Both hosts use `encoder_runtime.py` for input validation, finite-output checks
and timing boundaries. The adapters supply DRM replay or E5RT execution.
Each warm call reports preparation/upload, blocking backend execute, FP16
readback and their total; Asahi preparation also resets scratch. Dispatch includes driver scheduling
and waiting. It excludes CPU cross-K/V and decoder work, so compare dispatch
with Mac's roughly 10.9 ms execute boundary rather than its 15.83 ms complete
encode timer. Python readback is a reference implementation; retain the faster
native four-worker readback in native transcription benchmarks. A successful
encoder replay does not satisfy the separate full-decoder-logit gate.

The Python entry point verifies packing and execution; its readback timing differs
from the optimized native implementation. The current
[projection path](asahi-native.md) still performs 1,128 separate submissions
and is independently validated; its performance claims remain separate.

## Native transcription and CPU cross-K/V

The older Linux measurements below used a native full-encoder adapter whose
source is absent from this Mac checkout. The current
`whisper/asahi_full_encoder.cpp` supplies the external-encoder API for the new
multilingual tiny/base/small PR packages; use
[its preparation and build commands](pr3905-packing.md#native-whispercpp-integration).
It expects `native-layout.txt` and does not consume the older tiny.en layout
described below. The earlier repacker writes `layout.txt` with task count,
buffer sizes and native input strides. Its historical runner needed no JSON
parser or runtime Python, supported both tiny.en captures and retained the
four-worker readback. That source and build integration must be recovered
before the historical commands below are runnable from this checkout.

Prepare/build the isolated worktree as described in [asahi-native.md](asahi-native.md).
For the faster CPU cross-K/V path, use Fedora's **OpenMP** OpenBLAS variant:

```sh
# One-time package setup; ordinary inference runs without sudo.
sudo dnf install openblas-openmp openblas-devel
whisper/.venv/bin/cmake -S whisper/vendor/whisper-asahi \
  -B whisper/build/asahi-ane-blas -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_BUILD_TESTS=OFF -DGGML_VULKAN=OFF -DGGML_LLAMAFILE=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DBLAS_openblas_LIBRARY=/usr/lib64/libopenblaso.so \
  -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DANE_ROOT="$PWD"
whisper/.venv/bin/cmake --build whisper/build/asahi-ane-blas \
  --target whisper-cli -j4

flock "$HOME/ane.lock" flock "$HOME/gpu.lock" flock /tmp/m1-gpu.lock \
  taskset -c 4-7 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  whisper/.venv/bin/python whisper/scripts/benchmark_asahi.py \
  --build whisper/build/asahi-ane-blas --encoder complete \
  --payloads whisper/build/complete-encoder --diagnostic-timings \
  --output whisper/build/native-fast-benchmark
```

The local test used these Fedora RPM libraries extracted into ignored cache,
without installing system packages. Avoid the default single-thread OpenBLAS
variant. If using a Python wheel's OpenMP OpenBLAS instead, link ggml to that
same OpenMP runtime: loading both wheel and system OpenMP runtimes regressed
cross-K/V to about 50 ms. The tested Fedora build uses one system runtime and
needs no wheel library or `LD_PRELOAD`.

The benchmark checks all 80 full decoder vectors and token histories, then
measures encoder, prompt batch, token decode and whole transcription on all
three clips, with two backend-order rounds. `--diagnostic-timings` retains
failed accuracy gates and exits nonzero after collecting timing evidence.
The new graph still fails NRMSE < 0.005; all 80 raw argmaxes match.
Latest 11-second encode is 31.18 ms and whole transcription 142.90 ms, versus
Mac's 15.83 / 66.81 ms. Cross-K/V is 14.16 ms and native readback 1.34 ms.
These measurements do not establish speed parity or accepted model accuracy.

## Shared macOS / Asahi benchmark

`native.py` applies the same FP32 activations/cache, widened accumulation, exact
GELU and 1,500-key attention fixes to either native worktree. `validation.h`
captures the same native mel, encoder and full logit records. Both hosts run
`benchmark_native.py` and `benchmark_whisper.cpp`, with identical HF gates,
clip durations, warmup policy and stage parser. `benchmark_asahi.py` remains a
compatibility entry point. The old `benchmark_macos.py` retains the original
CPU/Metal baseline and uses the same parser.

On macOS, from the repository root (reuse the worktree if it already exists):

```sh
git -C whisper/vendor/whisper.cpp worktree add --detach \
  ../whisper-macos-matched 60c0be6ac8fa71b1a2ae2dd938a31a34a508e774
whisper/.venv/bin/python -m whisper.scripts.prepare_native \
  --backend macos --source whisper/vendor/whisper-macos-matched
WHISPER_CMAKE=whisper/.venv/bin/cmake
if [ ! -x "$WHISPER_CMAKE" ]; then
  WHISPER_CMAKE=$(whisper/.venv/bin/python -c 'import cmake; print(cmake.CMAKE_BIN_DIR + "/cmake")')
fi
"$WHISPER_CMAKE" -S whisper/vendor/whisper-macos-matched \
  -B whisper/build/macos-matched -DCMAKE_BUILD_TYPE=Release \
  -DANE_ROOT="$PWD" -DWHISPER_BUILD_TESTS=OFF -DGGML_METAL=OFF \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_LLAMAFILE=ON
"$WHISPER_CMAKE" --build whisper/build/macos-matched --target whisper-cli -j4
whisper/.venv/bin/python -m whisper.encoder_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --output whisper/build/matched-encoder
whisper/.venv/bin/python -m whisper.scripts.benchmark_native \
  --backend macos --build whisper/build/macos-matched \
  --model whisper/models/ggml-tiny.en.bin \
  --payloads whisper/build/matched-encoder --dylib ANEFORGE_DISPATCH_DYLIB \
  --warmups 2 --runs 10 --rounds 2 --profile-stages --profile-matmul \
  --diagnostic-timings \
  --output whisper/build/macos-matched-results
```

Replace `ANEFORGE_DISPATCH_DYLIB` with the existing ANEForge dispatch dylib.
The shared command acquires the ANE/GPU locks internally on both hosts.
The Linux command uses `--backend asahi` and its existing build/payloads,
without `--dylib`. Generated payloads, compiler outputs and reports belong in
`whisper/build`; reusable source is outside `.cache`. The encoder packer is
identical on both hosts; macOS executes the restored MIL/weights and Asahi
executes its already captured command stream.

`--profile-stages` records CPU cross-K/V and encoder host stages. Mac reports
input conversion, feed, blocking E5RT execute, read and conversion separately;
it verifies return codes and one actual execution per encode. API wall time
still includes runtime waiting. Per-matrix BLAS profiling, hot CPU assembly and
clock counters are requested in `todo.md`. `--profile-matmul` records all eight
cross-K/V products, with allocation, FP16-to-FP32 conversion, thread setup and
GEMM timings, dimensions/strides/transposes, the BLAS entry-point provider, and
one representative activation input per context. Actual Apple BLAS thread count
is reported as unknown when no query is available. The
[Mac sampling follow-up](macos-followup.md) identifies the selected AMX BLAS
kernel; [decoder profiling](decoder-profile.md) separately maps NEON vocabulary
and attention/probability routines on all three clips.
The shared instrumentation has been verified on Asahi without changing any of
the 18 native mel, encoder or full-logit captures.

The shared Mac run and CPU library ablations are now retained in
[the Mac follow-up](macos-followup.md). OpenBLAS reproduces the recorded Asahi
cross-K/V stage; the matched CPU reference passes all 80 vectors, while the
original ANE graph's numerical failure stays explicit. An opt-in
[cross-K/V fusion experiment](cross-kv-fusion.md) preserves the native outputs
but has not established a whole-transcription speedup. The
[native paired correction](native-precision.md) passes all 80 logit vectors on
Mac. A [batched encoder attention follow-up](encoder-attention.md) preserves
the full gate and improves paired whole time by about 16% in a same-binary
comparison; paired ANE remains slower than CPU. Existing exact Whisper
and Qwen arrays are staged locally; Linux replay and cross-host comparisons
remain pending.

Asahi verification after this refactor: all three CPU and ANE mel/encoder/full
logit captures are byte identical to the preceding run; all 80 full vectors
were checked again. Both Python replay packages match the native output on
all three clips. The production logit gate still fails. The Mac preparation
patch is idempotent and its native adapter passes a Clang syntax check on Linux;
Mac hardware execution remains to be tested after boot.

The current precision diagnostic is `experimental.whisper_precision`, with
`--hf-model`, `--traces` and `--output` paths. It uses the shared validation code
and reconstructs partial-task inputs/weights from the checkpoint. Old `.cache`
probe scripts are historical; they are no longer the active implementation.
Adding `--paired-outputs --paired-fc1` passes all 80 fixed-history logit checks
on Asahi (maximum NRMSE 0.255% / 0.392% / 0.384% for 11/5/23 seconds).
It retains CPU FP32 attention, normalization, GELU and residuals and requires
34 ANE submissions. This is a precision diagnostic, not the native production
encoder or a demonstrated speed improvement.

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
