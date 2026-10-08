# PR 3905 weight-free kernel packages

The multilingual tiny/base/small speed-reference graphs have complete offline
M1/H13G exports. The repository retains compiler instructions, zeroed weight
templates, byte-offset recipes and buffer metadata. No learned matrices,
biases, layer-norm values, positional buffers or complete packed weights are
included. External checkpoints supply those values during repacking.

The packages live in `whisper/kernels/pr3905/{tiny,base,small}`. Each has
`meta.json`, `proof.json`, `model.hwx.template.zlib`, `packing.json.zlib` and
`conv2-offsets.i32.zlib`. The corresponding large raw exports and timed bundles
remain under ignored `whisper/build/pr3905-hwx` and `pr3905-bundles`.
Medium is excluded at the user's request.

| Model | Tasks | Coefficient banks | Repository package | Original HWX bytes | Stripped learned bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1,779 | 1 | 123.8 KiB | 16,826,368 | 15,264,766 |
| base | 3,434 | 1 | 233.4 KiB | 42,598,400 | 39,645,180 |
| small | 10,015 | 2 | 700.7 KiB | 182,550,528 | 174,004,212 |

## Repack on Asahi

From the repository root, using Python and NumPy:

```sh
python -m whisper.pr_encoder_kernel \
  --checkpoint /external/path/whisper-tiny/model.safetensors \
  --kernel whisper/kernels/pr3905/tiny \
  --output whisper/build/pr3905-repacked/tiny
```

Select `base` or `small` with its matching checkpoint/package. The output
directory must be new. It contains the large regenerated `model.hwx`, a
tight channels-first `pos.f16`, and `layout.json`. Keeping outputs under
`whisper/build` ensures they stay ignored. Input/output transfers must use
the compiler's padded strides in the layout, rather than copying tight rows.

The checkpoint can be an HF-style safetensors file (F16/F32), or lossless
F16/F32 GGUF with the same logical encoder tensor names and shapes. GGUF
uses the existing optional `gguf` Python package. Names can be canonical
`model.encoder.*`, HF `encoder.layers.*`, or original Whisper
`encoder.blocks.*` / `encoder.positional_embedding`. Decoder tensors are
ignored. Quantized GGUF values do not reproduce these F16 captures; they
are rejected. Safetensors require only NumPy and the repository's reader.

Every encoder tensor's shape and F16 value hash must match the package.
This allows equivalent container formats without binding safetensors/GGUF
to a particular serialization hash. The original benchmark GGML files are
also supported, bound to their exact file hashes in metadata. The repacker
never downloads a checkpoint. It rejects changed weights, corrupt templates,
invalid offsets, overlapping writes, and retained learned bytes, then checks
the complete rebuilt HWX and position SHA-256 hashes.

The format verification used external safetensors and GGUF files made from
the exact benchmark encoder tensors. They reproduced all three original
HWX files and position buffers byte for byte. Independently downloaded HF
and GGUF containers were not exercised in this run. See
[verification](../results/pr3905-m1-20261008/packing-verification.json).
Tiny additionally passed F32 safetensors/GGUF reconstruction and the command
above through the CLI, including output layout and position files.

## Learned packing

Matrix tiles select consecutive output rows, flatten the input dimensions,
transpose to input-major order, and convert to little-endian F16. Tile
widths vary by compiled operation: 16/8 are common, with 14-row base and
10-row small FFN-down tiles, plus tail tiles. Bias slices precede their
corresponding matrix tiles. Recipes record every first row, count and byte
offset, including tiles in small's second coefficient bank.

Conv2 groups four output channels, then orders input channels, three filter
positions, and the four output values. The compiler scatters two-F16 pairs
through packets. The compressed signed-offset map preserves those locations;
sparse scalar recipes handle pairs whose exact zero is omitted. No nonzero
source coefficient may remain unmapped. Exact zero omissions account for
2 / 4 / 12 bytes in tiny/base/small respectively.

Most layer norms retain gamma and `F16(F32(beta)/F32(gamma))` in the constant
section. Small's layer 5 and 11 final layer norms instead store pairs
`[gamma, F16(2*F32(beta)/F32(gamma))]` ordered by 16 engines: engine zero
handles channels 0, 16, 32, and so on. Those complete blocks were matched
against the raw export and stripped too. A one-ULP source mutation confirmed
that the changed bytes lie in coefficients while task instructions stay
unchanged; [analysis receipt](../results/pr3905-m1-20261008/packing-analysis.json).

Positions are transposed from `[1500, state]` to `[state, 1500]` and checked
against the timed bundle's exact F16 bytes. `meta.json` records the original
BARs, both coefficient banks when present, workspace segments, task count and
first descriptor size, and all compiler I/O shapes, types and strides.

To learn a fresh package on Mac, the offline export and timed source bundle
must already exist:

```sh
python -m experimental.derive_pr_encoder \
  --checkpoint whisper/models/ggml-tiny.bin \
  --capture whisper/build/pr3905-hwx/tiny \
  --bundle whisper/build/pr3905-bundles/tiny \
  --model tiny \
  --output /new/path/to/stripped-tiny-package
```

## Prepare and verify Linux replay

The portable Python loader in `whisper/pr_encoder_replay.py` reconstructs the
checkpoint payloads, relocates the complete task chain and stages the padded
ports for all three packages. The first coefficient bank follows the commands
in buffer 0; the driver synthesizes its BAR 1 address. Constants move to BAR 2.
Small's second coefficient bank remains on BAR 8 in its own buffer. Task
register packets, dependencies and links are preserved; reversing the BAR
relocation reproduces every original command byte.

Optional preparation works on either OS and writes regenerated large payloads
to an ignored directory without accessing a device:

```sh
python -m whisper.pr_encoder_replay prepare \
  --checkpoint /external/path/whisper-tiny/model.safetensors \
  --kernels whisper/kernels/pr3905/tiny \
  --output whisper/build/pr3905-replay-plan/tiny
```

This also writes `native-layout.txt` for the C++ adapter. `plan.json` retains
SHA-256 provenance, while the native descriptor records exact file lengths,
destinations and CRC32 checks for detecting damaged payloads during loading.
All payloads, including the first descriptor used as bootstrap, are checked
before opening the accelerator.

On a base-M1 Asahi host with this repository's ANE DRM driver, verify against
the existing Mac fixtures:

```sh
python -m whisper.pr_encoder_replay verify \
  --checkpoint /external/path/whisper-tiny/model.safetensors \
  --kernels whisper/kernels/pr3905/tiny \
  --fixtures /external/path/pr3905-replay-fixtures-20261008/tiny \
  --manifest whisper/results/pr3905-m1-20261008/replay-fixtures.json \
  --warmups 3 --runs 20 \
  --output whisper/build/pr3905-asahi-tiny.json
```

Use matching `base` or `small` paths for the other models. The local fixture
directory is `whisper/build/pr3905-replay-fixtures-20261008`; its NPZ files stay
ignored and need a future external handoff. No large packed weights or HWX
files need transfer: regenerate them on Asahi from the external checkpoint.
The loader checks checkpoint, packet and fixture hashes before opening the
device. Device discovery requires native ARM64 Linux, the base-M1 device tree
and the `ane` driver; verification rejects unsupported hosts before device access.

Each model has zero-mel plus the three existing JFK mel fixtures, for 12
encoder cases. The three zero-input outputs reproduce the earlier timed Mac
direct-dispatch hashes exactly and are bitwise repeatable. The speech inputs
reuse the exact Mac arrays rather than regenerating a frontend on Linux.
Every warmup and measured replay must pass relative L2 < 0.005 and
`allclose(rtol=0.01, atol=0.03)` against its Mac encoder output, with repeatable
F16 output bits. This encoder comparison is separate from the strict full
decoder-logit gate.

Accuracy qualification requires a separate check for each multilingual model,
using its matching checkpoint, identical speech mel inputs and fixed reference
token histories. Feed both Mac-captured and Asahi-replayed encoder features
through the same model's FP32 HF decoder and compare with an independent
reference encoder through that decoder. Every full vocabulary logit vector
must have NRMSE < 0.005, matching histories/raw argmaxes and repeatable outputs.
Check the native whisper.cpp decoder against the same reference before
qualifying the complete path. Retain per-model hashes, vector counts and
pass/fail receipts; the original tiny.en 80-vector gate cannot qualify
multilingual tiny/base/small. Local Asahi speech checks now fail this separate
numerical gate for all three models; their native FP32 CPU references pass.
See the [tiny](../results/pr3905-asahi-tiny-accuracy-20261008.json),
[base](../results/pr3905-asahi-base-accuracy-20261008.json) and
[small](../results/pr3905-asahi-small-accuracy-20261008.json) receipts.
Comparisons with the exact saved Mac speech arrays remain unverified. This
broader numerical investigation is outside the completed tiny timing task in
[the repository TODO](../../todo.md).

The report separates input preparation/scratch clearing, blocking ioctl wall
time, readback and their total. Only the ioctl column has the corresponding
execution boundary to Mac's direct E5RT call; Linux scheduling and completion
wait remain included. CPU cross-K/V, decoding and cold loading are excluded.
Compare matching inputs and warmup counts, and retain all samples.

Host verification covers all 1,779 / 3,434 / 10,015 tasks, coefficient banks,
and byte-exact padded staging/readback for all 12 fixtures. The prepare CLI
also passed with external safetensors, and 24 unit tests cover packing,
relocation, rejection, submission staging and resource cleanup. See
[host preparation](../results/pr3905-m1-20261008/replay-preparation.json) and
[fixture manifest](../results/pr3905-m1-20261008/replay-fixtures.json).

## Native whisper.cpp integration

`whisper/asahi_full_encoder.cpp` implements the existing external-encoder API
with the new PR packages. It runs the complete task chain in one submission
and widens the encoder output for whisper.cpp's normal CPU cross-K/V and
decoder. Python is used for checkpoint repacking only. The adapter requires
the full 1,500-position audio context and matching model dimensions; an
explicitly requested backend fails initialization when unavailable. The
isolated patch preserves upstream CPU arithmetic and is separate from the
older strict-accuracy experiments.

Prepare an isolated checkout at the existing pinned revision on Asahi:

```sh
git -C whisper/vendor/whisper.cpp worktree add --detach ../whisper-asahi-pr3905 \
  60c0be6ac8fa71b1a2ae2dd938a31a34a508e774
python -m whisper.scripts.prepare_pr_asahi \
  --source whisper/vendor/whisper-asahi-pr3905
cmake -S whisper/vendor/whisper-asahi-pr3905 \
  -B whisper/build/pr3905-asahi-native -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_BUILD_TESTS=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DBLAS_openblas_LIBRARY=/usr/lib64/libopenblaso.so \
  -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DANE_ROOT="$PWD"
cmake --build whisper/build/pr3905-asahi-native \
  --target whisper-cli whisper-bench -j4
```

Skip worktree creation when it already exists. The preparation script is
idempotent and refuses other edits or a different revision. The OpenBLAS
paths above correspond to Fedora's OpenMP variant used by the earlier Linux
setup; use the installed paths on the target host. OpenMP enables four-worker
readback; the report records the actual worker count, with serial readback
when OpenMP is unavailable.

First run the Python fixture verifier above on the target. Then, using the
matching trained GGML model and regenerated native directory:

```sh
env WHISPER_ASAHI_ENCODER="$PWD/whisper/build/pr3905-replay-plan/tiny" \
  WHISPER_PROFILE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  whisper/build/pr3905-asahi-native/bin/whisper-cli \
  -m whisper/models/ggml-tiny.bin \
  -f whisper/vendor/whisper.cpp/samples/jfk.wav -l en -t 4 -ng

env WHISPER_ASAHI_ENCODER="$PWD/whisper/build/pr3905-replay-plan/tiny" \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  whisper/build/pr3905-asahi-native/bin/whisper-bench \
  -m whisper/models/ggml-tiny.bin -t 4 -ng
```

Repeat with the matching base/small model and payload paths. The CLI profile
separates conversion, packing, clearing, dispatch, reading and widening; the
stock bench encoder timer additionally includes CPU cross-K/V. Its stock
warmups and decoder heating differ from the isolated Mac library-substitution
harness, so keep those comparison boundaries explicit. A transcript or an
encoder replay pass does not establish the strict full-decoder-logit gate.

To reproduce the isolated Mac timing method on Asahi, start with tiny:

```sh
taskset -c 4-7 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  LD_LIBRARY_PATH=/home/asahi/.cache/applegpu-gpt2/runtime/usr/lib64 \
  whisper/.venv/bin/python -m whisper.scripts.benchmark_pr_encoder \
  --model tiny --build whisper/build/pr3905-asahi-native \
  --output whisper/build/pr3905-asahi-tiny-new
```

The command uses the unchanged Mac C++ timing harness: two fresh persistent
contexts, two excluded warmups and five measurements each, with four workers.
It reconstructs and validates the checkpoint payloads, checks three warmup and
20 measured zero-mel replays against the existing exact Mac output hash, then records every native
encode and host/dispatch/readback stage. Zero mel and its reference hash need
no NPZ transfer; the speech fixture and full-decoder gates remain separate.
The default checkpoint is `whisper/models/ggml-tiny.bin`; it must already
exist and match the package. The loader/runtime, model, payload, build and
source identities are retained in the result. The library path above supplies
the local extracted Fedora libgfortran; omit it when the system provides it.
Repeat with `--model base` or `--model small` and a new output directory. Measured
Asahi medians are **32.13 / 72.70 / 282.32 ms** for tiny/base/small; tiny uses
the [latest profiled run](../results/pr3905-asahi-tiny-profiled-20261008.json). The
[README table](../../README.md) compares the saved Mac runs. Each model has
ten native measurements and 23 exact Mac zero-mel output matches. The native API
passes frame count before mel channel
count; the initial adapter reversed those dimensions and rejected encoding.
The corrected adapter passed the public-API host regression and hardware run.
A separate three-warmup/twenty-measurement Python replay matched each exact Mac
zero-mel output hash on every call. This qualifies three zero-mel encoder cases;
speech fixtures and strict decoder accuracy remain pending.
Asahi receipts: [tiny](../results/pr3905-asahi-tiny-20261008.json),
[base](../results/pr3905-asahi-base-20261008.json),
[small](../results/pr3905-asahi-small-20261008.json).

The execution-stage comparison excludes input staging, output readback and CPU
cross-K/V. The saved [Mac direct-dispatch run](../results/pr3905-m1-20261008/direct-dispatch.json)
has three warmups and 20 measured E5RT executions; the native Linux stage comes
from the two-context encode benchmark above. The Python Linux column uses the
same three-warmup/twenty-measurement method as that Mac run.

| Model | Mac E5RT execute median | Asahi native blocking ioctl median | Asahi Python blocking ioctl median |
| --- | ---: | ---: | ---: |
| tiny | 10.89 ms | 14.21 ms | 14.17 ms |
| base | 23.29 ms | 30.22 ms | 39.93 ms |
| small | 79.41 ms | 106.86 ms | 237.63 ms |

The standalone Python measurements varied substantially: base dispatch drifted
from 30.09 to 69.93 ms, and small ranged 152.50–240.64 ms. Native context dispatch
was much steadier. Clocks were not controlled; driver profiling and clock/power
observations remain necessary before attributing this context-dependent gap.

The adapter passes host tests on all three real weighted exports and all 12
fixture transfers, using an in-memory fake submission transport. Serial and
four-worker OpenMP readback both reproduce the independently widened NumPy
reference bytes. Commands, both coefficient banks, bootstrap, padded inputs,
output conversion, rejection and resource cleanup are checked. The isolated `whisper-cli`/`whisper-bench`
build is checked on Mac with hardware initialization guarded. Linux build
and zero-mel hardware execution now pass for all three models. See
[native preparation evidence](../results/pr3905-m1-20261008/native-preparation.json).

## Execution boundary

Repacking and preparation require no Apple compiler or Accelerate. The Python
loader and native adapter run tiny/base/small on Linux with measured performance
and exact Mac zero-mel encoder agreement. Speech fixture comparison and strict
decoder accuracy still need verification.

The timed Mac E5RT program and offline HWX compile the same MIL/weights
separately. Their executed instruction identity has not been established.
The speed-reference runs also have no strict full-decoder-logit pass. Keep
them separate from the accuracy-passing comparison in
[the Mac follow-up](macos-followup.md).
