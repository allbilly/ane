# Paired projection replay on Linux

The 24 accuracy-corrected paired projection programs now have checkpoint-only
H13G replay packages. All command, constant and coefficient payloads reconstruct
byte for byte from the exports. The package contains 42,728 bytes across 11
files, including provenance and three shared instruction/layout templates;
14,155,776 learned coefficient bytes come from the external pinned checkpoint.
No new archive was created.

Exported-HWX execution now passes on Asahi through the shared paired Python
encoder. The [new Linux receipt](../results/asahi-paired-20261008.json) validates
all 80 full-vocabulary decoder vectors against the independent HF model:

| Clip | Vectors | Maximum logit NRMSE | Raw argmax matches |
| --- | ---: | ---: | ---: |
| JFK | 25 | 0.0949% | 25/25 |
| First five seconds | 8 | 0.3593% | 8/8 |
| Repeated JFK | 47 | 0.2894% | 47/47 |

All encoder outputs repeat byte for byte. Each encode submits all 24 programs
and 32 hardware tasks. The three exported shapes additionally pass raw
sparse-input hardware comparisons, with NRMSE below 0.007% and repeatable bits.
These checks use exact locally captured native mel inputs and fixed native token
prefixes with the shared HF decoder. Native C++ integration and warm timings
now also pass, as detailed below. Exact Mac speech-array comparison remains
unverified. The
historical [reconstruction proof](../kernels/tiny-en-paired/proof.json) retains
its preparation-only scope. The original fused fast graph still fails its
separate accuracy gate.

## Arithmetic and layout

These are the same exact MIL/weight files used by the passing
[native Mac paired encoder](native-precision.md), verified against its complete
80-vector accuracy receipt and the pinned checkpoint. Each projection splits
the contraction into two groups and uses two FP16 rounding grids, at gains
1 and 1.375. High and residual input planes for each grid occupy four consecutive
1,500-position ranges. The residual scale is 4,096. Separate products combine
in FP32 in the same grid/partition order; FP32 bias stays on the host.
Host convolution, attention, normalization, GELU and residual additions are
also unchanged.

| Input × output features | Programs per encode | Tasks per program | Coefficient tiles | Input bytes | Output bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 384 × 384 | 16 | 1 | 64 | 4,620,288 | 9,240,576 |
| 384 × 1,536 | 4 | 2 | 192 | 4,620,288 | 36,962,304 |
| 1,536 × 384 | 4 | 2 | 64 | 18,481,152 | 9,240,576 |

This is 24 logical submissions and 32 hardware tasks per encode, rather than
the historical 1,128 submissions of the 32-position projection path. Those
counts describe the exported structure and verified Python/native Linux execution.

The hardware input is `[1,k,1,6000]` and output `[1,2n,1,6000]`, with a
**12,032-byte channel stride**. A dense logical channel contains 12,000 bytes;
the additional 32 bytes are zero padding in the packer. Input/output banks are
4 and 5. Only active BAR selectors in task-header words 8/9 change: original
constant bank 1 maps to replay bank 2, and coefficient bank 6 maps to driver
bank 1. Register packets, task IDs, dependencies and next-task pointers are
preserved and checked separately.

Compiler coefficient packing uses transposed 16/8-output tiles. Every byte
is covered once, with no overlap or unexplained padding. The three command
and constant templates, port layouts and tile maps are identical across all
captured projections of their respective shape despite different weights.
Only checkpoint coefficient bytes vary.

## Reconstruct on Linux

Use the external tiny.en safetensors checkpoint SHA-256
`db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`:

```sh
whisper/.venv/bin/python -m whisper.paired_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --output whisper/build/asahi-paired-payloads-new
```

The reconstruction has been executed on both hosts with no macOS runtime or
compiler calls. It writes all 24 command/constant/coefficient/bootstrap
payload sets, checkpoint-bound grouped weight blobs and their native layout
descriptors. Choose a new output directory.
Every payload hash is checked before writing.

The eleven tests in `experimental/test_whisper_paired_kernel.py` verify all 24
payload hashes, 24/32 submission/task geometry, all four temporal planes and
zero padding for every shape. They reject dense or altered hardware layouts,
missing/overlapping coefficient tiles, non-FP16-exact weights, modified command
assets, incomplete program manifests, altered native payloads/descriptors,
partial native execution/fallbacks and a single failed full logit vector.
The tests require the external pinned checkpoint.

## Repeat the export/package preparation on Mac

The source programs must first pass
`whisper.scripts.prepare_macos_precision.validate_programs`. Export each
unchanged program with `experimental.capture_macos_program.export`, using
`gpt2/training/build/dump_hwx` and a new directory for each program. The validated
24 exports for this preparation are retained under the ignored
`whisper/build/asahi-paired-export-20261008/<projection>/hwx` paths.

Then package their instruction/layout recipes:

```sh
whisper/.venv/bin/python -m whisper.scripts.package_asahi_precision \
  --programs whisper/build/macos-batched-attention-programs-20261008 \
  --exports whisper/build/asahi-paired-export-20261008 \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --native-receipt whisper/results/macos-native-precision-20261007.json \
  --output whisper/build/asahi-paired-kernels-new
```

This accepts only the exact pinned programs and complete passing native
receipt. The source/compiled file hashes and compiler identity remain in the
compact proof. Offline ANECCompile and E5RT compile the same MIL separately;
matching their actual executable identity is unproven. The checkpoint-rebuilt
exported command/weight streams now execute on Asahi in both replay paths.

## Native whisper.cpp on Asahi

`whisper/macos_precision.cpp` now shares its original packing and combination
arithmetic with Linux. `whisper/paired_asahi.h` supplies only the DRM transport,
validates task chains and payload checksums, and shares maximum-sized cached
input/output views and padded BOs across all 24 projection plans. The native
grouped weights are checked byte for byte against whisper.cpp's F16 tensors.
The existing four graph workers partition packing, staging, readback and
combination; one worker submits each complete task chain. This avoids a nested
readback team. Linux requires a four-thread OpenMP graph, rejects unwritten or
nonfinite outputs, and has no ANE execution fallback. Mac's E5RT path is preserved.

The [native receipt](../results/asahi-native-precision-20261008.json) passes every
full-vocabulary decoder vector against CPU and an independently evaluated HF
encoder/decoder. Counts are 25 / 8 / 47; worst paired/HF NRMSE is
0.1479% / 0.4079% / 0.2655%. Every history and raw argmax matches. Repeated
mel/features/full decoder traces are byte identical on all three clips, and
match the [initial native implementation](../results/asahi-native-precision-baseline-20261008.json)
exactly after parallelizing the transport. Twenty-six packing/replay/paired
tests pass. The Apple conditional branch passes a Linux compiler syntax check;
that is not a new macOS runtime test.

| 11-second JFK, native path | Encode + cross-K/V | Decoder total | Whole |
| --- | ---: | ---: | ---: |
| Saved macOS CPU + Apple BLAS | 252.24 ms | 85.94 ms | 369.01 ms |
| Saved macOS paired + Apple BLAS | 366.88 ms | 87.14 ms | 483.76 ms |
| Asahi CPU + OpenBLAS | 216.57 ms | 72.95 ms | 307.35 ms |
| Asahi paired + OpenBLAS | 624.32 ms | 78.79 ms | 723.42 ms |

Both hosts use two contexts per backend, two excluded warmups and ten measures
per clip/context, four workers and the full 30-second encoder context. The
three-clip Asahi run retains 144 warm/timed transcriptions. Mac/Asahi use different
BLAS libraries and locally computed mel; exact Mac speech arrays remain absent.
Clocks are unfixed. All Asahi receipts preserve build/library/weight/source and
trace identities, every sample, gate and stage profile.

Linux spends 379.37 ms reading 332,660,736 padded output bytes through the
driver's write-combined mappings, plus 31.93 ms staging and 12.70 ms copying into
dense views. Blocking ioctl takes 13.34 ms. Packing/combination takes
10.43 / 12.50 ms, compared with 26.85 / 29.13 ms in the initial single-worker
native control. Graph-worker parallelization preserves all bits but leaves
the readback cost. These are enclosing wall timers, not ANE hardware timestamps;
component medians need not sum to the enclosing median. The numerical control
is accurate, but remains slower than CPU and the saved Mac result.

Prepare an isolated worktree at revision
`60c0be6ac8fa71b1a2ae2dd938a31a34a508e774`, then use the shared preparation:

```sh
whisper/.venv/bin/python -m whisper.scripts.prepare_native --backend asahi \
  --encoder paired --source whisper/vendor/whisper-asahi-paired
```

Build with the existing OpenBLAS CMake flags from
[PR native setup](pr3905-packing.md#native-whispercpp-integration), substituting
source/build directories `whisper/vendor/whisper-asahi-paired` and
`whisper/build/asahi-paired-native`. Set `LD_LIBRARY_PATH` to the extracted
libgfortran directory during configuration/build as well as execution. Use
the reconstruction command above to create a fresh paired payload directory.
The tested benchmark and shared collector are:

```sh
taskset -c 4-7 env OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
  LD_LIBRARY_PATH=/home/asahi/.cache/applegpu-gpt2/runtime/usr/lib64 \
  whisper/.venv/bin/python -m whisper.scripts.benchmark_native \
  --backend asahi --encoder paired --build whisper/build/asahi-paired-native \
  --precision-programs whisper/build/asahi-paired-payloads-new \
  --profile-stages --warmups 2 --runs 10 --rounds 2 \
  --output whisper/build/asahi-paired-native-new
whisper/.venv/bin/python -m whisper.scripts.summarize_native_precision \
  --validation whisper/build/asahi-paired-native-new \
  --output whisper/build/asahi-native-precision-new.json
```

The benchmark requires the pinned HF checkpoint and its matched GGML conversion
(`whisper/models/hf-ggml/ggml-model.bin`), validates all native payloads before
opening hardware, sets OpenBLAS to one thread, and admits timings only after
the unchanged complete accuracy and repeatability gates. Native opt-in is
`WHISPER_ASAHI_PRECISION=PAYLOAD_DIRECTORY`.

## Performance scope

The accurate paired path is a comparison control for numerical corrections.
Its Mac implementation remains slower than CPU. It is not an established
Asahi speed solution. The original fast Asahi graph's recorded 142.90 ms
whole time is retained as the diagnostic performance baseline while it fails
the strict accuracy gate. A cheaper numerical correction remains required.
Measured native Linux accuracy/performance follows; no projected whole-run time
is accepted as evidence.

`whisper/precision_encoder.py` shares the same packing, four-plane combination,
bias handling and HF host operations between Mac and Linux. The Linux transport
is `whisper/paired_replay.py`; it validates all executable/weight payloads before
opening the device and preserves the 12,032-byte hardware strides. Cached CPU
views avoid NumPy scalar arithmetic on uncached mappings. The existing
`qwen35/ane_matmul.c` cache-line reader can supply four-worker readback:

```sh
cc -std=gnu11 -O3 -march=armv8.2-a+fp16 -fopenmp -shared -fPIC \
  qwen35/ane_matmul.c -o whisper/build/libpaired-read.so -lm
taskset -c 4-7 env OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  OMP_WAIT_POLICY=PASSIVE HF_HUB_OFFLINE=1 \
  WHISPER_ASAHI_READER="$PWD/whisper/build/libpaired-read.so" \
  whisper/.venv/bin/python -m experimental.whisper_precision --exported-paired \
  --traces whisper/build/shared-validation-20261007 \
  --output whisper/build/asahi-paired-new.json
```

The native reference traces supply each clip's exact mel and frozen prefixes;
use a new output filename. The command validates every full logit vector,
raw argmax, 24/32 execution count and repeated encoder output. Cached byte copies
reduced observed Python encode calls from roughly 16.1 s to 2.9 s; shared NEON
readback reduced them further to 1.9 s. Encoder bits and all 80 logit checks are
identical across those three implementations. These first-call diagnostic
timings exclude model loading and are not the native warm benchmark boundary.
