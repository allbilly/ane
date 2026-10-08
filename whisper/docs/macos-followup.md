# Mac follow-up and CPU ablations, 2026-10-07

The shared native benchmark has run on the M1 MacBook Air. OpenBLAS reproduces
the recorded Asahi cross-K/V stage: **14.06 ms on Mac versus 14.16 ms on Asahi**,
compared with **4.75 ms using Accelerate**. This identifies a substantial CPU
library gap. It does not establish a fix that makes Asahi match Mac. Further flag
ablations were stopped at the user's request.

All original ANE configurations still fail the unchanged full decoder logit gate.
Their timings are excluded from the accuracy-qualified performance comparison.
The shared FP32 CPU reference passes all 80 vectors for each configuration, and
CPU/ANE histories and raw argmaxes match throughout.

## Accuracy-qualified encoder comparison

The following runs use the same binary and libraries on the M1, four workers,
the same 11-second clip and reversed algorithm/backend order in round two.
Every CPU and paired ANE run passes all 80 full logit vectors across the three
clips. See [batched encoder attention](encoder-attention.md) for the complete
accuracy and warm-run receipt.

Encoder time includes CPU cross-attention K/V preparation. The retained Asahi
rows below also pass every one of the 80 full CPU/ANE/HF vectors and all encoder
gates in the [optimized native receipt](../results/asahi-20261006/optimized-native/summary.json).
Their 10-call medians were recomputed from its individual measured calls.

| OS | Encoder / attention | BLAS library | CPU Accelerate vector routines | Encode |
| --- | --- | --- | --- | ---: |
| macOS | FP32 CPU / flash | Apple Accelerate BLAS | On | 176.06 ms |
| macOS | FP32 CPU / batched BLAS | Apple Accelerate BLAS | On | 146.88 ms |
| macOS | Paired ANE / flash | Apple Accelerate BLAS | On | 272.75 ms |
| macOS | Paired ANE / batched BLAS | Apple Accelerate BLAS | On | 205.92 ms |
| Asahi | Corrected CPU / flash | None; native ggml/tinyBLAS | Unavailable | 305.54 ms |
| Asahi | 32-position ANE projections / flash | None; native ggml/tinyBLAS | Unavailable | 345.80 ms |

These are accuracy-qualified available implementations. Mac measurements are
from October 8; the Asahi measurements are from October 6, before external
OpenBLAS integration. The Asahi projection route uses 1,128 submissions; the
Mac paired route uses 24 and different projection arithmetic. Same M1 class,
checkpoint, clip and four workers do not isolate the OS or BLAS effect, and
clocks/desktop load were uncontrolled.

No verified native Asahi result for the new paired implementation is available.
The earlier whole-run scaling projection has been withdrawn: its accuracy and
runtime differences make it too weak to guide the implementation decision.
Both accurate Mac ANE variants and the retained Asahi hybrid remain slower than
their respective CPU comparisons.

## Original fast graph: diagnostic timings only

Same M1, pinned tiny.en checkpoint, 1,779-task encoder, four workers, and
11/5/23-second clips. Each configuration has two excluded warmups and ten
measurements per clip per round, with CPU/ANE order reversed in round two. Every
repetition is retained in the [ablation receipt](../results/macos-cpu-ablation-20261007.json).
Configurations ran sequentially, not interleaved; active desktop load and unfixed
clocks limit estimates of each change's effect.

The medians below describe ANE encoder plus CPU decoder on the 11-second clip.
Cross-K/V stage time includes graph/backend overhead. The eight matrix-call sum
uses narrower allocation, conversion, thread setup and GEMM boundaries.

| BLAS | CPU Accelerate | ggml tiled kernels | Cross-K/V stage | Eight matrix calls | Encode | Decoder | Whole |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Accelerate | On | On | 4.75 ms | 3.58 ms | 15.94 ms | 74.57 ms | 104.21 ms |
| OpenBLAS | On | On | 14.06 ms | 12.68 ms | 25.34 ms | 67.09 ms | 106.20 ms |
| OpenBLAS | Off | On | 16.06 ms | 14.17 ms | 27.94 ms | 81.38 ms | 123.31 ms |
| OpenBLAS | Off | Off | 19.05 ms | 16.75 ms | 31.28 ms | 79.41 ms | 128.06 ms |
| Accelerate | Off | On | 5.29 ms | 3.85 ms | 17.18 ms | 87.24 ms | 121.89 ms |
| Recorded Asahi | Unavailable | On | 14.16 ms | Separate profile | 31.18 ms | No matching value retained here | 142.90 ms |

`GGML_ACCELERATE=OFF` disables ggml's Accelerate vector routines, while the separate
Accelerate BLAS backend remains enabled with `GGML_BLAS_VENDOR=Apple`.
`GGML_LLAMAFILE=OFF` changes matrix algorithms and leaves NEON fallback routines
enabled. OpenBLAS reports `0.3.34 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP neoversen1
MAX_THREADS=56`, uses four actual BLAS threads, and loads one OpenMP runtime.
No private Accelerate AMX disable flag was assumed or tested. Replacing BLAS
changes provider, algorithms and threading as well as the instruction path.

Disabling CPU Accelerate increases decoding time in these runs. Disabling the
tiled kernels gives no decoding improvement and changes rounding. A scalar-dot
control was built but not run after the request to stop; no scalar timing or
global NEON-disable claim is made.

A later [cross-K/V fusion experiment](cross-kv-fusion.md) uses the same OpenBLAS
binary and libraries for both layouts, reversing layout order in round two.
It preserves every native logit vector and encoder boundary byte, and passes
the CPU's independent 80-vector HF gate. Cross-K/V improves about 7.8%, but
whole transcription does not improve and memory grows. The experiment remains
opt-in; its later timing conditions differ from the library ablations above.

The roughly 20 ms gap between the closest Linux-library configuration and the
recorded Asahi whole time is unresolved. Medians of stages do not necessarily add
to the median whole time. The old Mac fast 66.81 ms result used different CPU
accuracy settings; it is not the shared FP32 decoder baseline shown here.

## Instruction and host evidence

The [environment receipt](../results/macos-followup-20261007/cpu-profile/environment.json)
retains OS/compiler/runtime identities and native binary hashes. The
[sample summary and loaded-image map](../results/macos-followup-20261007/cpu-profile/sample-summary.txt)
and [BLAS assembly](../results/macos-followup-20261007/cpu-profile/blas-assembly.txt)
map the sampled BLAS address to its loaded UUID and file offset. At that location,
opcode `0x00201189` is AMX FMA32. The
[decoding receipt](../results/macos-followup-20261007/cpu-profile/amx-decode.json)
records the mapping and [primary encoding source](https://github.com/corsix/amx/blob/main/aarch64.h).
This establishes AMX use by the selected routine beyond library initialization.

The selected [ggml assembly](../results/macos-followup-20261007/cpu-profile/ggml-assembly.txt)
shows FP16-to-FP32 NEON widening and FP32 accumulation for hot matrix/vector
products, plus attention. Asahi's recorded profile also identifies NEON ggml and
`sgemm_kernel_NEOVERSEN1`; NEON is not missing there. Generic matrix kernels serve
several projections. The [decoder profile](decoder-profile.md) now separately
attributes prompt/token vocabulary, online attention and probability processing
on all three clips. All 160 native vectors and boundaries remain byte identical,
and 606 transcriptions are verified. Actual compiler flags, loaded-image UUIDs,
representative stack paths and targeted vocabulary/softmax assembly are retained.

Mac `sample` captures wall-clock thread stacks, not user cycles or fractions of
whole transcription time. `powermetrics` requires administrator access here;
CPU/ANE clocks, power and exact physical core placement are explicitly unavailable.
E5RT exposes blocking API time without hardware execution timestamps. The
baseline execute median is 11.01 ms; recorded Asahi dispatch is about 14 ms and
includes completion waiting and scheduling. These measurements do not establish
a clock diagnosis or a parity fix.

## Exact fixture handoff

The [handoff manifest](../results/macos-followup-20261007/handoff.json) validates
and stages existing Whisper NPZ bytes without regenerating values. It records
file/array hashes, shapes, dtypes, logical strides and native padding. Arrays are
under `whisper/build/macos-handoff-20261007/whisper-fixtures`; no new archive was
made. All six Mac fixture replays, original and wrapped packages on all three
clips, preserve original outputs byte for byte. The
[replay receipt](../results/macos-followup-20261007/encoder-replay.json) retains
preparation, dispatch, readback and total times.

The existing Qwen `uzu-macos-all.npz`, `native-macos-all.npz` and report are staged
under `whisper/build/macos-handoff-20261007/qwen-oracles`. All 23 prefixes match
byte for byte on Mac, and the NPZ/report hashes match the committed package.
Asahi and its all-prefix arrays are not accessible from this Mac. External
transfer, strict Linux fixture replay and the first differing Qwen prefix/layer
comparison remain pending.

Restage from existing captures with:

```sh
whisper/.venv/bin/python -m whisper.scripts.prepare_macos_handoff \
  --fixtures qwen35/local-results/whisper-fast-recapture-20261007-complete \
  --oracle qwen35/local-results/macos-followup-20261005T175748Z/vendor-rerun \
  --output whisper/build/macos-handoff-new
```

## Numerical correction

The original graph's matched native maximum ANE logit NRMSE against HF is
3.08% / 5.49% / 2.54% on the 11/5/23-second clips, above the unchanged 0.5% gate.
Every raw argmax matches. The Mac numerical controls are retained in the
[precision receipt](../results/macos-followup-20261007/precision-experiments.json).

The six-GELU `erf` rewrite reduces error but still fails. The 18-mean matmul
rewrite regresses in the full graph even though a standalone probe on measured
activations is accurate. Neither correction replaced the compact kernels.

Host FP32 normalization/attention/residuals/nonlinearities with all 24 projections
using paired ANE arithmetic passes all 80 vectors at **0.120% / 0.426% / 0.220%**,
with matching argmaxes and bitwise repeated outputs. It takes roughly **1.03
seconds per Python encoder**. These times cover the entire CPU/ANE encoder,
including packing and combining products. This demonstrates a precision route;
the [native C++ integration](native-precision.md) now passes all 80 vectors with
maximum HF logit NRMSE 0.1685% / 0.4484% / 0.2439%. Its warm 11-second native
encode/whole medians are 366.88 / 483.76 ms, versus 252.24 / 369.01 ms for the
matched CPU baseline in that run. Native accuracy is corrected on Mac, while
speed and Linux native execution remain unresolved.

The [batched encoder attention follow-up](encoder-attention.md) keeps all 80
vectors below the same 0.5% gate, with maximum paired/HF NRMSE
0.1961% / 0.4594% / 0.2058%. In a new same-binary comparison, native paired
encode improves 272.75 → 205.92 ms and whole transcription
393.46 → 328.84 ms, about 16% faster overall. The CPU batched path remains
faster at 268.31 ms whole. All 288 transcriptions and actual attention BLAS
profiles are retained; graph memory increases and Linux performance remains
unverified. No further library/flag ablations were run.

Reusable sources are `whisper/precision_encoder.py` and
`experimental/whisper_mil_precision.py`. Learned weights, compiled programs,
raw arrays and full traces remain in ignored build/local-result paths.
`whisper.scripts.summarize_cpu_ablations` consolidates completed runs without
performing any new ablations.
