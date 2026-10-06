# Whisper on native Asahi M1: measured results

Executed on 2026-10-06 on Fedora Asahi, base M1 / T8103, kernel `7.1.13+`,
using `/dev/accel/accel0`. Trained `whisper-tiny.en` F16 weights were converted
from the pinned Hugging Face checkpoint on Linux. The converted model is
byte-identical to the model in the macOS benchmark. No new macOS compiler or
kernel dump was used; the existing `qwen35` matrix stream supplies ANE execution.

Whisper transcribes correctly here using CPU and the new native ANE projection
route. **The current ANE projection route is slower than CPU and macOS's full
ANE encoder.** Linux offloads the encoder's 24 dense projections and keeps
convolutions, attention, normalizations, activations, cross-K/V preparation and
the entire decoder on CPU. It does not replay the complete macOS encoder graph.

## Warm timings

Each row is the median of **10 warm transcriptions** of the same 11-second JFK
audio. All routes process the full 30-second padded encoder input. Times are
milliseconds; model loading and file I/O are excluded. All warmups and measured
calls produced the expected transcript words.

| OS and encoder / decoder | Encode | Short prompt setup | Decode ms/token | Decoder total | Whole transcription | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Asahi CPU / CPU, corrected reference | 305.54 | 4.80 | 3.80 | 95.82 | **417.49** | **0.0380** |
| Asahi ANE projections / CPU | 345.80 | 5.21 | 4.00 | 101.30 | **468.18** | **0.0426** |
| macOS CPU / CPU, upstream | 107.47 | 2.18 | 1.49 | 38.02 | 159.76 | 0.0145 |
| macOS full ANE encoder / CPU | 16.03 | 2.19 | 1.53 | 38.84 | 68.37 | 0.0062 |

The macOS rows come from the retained [M1 warm benchmark](benchmark-macos.md).
macOS uses Accelerate; this Linux build uses native ggml CPU without an external
BLAS library, with its existing tiled matrix implementation enabled. The Linux
reference also uses FP32 activations/K/V and the corrections described below.
These rows measure the available implementations, rather than isolate the OS.

The Asahi ANE hybrid takes 1.12 times the corrected CPU reference's whole
latency. It transcribes the 11-second clip in 0.47 seconds, about 23.5
times faster than real time. It is not on par with the retained macOS ANE route.
The current stream accepts 32 audio positions per submission, requiring
**1,128 hardware submissions per encoder**. A full exported graph or larger
batches could reduce dispatch and host overhead; that improvement is unmeasured.

| Asahi route | Observed whole-transcription range, 10 measured calls |
| --- | ---: |
| Corrected CPU / CPU | 401.70–469.09 ms |
| ANE projections / CPU | 456.88–521.38 ms |

Both Linux routes use four workers pinned to this M1's performance cores 4–7.
The desktop remained active; clocks, temperature and background activity were
not controlled. Hardware tests were serialized using the shared ANE/GPU locks.
The earlier two-grid run used unrestricted CPU affinity and is retained
separately; its timing does not isolate the effect of removing the second grid.

## Encoder speed investigation

The earlier [one-grid benchmark](../results/asahi-20261006/validated-native/summary.json)
measured 721.41 ms encode and 810.35 ms whole transcription. The optimized
candidate measures 345.80 ms and 468.18 ms: **2.09× faster encode**, **1.73×
faster whole transcription**. These are successive implementations on the same
machine, not simultaneous controlled trials. The decoder is slower in the new
candidate because FP32 activations/K/V and exact GELU improve reference parity.

Opt-in profiling found two large host costs: scalar `ldexp` calls during
activation scaling/restoration (about 210 ms), and reads from the driver's
uncached output mapping (about 180 ms). Vectorized power-of-two multiplication
retains an `ldexp` fallback for extreme exponents. Four workers now copy
disjoint cache-line ranges; nested OpenMP is enabled because ggml invokes the
custom projection inside its own parallel region. Shared Qwen native code
retains one read worker by default.

The CPU remainder also used an untiled matrix path. Enabling the existing
tiled implementation with FP32 accumulation, including F16-weight/F32-activation
GEMV, reduces convolution and cross-K/V work. Fixing a GCC NEON macro selects
the existing wider attention matrix kernel. These changes require no new
macOS export and no driver modification. Before/after profiles and uncached
copy measurements are retained alongside the benchmark.

| Profiled host work | Before, warm median | After, warm median |
| --- | ---: | ---: |
| Scale activations | 119.93 ms | 8.67 ms |
| Restore projection scale | 89.78 ms | 3.81 ms |
| Read uncached ANE outputs | 180.22 ms | 86.74 ms |
| Submit and wait in ioctl | 44.28 ms | 44.09 ms |

These profiling runs have three measurements each and are separate from the
10-call timing table. The submission count is unchanged at 1,128. The remaining
gap to macOS comes from the partial encoder offload, CPU work and uncached
transfer overhead; the existing projection stream does not establish the
speed of a complete exported encoder on Linux.

## Numerical validation

The same clips were run separately on CPU and ANE, capturing exact mel input,
encoder output, complete 51,864-value raw decoder logits and token histories.
An independent Hugging Face encoder ran on the exact captured native mel input.
The HF decoder also evaluated every captured prefix using its own encoder
features. The NRMSE gate remains **0.5%** for encoder CPU/ANE comparisons and
every complete CPU/ANE, CPU/HF and ANE/HF decoder vector; the independent HF
encoder cosine gate remains **0.999**.

| Audio | ANE encoder NRMSE vs CPU | HF encoder cosine, ANE | Maximum full-logit NRMSE vs CPU | Maximum full-logit NRMSE vs HF | Raw argmax matches, CPU and HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| JFK, 11 s | 0.0743% | 0.99999959 | 0.0995% | 0.1357% | 25/25 |
| JFK first 5 s | 0.0592% | 0.99999974 | 0.2498% | 0.3331% | 8/8 |
| JFK repeated with 1 s silence, 23 s | 0.0754% | 0.99999966 | 0.1356% | 0.0839% | 47/47 |

All captured token histories match exactly, all CPU/ANE transcript words match,
and both CPU and ANE pass the HF cosine gate on all three clips. Every ANE
encode verifies 24 projections, 24 resident plans and 1,128 actual submissions.
Missing hardware or a failed submission aborts; there is no CPU fallback for
the requested ANE projections. The 24 independent matrix checks also pass for
384×384, 384×1,536 and 1,536×384 with batches 1/8/28/32, against float64 products.
Serial and four-worker reads produce byte-identical results. Their worst NRMSE
is 0.0298% and their native kernel hashes match this run. Shared Qwen regression
checks also pass across real model dimensions, extreme input amplitudes and
compensated Mirai-style matrices (35 actual submissions).

These are three related constructed clips, not a diverse audio corpus or a
word-error-rate evaluation. Validation covers the F16 `tiny.en` model, not
other Whisper sizes or quantized checkpoints.

## Corrections made before accepting the measurements

Initial transcription passed on the full JFK clip, but the 5-second clip
exposed errors in both the CPU reference and the hybrid. The pinned upstream
encoder exposed 1,536 K/V positions to unmasked flash attention while copying
only 1,500 real positions. The extra 36 zero-filled positions changed softmax
probabilities. Using actual-length CPU K/V views reduced the short-clip HF
encoder discrepancy from about 5.73% NRMSE to 0.0718% on the final ANE path.

The stronger independent decoder check found the same padding problem in
decoder cross-attention: both native routes previously had about 2.86% full-logit
NRMSE versus HF despite matching each other's argmaxes. Cross-attention views
now expose only 1,500 real audio positions while preserving each layer's padded
allocation stride. All 80 full vectors now pass the unchanged 0.5% HF gate.

The isolated build also selects the existing widened FP32 NEON dot path for
CPU operations on F16 storage, FP32 activations and K/V caches, and exact GELU
in both encoder and decoder. Learned weights retain F16 storage. These apply
to both CPU and ANE execution modes, preserving a matched reference. The
unmodified upstream checkout and separate CPU binary remain available; their
earlier separate-process smoke timings are not these warm measurements.

A two-grid version first passed all numerical gates with 2,256 submissions.
After fixing the attention length, one grid passed the same gates, so the
final runtime uses all 32 rows for distinct audio positions and halves the
submission count. All failed numerical experiments and the successful
two-grid receipt remain separate from the final results.

## Reproduce and inspect

Follow [native Asahi setup](asahi-native.md) for model preparation, building,
transcription and the complete verification/benchmark commands. The runner
uses two persistent contexts per backend, two excluded warmups per context
and five measured calls per context, reversing backend order in the second
round. Settings: English, greedy best-of-one, temperature zero, no decoding
fallback, no timestamps, CPU decoder, flash attention, four workers.

The encode timer includes CPU cross-attention K/V preparation. This clip has
a two-token decoder prompt, charged to `batchd`, and 24 single-token
evaluations charged to `decode`. Decoder total is `batchd + prompt + decode`.
The external whole timer wraps `whisper_full`, including mel, sampling and
host orchestration. Medians are computed per metric, so rounded columns need
not sum exactly.

Retained evidence:

- [Optimized validation and 20 measured calls](../results/asahi-20261006/optimized-native/summary.json)
- [Optimized execution log](../results/asahi-20261006/optimized-native/execution.log), with six numerical-run and four warm-context logs in the same directory
- [Before profile](../results/asahi-20261006/profile-before.json) and [after profile](../results/asahi-20261006/profile-after.json)
- [Serial/parallel matrix checks](../results/asahi-20261006/parallel-read-matrices.json) and [shared Qwen checks](../results/asahi-20261006/qwen-shared-native-check.json)
- [Uncached mapping copy measurements](../results/asahi-20261006/ane-mapping-copies.jsonl) and [measurement tool](../scripts/benchmark_ane_mapping.c)
- [Earlier one-grid benchmark](../results/asahi-20261006/validated-native/summary.json)
- [Failed numerical experiments](../results/asahi-20261006/accuracy-investigation.json)
- [Successful two-grid receipt](../results/asahi-20261006/validated-two-grid/summary.json)
- [First one-grid smoke and matrix checks](../results/asahi-20261006/ane-smoke.json)
- [Pinned CPU smoke](../results/asahi-20261006/cpu-smoke.json)
- [Benchmark runner](../scripts/benchmark_asahi.py), [source preparation](../scripts/prepare_asahi.py), and [native projection adapter](../asahi_encoder.cpp)

The final report records model, checkpoint, source, binary and artifact SHA-256
hashes. Large weights and numerical arrays stay in the ignored workspace under
`whisper/models/` and `whisper/.cache/asahi-20261006/validated-cross-real-context/`.
The pending full-graph macOS export recovery is tracked in the root `todo.md`.
