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
| Asahi CPU / CPU, corrected reference | 486.65 | 3.48 | 2.57 | 65.24 | **567.62** | **0.0516** |
| Asahi ANE projections / CPU | 721.41 | 3.91 | 2.71 | 68.85 | **810.35** | **0.0737** |
| macOS CPU / CPU, upstream | 107.47 | 2.18 | 1.49 | 38.02 | 159.76 | 0.0145 |
| macOS full ANE encoder / CPU | 16.03 | 2.19 | 1.53 | 38.84 | 68.37 | 0.0062 |

The macOS rows come from the retained [M1 warm benchmark](benchmark-macos.md).
macOS uses Accelerate; this Linux build uses native ggml CPU without BLAS.
The Linux reference also uses the numerical corrections described below.
These rows measure the available implementations, rather than isolate the OS.

The Asahi ANE hybrid takes 1.43 times the corrected CPU reference's whole
latency. It still transcribes the 11-second clip in 0.81 seconds, about 13.6
times faster than real time. It is not on par with the retained macOS ANE route.
The current stream accepts 32 audio positions per submission, requiring
**1,128 hardware submissions per encoder**. A full exported graph or larger
batches could reduce dispatch and host overhead; that improvement is unmeasured.

| Asahi route | Observed whole-transcription range, 10 measured calls |
| --- | ---: |
| Corrected CPU / CPU | 512.04–659.18 ms |
| ANE projections / CPU | 786.44–952.49 ms |

Both Linux routes use four workers pinned to this M1's performance cores 4–7.
The desktop remained active; clocks, temperature and background activity were
not controlled. Hardware tests were serialized using the shared ANE/GPU locks.
The earlier two-grid run used unrestricted CPU affinity and is retained
separately; its timing does not isolate the effect of removing the second grid.

## Numerical validation

The same clips were run separately on CPU and ANE, capturing exact mel input,
encoder output, complete 51,864-value raw decoder logits and token histories.
An independent Hugging Face encoder ran on the exact captured native mel input.
The CPU/ANE NRMSE gate remains **0.5%** for encoder features and every full
decoder vector; the independent HF encoder cosine gate remains **0.999**.

| Audio | ANE encoder NRMSE vs CPU | HF encoder cosine, ANE | Maximum full-logit NRMSE vs CPU | Raw logit argmax matches |
| --- | ---: | ---: | ---: | ---: |
| JFK, 11 s | 0.0676% | 0.99999974 | 0.2594% | 25/25 |
| JFK first 5 s | 0.0638% | 0.99999977 | 0.3100% | 8/8 |
| JFK repeated with 1 s silence, 23 s | 0.1045% | 0.99999962 | 0.2549% | 47/47 |

All captured token histories match exactly, all CPU/ANE transcript words match,
and both CPU and ANE pass the HF cosine gate on all three clips. Every ANE
encode verifies 24 projections, 24 resident plans and 1,128 actual submissions.
Missing hardware or a failed submission aborts; there is no CPU fallback for
the requested ANE projections. The 12 independent matrix checks also pass for
384×384, 384×1,536 and 1,536×384 with batches 1/8/28/32, against float64 products.
Their worst NRMSE is 0.0298% and their native kernel hashes match this run.

These are three related constructed clips, not a diverse audio corpus or a
word-error-rate evaluation. Validation covers the F16 `tiny.en` model, not
other Whisper sizes or quantized checkpoints.

## Corrections made before accepting the measurements

Initial transcription passed on the full JFK clip, but the 5-second clip
exposed errors in both the CPU reference and the hybrid. The pinned upstream
encoder exposed 1,536 K/V positions to unmasked flash attention while copying
only 1,500 real positions. The extra 36 zero-filled positions changed softmax
probabilities. Using actual-length CPU K/V views reduced the short-clip HF
encoder discrepancy from about 5.73% NRMSE to below 0.07% on the final ANE path.

The isolated build also selects the existing widened FP32 NEON dot path for
CPU operations on F16 storage and exact encoder GELU. Those corrections apply
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

- [Final validation and 20 measured calls](../results/asahi-20261006/validated-native/summary.json)
- [Final execution log](../results/asahi-20261006/validated-native/execution.log), with six numerical-run and four warm-context logs in the same directory
- [Failed numerical experiments](../results/asahi-20261006/accuracy-investigation.json)
- [Successful two-grid receipt](../results/asahi-20261006/validated-two-grid/summary.json)
- [First one-grid smoke and matrix checks](../results/asahi-20261006/ane-smoke.json)
- [Pinned CPU smoke](../results/asahi-20261006/cpu-smoke.json)
- [Benchmark runner](../scripts/benchmark_asahi.py), [source preparation](../scripts/prepare_asahi.py), and [native projection adapter](../asahi_encoder.cpp)

The final report records model, checkpoint, source, binary and artifact SHA-256
hashes. Large weights and numerical arrays stay in the ignored workspace under
`whisper/models/` and `whisper/.cache/asahi-20261006/validated-one-grid-pcores/`.
The pending full-graph macOS export recovery is tracked in the root `todo.md`.
