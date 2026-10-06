# CPU, Metal and ANE benchmark

Measured on 2026-10-06 on this Apple M1 MacBook Air (8 GB), macOS 27.0.1.
Model: trained `whisper-tiny.en`, F16 weights, same HF checkpoint as the
[verification report](macos-test-results.md). Audio: the 11.0-second JFK sample.
All measured and warmup calls returned the expected transcript words.

## Results

Each row is the median of **10 warm transcriptions**. All times are milliseconds;
lower is better. Model loading, shader/model compilation and file I/O are
excluded. The encoder uses the full **30-second padded context** on every route.

| Encoder | Decoder | Encode | Decoder total | Single-token decode (ms/token) | Whole transcription | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | CPU | 107.47 | 38.02 | 1.49 | 159.76 | 0.0145 |
| Metal GPU | Metal GPU | 52.59 | 62.46 | 2.48 | 156.57 | 0.0142 |
| ANE | CPU | **16.03** | **38.84** | **1.53** | **68.37** | **0.0062** |
| ANE | Metal GPU | 31.80 | 108.93 | 4.42 | 175.02 | 0.0159 |
| ANE (Python) | ANE (Python) | 23.96 | 81.14 | 3.14 | 139.67 | 0.0127 |

The first four rows use the same whisper.cpp library and transcription settings.
The Python row uses ANEForge's separate implementation and timing wrappers;
its stage boundaries differ, as described below. It is not a comparison that
isolates decoder hardware. ANEForge Python was already checked against the HF
encoder output and transcript in the earlier verification run.

**ANE encoder + CPU decoder was fastest for this model, clip and configuration:**
68.37 ms to transcribe 11 seconds, about 161 times the audio duration and 2.34
times faster than CPU/CPU. Metal did not improve decoding here. The ANE/Metal
encode timer also includes Metal cross-attention K/V preparation, so its 31.80
ms is not a pure ANE dispatch measurement. These figures do not establish how
larger models, long audio or another chip perform.

| Route | Observed whole-transcription range across 10 runs |
| --- | ---: |
| CPU/CPU | 158.02-162.89 ms |
| Metal/Metal | 127.03-162.29 ms |
| ANE/CPU | 67.55-83.33 ms |
| ANE/Metal | 164.46-198.60 ms |
| ANE/ANE, Python | 132.73-145.37 ms |

The machine remained in its ordinary desktop session. Power, energy, thermal
state and background activity were not controlled or measured. Earlier
separate-process smoke-test numbers include startup and setup effects; use this
warm table for repeated inference.

## Which combinations exist?

The tested APIs expose these encoder/decoder choices:

| Encoder / decoder | CPU decoder | Metal GPU decoder | ANE decoder |
| --- | --- | --- | --- |
| CPU encoder | Measured, whisper.cpp | No separate stage selector | No provided mode |
| Metal GPU encoder | No separate stage selector | Measured, whisper.cpp | No provided mode |
| ANE encoder | Measured, whisper.cpp | Measured, whisper.cpp | Measured, ANEForge Python |

These are execution routes rather than a device on/off powerset. Metal routes
still use CPU for log-mel and sampling. ANE/CPU corresponds to CPU + ANE;
ANE/Metal uses **CPU + GPU + ANE**, with CPU host work, ANE audio encoding and
Metal decoder execution. ANE/ANE also needs CPU for feature extraction,
embeddings, cross-attention K/V preparation and sampling. None of these is a
fully CPU-free GPU-only or ANE-only transcription.

whisper.cpp has one `use_gpu` choice for its ggml backend and an external ANE
encoder override. It does not expose independent CPU/GPU choices for both
towers. Core ML's scheduler choosing among allowed devices is another runtime,
not an explicit additional combination; that runtime remains unmeasured here.

## Whisper versus LLM prefill/decode

Whisper is an encoder-decoder model. The audio encoder turns a padded log-mel
spectrogram into audio features. The decoder consumes a short start/task prompt
and then generates text autoregressively, attending to those audio features.
The encoder is a separate stage; it is not the decoder's text prefill.

For a useful benchmark, retain:

1. **Encode ms per audio window**: the audio stage.
2. **Decoder prompt setup ms**, analogous to LLM prefill, plus **single-token
   ms/token** or tokens/second for autoregressive evaluation.
3. **Full decoder evaluation ms**: prompt plus all generated-token evaluations.
4. **Whole transcription latency and RTF**: RTF = processing seconds / actual
   audio seconds. Less than 1 is faster than real time. Also verify transcript
   correctness; a faster route producing a different transcript is not enough.

For just a stage speed comparison, encode and full decoder totals are sufficient.
Whole latency/RTF answers how fast the usable transcription is. Generated token
count, prompt history, beam width, timestamps and retries affect decoder timing;
always hold those settings fixed. This sample has a two-token decoder prompt
and 24 subsequent single-token evaluations on all five routes.

| Route | Short prompt setup | Subsequent single-token evaluation total |
| --- | ---: | ---: |
| CPU/CPU | 2.18 ms | 35.82 ms |
| Metal/Metal | 2.77 ms | 59.61 ms |
| ANE/CPU | 2.19 ms | 36.66 ms |
| ANE/Metal | 3.06 ms | 106.00 ms |
| ANE/ANE, Python | 6.00 ms | 75.26 ms |

In this whisper.cpp revision, batches with 2-15 tokens go into the `batchd`
timer, while batches with at least 16 go into `prompt`. Thus `prompt time =
0.00 ms` in the logs does **not** mean prefill was absent: this clip's two-token
prompt appears in `batchd`. The decoder total above is
`decode + batchd + prompt`. Those are model evaluation timers and exclude
sampling. Medians are calculated per metric, so rounded column totals need not
sum exactly.

### Timing boundaries

whisper.cpp's **encode** timer includes input handling, audio encoder execution,
and cross-attention K/V preparation on its selected ggml backend. Its **decode**
timer includes single-token decoder evaluation; **batchd** includes the short
prompt in this configuration. Host mel and sampling have their own timers.
**Whole transcription** is an external timer around `whisper_full`, including
host processing and orchestration, with a loaded and warmed context.

The Python **encode** timer wraps `encode` and subtracts feature extraction;
it includes encoder input/output copies but excludes cross-K/V preparation.
The Python **decoder total** sums input feeds, ANE execution and logit reads for
every decoder step, including both prompt steps. It excludes host projection,
embedding/mask construction and sampling. Its complete decoder phase,
including that host work and tokenization, has median **110.36 ms**. Python's
whole-transcription timer includes all of those tasks. Timing wrappers preserve
the original inference calls and add a small amount of Python timer overhead.

## Method and reproduction

The C++ driver uses the public API and leaves upstream source unchanged. Each
backend is initialized once per round, performs two complete warmups, and then
five measured transcriptions. Two rounds give 10 measurements; the second
round reverses backend order. All hardware work executes serially. Timings and
text history reset between calls. Settings: English, four threads, greedy
best-of-one, temperature zero, no fallback, no timestamps, flash attention
requested with the same upstream backend handling, full audio context. Python
also uses four Torch host threads, greedy decoding, an initial compilation
call, two further warmups, and ten measured calls in a persistent model.

Run from the prepared workspace, choosing new output paths:

```sh
.venv/bin/python scripts/benchmark_macos.py \
  --dylib "$HOME/Desktop/ANEForge/aneforge/_lib/libane_e5rt_dispatch.dylib" \
  --output results/benchmark-macos-rerun

HF_HUB_OFFLINE=1 PYTHONPATH="$HOME/Desktop/ANEForge" .venv/bin/python \
  scripts/benchmark_python_whisper.py \
  --model "$PWD/.cache/huggingface/hub/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063" \
  --output results/benchmark-python-whisper-rerun.json
```

The first script builds the small C++ driver against `build/metal` and validates
all warmup/measured transcripts, ANE readiness, Metal selection and absence of
decoding fallback. The second validates both program device masks and every
transcript. The model was obtained with `hf download` and is reused offline;
this benchmark makes no new model download.

Raw evidence:

- [C++ summary and all 40 measured calls](../results/benchmark-macos/summary.json)
- [C++ execution log](../results/benchmark-macos.log), plus eight per-context
  logs under `results/benchmark-macos/`
- [Python summary and all 10 measured calls](../results/benchmark-python-whisper.json)
- [Python execution log](../results/benchmark-python-whisper.log)
- [C++ driver](../scripts/benchmark_whisper.cpp),
  [C++ benchmark runner](../scripts/benchmark_macos.py), and
  [Python benchmark runner](../scripts/benchmark_python_whisper.py)
