# Local macOS verification

Run on 2026-10-06 in Hong Kong (UTC timestamps in the raw reports start on
2026-10-05). This is a transcription smoke test on one 11-second JFK clip,
not a dataset-wide accuracy or power benchmark.

The follow-up [CPU/GPU/ANE benchmark](benchmark-macos.md) adds ANE + Metal
decoding and repeated warm timings for all five available routes. Use that
table for performance comparisons; the process-based timings below document
the earlier smoke test, including startup effects.

The workspace was subsequently moved from `~/Desktop/whisper` to
`~/ane/whisper` within the existing `ane` repository. The ANEForge checkout
stays at `~/Desktop/ANEForge`. Historical logs retain the paths used when those
runs were recorded; current reproduction commands use the new location.

The [relocation report](../results/relocation.json) records repaired virtualenv
launchers, both rebuilt CMake targets, preserved model hashes, and runtime
verification from the new path. All four
[C++ routes passed again](../results/relocation-whispercpp/summary.json).
[Python ANE encoder/decoder parity also passed](../results/relocation-python-whisper.json),
with encoder cosine 0.9999536 and the same HF transcript. These are functional
relocation checks; the original ten-run benchmark table remains unchanged.

## Environment

| Item | Observed value |
| --- | --- |
| Machine | MacBook Air `MacBookAir10,1`, Apple M1, 8 GB RAM, arm64 |
| macOS | 27.0.1, build `26A434` |
| Compiler | AppleClang 21.0.0.21000334, Xcode SDK |
| whisper.cpp | `60c0be6ac8fa71b1a2ae2dd938a31a34a508e774`, 1.9.4-dev |
| Local ANEForge | `/Users/yeren/Desktop/ANEForge`, revision `d0c666192ebd8d0428e0647fc4d87f57c11759fa` |
| Python | 3.12.7, arm64 |
| Packages | ANEForge 0.4.0, NumPy 2.2.6, PyTorch 2.7.0, Transformers 4.53.0, CMake 4.0.3 |
| Hugging Face CLI | `hf` 2.1.1 |
| Trained model | `openai/whisper-tiny.en`, revision `87c7102498dcde7456f24cfd30239ca606ed9063` |
| Audio | `vendor/whisper.cpp/samples/jfk.wav`, 16 kHz mono PCM16, 11.0 seconds |

The Python environment reuses existing Python 3.12 site packages; ANEForge and
NumPy were installed into the project environment. The sibling ANEForge and
upstream whisper.cpp tracked files were left clean. The broken global CMake
launcher was bypassed with `python -m cmake`.

## ANEForge Python encoder and decoder: PASS

[JSON report](../results/python-whisper.json) and
[execution log](../results/python-whisper.log) record the real run.

- A compiled arithmetic graph executed on ANE and exactly matched its NumPy oracle.
- Whisper encoder cosine similarity against HF fp32: **0.9999536**.
- The ANE decoder transcript exactly matched the HF reference text on this clip.
- Two consecutive transcriptions returned identical text with reset clip state.
- One warm transcription took **298.08 ms**, about 36.9 times faster than the
  audio duration. Loading/compiling took 23.99 seconds; the first transcription,
  including decoder compilation, took 26.82 seconds. Those cold figures also
  reflect concurrent work during setup.

Encoder and decoder graphs use ANE. Log-mel, token/position embeddings,
cross-K/V preparation and sampling still perform host work. Only one warm
transcription was timed; it is not a statistical comparison with whisper.cpp.

## Exported encoder and whisper.cpp: PASS

The trained channels-first exporter executed on this ANE and reported encoder
cosine **0.999968** against its PyTorch reference. The
[export log](../results/aneforge-export.log) and
[bundle manifest](../results/aneforge-export-manifest.json) identify the model,
dimensions and exported files. Tiny uses no weight compression in this run.

The verified C++ run is in
[summary.json](../results/whispercpp-macos-final/summary.json), with one `.log`
and `.txt` per invocation. All **9/9 runs passed**: three CPU, three Metal,
three ANEForge. The harness checked the ANE readiness log, the CPU decoder
configuration, Metal backend selection, and the expected transcript words.

Each invocation used English, four threads, greedy search (`-bs 1 -bo 1`),
temperature zero, no fallback, no timestamps and full audio context. CPU and
ANEForge used `-ng`; Metal used the GPU decoder. Each row is the median of
three separate CLI processes, not three warm calls in one process.

| Backend | Encoder time | whisper.cpp total time | Command wall time |
| --- | ---: | ---: | ---: |
| CPU encoder + CPU decoder | 463.42 ms | 964.32 ms | 1212.90 ms |
| Metal encoder + Metal decoder | 65.48 ms | 285.98 ms | 420.86 ms |
| ANEForge encoder + CPU decoder | 38.69 ms | 292.65 ms | 462.79 ms |

In this run, ANEForge's encoder was 12.0 times faster than CPU and 1.69 times
faster than Metal. Whole-transcription totals were similar to Metal because
these rows use different decoder backends. No energy measurements were taken.
The first CLI launch also incurred about 90 seconds of Metal shader compilation,
even with `-ng`, during device enumeration; subsequent runs used cached shaders.

Every route transcribed:

> And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.

The C++ log records `aneforge: encoder ready` and `ANEForge encoder loaded`.
The dispatch source requests the ANE device (`0x4`), and the result matches the
reference. It is an ANE execution test, with the decoder intentionally on CPU.

### Model provenance and download behavior

The trained checkpoint and tokenizer were obtained with **`hf download`**.
The original Xet transfer failed with a TLS/CAS error, so the successful CLI
download used `HF_HUB_DISABLE_XET=1`; compilation then used the local snapshot
with `HF_HUB_OFFLINE=1`.

The C++ run above used a local F16 ggml conversion of that same HF checkpoint,
via upstream `models/convert-h5-to-ggml.py`, because the initial separate ggml
download disconnected and restarted. Its model SHA-256 is
`776f39cd70d01a7df3c6098f87f121b62d1fb634d1272b5f36bb6f369fc34372`.
[Conversion log](../results/convert-ggml.log) records the tensor mapping.
The OpenAI `mel_filters.npz` asset used by the converter has SHA-256
`7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c`.

## Core ML encoder: build verified, runtime untested

The separate Core ML binary compiled successfully with `WHISPER_COREML=ON`
and fallback disabled; see the [build log](../results/build-coreml.log).
The pretrained encoder archive download repeatedly returned interrupted HTTP
responses. Download attempts were stopped after the ANEForge tests finished.
No Core ML inference, ANE placement or Core ML latency is claimed here.

The ready-to-run ggml file is the validated local conversion described above.
The successful checkpoint/tokenizer CLI logs are
[checkpoint download](../results/hf-checkpoint-download.log) and
[tokenizer download](../results/hf-tokenizer-download.log).

## Verification boundaries

Asahi driver execution and draft PR #3848's Core ML decoder were not run on
this macOS host. Their documents distinguish inspected source and prerequisites
from measured results. These tests establish one trained model/sample on this
physical M1 and macOS build; they do not establish other chips, larger models,
long recordings or future private-framework compatibility.
