# Whisper on the Apple Neural Engine

Instructions and reproducible tests for Whisper on Apple Silicon, covering
macOS ANEForge, Core ML, the experimental Asahi Linux driver, and ANE decoder
options. The macOS ANEForge paths were run on this M1 MacBook Air using the
existing `~/Desktop/ANEForge` checkout and checkpoints cached with `hf download`.
This directory lives at `~/ane/whisper` as part of the parent `ane` repository.

| Guide | Contents |
| --- | --- |
| [macOS setup](docs/macos.md) | Download models, export the ANE encoder, build whisper.cpp, transcribe, verify, and use Python encoder + decoder |
| [Native Asahi setup](docs/asahi-native.md) | Tested Linux CPU and ANE encoder projections using the existing matrix stream, without a new macOS dump |
| [Complete encoder kernels](docs/compact-encoder.md) | Original 1,779-task fast graph, native-stride packing, retained dense baseline and full-logit limits |
| [PR encoder packing and native replay](docs/pr3905-packing.md) | Compact tiny/base/small packages, safetensors/GGUF repacking, guarded replay and native whisper.cpp setup |
| [Asahi measurements](docs/benchmark-asahi.md) | Real hardware validation, warm encoder/prompt/decode timings and macOS comparison |
| [Historical Asahi PR](docs/asahi-linux.md) | Separate libane/anecc stack, `.anec` conversion and old PR prerequisites |
| [Decoder options](docs/decoder.md) | ANEForge Python versus the draft stateful Core ML decoder branch |
| [Local test results](docs/macos-test-results.md) | Hardware, versions, transcript parity, timings and raw evidence |
| [CPU / GPU / ANE benchmark](docs/benchmark-macos.md) | Five encoder/decoder routes, warm encode/decode timings, transcription latency and RTF |
| [Source overview](docs/README.md) | Status of the requested PRs/discussion and exact inspected revisions |

The macOS encoder backend is included in current whisper.cpp through merged
[PR #3905](https://github.com/ggml-org/whisper.cpp/pull/3905).
The Asahi [PR #1021](https://github.com/ggml-org/whisper.cpp/pull/1021) remains a
research proof of concept with separate dependencies. ANEForge's macOS runtime
does not run on Linux.

The [repository README](../README.md) includes the PR fast-encoder speed table
with measured Mac Accelerate/OpenBLAS timings and measured Asahi times of
**32.13 / 72.70 / 282.32 ms** for tiny/base/small. These include the ANE encoder
and CPU cross-attention K/V. Each model's zero-mel hardware output matches the
Mac hash exactly; speech and strict full-decoder accuracy remain unverified.
See [the replay method and Asahi receipts](docs/pr3905-packing.md#native-whispercpp-integration).

The separate [native tiny.en paired control](docs/asahi-paired-export.md#native-whispercpp-on-asahi)
passes all 80 full decoder vectors and byte-identical repeatability on Asahi.
Its warm JFK encoder/decoder/whole medians are **624.32 / 78.79 / 723.42 ms**,
versus **216.57 / 72.95 / 307.35 ms** for the matched CPU build. The
[README table](../README.md) compares the saved macOS paired result; Linux
readback remains the main measured cost, so this is an accurate control rather
than speed parity.

## Run on this Asahi machine

From the parent `~/ane` checkout:

```sh
flock "$HOME/ane.lock" flock "$HOME/gpu.lock" flock /tmp/m1-gpu.lock \
  taskset -c 4-7 env WHISPER_ASAHI_ANE=1 OMP_WAIT_POLICY=PASSIVE \
  whisper/build/asahi-ane/bin/whisper-cli \
  -m whisper/models/hf-ggml/ggml-model.bin \
  -f whisper/vendor/whisper.cpp/samples/jfk.wav \
  -l en -t 4 -bs 1 -bo 1 -tp 0 -nf -nt -ng
```

Set `WHISPER_ASAHI_ANE=0` for the same build's CPU reference. This native path
runs the 24 encoder dense projections on ANE and the remaining operations on
CPU. Across 5/11/23-second JFK variants, all 80 raw logit argmaxes and token
histories match CPU and all raw argmaxes match an independent HF model. Maximum
full-logit NRMSE versus HF is 0.333%. Optimizations reduced encode time from
721.41 to 345.80 ms. Warm median transcription time for the 11-second clip is
**468.18 ms with ANE projections**, versus **417.49 ms on CPU**.
The macOS full-encoder ANE route remains faster; its graph
and CPU implementation differ. See [setup](docs/asahi-native.md) and the
[complete table](docs/benchmark-asahi.md).

## Run in the prepared macOS workspace

```sh
cd ~/ane/whisper
ANEFORGE_ENCODER="$PWD/models/whisper-tiny.en-ane" \
ANEFORGE_DYLIB="$HOME/Desktop/ANEForge/aneforge/_lib/libane_e5rt_dispatch.dylib" \
  ./build/metal/bin/whisper-cli \
  -m models/ggml-tiny.en.bin -f vendor/whisper.cpp/samples/jfk.wav \
  -l en -t 4 -bs 1 -bo 1 -tp 0 -nf -ng
```

This uses ANE for the encoder and CPU for the whisper.cpp decoder. Replace
the sample path with your recording. Omit `-ng` to allow Metal decoding.
For a new machine or model, follow the [full setup](docs/macos.md) first.

Rerun the CPU/Metal/ANE transcription checks:

```sh
.venv/bin/python scripts/test_macos.py \
  --dylib "$HOME/Desktop/ANEForge/aneforge/_lib/libane_e5rt_dispatch.dylib"
```

Benchmark all four whisper.cpp routes with repeated warm calls, including
ANE encoder + Metal decoder:

```sh
.venv/bin/python scripts/benchmark_macos.py \
  --dylib "$HOME/Desktop/ANEForge/aneforge/_lib/libane_e5rt_dispatch.dylib" \
  --output results/benchmark-macos-rerun
```

The [benchmark table](docs/benchmark-macos.md) also includes ANEForge Python's
ANE encoder + ANE decoder, with the reproduction command and timing boundaries.

The separate ANEForge Python path also runs the decoder graphs on ANE:

```sh
PYTHONPATH="$HOME/Desktop/ANEForge" HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/test_python_whisper.py \
  --model "$PWD/.cache/huggingface/hub/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063" \
  --output results/python-whisper-rerun.json
```

Models, build outputs, caches and cloned dependencies are ignored by
`.gitignore`. `results/` retains the verification logs and reports. Python
package versions are pinned in [requirements-macos.txt](requirements-macos.txt).
