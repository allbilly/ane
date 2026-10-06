# Whisper on the Apple Neural Engine

Instructions and reproducible tests for Whisper on Apple Silicon, covering
macOS ANEForge, Core ML, the experimental Asahi Linux driver, and ANE decoder
options. The macOS ANEForge paths were run on this M1 MacBook Air using the
existing `~/Desktop/ANEForge` checkout and checkpoints cached with `hf download`.
This directory lives at `~/ane/whisper` as part of the parent `ane` repository.

| Guide | Contents |
| --- | --- |
| [macOS setup](docs/macos.md) | Download models, export the ANE encoder, build whisper.cpp, transcribe, verify, and use Python encoder + decoder |
| [Asahi Linux](docs/asahi-linux.md) | Separate kernel driver, libane, `.anec` conversion, historical PR setup and missing prerequisites |
| [Decoder options](docs/decoder.md) | ANEForge Python versus the draft stateful Core ML decoder branch |
| [Local test results](docs/macos-test-results.md) | Hardware, versions, transcript parity, timings and raw evidence |
| [CPU / GPU / ANE benchmark](docs/benchmark-macos.md) | Five encoder/decoder routes, warm encode/decode timings, transcription latency and RTF |
| [Source overview](docs/README.md) | Status of the requested PRs/discussion and exact inspected revisions |

The macOS encoder backend is included in current whisper.cpp through merged
[PR #3905](https://github.com/ggml-org/whisper.cpp/pull/3905).
The Asahi [PR #1021](https://github.com/ggml-org/whisper.cpp/pull/1021) remains a
research proof of concept with separate dependencies. ANEForge's macOS runtime
does not run on Linux.

## Run in this prepared workspace

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
