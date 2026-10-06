# Run Whisper on ANE on macOS

Use the ANEForge encoder with current whisper.cpp to place the encoder directly
on ANE. Keep the normal whisper.cpp decoder, or use ANEForge's separate Python
Whisper API to experiment with both encoder and decoder graphs on ANE.

## Prerequisites

Use a physical Apple Silicon Mac, macOS 14 or newer, Xcode command-line tools,
Git, Python 3.12 and the Hugging Face `hf` CLI. ANEForge needs Apple's private
Espresso/e5rt frameworks; a VM without ANE passthrough cannot run this test.
The local run uses the M1 MacBook Air and the existing `~/Desktop/ANEForge`
checkout. This project's root is `~/ane/whisper`.

From this project's root, create an isolated environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos.txt
```

Check `hf version` and `xcode-select -p`. If `hf` is absent, install the standalone
CLI with `uv tool install hf`; it stays separate from Transformers' Python
dependencies. If Xcode tools are absent, run `xcode-select --install`.
Allow a few GB for Python packages, two checkpoint formats, and compiled models.

## 1. Obtain source and trained checkpoints

The following source revision includes the merged ANEForge backend. These clone
commands are for a fresh folder; the supplied workspace already has the checkouts.

```sh
mkdir -p vendor models
git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp
git -C vendor/whisper.cpp checkout 60c0be6ac8fa71b1a2ae2dd938a31a34a508e774

# Reuse the user's local checkout, or clone ANEForge into vendor/ANEForge.
ANE_SOURCE="$HOME/Desktop/ANEForge"
# Alternative for a fresh machine:
# git clone https://github.com/sbryngelson/ANEForge.git vendor/ANEForge
# ANE_SOURCE="$PWD/vendor/ANEForge"

export PYTHONPATH="$ANE_SOURCE"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

hf download openai/whisper-tiny.en \
  config.json generation_config.json model.safetensors \
  preprocessor_config.json tokenizer_config.json tokenizer.json \
  vocab.json merges.txt normalizer.json added_tokens.json special_tokens_map.json \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063

hf download ggerganov/whisper.cpp ggml-tiny.en.bin --local-dir models

CHECKPOINT="$HF_HOME/hub/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063"
```

Use the same trained model for both formats: `openai/whisper-tiny.en` pairs with
`ggml-tiny.en.bin`; `tiny` and `tiny.en` are different checkpoints. The Hugging
Face checkpoint supplies exporter weights. The ggml file is still needed by
whisper.cpp, including for its decoder. Start with tiny on an 8 GB machine.
For other sizes, change both names and obtain the appropriate HF revision.

`HF_HUB_DISABLE_XET=1` uses the regular download path. It addressed a Xet TLS
download failure in this session; it does not change inference. `hf download`
caches the model explicitly before any ANE compilation.

If a separate ggml download fails, convert the already cached trained HF
checkpoint with upstream's converter. This was the route used for the first
local C++ test. It needs OpenAI's small mel-filter asset:

```sh
mkdir -p .cache/whisper-assets/whisper/assets
curl --fail -L https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz \
  -o .cache/whisper-assets/whisper/assets/mel_filters.npz
HF_HUB_OFFLINE=1 python vendor/whisper.cpp/models/convert-h5-to-ggml.py \
  "$CHECKPOINT" "$PWD/.cache/whisper-assets" "$PWD/models"
mv models/ggml-model.bin models/ggml-tiny.en.bin
```

This produces a local F16 model from the same checkpoint. Its file hash can
differ from a prebuilt ggml release; rerun transcription validation. It does
not create a Core ML encoder package.

## 2. Build the dispatch library and export the encoder

```sh
python -m aneforge.build

HF_HUB_OFFLINE=1 python "$ANE_SOURCE/bench/whisper_encoder_ane/export_bundle.py" \
  --model "$CHECKPOINT" --out "$PWD/models/whisper-tiny.en-ane"

ANEFORGE_DYLIB="$ANE_SOURCE/aneforge/_lib/libane_e5rt_dispatch.dylib"
test -f "$ANEFORGE_DYLIB"
```

The exporter uses trained weights, executes the compiled encoder, and prints its
cosine similarity against PyTorch. A random-weight export is a shape test and
cannot verify transcription. The bundle contains `model.mil`, `weights.bin`,
`ports.txt`, `pos.f16` and `cache/`. Its mel input is 80 x 3000 and encoder output
is 1500 x 384 for tiny. It uses full 30-second padded audio context.

Build/export with the same ANEForge source that provides the dylib. The local
checkout is recorded in [sources.json](sources.json). Without a checkout, the
[companion exporter](https://github.com/sbryngelson/whisper-aneforge/blob/master/export_encoder.py)
is another option with the installed `aneforge` package:

```sh
python vendor/whisper-aneforge/export_encoder.py \
  --model "$CHECKPOINT" --out "$PWD/models/whisper-tiny.en-ane"
```

Use one exporter per bundle. Optional `--compress int4` is intended for weight
compression experiments; validate the transcript again when changing precision.
Generate on the target Mac and keep the original bundle path. Native compiled
cache artifacts depend on the macOS build; the C++ backend compiles with cache
when each process initializes. An OS update can require regenerating the bundle.

## 3. Build whisper.cpp and transcribe

```sh
python -m cmake -S vendor/whisper.cpp -B build/metal \
  -DCMAKE_BUILD_TYPE=Release -DWHISPER_COREML=OFF
python -m cmake --build build/metal --target whisper-cli whisper-bench -j 4

ANEFORGE_ENCODER="$PWD/models/whisper-tiny.en-ane" \
ANEFORGE_DYLIB="$ANEFORGE_DYLIB" \
  ./build/metal/bin/whisper-cli \
  -m models/ggml-tiny.en.bin -f vendor/whisper.cpp/samples/jfk.wav \
  -l en -t 4 -bs 1 -bo 1 -tp 0 -nf -ng
```

Current whisper.cpp builds the ANEForge integration without a separate
`WHISPER_ANEFORGE` CMake flag. **Do not apply the companion's old patch** on this
revision. `ANEFORGE_ENCODER` activates the encoder; `ANEFORGE_DYLIB` supplies
the dynamically loaded runtime. Look for `aneforge: encoder ready` and a correct
transcript. A Metal log or `COREML = 1` alone does not prove ANE execution.

Here `-ng` disables whisper.cpp's GPU path, so decoding uses CPU while the
external encoder uses ANE. Omit `-ng` to allow the normal Metal decoder. Unset
`ANEFORGE_ENCODER` to return to whisper.cpp's normal encoder. Do not use `-ac` /
`--audio-ctx` with this fixed-context bundle.

Validate all three routes and preserve evidence:

```sh
python scripts/test_macos.py --dylib "$ANEFORGE_DYLIB"
```

The script runs three independent processes per route, compares JFK words to
the expected transcript, checks ANE readiness, and records raw logs, command
lines, hashes and timings under `results/`. CPU and ANE comparisons use the same
CPU decoder. Process wall time includes loading/compilation; encode timing is
the encoder call, including transfers. These are different measurements.

## Core ML encoder alternative

Core ML uses Apple's public API. whisper.cpp's default configuration permits
CPU, GPU and ANE scheduling, so a successful Core ML run proves Core ML usage
but requires compute-plan/profiling evidence to identify ANE placement.
[The source sets `MLComputeUnitsAll`](https://github.com/ggml-org/whisper.cpp/blob/60c0be6ac8fa71b1a2ae2dd938a31a34a508e774/src/coreml/whisper-encoder.mm).

Download a compiled model directly with `hf download`; the upstream
`download-coreml-model.sh` currently exits with a message that it is not functional.

```sh
hf download ggerganov/whisper.cpp ggml-tiny.en-encoder.mlmodelc.zip --local-dir models
unzip models/ggml-tiny.en-encoder.mlmodelc.zip -d models

python -m cmake -S vendor/whisper.cpp -B build/coreml \
  -DCMAKE_BUILD_TYPE=Release -DWHISPER_COREML=ON \
  -DWHISPER_COREML_ALLOW_FALLBACK=OFF
python -m cmake --build build/coreml --target whisper-cli -j 4

env -u ANEFORGE_ENCODER -u ANEFORGE_DYLIB ./build/coreml/bin/whisper-cli \
  -m models/ggml-tiny.en.bin -f vendor/whisper.cpp/samples/jfk.wav -l en -ng
```

Keep `ggml-tiny.en-encoder.mlmodelc` beside `ggml-tiny.en.bin` with that exact
name. Look for `Core ML model loaded`. Disabling Core ML fallback makes missing
model errors visible. To include this route in the test, pass
`--coreml-binary build/coreml/bin/whisper-cli` to `scripts/test_macos.py`.

To convert your own encoder instead, follow the
[upstream Core ML section](https://github.com/ggml-org/whisper.cpp#core-ml-support)
and `models/generate-coreml-model.sh tiny.en` with its conversion dependencies
installed. Core ML packages and ANEForge bundles are different formats.

## Both encoder and decoder with ANEForge Python

```sh
HF_HUB_OFFLINE=1 python scripts/test_python_whisper.py \
  --model "$CHECKPOINT" --output results/python-whisper.json
```

This loads ANEForge's `load_whisper`, validates a real ANE dispatch, compares
encoder features to HF fp32 with cosine > 0.999, and compares transcript words
to HF greedy decoding. A second transcription checks repeated execution with
resident state. It does not replace whisper.cpp's decoder.

For application use:

```python
import aneforge as af
import numpy as np

model = af.load_whisper("openai/whisper-tiny.en")
try:
    text = model.transcribe(audio)  # audio: mono float32 NumPy samples at 16 kHz
finally:
    model.release()
```

The current Python path truncates/pads each clip to 30 seconds. Split longer
recordings yourself. It uses greedy decoding; host work includes log-mel,
token/position embeddings, cross-attention K/V preparation and sampling.
Do not assume it has whisper.cpp's beam search, streaming, timestamps or
long-audio segmentation behavior.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Broken global `cmake` Python launcher | Use the environment's `python -m cmake`; this was necessary here |
| `dlopen` failure or missing symbols | Build the shim with the matching ANEForge source and pass its absolute dylib path |
| `no ports.txt`, positional embedding read failure | Complete the trained export and pass its bundle root |
| ANE compilation failure | Check physical Apple Silicon hardware, supported macOS/private frameworks, and regenerate on this Mac |
| Mel-size mismatch or wrong transcript | Match checkpoint identities, use full context, and verify both model formats |
| Xet TLS/CAS download error | Set `HF_HUB_DISABLE_XET=1`, rerun `hf download`, then export offline |
| Core ML model fails to load | Check the sibling `.mlmodelc` directory/name; redownload or regenerate for this OS |

ANEForge is a research runtime using private APIs that can change between OS
versions. The local tests establish behavior on the recorded Mac and sample.
