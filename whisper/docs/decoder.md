# Running the Whisper decoder on ANE

The encoder backend in [PR #3905](https://github.com/ggml-org/whisper.cpp/pull/3905)
does not move whisper.cpp's autoregressive decoder to ANE. ANEForge's separate
Python `load_whisper` API runs decoder graphs on ANE; see the tested commands in
[macOS](macos.md#both-encoder-and-decoder-with-aneforge-python).

[Discussion #3849](https://github.com/ggml-org/whisper.cpp/discussions/3849)
describes a different Core ML decoder experiment in draft
[PR #3848](https://github.com/ggml-org/whisper.cpp/pull/3848). Its author's design
uses decoder shards, self-attention K/V in `MLState`, and explicit FP16
cross-attention K/V inputs. Host code chains the shards and updates state.
Keeping cross K/V in state produced incorrect results in that experiment.

The author reports transcript parity, but encoder-plus-decoder Core ML was
slower than encoder-only Core ML in their end-to-end JFK measurements.
ANE placement and speed are separate questions. The discussion is evidence
of a working experiment on the author's machine, not a local result here.

## Reproduce the Core ML research branch

As of 2026-10-06, current upstream does not contain `WHISPER_COREML_DECODER`.
Passing that flag to upstream is insufficient: CMake can merely warn that an
unknown variable was unused. Use the PR branch in a separate checkout.
Stateful Core ML requires macOS 15 or later and appropriate coremltools/Xcode
versions. Start with a small model; the author's large-v2 experiment used an
M4 Max with 128 GB RAM.

```sh
git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper-coreml-decoder
git -C vendor/whisper-coreml-decoder fetch origin pull/3848/head
git -C vendor/whisper-coreml-decoder checkout --detach \
  95af5ca3ec4e95f0b890d26087e4cc8a2c3f2c9a
cd vendor/whisper-coreml-decoder

# In a separate Python environment, install this branch's converter dependencies
# following its README (openai-whisper, coremltools, ane_transformers, etc.).
./models/generate-coreml-model.sh --decoder tiny.en
hf download ggerganov/whisper.cpp ggml-tiny.en.bin --local-dir models

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_COREML=ON -DWHISPER_COREML_DECODER=ON
cmake --build build --target whisper-cli -j4
./build/bin/whisper-cli -m models/ggml-tiny.en.bin -f samples/jfk.wav \
  -l en -bs 1 -bo 1 -tp 0 -nf
```

These research-branch commands were checked against source, but were not run
here. The normal macOS test uses upstream's ANEForge encoder and separately
ANEForge's Python decoder. To use large-v2 as in the discussion, replace both
model names; allow sufficient memory and disk for conversion and all shards.

The inspected branch hard-codes decoder compute units to
`MLComputeUnitsCPUAndNeuralEngine`. Earlier experimental `all`/`cpu_gpu` settings
are not current runtime knobs in that revision. CPU operations can still be
present. Inspect Core ML's compute plan or Xcode profiling to verify which
operations execute on ANE. A Core ML load message does not prove every operation
uses it.

The encoder and decoder have separate build toggles. Decoder-only builds can
set `WHISPER_COREML_DECODER=ON` without `WHISPER_COREML=ON`. Generated shard
filenames and layouts must match the branch and the ggml model. It requires full
audio context, flash attention, and an F32/F16 ggml model. It does not support DTW token timestamps or partial `-ac`
context. Candidate state needs copying during beam search; the discussion's
initial greedy-only design should not be generalized to shared KV state across
candidates.

Asahi's `asahi/decoder.py` in PR #1021 is another partial demonstration. Neither
the macOS Core ML research branch nor ANEForge's e5rt dispatch runs on Linux.
