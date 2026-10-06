# Native Asahi Whisper projections

This path runs `whisper-tiny.en` on the base M1 / T8103 using this repository's
current ANE driver. It reuses `qwen35/ane_matmul.c` and its extracted, patchable
matrix stream. Model matrices are repacked from the F16 ggml checkpoint on Linux;
no new macOS compilation or dump is required.

The encoder's query, key, value, output and two MLP projections run on ANE:
24 resident matrices across four layers. CPU runs the two convolutions,
attention, normalization, encoder GELU, cross-attention K/V preparation and
the complete decoder. This differs from the macOS whole-encoder ANEForge graph.
It does not use the older single-convolution Whisper HWX as a complete encoder.

Each projection uses 32 audio positions per submission. A full 1,500-position
encoder makes 1,128 hardware submissions. Submission failures are fatal and the
runtime checks the complete projection and submission counts. The isolated
build uses exact encoder GELU and widened FP32 accumulation for CPU F16 NEON dot
products in both CPU and ANE modes. Encoder attention reads the 1,500 actual K/V
positions, excluding the allocation's 36 unmasked padding slots. Its CPU mode is a reference for this hybrid
implementation, rather than an unchanged upstream performance baseline.

Actual [Asahi measurements](benchmark-asahi.md) and numerical evidence are
retained. The current hybrid is accurate on the three tested clips, but slower
than CPU and the macOS whole-encoder ANE route.

## Run in the prepared workspace

From the parent `ane` checkout:

```sh
flock "$HOME/ane.lock" flock "$HOME/gpu.lock" flock /tmp/m1-gpu.lock \
  taskset -c 4-7 env WHISPER_ASAHI_ANE=1 OMP_WAIT_POLICY=PASSIVE \
  whisper/build/asahi-ane/bin/whisper-cli \
  -m whisper/models/hf-ggml/ggml-model.bin \
  -f whisper/vendor/whisper.cpp/samples/jfk.wav \
  -l en -t 4 -bs 1 -bo 1 -tp 0 -nf -nt -ng
```

Replace the audio path with your 16 kHz WAV recording. CPUs 4-7 are this M1's
performance cores; omit or adjust `taskset` on another machine.
Set `WHISPER_ASAHI_ANE=0` for the
hybrid build's CPU reference. The separate `whisper/build/asahi-cpu` binary is
the unchanged upstream CPU build. Metal and Apple's ANEForge dylib are not used
by this Linux path.

## Rebuild

Use whisper.cpp revision `60c0be6ac8fa71b1a2ae2dd938a31a34a508e774` and an
isolated worktree, preserving the normal CPU checkout:

```sh
git clone https://github.com/ggml-org/whisper.cpp.git whisper/vendor/whisper.cpp
git -C whisper/vendor/whisper.cpp checkout --detach \
  60c0be6ac8fa71b1a2ae2dd938a31a34a508e774
git -C whisper/vendor/whisper.cpp worktree add --detach ../whisper-asahi \
  60c0be6ac8fa71b1a2ae2dd938a31a34a508e774
uv venv --python 3.11 whisper/.venv
uv pip install --python whisper/.venv/bin/python -r whisper/requirements-asahi.txt
whisper/.venv/bin/python whisper/scripts/prepare_asahi.py
uv tool run --from cmake cmake -S whisper/vendor/whisper-asahi \
  -B whisper/build/asahi-ane -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_BUILD_TESTS=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=OFF -DANE_ROOT="$PWD"
uv tool run --from cmake cmake --build whisper/build/asahi-ane \
  --target whisper-cli -j4
```

Skip the clone, worktree and virtualenv creation commands when those paths
already exist. The preparation script accepts only the pinned revision and
refuses to overwrite other source edits.

The checkpoint is `openai/whisper-tiny.en` revision
`87c7102498dcde7456f24cfd30239ca606ed9063`, converted with the pinned upstream
`models/convert-h5-to-ggml.py`. Its safetensors SHA-256 is
`db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`.
The resulting F16 ggml model is byte-identical to the macOS model:
`776f39cd70d01a7df3c6098f87f121b62d1fb634d1272b5f36bb6f369fc34372`.

To prepare those weights in a fresh workspace after installing the requirements:

```sh
whisper/.venv/bin/hf download openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --include model.safetensors '*.json' merges.txt \
  --local-dir whisper/models/hf-tiny.en
git clone --depth 1 https://github.com/openai/whisper.git whisper/vendor/openai-whisper
git -C whisper/vendor/openai-whisper fetch --depth 1 origin \
  86098128c0b4f24f0e2aa2994de830614b474227
git -C whisper/vendor/openai-whisper checkout --detach \
  86098128c0b4f24f0e2aa2994de830614b474227
mkdir -p whisper/models/hf-ggml
env HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  whisper/.venv/bin/python whisper/vendor/whisper.cpp/models/convert-h5-to-ggml.py \
  whisper/models/hf-tiny.en whisper/vendor/openai-whisper whisper/models/hf-ggml
sha256sum whisper/models/hf-ggml/ggml-model.bin
```

The converter also needs the tokenizer assets from that same HF revision and
OpenAI Whisper's `whisper/assets/mel_filters.npz`, SHA-256
`7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c`.
All weights, build outputs and numerical captures are ignored by Git.

## Verify and benchmark

These commands must run serially with other ANE/GPU tests:

```sh
flock "$HOME/ane.lock" flock "$HOME/gpu.lock" flock /tmp/m1-gpu.lock \
  taskset -c 4-7 env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  whisper/.venv/bin/python whisper/scripts/verify_asahi_matrices.py \
  --output whisper/.cache/matrices-rerun.json

flock "$HOME/ane.lock" flock "$HOME/gpu.lock" flock /tmp/m1-gpu.lock \
  taskset -c 4-7 env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  whisper/.venv/bin/python whisper/scripts/benchmark_asahi.py \
  --output whisper/.cache/benchmark-asahi-rerun
```

Create the `.cache` directory before the matrix-only check, and choose new output
paths. The matrix check compares all three projection dimensions and batches
1/8/28/32 against independent float64 multiplication. The benchmark compares
5/11/23-second JFK variants, encoder features, complete decoder logit vectors
and independently generated token histories. It also computes an independent
HF encoder reference on the exact native mel input. Warm timings use two rounds
of five measurements per backend, reversing backend order in round two and
excluding two warmups per context. It measures encoder time, decoder prompt
setup, single-token evaluation, whole transcription and real-time factor.
