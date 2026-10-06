# Whisper on the Apple Neural Engine

Start with [native Asahi](asahi-native.md) or [macOS](macos.md) for an executable
setup, [historical Asahi](asahi-linux.md) for the separate libane/anecc recipe,
and [decoder options](decoder.md) for
what can move beyond the encoder. [Local test results](macos-test-results.md)
record the actual machine, versions, commands and evidence.
The [CPU/GPU/ANE benchmark](benchmark-macos.md) measures five available routes
with warm encode/decode timings, transcription latency and RTF.
The [native Asahi benchmark](benchmark-asahi.md) records the new Linux hardware
runs, independent numerical comparisons and measured CPU/ANE projection timings.

The ANE is separate from the Metal GPU. Whisper first converts audio to log-mel
features, then encodes the audio and repeatedly decodes text tokens. Moving only
the encoder to ANE still leaves decoding on CPU or GPU.

| Route | OS | What executes on ANE | Status checked 2026-10-06 |
| --- | --- | --- | --- |
| whisper.cpp + ANEForge | macOS on Apple Silicon | Encoder | [PR #3905](https://github.com/ggml-org/whisper.cpp/pull/3905) merged 2026-09-18; runtime opt-in |
| whisper.cpp + Core ML | macOS | Encoder when Core ML places it there | Upstream; default compute units also permit CPU/GPU |
| ANEForge Python Whisper | macOS on Apple Silicon | Encoder and decoder graphs | Separate [Python implementation](https://github.com/sbryngelson/ANEForge/blob/main/aneforge/models.py); does not enable whisper.cpp's decoder |
| Core ML decoder experiment | macOS | Sharded decoder graphs, with host work | [Discussion #3849](https://github.com/ggml-org/whisper.cpp/discussions/3849) links draft [PR #3848](https://github.com/ggml-org/whisper.cpp/pull/3848) |
| Asahi + libane | Native Asahi Linux | Encoder in the historical Whisper proof of concept | [PR #1021](https://github.com/ggml-org/whisper.cpp/pull/1021) open, unmerged; separate out-of-tree kernel driver |
| Native register stream + whisper.cpp | Native Asahi Linux, M1 / T8103 | 24 encoder dense projections; other operations and decoder on CPU | Executed and validated on three audio variants; 1,128 ANE submissions per encoder; [measurements](benchmark-asahi.md) |

## Sources and versions

The requested sources explain different layers of the stack:

- [whisper.cpp PR #3905](https://github.com/ggml-org/whisper.cpp/pull/3905)
  adds the C++ encoder integration. Its environment variables select an exported
  encoder bundle and a dispatch dylib.
- [ANEForge](https://github.com/sbryngelson/ANEForge) implements the macOS graph
  compiler and dispatch runtime. Its Whisper bundle exporter lives in
  `bench/whisper_encoder_ane/export_bundle.py`.
- [whisper-aneforge](https://github.com/sbryngelson/whisper-aneforge)
  provides the original integration and an alternative exporter. Its README still
  describes applying a patch; current whisper.cpp already includes that backend.
- [whisper.cpp PR #1021](https://github.com/ggml-org/whisper.cpp/pull/1021),
  [eiln/ane](https://github.com/eiln/ane) and [eiln/anecc](https://github.com/eiln/anecc)
  describe the independent Linux driver, userspace library and model conversion.
- [Discussion #3849](https://github.com/ggml-org/whisper.cpp/discussions/3849)
  describes a stateful Core ML decoder experiment. The reported correctness and
  speed measurements belong to its author's setup.

Exact inspected revisions and PR status are recorded in [sources.json](sources.json).
The upstream benchmark claims are not predictions for this M1 Mac; use the
local measurements and keep encoder timing separate from whole-command timing.
