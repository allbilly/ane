# macOS / Asahi follow-ups

- [ ] macOS: recapture the complete Whisper `tiny.en` ANE encoder from the
  graph used for the fast macOS benchmark. Start with
  `whisper/models/whisper-tiny.en-ane`; confirm its MIL against
  `whisper/results/benchmark-macos/summary.json` before dumping.
  Follow the GPT-2 approach: dump the task/register stream and static constants,
  strip learned weights, and commit compact templates plus a checkpoint repacker.
  Record the actual task count, dependencies, BAR mapping, scratch sizes,
  input/output shapes and strides, and checkpoint/compiler/MIL hashes.
  Document input packing and weight tile ordering, transposes, interleaving,
  padding, FP16 conversion, bias and layer-norm placement. Verify that repacking
  reproduces the original payload bytes. Keep full HWX files and weights local.
  Validate on the existing 5/11/23-second JFK cases: HF encoder cosine >= 0.999,
  every full decoder logit vector NRMSE < 0.005 and matching token choices.
  Record warm encoder, prompt setup, token decode and whole-transcription times.
  The current 1,783-task dump runs on Linux but fails the logit gate; its identity
  with the historical fast macOS export remains unverified.

- [ ] Asahi: replay the new compact kernels with inputs/weights repacked from
  the pinned HF checkpoint, without a macOS compiler. Validate the same cases
  and compare timings with matching boundaries and four CPU workers.
  Linux readback already preserves all output bits and improves its profiled
  stage from 5.31 to 1.34 ms; CPU cross-K/V still costs about 28 ms.
  Keep the existing dump and Linux diagnostic results as the baseline.

- [ ] Qwen: return the existing macOS all-prefix BF16 oracle files
  `uzu-macos-all.npz` and `native-macos-all.npz` with their reports/hashes.
  Compare with `qwen35/local-results/asahi-todo-20261007/vendor-all-prefix/`
  to locate the first differing prefix/layer. Same-host checks already pass;
  cross-host accuracy remains unverified.
