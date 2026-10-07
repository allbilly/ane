# macOS / Asahi follow-ups

- [x] macOS: recapture the complete Whisper `tiny.en` ANE encoder from the
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
  Done: the original MIL matches the fast benchmark; fresh 1,779-task commands
  and coefficients match the earlier original export byte for byte. The actual
  E5RT-staged ANE graph exports to the same stream after I/O BAR remapping.
  Default kernels are `whisper/kernels/tiny-en-encoder-fast` (149 KiB), with native
  padded input packing and byte-exact checkpoint reconstruction. Keep the
  1,783-task dense wrapper as a baseline: its two input reshapes add four tasks
  and about 13% execution overhead on Mac. The input packer now handles padding
  while preserving the original graph; this dump issue is fixed.
  All three outputs exactly match prior captures, HF encoder cosine passes,
  and all 80 raw token choices match. The warm 11-second rerun takes 15.83 ms
  encode including cross-K/V and 66.81 ms whole transcription. See
  `whisper/docs/fast-dump-review.md` and the linked receipts for all stage times.
  Recapture checklist:
  - [x] Compare original and wrapped MIL/weights and warm private-runtime execution.
  - [x] Export the unchanged original MIL and strip/repack every learned payload.
  - [x] Preserve native padded input strides in the Linux replay loader.
  - [x] Check decoder logits, record matching timing boundaries, and document limits.

- [ ] Whisper accuracy: achieve every full decoder logit vector NRMSE < 0.005
  with matching token choices on all three clips. The original fast graph also
  fails this gate on Mac: with the same HF CPU decoder, ANE encoder features
  produce maximum logit NRMSE 3.08% / 6.64% / 2.78% on the 11/5/23-second cases.
  Original fast Mac timing passed transcript/cosine checks, not this stricter
  gate. Dump identity alone cannot fix it; retain the failure and rebenchmark
  any numerical correction.
  - [ ] Locate the encoder operations responsible for the logit error, using
    the shared HF decoder to keep decoder differences out of the comparison.
  - [ ] Correct the numerical error, rerun every full logit vector on all three
    clips, and measure the resulting warm performance without weakening the gate.

- [ ] Asahi: replay the new original 1,779-task compact kernels with inputs/weights repacked from
  the pinned HF checkpoint, without a macOS compiler. Validate the same cases
  and compare timings with matching boundaries and four CPU workers.
  Linux readback already preserves all output bits and improves its profiled
  stage from 5.31 to 1.34 ms; CPU cross-K/V still costs about 28 ms.
  These latest Linux stage numbers come from the existing task notes; their raw
  reports are not present on this Mac. Compare exact encoder outputs and profile
  dispatch/readback/cross-K/V separately. CPU cross-K/V is a likely bottleneck:
  the reported 28 ms for that stage alone exceeds the Mac's 15.83 ms encode
  including cross-K/V. The Mac build uses Accelerate BLAS; the documented Linux
  build disables BLAS. The exact remaining gap is unverified until the Linux
  stage report is available; ANE clocks or driver scheduling are not established
  causes. Preserve the optimized native readback while switching to the original
  graph's padded input layout.
  Keep the existing dump and Linux diagnostic results as the baseline.
  - [x] Add a checkpoint/audio-only profiler and exact Mac output hashes;
    verify all three regenerated FP16 input hashes on Mac without hardware.
  - [ ] Run checkpoint/audio-only replay profiling with exact Mac input/output
    hashes and `--compare-baseline`; return dispatch, preparation, readback and
    total encoder times. Use the command in `whisper/docs/compact-encoder.md`.
  - [ ] Obtain the native four-worker full-encoder profile, including CPU
    cross-K/V and total encode; compare it with the Mac timing boundaries.
  - [ ] If the cross-K/V bottleneck is confirmed, optimize its eight matrix
    products and rebenchmark the complete transcription path, retaining the
    exact encoder-output check and full-logit accuracy gate.

- [ ] Qwen: return the existing macOS all-prefix BF16 oracle files
  `uzu-macos-all.npz` and `native-macos-all.npz` with their reports/hashes.
  Compare with `qwen35/local-results/asahi-todo-20261007/vendor-all-prefix/`
  to locate the first differing prefix/layer. Same-host checks already pass;
  cross-host accuracy remains unverified.
