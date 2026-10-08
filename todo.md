# macOS / Asahi follow-ups

Current priority: reproduce the PR 3905 fast encoder on base-M1 Asahi for
multilingual tiny/base/small. Medium is excluded. Further Mac reference
experiments and performance ablations are stopped.

The [README benchmark table](README.md) records measured Mac Accelerate/OpenBLAS
runs and conditional Asahi estimates of **≈29 / 63 / 335 ms**. Those estimates
assume Linux ANE and CPU performance match the Mac OpenBLAS surrogate; Linux
driver overhead and strict full-decoder accuracy remain unverified.
[Timing method and receipts](whisper/docs/benchmark-macos.md#cpu-cross-kv-library-substitution).

Only compact instruction templates, packing recipes and metadata belong in
the repository. Repack on Asahi from external safetensors or lossless F16/F32
GGUF; keep checkpoints, packed weights, HWX dumps and fixture arrays ignored
under `whisper/models` or `whisper/build`. Use the existing handoff files and
manifests without creating another archive.

## Completed preparation

- [x] Recreate the tiny/base/small Mac encoder benchmarks and OpenBLAS surrogate;
  add the measured/projected table to README and link it from the Whisper README.
- [x] Verify byte-exact reconstruction of all three offline exports, including
  small's two coefficient banks, from external checkpoints.
  [Packing instructions and proofs](whisper/docs/pr3905-packing.md).
- [x] Prepare Python replay and native whisper.cpp integration. All 12 fixture
  transfers pass host checks, including serial/four-worker readback; 24 Python
  tests, the Mac build, CPU smoke and device guard pass. These are host checks.
  [Native preparation evidence](whisper/results/pr3905-m1-20261008/native-preparation.json).
- [x] Recapture and repack the original 1,779-task tiny.en encoder, preserving
  padded input strides and the 1,783-task wrapper baseline.
  [Capture review](whisper/docs/fast-dump-review.md).
- [x] Finish Mac stage/BLAS/decoder profiling and the matched CPU accuracy checks.
  Historical ablations and their numerical limits are retained in
  [Mac follow-up](whisper/docs/macos-followup.md) and
  [decoder profiling](whisper/docs/decoder-profile.md).
- [x] Validate the Mac tiny.en paired numerical correction on all 80 logit
  vectors and prepare all 24 checkpoint-reconstructible projection exports.
  This remains a slower numerical control; an Asahi speedup is unverified.
  [Paired export](whisper/docs/asahi-paired-export.md).
- [x] Hash-verify and stage the existing Whisper and Qwen Mac fixtures locally.
  External transfer and cross-host verification remain pending.
  [Handoff manifest](whisper/results/macos-followup-20261007/handoff.json).

## Pending Asahi validation

- [ ] Hand off the existing exact fixtures and compact manifests to Asahi.
  Original three-clip Whisper fixtures are in
  `whisper/build/macos-handoff-20261007/whisper-fixtures`; the 12 PR cases are in
  `whisper/build/pr3905-replay-fixtures-20261008`. Preserve their bytes, shapes,
  strides and hashes; regenerate weights from the external checkpoint.
- [ ] Run all 12 PR encoder cases on native base-M1 Asahi with three warmups and
  20 measured runs per case. Check every output against its exact Mac reference
  with relative L2 < 0.005, `allclose(rtol=0.01, atol=0.03)`, matching position
  hashes and bitwise repeatability. Validate complete task chains and small's
  second bank. [Replay commands](whisper/docs/pr3905-packing.md).
- [ ] Independently qualify decoder accuracy for each multilingual PR model:
  tiny, base and small. Use each model's matching checkpoint, identical speech
  mel inputs and fixed reference token histories with its shared FP32 HF decoder.
  Compare both captured Mac and replayed Asahi encoder features against an
  independent reference encoder through that decoder. Require every full
  vocabulary logit vector to have NRMSE < 0.005, matching histories/raw argmaxes
  and repeatable outputs. Validate the native whisper.cpp decoder against the
  same reference before qualifying the complete path. Retain per-model input,
  checkpoint and history hashes, vector counts and pass/fail receipts; the
  original tiny.en 80-vector result does not qualify these models. Recheck and
  remeasure any numerical correction, and admit only passing model/backend
  combinations to the accuracy-qualified performance table.
- [ ] Build/run the native PR whisper.cpp adapter with matching checkpoints and
  OpenBLAS. Replace README projections only after measuring the same full-context
  zero-mel encode + CPU cross-K/V boundary: four threads, two warmups and five
  measurements per persistent context, two contexts per model. Retain every
  sample, library/build identity and preparation/dispatch/readback/total stage.
  Compare blocking Linux ioctl time separately with Mac E5RT execute time;
  compilation, model loading and decoding stay outside the encode timer.
- [ ] Verify the original tiny.en fast package on the exact 11/5/23-second Mac
  inputs, using four workers and `--compare-baseline` against the retained wrapper.
  Return input/output hashes and preparation/dispatch/readback/total profiles,
  including CPU cross-K/V and native readback. Recover the older native adapter
  and its absent `whisper/results/asahi-fast-20261007.json` receipt before treating
  the historical whole-transcription number as verified.
  [Original-package setup and source limits](whisper/docs/compact-encoder.md).
- [ ] Resolve the original tiny.en fast encoder's strict accuracy failure.
  Use the shared HF decoder and identical mel inputs to isolate and correct
  encoder arithmetic; then require every one of the 80 full decoder logit vectors to have
  NRMSE < 0.005, matching histories/raw argmaxes and repeatable outputs on all
  three clips. Preserve the failing baseline and remeasure any correction.
  Match the reference precision: FP32 activations/KV caches, widened FP32 CPU
  accumulation, exact GELU and 1,500 real cross-attention keys.
  Transcript/cosine or captured-encoder agreement alone does not pass this gate.
  Only accuracy-passing paths enter the accuracy-qualified performance table.
- [ ] Integrate/replay the prepared tiny.en paired projection control on Asahi:
  24 logical submissions and 32 hardware tasks with native padded ports. Verify
  executable identity, the unchanged 80-vector gate and warm native timings. Evaluate any
  resulting correction against the matched CPU reference; the slower Mac control
  does not establish speed parity.
  [Numerical control and export](whisper/docs/asahi-paired-export.md).
- [ ] Close the remaining Linux dispatch/whole-transcription gap with hardware
  evidence. Recover the existing driver profiling/`poll_sleep_us` control patch
  where needed; compare busy/longer polling with the original 1-us default,
  performance-core placement and CPU-policy samples before changing defaults.
  Retain enqueue/completion-wait/IRQ times and bitwise output checks. Record
  available clock, power and thermal evidence; explicitly identify unavailable
  counters. Keep OpenBLAS on one OpenMP runtime and preserve strict accuracy
  while evaluating CPU cross-K/V or decoder changes.
- [ ] Transfer `uzu-macos-all.npz`, `native-macos-all.npz` and their existing
  reports/hashes from `whisper/build/macos-handoff-20261007/qwen-oracles`.
  Compare all 23 prefixes against
  `qwen35/local-results/asahi-todo-20261007/vendor-all-prefix/` and identify the
  first differing prefix/layer from array values. Same-host parity or differing
  file hashes alone cannot establish cross-host accuracy.

Asahi hardware, the named Linux receipt and Qwen Linux all-prefix arrays are
not accessible from this Mac. Hardware execution, strict accuracy and parity
performance remain blocked until access or returned results are available.
