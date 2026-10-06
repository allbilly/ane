# macOS follow-ups

- [x] Benchmark full-model Qwen3.5-0.8B-M on macOS with ANE body projections
  and CPU recurrence, attention and vocabulary head. Use the same pinned Mirai
  M checkpoint and accurate compensation policy as Asahi (96 ANE submissions
  per input). The existing macOS recurrence probe is not a full-model benchmark.
  Match the Asahi six prompts (23/19/21/18/1,024/2,048 tokens), 64 decode calls
  per prompt, 2,112-token context capacity and four CPU workers. Measure prefill
  and decode separately, exclude warmup/setup, and retain at least three repeated
  CPU and ANE runs with host-load/paging observations. Verify every timed choice
  and full logits against the floating reference using the unchanged 0.5% NRMSE
  gate. Save raw per-prompt results, source/model hashes and runtime details so
  the Asahi comparison can be repeated with identical settings.
  Completed 2026-10-07: three CPU and three private-E5RT ANE runs pass all
  1,170 prediction choices per path. Maximum CPU/ANE NRMSE is 0.05817%/0.48789%.
  All timings are marked affected by host activity/paging. Raw compressed
  receipts, observations and measured source snapshots are committed under
  `qwen35/provenance/macos-full-model-20261007/`; see
  `qwen35/macos-full-model.md`. Prompt lengths/policy match Asahi; the saved
  cross-host trace hashes differ, so identical reference histories and an
  isolated operating-system speed comparison are not established.

- [ ] Dump the complete Whisper `tiny.en` ANE encoder kernels on macOS for
  native Asahi replay. Recover the already validated 1,783-task dense encoder
  export first; regenerate and dump on macOS if it cannot be recovered.
  Commit compact command templates and checkpoint packing recipes, keeping
  learned weights and full dumps outside Git, as requested. Include actual
  register/command streams in task order, reconstructible coefficient layouts,
  buffer sizes and relocations,
  input/output ports and tensor layouts, scratch/intermediate buffers, source
  MIL, checkpoint revision/hashes, and target ANE generation plus compiler/runtime
  versions. Retain the three real-audio FP16 fixtures and HF references locally,
  with their hashes and reproduction/validation commands in Git. Export hashes
  alone do not supply executable tasks.
  Use the complete dump to build a compact encoder replay path, replacing the
  CPU convolution/attention work and 1,128 separate projection submissions.
  The older `~/old_whisper.cpp/asahi/whisper-tiny-encoder.hwx` contains one
  convolution task, not the complete encoder. After Linux validation, benchmark
  matching CPU/CPU and ANE/CPU encode, decoder prompt setup, token decode and
  whole-transcription latency; retain transcript checks and the same timing
  boundaries as the macOS warm benchmark. The separate Linux projection path
  now runs without a fresh dump: all 80 raw logit argmaxes match CPU and an
  independent HF model across three clips; maximum full-logit NRMSE versus HF
  is 0.333%. Host and CPU optimizations reduced warm encode time from 721.41
  to 345.80 ms. Whole-transcription latency is 468.18 ms versus CPU's 417.49 ms
  for the 11-second JFK sample.
  That path leaves convolutions, attention and decoding on CPU and is not a
  replay of this complete exported encoder.
  Completed macOS packaging 2026-10-07: `whisper/kernels/tiny-en-encoder/`
  contains approximately 147 KiB of weights-free kernels and recipes.
  Repacking from the pinned safetensors reproduces every command, constant,
  coefficient, source MIL weight and position byte. The complete one-submission
  Python Asahi replay entry point is ready; hardware validation, whisper.cpp
  decoder integration and matching transcription benchmarks remain pending
  native Linux. See `whisper/docs/compact-encoder.md`.

- [ ] Transfer the existing macOS all-prefix Qwen BF16 oracle captures
  (`uzu-macos-all.npz` and `native-macos-all.npz`) with their report and hashes.
  Compare them with the new Asahi captures to locate the first differing prefix
  and layer behind the saved 1.5815% cross-host final-logit NRMSE. Same-host Uzu
  parity already passes on both systems.
  Prepared 2026-10-07: both Mac all-prefix arrays and the original report are
  archived under ignored `qwen35/local-results/macos-todo-20261007/`; the
  committed `macos-bf16-oracle-package-20261007.json` manifest verifies all
  hashes and same-host bitwise equality. The comparison helper accepts saved
  arrays directly. Actual transfer and comparison still need an Asahi
  destination/new all-prefix captures; neither is available in this workspace.
