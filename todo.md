# macOS and Asahi follow-ups

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

- [x] Capture and commit the complete Whisper `tiny.en` ANE encoder kernels
  and checkpoint packing logic for Asahi reproduction.
  Completed 2026-10-07 in commit `7aba9fe`: all 1,783 tasks are included in
  `whisper/kernels/tiny-en-encoder/`, totaling 150,350 bytes (approximately
  147 KiB). Learned weights, full HWX dumps and numerical arrays stay outside
  Git. The package includes command templates, coefficient recipes, buffer
  and port layouts, relocations, scratch sizes, source MIL and compiler/runtime
  provenance. Three real-audio fixtures and independent HF references remain
  local with committed hashes and macOS validation evidence.
  Repacking reproduces every command, constant, coefficient, source MIL weight
  and position byte, with SHA-256 checks. A fresh Asahi checkout needs only the
  pinned HF safetensors checkpoint and Python/NumPy to reconstruct the payloads;
  no macOS compiler, ANEForge or additional full dump is required.
  See `whisper/docs/compact-encoder.md` and `whisper/encoder_kernel.py`.

- [ ] Validate the complete Whisper encoder on native base-M1/T8103 Asahi.
  Use `whisper/replay_encoder.py` with the committed kernels and external
  `openai/whisper-tiny.en` safetensors revision
  `87c7102498dcde7456f24cfd30239ca606ed9063`. Reconstruct and check all payload
  hashes, then run the complete 1,783-task chain through the ANE accel driver.
  Supply frontend FP16 mel input `[80,3000]`; checkpoint position embeddings
  are repacked automatically. Check captured-output relative L2 < 0.005,
  allclose rtol=0.01/atol=0.03 and independent HF encoder cosine >= 0.999
  across all three clips. Byte-exact reconstruction is verified; Linux
  hardware execution remains unverified. Hold the shared ANE/GPU locks.

- [ ] Integrate the validated complete encoder with the whisper.cpp decoder
  on Asahi, replacing CPU convolution/attention and the current 1,128 separate
  projection submissions. Preserve transcript and full-logit checks.
  The existing projection path is independently validated across three clips:
  all 80 raw argmaxes match CPU/HF and maximum HF logit NRMSE is 0.333%.
  It remains a separate path until complete-encoder integration passes.

- [ ] After complete-encoder Linux validation and integration, benchmark
  matching CPU/CPU and ANE/CPU encode, decoder prompt setup, token decode and
  whole-transcription latency. Retain repeated warm runs, transcript checks,
  host observations and the same timing boundaries as macOS.
  Existing projection-path baselines: warm encode 345.80 ms, versus its earlier
  721.41 ms; whole transcription 468.18 ms versus CPU's 417.49 ms for the
  11-second JFK sample. See `whisper/docs/benchmark-asahi.md`.

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
