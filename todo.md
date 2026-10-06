# macOS follow-ups

- [ ] Benchmark full-model Qwen3.5-0.8B-M on macOS with ANE body projections
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

- [ ] Make the already validated macOS Whisper `tiny.en` dense encoder kit
  available for native Asahi replay. Recover the existing 1,783-task export first;
  regenerate on macOS only if it cannot be recovered. Include the actual
  `model.hwx` and coefficient payloads, task/port metadata, source MIL, checkpoint
  revision/hashes, and the three real-audio FP16 input/output fixtures with their
  independent HF references. Export hashes alone do not supply executable tasks.
  The older `~/old_whisper.cpp/asahi/whisper-tiny-encoder.hwx` contains one
  convolution task, not the complete encoder. After Linux validation, benchmark
  matching CPU/CPU and ANE/CPU encode, decoder prompt setup, token decode and
  whole-transcription latency; retain transcript checks and the same timing
  boundaries as the macOS warm benchmark. The separate Linux projection path
  now runs without a fresh dump: all 80 raw logit argmaxes match CPU across
  three clips, maximum logit NRMSE is 0.310%, and warm whole-transcription
  latency is 810.35 ms versus CPU's 567.62 ms for the 11-second JFK sample.
  That path leaves convolutions, attention and decoding on CPU and is not a
  replay of this complete exported encoder.

- [ ] Transfer the existing macOS all-prefix Qwen BF16 oracle captures
  (`uzu-macos-all.npz` and `native-macos-all.npz`) with their report and hashes.
  Compare them with the new Asahi captures to locate the first differing prefix
  and layer behind the saved 1.5815% cross-host final-logit NRMSE. Same-host Uzu
  parity already passes on both systems.
