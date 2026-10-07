# macOS / Asahi follow-ups

Shared source is ready for the next macOS run: `whisper.native` applies the
same CPU precision fixes; `benchmark_native.py` uses the same native runner,
traces, timing parser and full-logit gates on both hosts. The Asahi rebuild
preserves all three CPU/ANE captures byte for byte, including all 80 vectors.
Both Python replay packages also preserve all three outputs. Mac preparation
and adapter syntax checks pass on Linux; Mac hardware execution is pending.
Use the short command sequence in `whisper/docs/compact-encoder.md` under
“Shared macOS / Asahi benchmark”. Keep reusable code outside `.cache`; generated
payloads/results go in `whisper/build`. The macOS handoff includes the shared
runner, repacker, profiling/validation code and commands; Linux-only driver,
bridge, precision experiments and generated reports remain local.
Linux hardware runs are finished; the next step is the shared macOS run and
returning the existing fixtures plus host profiles below. Transfer the handoff
commit to the macOS checkout before running the commands.

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
  Asahi recapture rerun: all 80 histories/argmaxes match, but maximum HF logit
  NRMSE remains 2.93% / 5.49% / 2.73%. Controlled HF-decoder tests isolate the
  error from checkpoint and boundary rounding: rounding mel or final features
  alone stays below 0.08%; full FP16 encoder arithmetic reaches 1.05% on the
  five-second clip. Earlier same-input traces also identify inaccurate fused
  GELU; replacing GELU alone was insufficient. Correct intermediate arithmetic
  as well as nonlinear operations, rather than recapturing the same graph.
  Further Linux controls: FP32 host normalization/GELU/residuals still leave
  3.44% error on the five-second clip. FP32 attention alone reduces it to 0.85%.
  A diagnostic using ANE convolutions/FFN first projections and FP32 CPU
  Q/K/V, attention and output projections passes all 80 vectors at
  0.192% / 0.495% / 0.347%, with matching argmaxes and bitwise repeatability.
  Its Python encoder alone takes 575–614 ms; it identifies a precision route,
  but does not improve the native benchmark. Preserve the production failure.
  Existing-kernel follow-up: split FP32 Q/K/V activations into two FP16 planes
  and combine their ANE products in FP32. With CPU attention/output projections,
  this passes all 80 vectors at 0.190% / 0.316% / 0.398%, with bitwise warm
  repeatability. Checkpoint Q/K/V weights already fit FP16 exactly; zero weight
  residual planes can be omitted. Moving output projections back to ANE still
  fails the five-second gate (0.543%; two rounding grids give 0.655%; two
  input partitions give 0.634%; four/eight give 0.532% / 0.607%). Splitting the
  FFN first projection too, after host FP32 normalization, now keeps all encoder
  matrix projections on ANE and passes all 80 vectors at
  0.255% / 0.392% / 0.384%, with matching argmaxes and byte-identical repeat
  outputs. This uses 34 submissions and CPU FP32 attention/nonlinearities;
  diagnostic time includes CPU reference checks and is not a native speed result.
  Runnable source: `experimental.whisper_precision --paired-outputs --paired-fc1`.
  - [ ] Locate the encoder operations responsible for the logit error, using
    the shared HF decoder to keep decoder differences out of the comparison.
  - [ ] Correct the numerical error, rerun every full logit vector on all three
    clips, and measure the resulting warm performance without weakening the gate.
    ANEForge candidates: matmul-based mean/variance and paired FP16 arithmetic.
    The current MIL has 18 `reduce_mean` nodes, so its existing `reduce_sum`
    rewrite needs adaptation. Test measured intermediates before a new capture.

- [ ] macOS: capture the missing performance evidence for the same M1, checkpoint
  and three clips. The ANE task/register dump is already verified; capture the
  host work and runtime behavior around it before requesting another graph.
  - [ ] Return the existing exact FP16 mel inputs and encoder outputs for all
    three clips, with array shapes, strides and byte hashes. Preserve the
    captured values rather than regenerating them; checkpoint weights remain
    reconstructible from safetensors. These also unblock the strict Linux
    input/output comparison below.
  - [ ] Profile the native ANE + CPU transcription path with separate timings
    for input conversion/packing, feed/copies, ANE submit/completion wait,
    output read/conversion, CPU cross-K/V, prompt batch and token decode.
    Label API wall time separately from device execution if hardware timestamps
    are available. Use four workers, two excluded warmups and ten measurements
    per clip per round, with reversed CPU/ANE backend order in round two; retain
    each repetition.
  - [ ] Time all eight cross-K/V matrix products separately, including weight
    conversion/packing and GEMM. Record actual backend/routine, matrix dimensions,
    operand/output types, strides, transposes and thread counts. Retain one
    representative activation input and its packing description for a matched
    Linux microbenchmark; reconstruct learned matrices from the checkpoint.
    The shared `--profile-matmul` implementation is verified on Asahi: 1,152
    matrix calls, eight products per encode, four actual OpenBLAS threads,
    separate allocation/conversion/thread-setup/GEMM times and captured inputs.
    All 18 native mel/encoder/logit files remain byte identical. Run it on Mac
    too; its BLAS entry-point identity still needs the sampling profile below
    to identify the internal hot routine.
  - [ ] Capture a CPU sampling profile and symbol/address map for cross-K/V,
    prompt evaluation and token decode. Dump assembly only for the selected
    hot routines: BLAS GEMM, ggml matrix/vector products, attention/softmax and
    vocabulary projection as identified by the profile. Include loaded image
    names/UUIDs, compiler version, build flags and backend/thread settings.
    Establish whether the selected BLAS routine uses NEON or Apple AMX;
    Accelerate initialization alone does not prove which kernel ran.
    Asahi user-cycle sampling now works without sudo: the two main ggml FP16
    weight/FP32 activation GEMV routines account for 52.73% of sampled cycles,
    attention 14.54%, and OpenBLAS `sgemm_kernel_NEOVERSEN1` 12.48%.
    Targeted assembly is saved under `whisper/build/cpu-profile-20261007`;
    these are CPU cycle shares, not fractions of end-to-end wall time.
  - [ ] Run the shared `benchmark_native.py --backend macos --profile-stages --profile-matmul`
    with `prepare_native.py --backend macos` applied first. It uses the Linux
    accuracy settings: FP32 activations/KV caches,
    widened FP32 accumulation, exact GELU and 1,500 real cross-attention keys.
    Check all 80 full logit vectors against HF and retain the same stage timings.
    Keep the original fast configuration as a separate baseline; its failed
    numerical gate makes its CPU decode speed an unmatched comparison.
  - [ ] Record available CPU/ANE clock, power/thermal and core-placement evidence
    during warm runs, plus OS/runtime/library identities. Mark unavailable
    counters explicitly so timing alone is not mistaken for a clock diagnosis.
  Return compact timing/profile logs, targeted assembly text and the required
  fixture arrays; reuse existing captures instead of creating another archive.

- [ ] Asahi: replay the new original 1,779-task compact kernels with inputs/weights repacked from
  the pinned HF checkpoint, without a macOS compiler. Validate the same cases
  and compare timings with matching boundaries and four CPU workers.
  New graph runs in one submission with padded inputs and four-worker native
  readback. All three native outputs match the old wrapper bit for bit; Python
  replay also matches both packages on identical locally generated inputs.
  OpenBLAS/OpenMP reduces CPU cross-K/V to 14.16 ms. Native stages: input
  0.59 ms, clear 0.18 ms, dispatch 14.09 ms, readback 1.34 ms, conversion/check
  0.79 ms. Latest 11-second encode is 31.18 ms and whole transcription 142.90 ms,
  versus Mac 15.83 / 66.81 ms. CPU decoder is also slower. Not on par yet;
  timings are diagnostic while accuracy fails, with active-desktop variability.
  See `whisper/results/asahi-fast-20261007.json` and the runnable commands in
  `whisper/docs/compact-encoder.md`. Keep the old dump/results as baseline.
  - [x] Add a checkpoint/audio-only profiler and exact Mac output hashes;
    verify all three regenerated FP16 input hashes on Mac without hardware.
  - [ ] Run checkpoint/audio-only replay profiling with exact Mac input/output
    hashes and `--compare-baseline`; return dispatch, preparation, readback and
    total encoder times. Use the command in `whisper/docs/compact-encoder.md`.
    Local profiling is done; the strict cross-host check fails before submission
    because Linux-generated mel hashes differ. Fifteen frontend variants did
    not reproduce the Mac inputs. Need the three existing input/output fixtures
    for an exact comparison; no new compiler/register dump is needed for this.
  - [x] Obtain the native four-worker full-encoder profile, including CPU
    cross-K/V and total encode; compare it with the Mac timing boundaries.
  - [x] If the cross-K/V bottleneck is confirmed, optimize its eight matrix
    products and rebenchmark the complete transcription path, retaining the
    exact encoder-output check and full-logit accuracy gate.
    Use the OpenMP OpenBLAS variant and a single OpenMP runtime. Loading both
    system and Python-wheel OpenMP runtimes regressed cross-K/V to about 50 ms;
    the standard Fedora OpenMP library avoids that. All 80 vectors were retested
    and failure remains explicit; the numerical gate has not been weakened.
  - [ ] Close the remaining dispatch gap: Linux ioctl is about 14 ms versus
    Mac execute 10.88 ms. Driver stage profiling is now hardware verified on
    all three clips with bitwise output checks: 14.20 ms dispatch includes
    14.16 ms completion wait, 0.009 ms enqueue and 0.011 ms IRQ drain.
    Clearing diagnostic-event masks reduces IRQ cleanup but does not close the
    gap. An efficiency-core ioctl takes 32.42 ms; keeping a performance core
    busy reduces it to 13.93 ms. This establishes sensitivity to CPU/cluster
    activity, not the ANE frequency or a confirmed clock cause. The bounded
    `poll_sleep_us` control is built with the original 1-us default; reload it
    and compare busy/longer polling with CPU policy samples and bitwise checks
    before changing defaults. Wait timing includes hardware and scheduling.

- [ ] Qwen: return the existing macOS all-prefix BF16 oracle files
  `uzu-macos-all.npz` and `native-macos-all.npz` with their reports/hashes.
  Compare with `qwen35/local-results/asahi-todo-20261007/vendor-all-prefix/`
  to locate the first differing prefix/layer. Same-host checks already pass;
  cross-host accuracy remains unverified.
  Checked again on Asahi: the Mac NPZ files are absent. The local captures and
  Mac package have different file hashes; this alone cannot identify an array
  difference. Need the existing files to locate the first differing layer.
