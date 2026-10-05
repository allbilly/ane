# Speculative decoding lessons from Asahi MLX

Measured on 2026-10-05 on a base M1 MacBook Air with 8 GB RAM, Fedora
Asahi Remix 44, kernel 7.1.13+, and stock Mesa 26.2.3 Honeykrisp. These are
Linux GPU findings, not a reproduction of the M4 Pro CoreML/ANE results.

The experiment used the omarchy-mlx Vulkan build
`mlx-omarchy 0.32.4.dev202610041653+6edd258` (release v0.7.27),
`mlx-lm 0.31.3`, a Qwen3-0.6B bf16 target, and its 4-bit version as an
autoregressive draft. This is a small feasibility test, not DFlash. The
4-bit model is the **draft**, not the target.

## What happened

Four diverse prompts, 100 generated tokens each, two passes with reversed
baseline/SD order on the second pass:

| Prompt | Greedy target tok/s | SD tok/s | Token equality |
|---|---:|---:|---|
| capital | 33.31 | 10.98 | first difference at index 3 |
| fibonacci | 33.21 | 10.88 | all 100 equal |
| math | 34.86 | 11.14 | first difference at index 29 |
| story | 33.97 | 10.28 | first difference at index 96 |
| **Mean** | **33.83** | **10.82** | **3 of 4 differ** |

The mean paired speedup was **0.32x**, with a best prompt of 0.33x.
Draft-origin tokens made up 62–73% of generated tokens; that is not the
fraction of all proposed draft tokens accepted. Both passes reproduced the
same divergence indices. Prefill was outside the measured decode interval;
the full first verification block was inside it.

GPU tensor, attention, RoPE, and tiny-model cache/accept/reject smoke tests
passed. GPU tracing confirmed dispatch. Those checks establish that the
runtime works for the tested operations; they do not establish pretrained
SD correctness or speedup.

Detailed receipts, model revisions, hashes, timings, and reproducer commands
are in [the Asahi findings](../mlx-ane-sd/notes/asahi_findings.md) and
[raw artifacts](../mlx-ane-sd/notes/asahi/).

## What the ANE source actually tells us

The current Linux driver and portable GPT-2 path are a verified research
implementation, with specific limitations worth improving. Calling them
poorly written is broader than the evidence supports. The MLX SD experiment
above did not call this ANE driver at all.

| Observed in this repository | Implication and next check |
|---|---|
| [Submission](kmod/ane_drv.c) holds `engine_lock` through synchronous execution; [completion](kmod/ane_tm.c) uses `readl_poll_timeout` with a 1 microsecond polling interval and a 1 second timeout. | ANE submissions serialize. Measure lock wait, submit-to-completion time, and polling cost before proposing asynchronous queues. The polling interval is not a measured 1 microsecond dispatch latency. |
| [Decode replay](gpt2/replay.py) pads one vector into a `[768,32]` tile, rewrites inputs, clears scratch, poisons outputs, submits, and copies outputs back. | Account for preparation/copies separately from ioctl time. Test reusable input/scratch buffers or a verified narrower tile. Padding alone does not prove that 32 times the arithmetic is executed; inspect the program and measure. Preserve unwritten-output checks in validation. |
| [The model loop](gpt2/model.py) makes two synchronous ANE calls per layer, across 12 layers. Attention, output projection, final normalization, and the vocabulary head stay on CPU. | There are 24 ANE calls and repeated CPU/ANE boundaries per token. Fusion or moving a large projection may matter more than tuning one elementary kernel. Profile each component first. |
| Kernel programs and device buffers are cached in `Device.kernels`; compiled coefficients are cached and hash checked. | Preserve this existing reuse. Separate first-use allocation/packing from warm execution instead of treating startup as steady-state ANE compute. |
| The driver retains a polling fault design with the combined ANE/DART fault IRQ masked, as documented in [the KMD notes](kmod/README.md). | Timeout and numerical checks remain necessary. IRQ handling and recovery deserve review before an asynchronous implementation; removing the mask without solving the secondary DART handling is not a performance fix. |
| This target currently exports neither `agx_stats` nor `ane_stats`. | Add observability before attributing idle time or claiming an ANE ceiling. Coreglass can capture host cost now, but engine busy time is unavailable. |

Existing [hardware receipts](gpt2/asahi-runtime-validation.json) show all
49 inference kernels and full-generation parity passing on this base M1.
That establishes useful correctness evidence, not optimal performance.
Likewise, older CPU-versus-ANE figures in the GPT-2 notes include **macOS**
measurements and cold compilation; they must not be relabeled as a current
Linux driver comparison.

The most transferable lesson from the SD project's M4 work is to profile
the whole cycle, then reduce costly engine boundaries and offload the actual
hot projection. Its CoreML LUT6/fusion results are hypotheses for a Linux
port, not register recipes. Revalidate real hidden states and token outputs
after changing packing, precision, tile shape, fusion, or cache positions.

## Lessons to carry into ANE work

1. **Separate bring-up, correctness, and performance.** Discovering an ANE
   device or dispatching a GPU kernel is the first gate. Then verify real
   model outputs, then benchmark. A speed number with divergent greedy
   tokens cannot be labeled an exact speculative-decoding result.

2. **Test one-token and block verification on identical cache histories.**
   The same bf16 target, without a draft or quantized weights, changed its
   winner between sequential and four-token evaluation for the capital and
   math prefixes. Maximum logit differences were about 0.16 and 0.125;
   winning margins were 0–0.125. Fibonacci retained its winner with a
   margin of 12.875. This isolates a numerical difference in target
   evaluation for two prompts, rather than attributing every mismatch to
   draft quantization. The controlled probe did not isolate the story
   mismatch. Use the runtime's actual `argmax`: sorting tied logits can
   choose a different token and give a misleading diagnosis.

3. **High acceptance alone does not imply a win.** Account for draft
   execution, block verification, cache trimming, synchronization, sampling,
   and CPU/device transfers. These results show a loss for this model pair
   and runtime, not a universal Asahi SD ceiling. Profile those components
   before increasing the draft length or choosing a larger draft.

4. **Keep a precise measurement contract.** Use at least four varied
   prompts, repeat with alternating run order, record every output token,
   define EOS counting, and report means and the best prompt. Distinguish
   prefill, time to first token, and decode. A streaming API's reported
   generation rate may exclude its first-token interval; it is not directly
   interchangeable with this full decode-wall-time measurement.

5. **Pin the runtime and the data.** Record the wheel release, native
   binary hashes, actual selected device, kernel/Mesa versions, model
   revisions, tokenizer hashes, and weight hashes. Install the Vulkan MLX
   fork in an isolated environment; avoid accidentally replacing it with
   upstream MLX while installing mlx-lm. This machine also needed local
   OpenBLAS/libgfortran libraries. Check proxy configuration early if model
   downloads stall, and verify mirrored weights against upstream hashes.

6. **Start within the hardware's memory budget.** This machine exposes
   roughly 3.9 GB of usable GPU heap. The original Qwen3-4B bf16 target is
   about 8 GB before its draft and caches. The M4 Pro's 64 GB experiment
   cannot be transferred unchanged. Use a small dense target first; do not
   use Qwen3.5/GatedDeltaNet caches with a loop that requires trimming.

7. **Treat heterogeneous execution as a new experiment.** The Linux
   register-programmed GPT-2 runtime and the native Qwen3 backend in
   `../qwen3.c` already provide concrete ANE paths. CoreML packages and the Swift CoreML runner are not Linux
   deployment artifacts. A real ANE draft for Qwen needs compatible trained
   weights, tokenizer, positions, attention, cache rollback, and a verified
   accept/reject loop. Existing GPT-2 inference is useful for independent
   engine contention tests, but is not a compatible Qwen draft.

8. **Measure placement and overlap honestly.** Lock shared GPU/ANE
   workloads, capture an idle baseline, mark each workload phase, and save
   a manifest. Missing driver busy counters mean **not captured**, not
   zero utilization. IRQ counts and call latency are not utilization.
   Separate sequential ANE/GPU phases establish each path; a subsequent
   simultaneous run is needed to measure interference and aggregate
   throughput. Include host copies and coordination in the end-to-end
   result.

## Useful next experiments

- Probe bf16 target block widths 1, 2, 4, 8, and 16 on real prefixes with
  identical caches. Save logits, margins, and actual argmax decisions before
  optimizing speculative verification.
- Profile GPU draft and target timings independently. Only test larger
  models or longer speculation when measured costs leave headroom.
- Use Coreglass to capture CPU, GPU, and known-good ANE workloads in marked
  phases. Keep its streaming generation metric distinct from the SD
  benchmark's decode metric, and preserve missing-counter labels.
- Once each engine's solo run passes, compare solo versus concurrent ANE
  GPT-2 and GPU Qwen execution with identical per-engine workloads. That
   tests contention; it does not itself demonstrate speculative decoding.

The reusable outcome is the validation method and the verified Linux MLX
environment. The measured SD configuration is currently a negative result.

## Existing Qwen3 ANE backend: a better starting point

Inspection of `../qwen3.c` on 2026-10-05 found a working **native C Qwen3
ANE backend**, not just the GPT-2 path. The checkout was at
`69d81fc`. [Its backend](../qwen3.c/ane/ane_matmul.c) creates resident
projection weights and reusable device/host buffers; its
[causal batched forward](../qwen3.c/ane/prefill.h) processes up to 32 tokens
with Qwen3 normalization, RoPE, attention and KV handling. Linear projections
run on ANE, while attention, normalization, nonlinearities, and the
vocabulary head run on CPU. This materially narrows the remaining Linux
work; claiming that no Qwen ANE path exists would be incorrect.

The retained five-trial Qwen3-0.6B **Q8** benchmark reports 1.97–2.37x faster
prefill for 16–65-token prompts. At 32 prompt tokens, CPU prefill was 397.8 ms
and ANE prefill 184.8 ms. CPU Q8 decode was 58.9 tokens/s; explicitly routing
single-token projections to ANE measured about 24.4 tokens/s in a separate
three-trial comparison. Initialization is excluded and these are shared
desktop measurements. The prompt-length cases are not a four-diverse-prompt
SD benchmark. See [the saved comparisons](../qwen3.c/ane/benchmarks/).

This is **not the same precision path as our MLX bf16 test**: the checkpoint
is Q8, ANE plans expand its projection weights to FP16, and activation
arithmetic differs between CPU and ANE. Saved full-model checks matched
63/64 top-1 decisions on fixed inputs, with one near-tie change. A fresh
four-case, four-prediction diagnostic matched 15/16, reproducing that issue;
it is a numerical check, not a new performance estimate.

For speculative decoding, the promising reuse is **batched target
verification**, rather than simply routing every autoregressive token to
ANE. The remaining concrete changes are:

- Return final hidden states/logits for every candidate position. The current
  prefill computes the vocabulary head only for the final row; an SD verifier
  must check the whole candidate prefix and obtain the correction/bonus token.
- Add logical KV commit/rollback and the accept/reject loop. The existing
  position-indexed cache is useful infrastructure, but no SD loop is present.
- Implement a compatible trained draft. The current causal Qwen forward is
  not the DFlash block-diffusion architecture; DFlash also needs target hidden
  features and its own attention/cache contract.
- Benchmark realistic verification widths. Short batches below 16 positions
  normally route to CPU; a proposed ANE verifier needs an explicit batch policy
  instead of turning on slow ANE single-token decode accidentally.
- Align and validate the target arithmetic. Near-tie disagreement must be
  addressed before comparing against the CPU Q8 or MLX bf16 greedy stream.
- Measure classifier and transfer costs. There are seven ANE projections per
  layer, hence 196 projection calls per full 28-layer batch. The current plan
  caps output width at 32736, below Qwen's 151936-token vocabulary; an ANE
  vocabulary head would need chunking or a different verified program.

The native matmul test passed 15 shape/trial cases and three batch cases,
including in-place outputs, with 27 ANE submissions. All three shared-queue
tests passed. Review receipts and source/binary hashes are in
`~/.cache/qwen3.c/sd-review/`; the Qwen repository was left unchanged.
The next useful gate is all-position verification over real prefixes with
identical cache histories, before building or timing a full SD pipeline.

## Coreglass follow-through on this machine

Coreglass was run using the same verified omarchy-mlx environment and pinned
bf16 checkpoint. The local target lives in the user's Coreglass config, not
in the public repository. A corrected five-phase capture recorded **414
samples over 41.6 seconds**: P-core spin, E-core spin, GPU matmul, MLX Qwen
generation, and ANE GPT-2 generation. Every phase returned success. The ANE
phase executed two batches, each checking 24 decode kernels and full
generation parity before generating 16 tokens.

The capture is an active-desktop functionality test; the quiet-host load gate
was explicitly overridden with free GPU locks and the override retained in
the manifest. One Qwen request generated 64 tokens at the stream API's
reported 30.3 tok/s. This is not a controlled throughput comparison or the SD
benchmark's timing contract. Neither driver exports engine busy counters;
the UI shows unavailable ANE busy time and labels GPU firmware interrupts as
an activity proxy. Host telemetry does not reveal ANE kernel compute time.

Bring-up exposed and fixed Coreglass issues: its workload/preflight path
needed the sampler's existing local transport; fractional ANE durations
failed in shell arithmetic and falsely returned success; bf16 model labels
said `None-bit`; Brave was not recognized for export; and capture-only
summaries omitted missing engine counters. Loopback server discovery now
bypasses desktop HTTP proxies. All 19 Coreglass tests passed, and browser
checks covered live sampling, saved-capture replay, and an anonymized PNG
built from this capture alone, without bundled reference measurements.

Local receipts are under `~/.local/share/coreglass/`: `experiments.md`,
`verified.log`, `phases.json`, `verification.md`, provenance/source hashes,
and `captures/asahi-20261005-verified.jsonl` with its `.run.json` manifest.
The initial failed ANE receipt is retained as
`captures/asahi-20261005-bringup.jsonl`; its success exit code must not be
mistaken for actual ANE execution.

The app runs at <http://127.0.0.1:8777/> as the transient user service
`coreglass-asahi.service`. Stop it with
`systemctl --user stop coreglass-asahi.service`. It does not start at boot.

The next ANE optimization measurement should add component timings around
input preparation, each ioctl, output copies, CPU attention, and the head,
then compare those costs across four real prompts. This capture establishes
that both execution paths work, while leaving kernel efficiency unproven.

## Native Linux DFlash follow-through (2026-10-05)

The project now has a trained DFlash draft running on the base M1 ANE with a
Qwen3-0.6B bf16 MLX/Vulkan target. This is a smaller-model reproduction of the
block-diffusion mechanism, not the M4 Pro Qwen3-4B full-ANE result. Across four
prompts, two passes and 100 generated tokens, the lossless scalar verifier
matched all 800 greedy tokens but slowed from **22.30 to 15.56 tok/s (0.70x;
best trial 0.75x)**. Batched bf16 verification averaged **20.59 to 9.22 tok/s
(0.45x)** and only four of eight runs matched token-for-token.

| Verification path | Greedy mean | DFlash mean | Paired speedup | Token identity |
|---|---:|---:|---:|---|
| Scalar, stop at first rejection | 22.30 tok/s | 15.56 tok/s | 0.70x | 800/800 tokens |
| Batched target | 20.59 tok/s | 9.22 tok/s | 0.45x | 4/8 runs |

The scalar path uses the same single-token target arithmetic as greedy
decoding, which preserves identity but gives up parallel verification's
amortization. The batched path diverged on capital and story, consistent with
the separate block-versus-scalar numerical differences. Its average was only
2.106 emitted tokens per cycle and target verification took about 200 ms per
cycle; even a zero-cost draft would reach only about 10.5 tok/s at that rate.
The target batching path and draft acceptance both need improvement before
this configuration can beat its greedy baseline. An FP32 24-token diagnostic
is useful for investigating numerical drift, but is not a bf16 speedup or a
100-token quality result.

The Linux draft reuses a copied native C Qwen3 ANE linear backend. Context and
transformer projections plus the vocabulary head run on ANE; attention,
normalization, nonlinearities, residuals and host decisions run on CPU. The
target remains on MLX/Vulkan. Each proposal makes 59 ANE submissions (21 body
projections and 38 vocabulary-head chunks). Profiling measured about 39.85 ms
per proposal including packing, transfers and host work, with about 17.23 ms
inside synchronous submission calls; that interval includes compute and
driver waiting, not just overhead. This is a useful native Linux port, but it
does not have the fused graphs, LUT6 weights or target offload of the macOS
runner.

The next cross-OS evidence is pending: a prepared kit captures 20 matched
real-input ANE cases and measures the same target on Metal on this same M1
under macOS. After Git pull, follow the [capture task](../mlx-ane-sd/task.md)
and [kit instructions](../mlx-ane-sd/asahi/macos/README.md); macOS execution
and Linux replay have not yet been completed. The full measurements, raw
receipts and bottleneck analysis are in [the DFlash findings](../mlx-ane-sd/notes/asahi_dflash_findings.md)
and [the project survey](../mlx-ane-sd/notes/asahi_sd_project_survey.md).

The original 2.21x result remains specific to an M4 Pro with 64 GB, a Qwen3-4B
bf16 target, a different trained draft and the fused CoreML/LUT6 stack. The
base M1 Linux result neither reproduces nor directly benchmarks that setup.
