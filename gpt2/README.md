# Orion GPT-2 on Asahi

Complete GPT-2 124M text generation, ported from `~/Desktop/Orion`. Standard
model weights stay in your Hugging Face cache. ANE coefficients are regenerated
from that checkpoint on the target machine; the dump supplies reference hashes
and packing recipes. Linux execution uses Python, NumPy,
`regex`, `safetensors`, and the `ane` kernel driver; it needs no Apple
frameworks, `anecc`, or Python `libane` binding.

The runner, replay/reference data, and benchmark reports are about **12 MiB**, versus about
**974 MiB** for the source dump and its surrounding experiments. Standard model
weights and compiled learned coefficients are omitted. Kernel templates have
their learned layer-norm constants zeroed. The original dump remains available
as a reference and backup.

## First run on Asahi

Copy this entire `gpt2` directory into `~/ane/gpt2`. On a base **M1 / t8103**
running Asahi Linux with the ANE device tree and `ane` driver installed:

```sh
~/ane/gpt2/first-run.sh --prompt 'Hello world' --max-tokens 32
```

The script creates a local Python environment and installs NumPy, `regex`, and
`safetensors`. It discovers GPT-2 in `~/.cache/huggingface/hub` (respecting
`HF_HOME`, `HF_HUB_CACHE`, and `XDG_CACHE_HOME`) and reads `model.safetensors`
directly. There is no model copying or download when a matching cache exists.
On the first ANE run, the loader packs the required matrices, biases, and folded
layer-norm coefficients into `~/.cache/orion-gpt2/h13g-packed-v1` (under
`XDG_CACHE_HOME` when set). Every generated payload must match the original
dump's SHA256 before submission. Later runs reuse verified cache entries;
corrupt entries are regenerated. Nothing is written back into the HF cache.
If no weights exist, it downloads the standard checkpoint into an external
cache. `--weights PATH` or `GPT2_WEIGHTS` selects an existing checkpoint or an
Orion BLOBFILE directory explicitly.
Python 3.10 or newer, Python's venv/pip support, and internet access for the
initial dependency install are required. A model download is only needed if
the checkpoint is absent from the cache. `GPT2_PYTHON=/path/to/python3` selects
another Python. There is no network access during inference.
The CLI defaults `OPENBLAS_NUM_THREADS` to `1` before importing NumPy for
batch-one matrix-vector operations. An explicitly set value is preserved.
Subsequent launches reuse the installed dependencies; pip runs again only when
requirements change or a required import is missing. Downloads use a pinned HF
revision and separate temporary files, so concurrent setup processes cannot
overwrite one another's partial download.

First-run verification is part of generation: every packaged file is hashed,
the ANE device is identified through its driver, weight packing is checked
against the reference hashes, all 24 decode kernels are
compared against recorded macOS ANE outputs, and the complete generation flow
is checked against macOS ANE logits and four greedy tokens. A failure exits
with a diagnostic. The selected ANE backend never falls back to CPU.

**Linux ANE decode and generation were verified on base M1 Asahi on
2026-10-04.** All 24 decode kernels matched the macOS ANE fixtures, and the
full generation check passed its reference logits and four greedy tokens.
Repeated 32-token generations completed successfully. The 25 reference
prefill kernels have not been replayed on Linux in these runs.
`package.json` and [asahi-decode-performance.json](asahi-decode-performance.json) record this
verification scope.

Packing can be tested without an ANE device, on macOS or Linux:

```sh
~/ane/gpt2/first-run.sh pack --all-kernels
```

`setup` finds/downloads the external checkpoint and packs the 24 decode kernels;
`--all-kernels` additionally packs the prefill reference kernels. The generated
cache uses about 149 MiB for decode, or 204 MiB for all kernels. This storage is
created on the target machine; it is excluded from the portable directory.

Stock Asahi installation alone is insufficient if it lacks the ANE device
tree/driver. Follow the parent [ANE setup instructions](../README.md) and
[allbilly/libane](https://github.com/allbilly/libane) for that prerequisite.
`doctor` checks the package, M1 device-tree compatibility, ANE driver binding,
and device permissions before submitting work:

```sh
~/ane/gpt2/.venv/bin/python ~/ane/gpt2/gpt2.py doctor
~/ane/gpt2/.venv/bin/python ~/ane/gpt2/gpt2.py verify
~/ane/gpt2/.venv/bin/python ~/ane/gpt2/gpt2.py verify --all-kernels
```

`--device /dev/accel/accelN` can select another ANE node. M1 Pro/Max, M2, and
later chips are rejected: the source capture is base M1 and those chips need
their own validated artifacts. Kernel/device-tree installation is deliberately
outside this startup script.

## CPU generation and sampling

The explicit CPU backend works on macOS and Linux and verifies its logits
against Orion's independent C/Accelerate implementation before generation:

```sh
~/ane/gpt2/first-run.sh --backend cpu --prompt 'Hello world' --max-tokens 18
~/ane/gpt2/.venv/bin/python ~/ane/gpt2/gpt2.py generate \
  --prompt 'The capital of France is' --temperature 0.7 --top-k 40 --seed 123
```

Greedy generation is the default (`--temperature 0`). Context is 1024 tokens,
including processed prompt and continuation tokens. EOS ends generation. UTF-8
output is decoded incrementally, so multibyte tokens can span printed chunks.

The inference flow follows Orion's decode path:

1. CPU token + position embedding.
2. For each of 12 layers: ANE LN1/Q/K/V projection; CPU KV-cache attention,
   output projection and residual; ANE LN2/FFN/residual.
3. CPU final layer norm, tied embedding output head, and token selection.

Prompt tokens are ingested sequentially through the decode path. The 32-wide
ANE tensor stride is retained, with each token at sequence position zero.
This permits prompts of up to 1024 tokens without treating the captured
32-position prefill bucket as a larger graph. It may be slower than Orion's
bucketed prefill. The 25 captured prefill kernels remain available for replay
and reference, and are covered by `verify --all-kernels`.

## Measured Asahi generation

Measured on **base M1 / 8 GB / Fedora Asahi Remix 42 / Linux 6.19.11+** on
2026-10-04, using the pinned GPT-2 124M checkpoint, batch one, greedy sampling,
and the two-token prompt `Hello world`. With one OpenBLAS thread and decode
copying only the first output position, four fresh CLI invocations measured:

| Run | Generated tokens | Generation elapsed | Generated tokens/s |
| --- | --- | --- | --- |
| Trial 1 | 32 | 0.53 s | 60.38 |
| Trial 2 | 32 | 0.58 s | 55.17 |
| Trial 3 | 32 | 0.54 s | 59.26 |
| Trial 4 | 32 | 0.55 s | 58.18 |
| All four combined | 128 | 2.20 s | 58.18 |

Rates are generated tokens divided by the CLI's reported elapsed time,
which is rounded to 0.01 s. The timer includes sequential prompt processing,
KV-cache reset, generation, token selection, and continuation printing.
Dependency setup, checkpoint loading, packing, and the built-in parity checks
finish before the timer starts. Those checks exercise all 24 decode kernels
and the generation path before each measured generation. No additional
benchmark warmup or resource isolation was used for these CLI runs. CPU
generation and time to first token were not measured in this session.

All CLI runs used `/dev/accel/accel0`; all 24 decode-kernel checks and full
generation parity passed, with identical generated text. ANE executes
projection and FFN kernels; attention, embeddings, output projection, logits,
and token selection run on CPU. [asahi-decode-performance.json](asahi-decode-performance.json)
records the runtime versions, timing scope, and retained measurement evidence.

Reproduce from the repository root:

```sh
./gpt2/first-run.sh --backend ane --prompt 'Hello world' --max-tokens 32
```

## GPT-2 implementation comparison on M1

Measurements on **M1 / 8 GB**, batch one, with model loading and warmup
excluded. The Asahi row uses Fedora Asahi Remix 42 / Linux 6.19.11+;
the remaining rows are saved macOS 27.0.1 references. Every measured cell
uses 64 decode steps per trial and the same checkpoint/token trace for its
prompt. Asahi and native Orion each have four trials per prompt and two
16-step warmups immediately before each trial. These are separate sessions.

Decode **engine steps/s**:

| Implementation | 2-token prompt | 32-token prompt | 64-token prompt |
| --- | --- | --- | --- |
| Asahi Python ANE + CPU attention § | 70.18 | 63.58 | 57.60 |
| Orion CPU ‡ | 50.03 | 51.50 | 47.83 |
| Orion ANE + CPU attention ‡ | 52.57 | 52.00 | 48.42 |
| CoreML CPU | 65.21 | 64.83 | 62.52 |
| CoreML CPU + GPU | 51.41 | 51.32 | 51.82 |
| CoreML CPU + ANE | 49.66 | 49.18 | 50.51 |
| CoreML ALL | 52.45 | 52.50 | 52.41 |
| MLX-LM GPU FP16 † | 108.92 | 123.03 | 132.47 |
| MLX-LM GPU FP32 | 70.72 | 70.68 | 70.64 |
| vllm.cpp | Unavailable | Unavailable | Unavailable |

§ Asahi uses one OpenBLAS thread and copies only the consumed first output
position during decode. Full replay still checks all 32 output positions.
All 24 kernel fixtures and generation parity passed; all three saved
65-prediction HF token-choice traces and 16-token greedy continuations matched.
HF KL and raw-logit parity were not measured. The synchronous `model.step`
timer includes embeddings, attention, full-vocabulary logits, ANE dispatch
and transfers, and internal finite checks; prompt processing, allocation,
argmax, printing, and diagnostic replays are excluded. Prompt ingestion uses
sequential decode, while native Orion uses bucketed prefill. This compares
implementations across sessions; it does not isolate an OS speed advantage.

The reproducible Asahi report, including every timed sample, is
[asahi-decode-performance.json](asahi-decode-performance.json):

```sh
./gpt2/.venv/bin/python gpt2/tools/bench_asahi.py --output /tmp/asahi-decode.json
```

† MLX FP16 changes one of 65 HF argmax predictions for `Hello world`; its
16-token free greedy continuation differs. The 32/64-token cases pass the
choice/KL gate. Raw logit NRMSE reaches 0.175 across these cases, despite small
probability KL; this is not full-logit parity. MLX FP32 passes all three
65-prediction choice/KL gates and all three greedy checks, with maximum raw
logit NRMSE below 0.00005. These checks cover these prompts only.

‡ Both Orion backends pass all three tested HF token-choice/KL gates and
16-token greedy checks. Both fail the raw full-logit NRMSE threshold of 0.005;
FP16 stored weights and arithmetic differ from HF FP32. See the current Orion
reference below. These timings do not establish full-logit parity.

**MLX timings are provisional active-desktop measurements.** Brave's GPU
process was active at the start/end snapshots; the snapshots show process CPU
activity, not GPU utilization. FP16's two-token trial range was
91.81–119.19 steps/s. All timed MLX trials began and ended at nominal thermal
state. Owned repository checks were stopped before this final run; user
applications remained active. Earlier exploratory runs are excluded and kept
in the external cache, as recorded in [mlx-performance.json](mlx-performance.json).

This table compares implementations. Asahi uses stride 32 and 1024-token
context, with sequential prompt ingestion. CoreML uses a 64-wide input window and
512-token context; MLX uses one-token KV-cached decode and 1024-token context;
Orion uses stride 32 and 1024-token context. CoreML has FP16 weights with FP32
layer normalization; MLX rows use the stated unquantized parameter precision.
Orion stores FP16 weights and uses FP32 CPU arithmetic. Each engine returns
full-vocabulary logits; MLX timing includes Python graph construction and
synchronized evaluation of logits and KV state. Input assembly and argmax are
excluded from engine timings. CoreML and MLX also retain request timings that
include those operations. The current Orion, CoreML and MLX measurements all
use separate diagnostic replays. Sessions and precision order were not balanced
across runtimes; these are implementation measurements, with active desktop
applications during the new Orion session as well. The controlled same-artifact
CoreML configuration comparison remains in its own section below.

**vllm.cpp is unavailable for this GPT-2 comparison.** At local commit
`9157ba9c855c1b8a1c7240a2affa56c85da046b2`, `GPT2LMHeadModel` is absent from
the model-registration paths. Its GPT-2 code is a float32 host backbone
reference for IndexTTS, with full-sequence forward rather than the cached text
runner required here. It was inspected, not throughput-tested; see the source
hashes and evidence in [vllm-performance.json](vllm-performance.json).

[benchmark-comparison.json](benchmark-comparison.json) links the raw reports
and their hashes. [mlx-performance.json](mlx-performance.json) includes trial
rates, request throughput, TTFT, accuracy and source/runtime versions. MLX-LM
was cloned to `~/mlx-lm`; the benchmark reused the existing oMLX Python
environment without changing it and invokes MLX-LM directly. The GPT-2,
attention and cache source files hash-match that clone; the installed package
version is recorded separately. There is no HTTP server or prefix-cache reuse.

Reproduce either precision with a new output path:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /opt/homebrew/Cellar/omlx/0.7.0/libexec/bin/python \
  ~/ane/gpt2/tools/bench_mlx_gpt2.py --dtype float32 \
  --output /tmp/gpt2-mlx-fp32.json
```

Use `--dtype float16` for the FP16 row. The tool checks the pinned cached
checkpoint and cloned sources, acquires `~/gpu.lock`, and runs four trials per
prompt with two 16-step warmups before every timed trial. Full-logit checks
and greedy continuations run separately. Runtime dependencies and checkpoint
weights stay outside this portable directory.

## Orion macOS performance references

The primary Orion row now uses the [three-context measurement](provenance/orion-performance/m1-contexts.json),
with the exact same 2/32/64-token HF prompts and 64 teacher-forced decode
inputs as CoreML and MLX. Each backend has four trials per prompt; CPU/ANE
first position alternates evenly, with two fresh 16-step warmups immediately
before every timed trial. Logit checks occur in a separate 65-prediction replay
and each backend also has a 16-token free greedy check. The ANE program cache
is cleared between prompt cases, warmed before trials, and stays at 49 cached
programs per case. All 24 timed trials were thermally nominal and had **zero
timed compilations**.

All six backend/prompt cases have zero HF top1 mismatches, KL below 0.01 nats,
and matching tested greedy continuations. Raw logit NRMSE nevertheless reaches
0.054 on CPU and 0.379 on ANE, exceeding the separate 0.005 full-logit gate.
Both paths load the original FP16 weight blobs; CPU computes in FP32. These
checks cover these prompts only and do not establish full-logit parity.

The two-token cell was remeasured with the longer cases, rather than mixed
with the old 69.82/61.41 pair. This fresh checkout's captured source hashes
differ in five files from that older run; desktop applications were also
active. Differences between sessions cannot be attributed to the prompt length
or the new harness alone. [orion-context-performance.json](orion-context-performance.json)
records the source differences, trial rates, prefill and accuracy; the raw
report records thermal, memory and background-process snapshots.

Reproduce with existing external Orion/HF weights and a new output path:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/more-ane-transformers/.venv/bin/python \
  ~/ane/gpt2/tools/bench_orion_contexts.py \
  --output /tmp/orion-contexts.json
```

The runner builds fresh native objects, verifies all 196 external blob hashes
and the pinned HF checkpoint, checks prompt/token-trace equality with the saved
CoreML run, acquires `~/gpu.lock`, and refuses ANE fallback or timed compilation.
It leaves the external Orion checkout unchanged.

### Earlier two-token source snapshot (historical)

After prewarming both native Orion backends in one process, five alternating
64-step trials on this **base M1** measured:

| Prewarmed M1 run | Decode mean | Decode p50 | Decode p90 | Decode throughput |
| --- | --- | --- | --- | --- |
| Orion CPU | 14.32 ms/step | 14.13 ms/step | 14.74 ms/step | 69.82 steps/s |
| Orion ANE projection/FFN + CPU attention | 16.28 ms/step | 16.21 ms/step | 17.16 ms/step | 61.41 steps/s |

CPU throughput was about **14% higher**. Each backend was prewarmed with the
two-token prompt and 16 decode steps; all 49 ANE programs were cached before
measurement, and the compile counter stayed at 51 throughout the trials
(**zero timed compiles**, including prefill). Weight loading and warmup were
excluded. ANE warmup took 7.94 s in this run; this is recorded separately,
rather than subtracted from a cold average.

Both backends received the same 64-step CPU greedy token trace. Each timed
decode call includes embeddings, logits, CPU attention, and ANE transfers and
dispatch; prompt prefill, KV allocation, token selection, printing, and numerical
checks are outside that interval. This is native Orion decode throughput,
not end-to-end application throughput. Warm prefill mean was 29.75 ms on CPU
and 35.29 ms on ANE, measured separately.

All 320 ANE next-token argmax checks matched CPU, but full logits were not
equivalent: maximum normalized logit RMSE was **0.292**, above the harness's
0.005 diagnostic threshold (maximum absolute error 4.15). The untimed control
with CPU prompt prefill followed by eight ANE decode steps had maximum RMSE
0.00149. The full ANE-prefill path therefore needs further numerical
investigation; these timings do not establish full-logit parity or accuracy
across other prompts. The portable runtime ingests prompts through decode,
so this is also a different prefill path from the port.

The [full measurement](provenance/orion-performance/m1-prewarmed.json) includes
all samples, trial order, numerical diagnostics, excluded warmup times, host
information, source hashes, and build provenance; its
[log](provenance/orion-performance/m1-prewarmed.log) records cache reuse.
To reproduce on macOS with an external Orion checkout and matching weight
blobs (no weights are packaged here):

```sh
python3 ~/ane/gpt2/tools/bench_orion_macos.py \
  --orion ~/Desktop/Orion --output /tmp/orion-prewarmed.json
```

The output and adjacent `.log` must be new paths. The tool verifies all 196
weight hashes and builds fresh objects in a temporary directory. It stops on
ANE execution failure, unexpected I/O layout, or compilation during trials;
logit differences are retained as diagnostics.

The saved base-M1 Orion run used `Hello world` (2 prompt tokens), greedy
generation, and 16 generated tokens. CPU and ANE produced identical text;
the ANE run did not fall back to CPU.

| Saved M1 run | Prefill | Decode p50 | Decode p90 | Reported decode throughput |
| --- | --- | --- | --- | --- |
| Orion CPU | 52.7 ms | 16.2 ms/token | 20.4 ms/token | 58.7 tokens/s |
| Orion ANE projection/FFN + CPU attention | 5006.6 ms | 15.7 ms/token | 24.7 ms/token | 4.6 tokens/s |

Sources: the retained [CPU log](provenance/orion-performance/m1-cpu.log) and
[ANE log](provenance/orion-performance/m1-ane.log), from the dump's
`ablation/final-verification` run. The original two-token export log in
`provenance/original-inference.log` is retained for provenance, rather than used
as the corrected performance comparison.

The short-run average favors CPU, while the median decode latencies are close.
Orion reports throughput as `1000 * sample_count / sum(decode_ms)`. It excludes
prefill, token sampling, and printing, but includes lazy compilation inside
the first timed ANE decode step. The ANE prefill also includes 25 fresh program
compilations. That earlier run had no explicit warmup, so 4.6 tokens/s is a
cold short-run average. Use the separate prewarmed measurement above for
decode throughput with startup excluded.

For context, the local Orion checkout's `RESULTS.md` reports **M4 Max 64GB**
results: CPU **283 tokens/s, 3.5 ms/token p50**; ANE **170+ tokens/s,
5.78 ms/token**. Those are upstream-reported numbers for different hardware,
not measurements of this M1 port.

[orion-performance.json](orion-performance.json) records the values, log hashes,
output equality, timing boundaries, and hardware scope. These are Orion macOS
references. The Python port now also passes its decode and generation checks
on base M1 Asahi; its observed generation rates are recorded in the
[Asahi section](#measured-asahi-generation). No Asahi CPU comparison was run.

## Controlled GPT-2 CoreML benchmark

This is the primary **GPT-2 124M** CoreML comparison on this **M1 / 8 GB / macOS
27.0.1**. Every row uses the same pinned checkpoint, converted model artifact,
FP16 policy with FP32 layer normalization, 64-token input window, and 512-token
context. Batch size is one. Each prompt case uses the same 64 HF greedy decode
inputs across configurations. Prompt lengths are exactly 2, 32, and 64 tokens;
these are different starting contexts and are reported separately.

Only **one model configuration is loaded at a time**. Each timed trial receives
two 16-step warmups immediately beforehand and starts with an empty KV cache.
Four trial blocks use a balanced Latin square to vary configuration order.
Full-logit accuracy checks occur in a separate replay, outside timed trials.
Loading and compilation are excluded; loading durations are retained in the
raw report. No checkpoint download or reconversion was needed.

Decode engine throughput, **steps/s** (256 predictions per cell):

| CoreML configuration | 2-token prompt | 32-token prompt | 64-token prompt |
| --- | --- | --- | --- |
| CPU only | 65.21 | 64.83 | 62.52 |
| CPU + GPU | 51.41 | 51.32 | 51.82 |
| CPU + ANE | 49.66 | 49.18 | 50.51 |
| All devices allowed | 52.45 | 52.50 | 52.41 |

The engine timer covers synchronous `MLModel.prediction(from:)`. The native
request timer additionally covers input assembly, KV handoff, full-vocabulary
argmax, and autorelease cleanup. It uses the same fixed teacher-forced input
trace; it does not measure HTTP/UI serving or detokenization. For the
**64-token prompt**, the second boundary measures:

| Configuration | Native request decode, steps/s | Warm time to first token, ms |
| --- | --- | --- |
| CPU only | 54.59 | 17.98 |
| CPU + GPU | 45.39 | 22.52 |
| CPU + ANE | 45.39 | 22.09 |
| All devices allowed | 45.83 | 22.27 |

Prompt processing is measured independently. Mean first-prediction latency,
**ms** (four requests per cell):

| Configuration | 2-token prompt | 32-token prompt | 64-token prompt |
| --- | --- | --- | --- |
| CPU only | 15.28 | 15.24 | 15.63 |
| CPU + GPU | 19.59 | 19.58 | 19.72 |
| CPU + ANE | 20.20 | 19.98 | 19.80 |
| All devices allowed | 20.11 | 19.05 | 19.57 |

Actual prompt throughput is `actual_prompt_tokens * 1000 / mean_prefill_ms`,
recorded in the summary ledger. The graph still evaluates a fixed width of
64, so neither padded positions nor the decode rate should be presented as
actual prompt tokens/s. Warm time to first token starts before cache reset
and ends after the first native greedy token selection.

Recorded static compute plans prefer:

| Configuration | CPU operations | GPU operations | ANE operations |
| --- | --- | --- | --- |
| CPU only | 418 | 0 | 0 |
| CPU + GPU | 0 | 418 | 0 |
| CPU + ANE | 117 | 0 | 301 |
| All devices allowed | 0 | 418 | 0 |

These are preferred-device operation counts, not runtime utilization or proof
of concurrent execution. CoreML `ALL` permits all processors but its recorded
plan here prefers GPU exclusively. Its small timing difference from CPU + GPU
cannot be attributed to ANE; all three exploratory paired-trial intervals
include a throughput ratio of one. CPU only was fastest on these workloads.
This compares CoreML configurations on the same graph, not optimized hardware
peaks. Orion's different graph, cache contract, and arithmetic remain a
separate implementation reference above.

All twelve prompt/configuration cases passed the predefined trace gate:
finite logits, zero top1 mismatches across the prompt plus 64 decode checks,
and maximum KL(HF || CoreML) no greater than 0.01 nats. All twelve free greedy
16-token continuations also matched HF. These checks cover three prompts and
do not establish full-logit parity or general task accuracy.

Every timed trial began and ended at nominal thermal state. A preliminary
four-resident-model run increased swap use and produced prefill outliers;
its timings are excluded from the primary comparison and retained externally
in `~/.cache/coreml-gpt2/fair-resident-pilot`. The single-resident run had no
increase in the system-wide swapout counter; swapins occurred during the
session, including model loading, so zero paging within trials is not claimed.
Four blocks in one session support an exploratory comparison. Separate
sessions are needed before treating small differences as stable speedups.

See the [protocol](benchmark-protocol.json),
[summary ledger](coreml-fair-performance.json),
[full measurements](provenance/coreml-performance/m1-fair.json), and
[run log](provenance/coreml-performance/m1-fair.log). The report includes raw
samples, warm request metrics for all prompt lengths, trial order, quality
checks, model/source hashes, and paired whole-trial bootstrap intervals.
Device plans were captured in the first session on this Mac with the same
artifact hashes. A subsequent fresh plan query failed inside CoreML while
reading a missing `manifest.plist`; the corrected timing run reused those
verified plans. The portable [plan snapshot](provenance/coreml-performance/m1-compute-plans.json)
contains the placement metadata and model hashes, without the excluded pilot's
performance numbers.

Reproduce on this Mac using the existing isolated environment and converted
model:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/more-ane-transformers/.venv/bin/python ~/ane/gpt2/tools/bench_coreml_fair.py \
  --model ~/.cache/coreml-gpt2/gpt2-more-ane-cmt9.mlpackage \
  --plan-report ~/ane/gpt2/provenance/coreml-performance/m1-compute-plans.json \
  --output /tmp/gpt2-coreml-fair.json
```

The output and adjacent log must be new paths. Cached plans are specific to
this Mac and artifact; omit `--plan-report` to query plans on another machine.
`--trials 8` repeats the balanced order twice. Model conversion and environment
setup are described in the historical section below. CoreML's processor
permissions are documented in the [official API reference](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel).

## Earlier CoreML macOS reference (historical)

These earlier timings are retained for provenance; use the controlled
comparison above for the current benchmark numbers.

The tested CoreML implementation is
[more-ane-transformers](https://github.com/smpanaro/more-ane-transformers),
cloned to `~/more-ane-transformers` at commit
`14ef84fab12b8df11fd53fa0ba261ad451f91b89`. This is **GPT-2 124M**, freshly
converted with coremltools 9 from the same pinned Hugging Face checkpoint used
by this port. It uses the upstream FP16 policy with FP32 layer normalization.
On this **Apple M1, 8 GB, macOS 27.0.1**, the native Swift harness measured:

| CoreML backend | Decode mean | Decode p50 | Decode p90 | Decode throughput |
| --- | --- | --- | --- | --- |
| CPU + ANE (`CPU_AND_NE`) | 20.54 ms/step | 19.66 ms/step | 23.22 ms/step | **48.68 steps/s** |
| CPU only (`CPU_ONLY`) | 18.06 ms/step | 14.90 ms/step | 24.82 ms/step | **55.37 steps/s** |

Each backend received two 16-step warmup generations, followed by five
alternating 64-step trials using `Hello world` (two prompt tokens) and the
same HF greedy token trace as the prewarmed Orion report above. All trials
reported nominal thermal state. Throughput is `1000 * steps / sum(decode_ms)`.
The interval covers synchronous `MLModel.prediction(from:)`, including the
embedding, attention, FFN, output head, and runtime dispatch. Input assembly,
token selection, printing, numerical checks, loading, compilation, and warmup
are excluded. The native runner passes the prior prediction's KV-cache
`MLMultiArray` directly to the next prediction.

GPU execution is disabled for both configurations. `MLComputePlan` assigns
301 nonconstant operations to ANE and 117 to CPU in the CPU + ANE configuration;
these are static preferred-device counts, not runtime percentages. This model
uses a fixed 64-token input window and a 512-token context, whereas Orion uses
a different graph and a 1024-token context. These results measure the two
implementations on this Mac; they do not isolate hardware speed or establish
Asahi performance. Warm prompt prediction averaged 43.74 ms on CPU + ANE and
20.39 ms on CPU only; this excludes application startup and is not end-to-end
time to first token.

All 320 timed next-token choices per backend matched HF on the main trace,
and both generated the same 16-token `Hello world` continuation as HF.
Full logits differ: maximum normalized logit RMSE was 0.406 on CPU + ANE and
0.756 on CPU, with maximum KL(HF || CoreML) of 0.00618 and 0.00598 nats.
The capital-of-France prompt also matched the 16-token HF continuation, but
the Mars packing prompt had 3/17 teacher-forced next-token mismatches on both
backends and free generation diverged after two matching tokens. Successful
generation and the throughput measurement therefore do not establish full
logit parity or general accuracy equivalence.

The [summary ledger](coreml-performance.json) and
[full report](provenance/coreml-performance/m1-cmt9.json) retain raw samples,
numerical checks, generation output, device placement, model and source hashes,
and environment versions. The [native log](provenance/coreml-performance/m1-cmt9.log)
records the measured run. Converted weights and compiled models remain outside
this directory in `~/.cache/coreml-gpt2`.

To reproduce with the already cloned repository, create its isolated
environment if needed:

```sh
uv venv --python 3.11 ~/more-ane-transformers/.venv
uv pip install --python ~/more-ane-transformers/.venv/bin/python \
  coremltools==9.0 numpy==1.26.4 torch==2.5.1 transformers==4.44.2 \
  safetensors==0.8.0 regex stopwatch.py==2.0.1 os-signpost==0.0.3
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/more-ane-transformers/.venv/bin/python ~/ane/gpt2/tools/bench_coreml_macos.py \
  --repo ~/more-ane-transformers --convert \
  --model ~/.cache/coreml-gpt2/gpt2-reproduction.mlpackage \
  --output /tmp/coreml-gpt2-reproduction.json
```

The cached checkpoint must match revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; `--weights PATH` selects it explicitly.
The output and adjacent `.log` must be new paths. For an existing converted
model, omit `--convert` and pass its `.mlpackage` path instead.

The stock upstream Python CLI is incompatible with coremltools 9's private
model-proxy constructor. [run_more_ane_cli.py](tools/run_more_ane_cli.py) adapts
that call to the public compiled-model API and uses the cached tokenizer,
without editing the cloned repository. Its successful
[16-token CLI smoke run](provenance/coreml-performance/upstream-cli-adapted.log)
reported 31.20 tokens/s, including Python cache copies and printing, with no
explicit warmup. This single smoke result has a different timing boundary
from the native table. The CLI's reported prompt rate divides by 64 padded
positions rather than the two actual prompt tokens and is not a valid prompt
throughput reference.

Also cloned to `~/ane-llm-measurements`:
[ane-llm-measurements](https://github.com/shershah1024/ane-llm-measurements),
commit `d1cb2f4d0e936c23a67283b732b28e1e899926e0`, for its CoreML placement
methodology. It supplies no GPT-2 checkpoint. Its optional operator smoke
test built successfully but exited with signal 11 before producing results;
no benchmark number is attributed to it. A slow optional download of the old
more-ane-transformers release model was stopped; the numbers above use the
successful fresh conversion.

## Replay layout and audit

- `model-checksums.json`: hashes of Orion's 196 original fp16 tensors. The
  matching HF weights are loaded externally and rounded to fp16 in memory;
  no weight blobs are copied into this directory.
- `tokenizer/`: GPT-2's 50257-token vocabulary and all 50000 merge rules.
- `kernels/`: 49 MIL programs, buffer/I/O metadata, and compressed strict
  register reports covering 1574 tasks and 187848 register values.
- `objects/`: content-addressed task/constant templates with learned parameters
  zeroed. No compiled learned coefficient banks are bundled.
- `packing.py` and `kernels/*/meta.json`: portable fp16 tile-packing operations,
  folded affine operations, fixed activation LUTs, and expected payload hashes.
- `packing-validation.json`: byte-exact reconstruction results for all 147
  program/constant/coefficient payloads across the 49 kernels.
- `fixtures/`: input/output captures for all 49 macOS ANE kernels, independent
  Orion CPU logits, and a complete hybrid generation reference.
- `provenance/`: original dump manifest and inference log.
- `checksums.json`: checksums of portable source and data files.

362 packed MIL parameter tensors were compared byte-for-byte with the Orion
CPU weights. All 196 tensors derived from the cached HF checkpoint were also
checked against the original Orion fp16 blobs. The HF checkpoint SHA256 is
pinned, so a different model cannot silently be combined with this dump.

The packing flow is HF float32 → Orion-equivalent fp16 tensors → captured
H13G tile layout. For each of 16 engines, each output tile contains its fp16
bias followed by the matrix transposed to `[input, tile_output]`; each engine's
payload is aligned to 64 bytes. Decode Q/K/V and output projections use three
16-channel tiles per engine; the FFN expansion uses six 32-channel tiles.
Prefill Q uses a mixed 32/16-channel schedule. Recipes preserve the compiler's
matrix order, offsets, LUTs, and final 16 KiB segment padding.

The compiler also folds layer norm into `fp16(beta/gamma)` and `fp16(gamma)`.
Most kernels store these arrays linearly in `__TEXT.__const`. FFN layers 6 and
10 instead store engine-interleaved `(gamma, beta/gamma * scale)` pairs in the
coefficient bank, with scales 32 and 2 respectively. The division/scaling is
done in float32 from fp16-rounded inputs before final fp16 rounding.

All 49 coefficient banks, constants, and relocated programs were reconstructed
from the cached HF checkpoint and compared byte-for-byte against the original
HWX dump. Only model-independent activation tables remain as literal data;
unexplained coefficient bytes cause recipe derivation to fail. Runtime checks
the corresponding full-payload hashes. This implements the captured GPT-2/M1
layouts; it is not a general ANE compiler or a packer for arbitrary checkpoints.

Orion itself assembles BLOBFILEs and relocates their offsets in
`core/mil_builder.m`; `core/ane_runtime.m` passes them to Apple's ANE compiler
through `compileWithQoS`. The hardware coefficient layout was handled by
macOS, and is reconstructed here for Asahi.

The upstream implementations were checked as well. oMLX's
[ANE backend](https://github.com/jundot/omlx/blob/5dcfe2430b73a86e871de13019c93e047f0aba9a/omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm#L295-L323)
prepares fp16/int8 BLOBFILE inputs and invokes Apple's `compileWithQoS`.
[ane-ex](https://github.com/eiln/ane-ex/blob/21bc510b2bc88eb41f8f9c17177d7a5ac76df682/c/Makefile)
invokes `anecc` on existing HWX files; its
[source instructions](https://github.com/eiln/ane-ex/blob/21bc510b2bc88eb41f8f9c17177d7a5ac76df682/sources.md)
start from CoreML conversion. Neither source tree supplies a raw-HF-to-H13 tile
packer that could replace the reconstruction above.

The existing `experimental/hwx2py.py` is not used. It hardcodes a single
628-byte task, misses this dump's compiled coefficient segment, and does not
handle the 288 extended task headers. This port uses the H13 load-command BAR
table and actual first-task size (504 bytes), follows every NextPtr/NextSize,
and preserves register packets and dependency bits.

The Linux driver synthesizes BAR 1 at `align16(tsk_size)`. The command buffer
therefore contains the complete original `__TEXT` followed by `__KERN_0`.
The compiler's constant BAR 1 moves to free BAR 2; its kernel BAR moves to 1.
Only enabled BAR selectors in task header words 8 and 9 change. Inputs,
outputs, and scratch buffers retain the compiler's indices, sizes, and names;
Q/K/V are selected by name. Bootstrap NID is set to 0x40. Buffers and handles
are freed on exit, including allocation failure paths.

The ABI is based on
[allbilly/libane `ane_accel.h`](https://github.com/allbilly/libane/blob/1e0afd832cf171be543d18069cef726aae2b9634/ane/src/uapi/drm/ane_accel.h)
and its command/bootstrap setup. The tokenizer follows
[OpenAI's GPT-2 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py),
including valid hash-character merges that must not be discarded as comments.

## Verification and rebuilding

```sh
~/ane/gpt2/.venv/bin/python -m unittest discover -s ~/ane/gpt2/tests -v
```

Tests cover checksum corruption, every task's reversible bank relocation,
truncated chains, shared coefficients, independent CPU logits, generation,
Unicode/BPE, context bounds, seeded sampling, the driver ABI, command/weight
placement, multi-output buffers, bootstrap headers, and cleanup. They also
reconstruct all 147 payloads from external weights, verify mixed tile placement,
reject wrong tensors, recover corrupt generated caches, and check that no
compiled learned coefficient bank remains in the portable directory. Mocked DRM
tests validate requests and lifecycle; they do not substitute for Linux
hardware verification.

To regenerate data from the original sources:

```sh
python3 ~/ane/gpt2/tools/prepare.py \
  --dump ~/Desktop/GPT2-ANE-Dump --orion ~/Desktop/Orion
```

This changes packaged data, so recreate the macOS fixtures if source weights
or MIL changed, then run `tools/seal.py` to update checksums and rerun tests.
`tools/capture_macos.m` captures each kernel through Orion's runtime;
`tools/capture_cpu.m` records independent CPU logits; `tools/macos_bridge.m`
and `tools/capture_hybrid.py` check the full Python flow with macOS ANE kernels.
These tools are only for rebuilding reference data, not Linux execution.

Orion-derived logic is covered by `LICENSE-Orion`. The bundled GPT-2 tokenizer
resources retain OpenAI's license in `LICENSE-GPT2`. Standard model weights are
loaded from the external HF cache; original GPT-2 comes from
[OpenAI](https://github.com/openai/gpt-2).
