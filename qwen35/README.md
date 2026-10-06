# Mirai Qwen3.5 on Asahi

The loader and packed CPU kernels support the exact
[`trymirai/Qwen3.5-0.8B-M`](https://huggingface.co/trymirai/Qwen3.5-0.8B-M)
checkpoint at revision `c12202e4c764e559960827761566aaa1fd15a87a`.
Model weights remain outside the repository in `~/.cache/ane-qwen35/model/`.
`setup` downloads and verifies all three pinned files. Select another local
directory with `--model` when it contains this same checkpoint.

```sh
./qwen35/first-run.sh setup
./qwen35/first-run.sh inspect --check-hashes
./qwen35/first-run.sh verify
./qwen35/first-run.sh generate --prompt 'What is 2 + 2?' --max-tokens 32
./qwen35/first-run.sh generate --precision bf16 --prompt 'Hello'
./qwen35/first-run.sh bench --raw --prompt 'Hello' --max-tokens 32 --output /tmp/qwen35.json
./qwen35/.venv/bin/python -m unittest qwen35.tests -v
```

CPU execution also works on macOS. Use a compiler with OpenMP support; Apple's
default `cc` rejects `-fopenmp`. On the tested M1, Homebrew GCC 16 works:

```sh
env CC=gcc-16 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  ./qwen35/first-run.sh generate --threads 4 --prompt 'Hello'
env CC=gcc-16 ./qwen35/.venv/bin/python -m unittest qwen35.tests -v
```

Use `--model` for an existing copy of the pinned checkpoint. Select worker
counts explicitly for macOS comparisons; Linux CPU affinity selection is
unavailable there.

Mirai M uses asymmetric 4-bit weights, groups of 32, BF16 scales, packed
4-bit zero points and signed 32-element Hadamard transforms. This differs
from GGUF Q4_0 and Mirai S's trellis codec. Body matrices store scales and
zero points in group/output order; embeddings use output/group order.
The loader retains the original packed weights and normalizes only those
small parameter tables. Native matvec accumulates in FP32. On M1 the default
FP32 mode uses SDOT with two signed byte activation planes, retaining 16-bit
precision per group. It subtracts each group's zero point before the integer
dot product, keeping the original scales and packed weights.
`--kernels native` selects floating packed matvec. DeltaNet's convolution
and recurrent states use FP32. `--precision bf16` reproduces the vendor's
BF16 activation boundaries and sequential accumulation; this slower
reference mode matched Uzu exactly on the retained Asahi fixtures. The
[retained Asahi arithmetic fixture differs on macOS](#macos-cpu-measurements),
while a fresh independent macOS Uzu capture matches the macOS implementation exactly.
FP32 is the default. Four workers use M1 performance cores within the
process's allowed affinity; `--threads` and explicit OpenMP settings can override this.
Builds require a C compiler with OpenMP. `first-run.sh` installs the Python
requirements into `qwen35/.venv`.

There are 18 recurrent layers and six attention layers, width 1024 and
FFN width 3584. Attention uses eight query heads, two KV heads and partial
64-dimensional RoPE. The tokenizer and chat template come from the exact
checkpoint; generation currently uses greedy selection. Prompt ingestion
is sequential, and earlier prompt tokens skip the vocabulary projection.

The independent NumPy checks cover distinct group/output parameter values,
both sides of the transform, tied embedding readout, and an explicit
recurrent-state update. They do not establish full-model parity. An optional
Uzu CPU oracle can be built with `tools/build_reference.py`; its instrumentation
exports decoder logits without changing the decoder or its kernels.
The retained Uzu fixtures cover all 248,320 logits after one token, two
tokens and a 23-token chat prompt. All matched bit for bit in BF16 mode on Asahi;
the first two fixtures also matched all 24 residual layer outputs.
`verify` replays these fixtures and checks a rolling FP32 integer-dot trace.
[Vendor validation](provenance/vendor-validation.json) and
[source revisions and hashes](provenance/sources.json) are retained.

## Measured performance

The 2026-10-06 Asahi follow-up checked macOS commits `b40306e` and
`fd0831e`. All 70 capture, Qwen and GPT-2/GGUF unit tests passed. Floating
CPU references prepared locally from the pinned Hugging Face checkpoint
matched all six first predictions and 384 saved-input decode choices through
SDOT, including 1,024- and 2,048-token prompts. Maximum full-logit NRMSE was
0.0432%. Four independently evolving greedy histories also matched all
256 choices, with maximum NRMSE 0.0122%.

The rebuilt pinned Uzu CPU oracle also matched the native BF16 implementation
bit for bit at all 23 arithmetic prefixes: 5,711,360 vocabulary logits and
552 residual layer vectors (565,248 values). The original one-token,
two-token and arithmetic fixtures still matched, and all-prefix
instrumentation preserved the final logits. This establishes same-host
Asahi parity; it does not compare against missing macOS all-prefix arrays.

The accurate ANE backend also passed all six first predictions and 384 decode
choices, with maximum full-logit NRMSE 0.3813%, below the unchanged 0.5% gate.
It made 339,552 timed submissions for 3,153 prompt tokens and 384 decode calls:
all 96 body projections ran on ANE for every input. The separate dimension,
Mirai projection and extreme-value regression passed all 35 submissions.

| Prompt | CPU SDOT prefill tok/s | ANE prefill tok/s | CPU SDOT decode tok/s | ANE decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Chat, 23 tokens | 21.35 | 13.49 | 17.54 | 12.96 |
| Chat, 19 tokens | 20.99 | 13.28 | 17.40 | 12.95 |
| Chat, 21 tokens | 21.23 | 13.45 | 15.87 | 12.80 |
| Chat, 18 tokens | 20.59 | 13.06 | 15.95 | 12.26 |
| 1,024 tokens | 20.24 | 12.94 | 16.12 | 11.43 |
| 2,048 tokens | 19.33 | 12.10 | 15.04 | 10.91 |
| Aggregate | 19.66 | 12.39 | 16.28 | 12.17 |

Both paths use the pinned Mirai M asymmetric W4 checkpoint. CPU SDOT reads
packed W4 matrices; ANE expands body weights to resident FP16 and uses the
accurate compensation policy, with its head and recurrence on the CPU.
These are single active-desktop runs at different times, with 64 decode calls
per prompt. The ANE run held all three shared ANE/GPU locks; the earlier CPU
run did not. Clocks were not fixed, and continuous host-load and swapout
accounting were not retained. These timings do not establish an isolated
speed comparison. The initial sandbox rejection and a subsequent lock timeout
remain in the receipts; both occurred before hardware submission.
[Follow-up receipts and artifact hashes](provenance/m1-asahi-macos-update.json)
retain completed CPU and ANE checks, inputs and result hashes.

Recurrence inputs and constants can now be prepared on Linux without an
Apple compiler or copied macOS weights. Choose fresh output directories:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  qwen35/.venv/bin/python -m qwen35.tools.prepare_recurrence \
  --output qwen35/local-results/recurrence-inputs
env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  qwen35/.venv/bin/python -m qwen35.tools.check_long_context \
  --model ~/.cache/ane-qwen35/model --greedy-only \
  --output qwen35/local-results/independent-greedy
```

The recurrence tool captures layer zero at positions 0, 1, 7 and 22 using
floating packed W4 projections. It emits the reusable `native-inputs.npz`,
dense FP16 semantic I/O for four groups of four heads, FP16 constants and
independent native FP32 references. All 32 component checks passed after
rounding the MIL CPU result to FP16: maximum state/output NRMSE was
0.0478%/0.0846%. The compiled program's port names and coefficient layout
must still be mapped before dispatch. The committed macOS receipt has
hashes and port metadata; the new recurrence command bytes are not embedded
in it. The tool's CPU preparation does not compile or execute an ANE program.
The existing hybrid decoder already repacks all 96 body projections through
`linear_template.h` and keeps recurrence on the CPU. Running that decoder
needs no additional kernel capture. Mapping a fused recurrence program is a
separate optimization and does not block the current inference path.
`QWEN35_CACHE_DIR` now selects an alternate build cache for both CPU and ANE
libraries.

The matched benchmark now measures the complete prompt, including its last
token and the vocabulary head, as prefill. Warm TTFT also includes greedy
selection of the first output token. Decode starts by feeding that generated
token and measures 32 further model calls. Model loading, tokenization, state
reset and full-logit comparisons stay outside both timers. Prefill and decode
component timers and ANE submission counts are reported separately.

```sh
# Use a new reference directory; the benchmark refuses to overwrite one.
env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  qwen35/.venv/bin/python -m qwen35.tools.benchmark \
  --prepare-traces --traces ~/.cache/ane-qwen35/prefill-decode-traces
env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  qwen35/.venv/bin/python -m qwen35.tools.benchmark \
  --kernels dot --backend cpu --traces ~/.cache/ane-qwen35/prefill-decode-traces \
  --output /tmp/qwen35-prefill-decode.json
```

The default suite has four short chats and two longer chats cropped to exactly
128 and 512 input tokens. Each path replays the floating CPU reference's saved
tokens, continues beyond EOS for a fixed number of steps, and checks every
logit outside the timers. Use `--backend ane --kernels dot` for the experimental
ANE body; coordinate the shared hardware locks for both CPU and ANE runs.
The `bench` CLI also reports prefill tok/s, warm TTFT and decode tok/s; its
`--max-tokens 32` includes the first prediction from prefill, so it times 31
decode calls.

The corrected ANE policy was compared with the legacy policy in three
paired fresh-process runs on 2026-10-05. Both replayed the same six prompts:
2,163 timed prefill tokens and 576 timed generated-input decode calls per
policy. Each check compares all 248,320 logits against the retained floating
CPU reference. The 192 distinct decode inputs were replayed three times.

| ANE policy | Prefill tok/s | Decode tok/s | First choices | Decode choices | Maximum logit NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy, unscaled | 18.86 | 17.38 | 18/18 | 570/576 | 17.294% |
| Accurate, default | **12.37** | **11.57** | **18/18** | **576/576** | **0.381%** |

The corrected policy passes the unchanged 0.5% full-logit gate and every
argmax check. Maximum NRMSE is 45.4 times lower. The throughput cost is 34.4%
for prefill and 33.4% for decode in this paired capture. It uses the same
coefficient memory and 96 ANE submissions per input: 262,944 timed
submissions per policy. Active desktop activity and paging still limit
isolated speed claims. The ANE lock remained reserved throughout the
capture, with both shared GPU locks held per workload. Only the desktop
load gate was overridden after a free-lock preflight.

[Accuracy diagnosis, hardware regressions, repeated timings and capture hashes](provenance/m1-ane-accuracy.json)
retain the unsuccessful precision experiments as well as the selected
configuration. Coreglass recorded 4,429 samples over 443.3 seconds; the
paired-capture warning excerpts contained only network rate-limit messages.
The [local ANE accuracy report](http://127.0.0.1:8777/out/qwen35-ane-accuracy/index.html)
uses this capture. These timing receipts are pinned to implementation commit
`de3c0b6`; a later extreme-value restoration guard is validated separately
in the same provenance file.

The earlier, unscaled 2026-10-05 run on base M1, 8 GB, Asahi Linux used
three fresh processes per path and 32 teacher-forced decode calls per prompt. Each path ingested
2,163 timed prompt tokens and made 576 timed decode calls. Rates below are
aggregate token counts divided by aggregate engine time; warm TTFT is the
mean per request. All paths used four performance-core workers, one OpenBLAS
thread and passive OpenMP waiting.

| Path | Prompt tokens | Prefill tok/s | Warm TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| CPU floating packed W4 | 18–23 | 16.81 | 1.205 | 12.60 |
| CPU floating packed W4 | 128 | 11.73 | 10.908 | 7.52 |
| CPU floating packed W4 | 512 | 10.51 | 48.723 | 7.56 |
| CPU SDOT packed W4 (default) | 18–23 | 18.33 | 1.105 | 13.84 |
| CPU SDOT packed W4 (default) | 128 | 17.57 | 7.283 | 13.48 |
| CPU SDOT packed W4 (default) | 512 | 11.97 | 42.757 | 10.83 |
| ANE FP16 body + CPU SDOT head (legacy, unscaled) | 18–23 | 19.51 | 1.038 | 18.22 |
| ANE FP16 body + CPU SDOT head (legacy, unscaled) | 128 | 19.53 | 6.554 | 17.95 |
| ANE FP16 body + CPU SDOT head (legacy, unscaled) | 512 | 19.23 | 26.623 | 17.53 |

Across all six prompts, CPU floating, CPU SDOT and ANE measured respectively
**10.30, 13.17 and 18.06 decode tok/s**. Every CPU prediction matched the
floating reference: 18 first predictions and 576 decode predictions per
path. CPU SDOT's maximum full-logit NRMSE was 0.0433%. ANE matched all 18
first predictions and 570/576 decode predictions, with maximum full-logit
NRMSE 17.30%; that legacy path failed the accuracy gate. The corrected default
ANE policy passes the retained suite above. ANE counters confirmed all 96
projections for every timed input: 262,944 submissions across its three runs.

These are active-desktop measurements with substantial CPU variation and
observed paging. Individual prompt decode rates ranged 4.39–14.55 tok/s
for floating CPU, 6.20–18.51 for CPU SDOT and 17.11–18.91 for ANE. Shared
hardware locks serialized jobs, but setup from queued jobs and desktop
activity were present. Two 60-second lock timeouts never reached inference;
their attempts were retained and the missing paths were rerun successfully.
The retry capture held the ANE reservation throughout both workloads.
These conditions do not establish an isolated kernel speed comparison.

[Full prefill/decode receipts, saved inputs, component timers and numerical checks](provenance/m1-prefill-decode.json)
include both Coreglass capture hashes and the failed lock attempts. The two
captures retained 9,628 samples over 964.7 seconds. Whole marked process
counter averages include setup, comparisons and lock waits; engine timers
exclude them. The [local prefill/decode report](http://127.0.0.1:8777/out/qwen35-prefill-decode/index.html)
uses only this workload and these captures. Prompt ingestion remains
sequential; batching prefill is still an optimization opportunity.

The earlier decode-only measurements below used the previous timing contract:
their first timed step consumed the last prompt token. Retained receipts are
unchanged; new measurements use the boundary described above.

Base M1, 8 GB, Asahi Linux: three rotated fresh processes per path, four
actual chat prompts per process and 32 saved input tokens per prompt.
Each path received 384 timed teacher-forced decode steps. All used four
performance-core workers, one OpenBLAS thread and passive OpenMP waiting.
Loading, packing, tokenization, prompt ingestion and logit comparison were
excluded from the timers. Traces continue beyond EOS to measure a fixed
number of steps; these rates are not complete application throughput.

| Path | Aggregate steps/s | Prompt rate range | Argmax matches | Maximum logit NRMSE |
| --- | ---: | ---: | ---: | ---: |
| CPU floating packed W4 | 18.56 | 15.17–20.87 | 384/384 | Reference |
| CPU SDOT packed W4 (default) | **28.78** | 21.89–33.43 | **384/384** | **0.0433%** |
| ANE FP16 body + CPU SDOT head (legacy, unscaled) | 19.34 | 19.05–19.44 | 378/384 | 17.30% |

The SDOT path completed these traces 55.1% faster than floating packed W4;
the vocabulary projection fell from 12.66 to 5.34 ms/step. This was an active
desktop run and CPU rates varied considerably, so it does not establish
isolated sustained throughput. Coreglass retained 1,068 samples over 106.9
seconds. Its load gate was overridden after confirming the shared locks
were free. GPU/ANE engine busy counters, DRAM bandwidth and process-specific
counters were unavailable. The per-phase journal excerpts contained network
rate-limit messages; no ANE warning or error appeared in those excerpts.
[All samples, component timers, numerical checks and capture hashes](provenance/m1-performance.json)
are retained. Raw capture and run manifest stay in the user's Coreglass data
directory. The [local report](http://127.0.0.1:8777/out/qwen35-mirai/index.html)
combines the measured benchmark data and that capture.

## macOS CPU measurements

The same base M1 / 8 GB machine was measured on macOS 27.0.1 on 2026-10-06,
using GCC 16, one BLAS thread and passive OpenMP waiting. Three rotated
fresh-process runs per configuration used the same six prompts and 32 decode
calls as the Linux prefill/decode suite: 2,163 prefill tokens and 576 decode
calls per configuration. Regenerated floating references reproduced every
saved Linux prompt token, generated input and prediction.

| CPU kernels | Workers | Prefill tok/s | Decode tok/s | Per-prompt decode range |
| --- | ---: | ---: | ---: | ---: |
| Floating packed W4 | 1 | 7.73 | 5.82 | 3.22–7.70 |
| Floating packed W4 | 2 | 7.70 | 5.49 | 2.77–11.35 |
| Floating packed W4 | 4 | 9.10 | 6.66 | 3.03–14.40 |
| SDOT packed W4 | 1 | 16.18 | 12.66 | 8.69–17.11 |
| SDOT packed W4 | 2 | 9.83 | 5.14 | 0.78–20.70 |
| SDOT packed W4 | 4 | 15.01 | 11.78 | 6.42–19.41 |

Every configuration passed all 18 first choices and 576 decode choices.
Floating outputs matched the local reference exactly; SDOT's maximum full-logit
NRMSE was 0.0582%, below the unchanged 0.5% gate. The separate 16-step rolling
SDOT verification also passed. All eight small-matrix/recurrent tests passed.

These are active-desktop measurements. Another session was assigned Metal/ANE
work, shared workload isolation was not established, and the sampled one-minute
host load ranged from 1.90 to 168.88. The two-worker SDOT configuration's second
run was especially affected. These aggregate rates do not establish a best
worker count or an isolated macOS-versus-Linux speed difference.

Strict BF16 parity against the retained Asahi fixtures fails on macOS: the one- and
two-token fixtures, including all 24 layer outputs, matched exactly, while the
23-token arithmetic fixture had 1.5815% logit NRMSE and maximum absolute error
0.1875. Its argmax still matched. GCC and Clang produced identical macOS
outputs and layer traces, so switching these compilers did not resolve the
fixture difference. A fresh build of pinned Uzu on macOS, with all-token
instrumentation, matched native BF16 bit for bit at all 23 arithmetic tokens:
552 residual-layer vectors and 5,711,360 logits. The instrumented final logits
also matched the uninstrumented final-token capture exactly. macOS Uzu itself
has the same 1.5815% difference from the retained Asahi arithmetic logits.
This establishes same-host vendor parity and narrows the unresolved question
to cross-host reference reproducibility; it does not establish the underlying
cause. The original fixtures and exact gate remain unchanged.

[The macOS CPU receipt](provenance/m1-macos-cpu.json) records all configurations,
failed checks, successful reruns, host observations and the evidence archive's
SHA256. Raw timings, reference logits, compiler diagnostics and layer dumps
remain under the ignored `local-results/` directory. GPT-2's accompanying
macOS checks passed all 49 tests, packed all 49 kernels/110 unique payloads,
and passed NumPy, exact floating and native CPU parity. Its four-trial warmed
CPU decode measurement ranged from 108.07 to 115.03 steps/s across three prompts.

### Follow-up validation

Review on 2026-10-06 rechecked all 97 baseline artifact hashes, the archive
SHA256 and all 18 baseline benchmark receipt hashes. The subsequent
[follow-up receipt](provenance/m1-macos-followup.json) records the expanded
checks and [capture commands](../experimental/macos-followups.md).
The subsequent [unrestricted rerun](provenance/m1-macos-unrestricted.json)
provides actual macOS ANE training and Whisper fixtures, successful Metal
transcriptions and Core ML hardware placement. Its repeated CPU sweep uses
complete host observations again. Earlier restricted attempts remain retained.

1. **Compare the two hosts' BF16 oracle captures.** The macOS capture is
   complete and matches native BF16 exactly at every arithmetic prefix and
   layer. Run `tools/capture_vendor_macos.py` on Asahi with `--tag asahi` and
   `--compare-capture` pointing to the saved `uzu-macos-all.npz`; the helper
   reports the first differing prefix and layer between independent oracles.
   `tools/build_reference.py --lockfile` accepts the retained Cargo lock so
   the cross-host build can use the same pinned dependency resolution.
   Capture operation inputs and recurrent state around that first cross-host
   mismatch if layer outputs alone do not explain it. Keep the bitwise gate.
2. **Repeated CPU correctness passed; quiet timing remains limited.**
   Three additional floating/SDOT, 1/2/4-worker sweeps passed all 54 numerical
   runs: 324 first predictions and 10,368 decode choices matched. Thirteen runs
   passed the predeclared host-activity gates and 41 timings were affected.
   All 18 second-phase runs lacked process/swap queries under the workspace
   sandbox and were marked affected. The unrestricted third phase passed all
   18 runs with complete observations: 108 first predictions and 3,456 decode
   choices matched; five timings passed the host gates and 13 were affected.
   All attempts are retained. Desktop activity and repeated phases on one
   boot do not establish an optimum worker count or an isolated OS comparison.
3. **Longer-context and free greedy CPU checks passed.** Four short prompts
   and exact 1,024/2,048-token prompts each completed 64 decode calls: all
   six first predictions and all 384 decode choices matched the floating
   reference. Maximum full-logit NRMSE was 0.058167%, below the unchanged
   0.5% gate. Four separate 64-step greedy streams covering arithmetic,
   Python, Chinese and household advice matched all 256 choices, with
   maximum NRMSE 0.010210%. The fixed calls continue beyond EOS and do not
   measure response quality or application throughput. The saved floating
   traces and `long-context/asahi-replay.json` are ready for the accurate
   ANE policy on Asahi; Linux hardware parity remains pending. These checks
   use the existing projection shapes and do not resolve the cross-host
   BF16 oracle difference.

The unrestricted repeat retained three runs per configuration. The rates
below aggregate every attempt, including affected timings; the final column
counts runs passing the host-activity gates. Each configuration covers 2,163
timed prefill tokens and 576 decode calls. Maximum SDOT full-logit NRMSE
remained 0.058167%, below the unchanged 0.5% gate.

| CPU path | Workers | Prefill tok/s | Decode tok/s | Host gates passed |
| --- | ---: | ---: | ---: | ---: |
| Floating packed W4 | 1 | 10.51 | 7.55 | 1/3 |
| SDOT packed W4 | 1 | 20.69 | 16.13 | 0/3 |
| Floating packed W4 | 2 | 15.26 | 11.83 | 1/3 |
| SDOT packed W4 | 2 | 26.66 | 22.50 | 1/3 |
| Floating packed W4 | 4 | 20.00 | 16.95 | 0/3 |
| SDOT packed W4 | 4 | 29.62 | 26.18 | 2/3 |

These historical host-gate counts use periodic observations. The
[subsequent review](provenance/m1-macos-review.json) found that the final
interval before process exit could miss paging; the helper now brackets
launch and exit with observations. Existing classifications retain that
sampling limit and do not establish isolated throughput.

## Experimental ANE backend

```sh
./qwen35/first-run.sh generate --backend ane --prompt 'Hello' --max-tokens 32
./qwen35/.venv/bin/python -m qwen35.tools.verify_ane
```

The default `--ane-mode accurate` normalizes dynamic activation coefficients
with exact powers of two and restores the scale in FP32. On identical CPU
projection inputs, the worst unscaled ANE error was 2.509%; scaling reduced
it to 0.0416%. Small coefficient values were lost by the original stream,
beyond ordinary FP16 rounding. This diagnosis does not establish which
internal hardware stage loses those values.

Scaling alone still accumulated too much full-model error. The corrected
policy uses two disjoint input partitions, a high and residual FP16
activation plane per partition, and two replicas with gains 1 and 11/8.
These occupy eight of the existing 32 batch rows. Different gains give
different FP16 output rounding grids; their FP32 average reduces rounding
error. Native NEON code packs the inputs and combines the partial results,
retaining the order of the per-replica reductions. All matrix products run
on ANE in one submission per projection. The 96 resident FP16 matrices
occupy 995,229,696 coefficient bytes (0.927 GiB), as before. The vocabulary
head, recurrent updates and attention remain on CPU.

`--ane-mode legacy` reproduces the unscaled policy for comparisons;
`--ane-mode scaled` isolates input normalization. Both are diagnostic
policies. The benchmark enforces the unchanged 0.5% full-logit NRMSE gate
and every argmax match for `accurate`, while retaining failed receipts.
It checks the exact 96-submissions-per-input contract on all ANE policies.

The Linux backend derives its dimension patches from the existing training
linear primitives. It accepts decoded FP16 matrices without re-quantizing
them to Q8, retains all 96 body projections on ANE, and leaves recurrent
updates, attention and the vocabulary head on CPU. Every projection shape
passed against FP32 matmul at batches 1, 8 and 32, with normalized error below
0.035%. No new macOS compilation or dump was needed.
[Template provenance and shape results](provenance/ane-template.json) are retained.

The historical unscaled full-model path accumulated error beyond the
projection-level tolerance. The corrected path is still experimental:
passing these traces does not establish exact logits or parity on arbitrary
prompts. It copies more output rows and combines more partial sums, so the
accuracy improvement has a throughput cost. CPU SDOT remains the default.
Hardware submission failures raise an error; there is no silent CPU
projection fallback.
Concurrent hardware benchmarks should hold the site's shared ANE/GPU locks.
