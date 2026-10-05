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
reference mode matched Uzu exactly on the retained fixtures. FP32 is the
default. Four workers use M1 performance cores within the process's allowed
affinity; `--threads` and explicit OpenMP settings can override this.
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
tokens and a 23-token chat prompt. All matched bit for bit in BF16 mode;
the first two fixtures also matched all 24 residual layer outputs.
`verify` replays these fixtures and checks a rolling FP32 integer-dot trace.
[Vendor validation](provenance/vendor-validation.json) and
[source revisions and hashes](provenance/sources.json) are retained.

## Measured performance

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
