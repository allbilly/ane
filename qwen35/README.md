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
| ANE FP16 body + CPU SDOT head (experimental) | 19.34 | 19.05–19.44 | 378/384 | 17.30% |

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

The Linux backend derives its dimension patches from the existing training
linear primitives. It accepts decoded FP16 matrices without re-quantizing
them to Q8, retains all 96 body projections on ANE, and leaves recurrent
updates, attention and the vocabulary head on CPU. Every projection shape
passed against FP32 matmul at batches 1, 8 and 32, with normalized error below
0.035%. No new macOS compilation or dump was needed.
[Template provenance and shape results](provenance/ane-template.json) are retained.

Full-model FP16 error accumulates beyond the projection-level tolerance:
this path fails the 0.5% full-logit gate and changed six of 384 token choices.
It is experimental and slower than CPU SDOT in these measurements. Hardware
submission failures raise an error; there is no silent CPU projection fallback.
Concurrent hardware benchmarks should hold the site's shared ANE/GPU locks.
