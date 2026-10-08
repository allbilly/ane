# Native batched encoder attention

Contiguous batched FP32 attention improves the accurate native paired encoder
on the M1 MacBook Air. In the same-binary comparison, 11-second encode falls
from **272.75 to 205.92 ms**, and whole transcription from **393.46 to 328.84
ms**, about **16% faster overall**. All 80 full decoder vectors still pass the
unchanged NRMSE < 0.005 gate with matching histories and raw argmaxes. The
paired encoder remains slower than the matched CPU encoder. Asahi execution
and performance remain unverified.

The [compact receipt](../results/macos-encoder-attention-20261008.json) retains
both complete native validations, all 160 cross-algorithm native logit
comparisons, all 288 warmup/measured transcriptions and every matrix profile.
It preserves exact matrix metadata, including addresses, by interning only
repeated fields. Every original log and native validation artifact hash was
verified after execution, and compacting the raw report reproduces the receipt.
Raw logs, native features/logits and learned programs remain in ignored
`whisper/build` paths.

## Algorithm and execution

The existing encoder flash-attention path processes individual queries. The
existing non-flash graph has two matrix products per layer, but its permuted
operands fail the BLAS backend's contiguous-operand requirement. The opt-in
graph materializes contiguous Q/K/V head planes and uses the existing FP32
matrix products and scaled softmax. Decoder attention is unchanged.

Each of four encoder layers executes two BLAS batches with six heads:

| Product | Per-head dimensions m × n × k | Arithmetic |
| --- | --- | --- |
| QK | 1500 × 1500 × 64 | FP32 GEMM |
| Probabilities × V | 1500 × 64 × 1500 | FP32 GEMM |

Profiles identify the actual `cblas_sgemm` entry point, loaded Apple BLAS
library, strides, transposes, dimensions and six-head batch count. Each encode
requires all eight named batches, covering 48 GEMM calls. All 1,152 attention
batch profiles from the warm comparison are retained; correctness adds 48.
The unchanged cross-K/V path contributes 2,304 warm batch profiles. Four host
workers are requested; the actual Apple BLAS thread count remains unavailable.
The new attention shapes were not separately sampled for internal AMX kernel
selection.

`WHISPER_ENCODER_BLAS_ATTENTION=1` selects the graph only for tiny.en with a
FP32 CPU host. `WHISPER_PROFILE_ENCODER_ATTENTION=1` records its actual BLAS
calls. The shared benchmark requires those calls and rejects incomplete
execution, wrong dimensions/types/head counts, or unexpected attention in
the flash baseline. FP32 operands require no numerical conversion; an empty
timing interval can nevertheless span a timer tick. Convolution and the 24
paired ANE programs, FP32 normalization/GELU/residuals, cross-K/V and the
decoder retain their existing arithmetic.

Encoder compute buffers grow from **17.72 to 64.81 MB** in both native modes.
The graph uses portable ggml/CBLAS interfaces, with no Accelerate API calls in
the graph patch. The preparer currently recognizes the shared Mac preparation;
Linux native integration and hardware measurements are still required before
an OpenBLAS or Asahi speedup can be claimed. Further library/flag ablations
were not run.

## Accuracy

Both attention algorithms were independently validated against HF on exact
native mel inputs, using the shared FP32 decoder configuration. The table
shows the new batched path; every full vector, not only the worst vector,
passes the 0.5% gate.

| Clip | Vectors | Maximum CPU/HF logit NRMSE | Maximum paired/HF logit NRMSE |
| --- | ---: | ---: | ---: |
| 11 seconds | 25 | 0.0585% | 0.1961% |
| 5 seconds | 8 | 0.1207% | 0.4594% |
| 23 seconds | 47 | 0.0736% | 0.2058% |

Every encode verifies 24 actual ANE projection submissions. Changing attention
alters accumulation order, so native encoder/logit arrays are not bit identical
across algorithms; their measured errors and every matching token history are
retained separately. The worst cross-algorithm ANE logit NRMSE is 0.0994%.
The original fast FP16 graph remains unchanged and still fails this strict
gate. Broader audio accuracy is unverified.

## Warm performance

Both algorithms use the same native binary and libraries, four workers,
three clips, two excluded warmups and ten measurements per clip per round.
Algorithm and CPU/ANE backend order are both reversed in round two.
For the 11-second clip:

| Encoder | Attention | Encode | Decoder | Whole transcription |
| --- | --- | ---: | ---: | ---: |
| FP32 CPU | Flash | 176.06 ms | 101.52 ms | 315.30 ms |
| FP32 CPU | Batched BLAS | 146.88 ms | 100.16 ms | 268.31 ms |
| Paired ANE | Flash | 272.75 ms | 103.07 ms | 393.46 ms |
| Paired ANE | Batched BLAS | 205.92 ms | 100.18 ms | 328.84 ms |

The paired batched transformer stage is 183.58 ms. Projection packing,
blocking dispatch and combining products have medians of 29.16, 33.79 and
28.33 ms. Remaining transformer work, computed within each repetition, is
88.24 ms, versus 139.50 ms with flash attention. The eight attention BLAS
batches take 34.17 ms; the remaining host work is not separately attributed
by this profile. Medians of components need not sum to enclosing medians.
Blocking dispatch includes runtime waits and is not a device timestamp.

Whole paired medians for the 5/23-second clips improve 299.20 → 245.94 ms and
482.29 → 447.38 ms. Active desktop load and unfixed clocks cause substantial
variation: 11-second paired flash round medians are 375.15/403.95 ms, and
batched medians are 338.82/259.73 ms. Both rounds improve, but these results
do not isolate attention as the cause of every component's timing change.
Compare these paired results within this experiment, not against older runs
under different timing conditions.

## Reproduce

Use a fresh isolated worktree with the pinned shared Mac FP32 preparation and
the [native paired preparation](native-precision.md). Apply attention after
preparing the 24 paired projections:

```sh
whisper/.venv/bin/python -m whisper.scripts.prepare_encoder_attention \
  --source whisper/vendor/whisper-macos-batched-attention
```

Build with the same Apple BLAS settings as the native paired experiment.
Run `whisper.scripts.benchmark_native --backend macos --encoder paired`
with the same model, programs, dylib and two-warmup/ten-measurement/two-round
policy, once with `--encoder-attention flash` and once with
`--encoder-attention blas`. Both require `--profile-stages --profile-matmul`
for this comparison; use distinct new output directories. Both full 80-vector
validations must pass before comparing warm execution:

```sh
whisper/.venv/bin/python -m whisper.scripts.benchmark_encoder_attention \
  --build whisper/build/macos-batched-attention \
  --reference whisper/build/macos-flash-attention-validation-20261008 \
  --validation whisper/build/macos-batched-attention-validation-fixed-20261008 \
  --precision-programs whisper/build/macos-batched-attention-programs-20261008 \
  --dylib ANEFORGE_DISPATCH_DYLIB \
  --output whisper/build/macos-encoder-attention-paired-new \
  --compact-receipt whisper/build/macos-encoder-attention-receipt-new.json
```

Replace the dylib placeholder with the existing runtime path. This command
reverifies every full native vector and artifact hash, checks binary/library/
program identities, and records 288 transcriptions with reversed order.
