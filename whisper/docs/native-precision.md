# Native paired encoder precision

The native C++ paired encoder passes the unchanged full-logit NRMSE < 0.005 gate
on all 80 tiny.en decoder vectors, with every token history and raw argmax
matching the CPU reference and HF. This corrects accuracy on the M1 MacBook Air;
**it is slower than the matched CPU encoder and does not establish Asahi parity**.
The original fast FP16 graph remains unchanged and still fails the strict gate.

A subsequent [batched encoder attention change](encoder-attention.md) preserves
the all-vector gate and reduces paired whole time 393.46 → 328.84 ms in a new
same-binary comparison. It remains slower than that experiment's CPU baseline;
these later timings should not be compared causally with the initial run below.

| Clip | Vectors | Maximum CPU/HF logit NRMSE | Maximum paired/HF logit NRMSE |
| --- | ---: | ---: | ---: |
| 11 seconds | 25 | 0.0584% | 0.1685% |
| 5 seconds | 8 | 0.1210% | 0.4484% |
| 23 seconds | 47 | 0.0736% | 0.2439% |

The [native receipt](../results/macos-native-precision-20261007.json) retains
every full-vector comparison, every warmup and measured transcription, program
and checkpoint identities, matrix metadata and all stage timings. It reparses
the original logs and verifies every artifact hash before collecting results.
The raw validation, arrays, programs and learned weights remain in ignored
`whisper/build` paths.

## Native arithmetic and runtime

The native encoder uses the same paired projection programs that passed the
Python correction. Each of the 24 projections splits the contraction in two
and supplies four FP16 temporal planes: high and residual values on rounding
grids with gains 1 and 1.375. Residuals are multiplied by 4096 before conversion.
One ANE-only E5RT submission returns both partial products for both grids. The
host combines them in FP32 in the same order as the Python encoder; existing
native FP32 bias addition follows. Convolution, attention, normalization,
GELU and residual additions retain the shared FP32 CPU arithmetic.

Native packing and combination use four-position NEON transposes and persistent
E5RT buffer views. The runtime checks each compiled weight payload against the
native checkpoint tensor byte for byte and rejects port-size mismatches or
nonfinite arithmetic. It requires exactly 24 projections and submissions per
encode; it has no CPU fallback for failed ANE execution. The CPU baseline does
not initialize the paired runtime. The existing full-encoder route remains
available separately.

## Warm performance

Both native paths use the same Apple BLAS build, four workers, three clips,
two excluded warmups and ten measurements per clip per round. CPU/ANE order is
reversed in round two. For the 11-second clip:

| Path | Encode | Decoder | Whole transcription |
| --- | ---: | ---: | ---: |
| Shared FP32 CPU encoder | 252.24 ms | 85.94 ms | 369.01 ms |
| Native paired ANE projections | 366.88 ms | 87.14 ms | 483.76 ms |

The paired transformer stage has a 333.60 ms median. Projection packing takes
52.02 ms, blocking dispatch 43.12 ms and combining products 43.32 ms. The median
remaining transformer work, computed within each repetition before aggregating,
is 195.93 ms. It includes host attention, normalization/nonlinearities,
residual work and graph execution overhead; these are not separately attributed
by this measurement. Blocking dispatch time includes runtime waits and is not a
hardware execution timestamp. Medians of components need not add to the median
of an enclosing timer.

All 144 timed/warmup transcriptions are retained. The paired timed contexts make
1,728 actual ANE submissions, and the three correctness runs add 72. All 1,152
timed/warmup cross-K/V matrix profiles are retained; correctness adds 48.
The active desktop and unfixed clocks limit comparisons with older runs.
This correction passes the requested fixtures; broader audio accuracy and
Linux execution have not been verified. The [paired Linux export preparation](asahi-paired-export.md)
now provides checkpoint-only H13G command, constant and coefficient reconstruction
for all 24 programs; native Linux integration and hardware verification remain pending.

## Reproduce on Mac

Create a new isolated worktree at the pinned revision and apply the shared
preparation described in [compact-encoder.md](compact-encoder.md). Stage the
existing paired programs rather than recompiling a different numerical graph:

```sh
whisper/.venv/bin/python -m whisper.scripts.prepare_macos_precision \
  --source whisper/vendor/whisper-macos-native-precision \
  --programs whisper/build/macos-paired-precision-20261007/paired-host-fp32/projections \
  --receipt whisper/build/macos-paired-precision-20261007/report.json \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --output whisper/build/macos-native-precision-programs-new
```

This checks the passing Python 80-vector receipt, each MIL hash and every
grouped weight against the pinned checkpoint. It patches the six projection
sites in the isolated native source and adds the runtime to its CMake target.
Use a new output directory on each staging run. Source patching is idempotent.

Build with the shared Apple BLAS settings, then run:

```sh
whisper/.venv/bin/python -m whisper.scripts.benchmark_native \
  --backend macos --encoder paired --build whisper/build/macos-native-precision \
  --model whisper/models/ggml-tiny.en.bin \
  --precision-programs whisper/build/macos-native-precision-programs-new \
  --dylib ANEFORGE_DISPATCH_DYLIB \
  --warmups 2 --runs 10 --rounds 2 --profile-stages --profile-matmul \
  --output whisper/build/macos-native-precision-validation-new
whisper/.venv/bin/python -m whisper.scripts.summarize_native_precision \
  --validation whisper/build/macos-native-precision-validation-new \
  --output whisper/build/macos-native-precision-receipt-new.json
```

Replace `ANEFORGE_DISPATCH_DYLIB` with the existing ANEForge runtime path. The
benchmark acquires the same hardware locks and validates all 80 full vectors
with the same 0.005 gate. Compilation occurs during context initialization and
is excluded from warm transcription time. Native opt-in is
`WHISPER_MACOS_PRECISION=PROGRAM_DIRECTORY` with `ANEFORGE_DYLIB` set.
This runtime uses macOS E5RT; an Asahi native implementation/export and hardware
validation are still required.
