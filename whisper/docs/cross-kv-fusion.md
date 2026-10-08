# Experimental cross-K/V fusion

Combining the eight tiny.en cross-attention projections into one OpenBLAS GEMM
preserves the native results but has **not established an end-to-end speedup**.
It remains opt-in. This is an algorithm change; no further CPU library or flag
ablations were run.

On the M1 MacBook Air, paired runs using the same binary, OpenBLAS 0.3.34 and
four actual BLAS threads produced these 11-second ANE/CPU medians:

| Graph | Cross-K/V stage | Matrix calls | Weight conversion | Encode | Decoder | Whole |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eight separate products | 33.16 ms | 30.98 ms | 1.02 ms | 52.24 ms | 143.91 ms | 232.11 ms |
| One fused product | 30.57 ms | 28.48 ms | 0.00 ms | 49.81 ms | 144.94 ms | 232.14 ms |

Cross-K/V improves about 7.8%, but whole time does not improve. Each layout and
backend used two persistent contexts: two excluded warmups and ten measurements
per clip per round, with layout order and CPU/ANE order reversed in round two.
All 288 transcriptions and 1,296 matrix profiles, including warmups, are retained
in the [receipt](../results/macos-cross-kv-fusion-20261007.json). Matrix metadata
is interned without discarding runtime addresses or any timing stage.

These runs are slower than the earlier library ablations. Active desktop load
and unfixed clocks prevent a causal comparison with those earlier timings.
The paired comparison uses its own contemporary separate-product baseline.
This is a macOS/OpenBLAS measurement, not an Asahi hardware result.

All six native cases preserve their mel and encoder-output hashes. All 160 captured
native decoder logit vectors, CPU and ANE combined, match the earlier separate
OpenBLAS implementation bit for bit, including token histories and argmaxes.
Every CPU vector passes the unchanged independent HF NRMSE < 0.005 gate. The
original ANE graph still fails that gate; these timings remain diagnostic.

The cache widens checkpoint FP16 weights to FP32 once per persistent decoder
state. Weight rows are concatenated as K0,V0,K1,V1,K2,V2,K3,V3 into `[3072,384]`.
One GEMM multiplies encoder features `[1500,384]` by its transpose, producing
`[1500,3072]`. Eight contiguous slices then use the existing K scaling, V bias
and KV cache writes. FP16 widening is exact; checkpoint values are unchanged.
The ggml cross compute buffer increases from **3.89 MB to 22.33 MB**, plus a
**4,718,592-byte** persistent FP32 weight cache. The prototype supports tiny.en,
CPU execution, flash attention and 1,500 encoder positions only.

## Reproduce

First prepare an isolated worktree using the shared native precision/profiling
workflow in [compact-encoder.md](compact-encoder.md). Then apply:

```sh
whisper/.venv/bin/python -m whisper.scripts.prepare_cross_kv \
  --source whisper/vendor/whisper-macos-fused-cross-kv
```

The patcher checks the pinned revision, rejects unknown cross-graph edits and
can be run repeatedly without changing an already patched graph. Its source
patch compatibility was checked against both Mac and Asahi prepared graphs;
Linux compilation and hardware execution remain unverified.

Build with the same OpenBLAS settings as the reference, then validate:

```sh
whisper/.venv/bin/python -m whisper.scripts.benchmark_native \
  --backend macos --build whisper/build/macos-fused-cross-kv \
  --model whisper/models/ggml-tiny.en.bin \
  --payloads whisper/build/matched-encoder --dylib ANEFORGE_DISPATCH_DYLIB \
  --warmups 2 --runs 10 --rounds 2 --profile-stages --profile-matmul \
  --fused-cross-kv --diagnostic-timings \
  --output whisper/build/cross-kv-fused-validation-new
```

Replace `ANEFORGE_DISPATCH_DYLIB` with the existing runtime path. The command
still exits with FAIL if the full-logit gate fails. It explicitly requests one
`1500 x 3072 x 384` FP32-weight matrix profile. Without `--fused-cross-kv`, the
validator still requires all eight `1500 x 384 x 384` profiles.

After a completed fused validation and a completed separate-product reference:

```sh
whisper/.venv/bin/python -m whisper.scripts.benchmark_cross_kv \
  --build whisper/build/macos-fused-cross-kv \
  --reference whisper/build/macos-openblas-ablation-20261007 \
  --validation whisper/build/cross-kv-fused-validation-new \
  --payloads whisper/build/matched-encoder --dylib ANEFORGE_DISPATCH_DYLIB \
  --output whisper/build/cross-kv-paired-new \
  --compact-receipt whisper/build/cross-kv-paired-new-receipt.json
```

This verifies model, payload, runtime, driver and library identities, compares
all full vectors and boundary bytes, then measures both layouts on the same
binary. On Asahi, validate with `--backend asahi` and omit `--dylib` from both
commands; supply the corresponding Linux reference and validation directories.
The native opt-in switch is `WHISPER_FUSED_CROSS_KV=1`; the default is separate
products. No Apple AMX or Accelerate dependency is introduced by the cache.
