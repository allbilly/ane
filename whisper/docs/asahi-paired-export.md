# Paired projection export for Linux preparation

The 24 accuracy-corrected paired projection programs now have checkpoint-only
H13G replay packages. All command, constant and coefficient payloads reconstruct
byte for byte from the exports. The package contains 42,728 bytes across 11
files, including provenance and three shared instruction/layout templates;
14,155,776 learned coefficient bytes come from the external pinned checkpoint.
No new archive was created.

This completes export and reconstruction preparation. It does not establish
native Linux integration, exported-HWX execution, Linux accuracy or speed.
The [proof](../kernels/tiny-en-paired/proof.json) explicitly retains those limits.
The original fast graph and its failed accuracy checks remain unchanged.

## Arithmetic and layout

These are the same exact MIL/weight files used by the passing
[native Mac paired encoder](native-precision.md), verified against its complete
80-vector accuracy receipt and the pinned checkpoint. Each projection splits
the contraction into two groups and uses two FP16 rounding grids, at gains
1 and 1.375. High and residual input planes for each grid occupy four consecutive
1,500-position ranges. The residual scale is 4,096. Separate products combine
in FP32 in the same grid/partition order; FP32 bias stays on the host.
Host convolution, attention, normalization, GELU and residual additions are
also unchanged.

| Input × output features | Programs per encode | Tasks per program | Coefficient tiles | Input bytes | Output bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 384 × 384 | 16 | 1 | 64 | 4,620,288 | 9,240,576 |
| 384 × 1,536 | 4 | 2 | 192 | 4,620,288 | 36,962,304 |
| 1,536 × 384 | 4 | 2 | 64 | 18,481,152 | 9,240,576 |

This is 24 logical submissions and 32 hardware tasks per encode, rather than
the historical 1,128 submissions of the 32-position projection path. Those
counts describe the exported program structure; no Linux execution was observed.

The hardware input is `[1,k,1,6000]` and output `[1,2n,1,6000]`, with a
**12,032-byte channel stride**. A dense logical channel contains 12,000 bytes;
the additional 32 bytes are zero padding in the packer. Input/output banks are
4 and 5. Only active BAR selectors in task-header words 8/9 change: original
constant bank 1 maps to replay bank 2, and coefficient bank 6 maps to driver
bank 1. Register packets, task IDs, dependencies and next-task pointers are
preserved and checked separately.

Compiler coefficient packing uses transposed 16/8-output tiles. Every byte
is covered once, with no overlap or unexplained padding. The three command
and constant templates, port layouts and tile maps are identical across all
captured projections of their respective shape despite different weights.
Only checkpoint coefficient bytes vary.

## Reconstruct on Linux

Use the external tiny.en safetensors checkpoint SHA-256
`db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`:

```sh
whisper/.venv/bin/python -m whisper.paired_kernel \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --output whisper/build/asahi-paired-payloads-new
```

The command was executed on Mac with the committed-intended templates. Its
reconstruction has no macOS runtime or compiler calls; execution on Linux is
still unverified. It writes all 24 command/constant/coefficient/bootstrap
payload sets and their native layout descriptors. Choose a new output directory.
Every payload hash is checked before writing.

The six tests in `experimental/test_whisper_paired_kernel.py` verify all 24
payload hashes, 24/32 submission/task geometry, all four temporal planes and
zero padding for every shape. They reject dense or altered hardware layouts,
missing/overlapping coefficient tiles, non-FP16-exact weights, modified command
assets, incomplete program manifests and a single failed full logit vector.
The tests require the external pinned checkpoint.

## Repeat the export/package preparation on Mac

The source programs must first pass
`whisper.scripts.prepare_macos_precision.validate_programs`. Export each
unchanged program with `experimental.capture_macos_program.export`, using
`gpt2/training/build/dump_hwx` and a new directory for each program. The validated
24 exports for this preparation are retained under the ignored
`whisper/build/asahi-paired-export-20261008/<projection>/hwx` paths.

Then package their instruction/layout recipes:

```sh
whisper/.venv/bin/python -m whisper.scripts.package_asahi_precision \
  --programs whisper/build/macos-batched-attention-programs-20261008 \
  --exports whisper/build/asahi-paired-export-20261008 \
  --checkpoint whisper/models/hf-tiny.en/model.safetensors \
  --native-receipt whisper/results/macos-native-precision-20261007.json \
  --output whisper/build/asahi-paired-kernels-new
```

This accepts only the exact pinned programs and complete passing native
receipt. The source/compiled file hashes and compiler identity remain in the
compact proof. Offline ANECCompile and E5RT compile the same MIL separately;
matching their actual executable identity is unproven. The exported HWX files
have not been executed on either host.

## Performance scope

The accurate paired path is a comparison control for numerical corrections.
Its Mac implementation remains slower than CPU. It is not an established
Asahi speed solution. The original fast Asahi graph's recorded 142.90 ms
whole time is retained as the diagnostic performance baseline while it fails
the strict accuracy gate. A cheaper numerical correction and measured Linux
accuracy/performance remain required; no projected whole-run time is accepted
as benchmark evidence.
