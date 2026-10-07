# Original fast encoder dump review

The original fast graph is now the default compact package:
[`tiny-en-encoder-fast`](../kernels/tiny-en-encoder-fast/meta.json). It contains
**1,779 tasks in 152,479 bytes (149 KiB)**, with learned weights stripped and
reconstructed from the pinned HF checkpoint. The previous 1,783-task dense
wrapper is retained as a baseline. Full HWX, weights and numerical arrays stay
outside Git.

## Identity checks

The original MIL SHA-256 is
`e7320a59f2988a48add91afe6eda18beec615006cae1e613ff68041f6b2c91ed`,
matching [`benchmark-macos/summary.json`](../results/benchmark-macos/summary.json).
Its weight blob matches the previous fast export too. The wrapper changes only
the two input signatures and adds two reshape nodes; the trained graph body
is identical. Those reshapes compile into four additional tasks and increase
scratch allocation from 8,093,696 to 10,371,072 bytes.

The fresh original export reproduces the earlier original command TEXT and
compiled coefficient segments byte for byte. Its full HWX hash changes because
container metadata changes; full-file hash equality is not the executable
payload comparison.

During E5RT compilation, the actual staged ANE MIL was also captured locally.
Offline-exporting that lowered MIL produces an identical complete TEXT payload
after swapping mel/output BAR selectors 5 and 6. Its coefficients are identical.
This rules out a different lowered model in that comparison. It remains an
offline export of the staged graph, rather than a readback of the live hardware
command buffer. macOS executes MIL through private E5RT; custom HWX is reserved
for Linux replay.

The [receipt](../results/fast-recapture-20261007/receipt.json),
[lowered-graph comparison](../results/fast-recapture-20261007/runtime-lowered-comparison.json)
and [byte reconstruction proof](../kernels/tiny-en-encoder-fast/proof.json)
retain hashes and numerical evidence. Compressed JSON reports beside the receipt
retain every timing repetition and full-vector error measurement.

## Warm performance

The direct wrapper comparison used fresh private-runtime programs, two warmups,
ten measurements per clip and alternating execution order. Every output was
bit-identical between graphs and to the earlier captured ANE output.

| Clip | Original execute | Dense-wrapper execute | Extra time |
| --- | ---: | ---: | ---: |
| JFK, 11 s | 10.880 ms | 12.278 ms | 1.398 ms |
| First 5 s | 10.855 ms | 12.200 ms | 1.345 ms |
| Repeated JFK, 23 s | 10.974 ms | 12.341 ms | 1.367 ms |

That wrapper costs about **13%** in this Mac comparison. The recapture's separate
original-only run again takes approximately 10.84–10.88 ms for execution.

The native transcription rerun used the pinned ggml model, four whisper workers,
one persistent context per backend across all three clips, two excluded warmups
and ten measurements per clip. All repeated transcripts match the earlier
CPU transcripts.

| Clip | Backend | Encode incl. cross-K/V | Prompt batch | Token decode total | Per decode call | Whole transcription |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11 s | CPU | 106.31 ms | 2.15 ms | 34.39 ms | 1.433 ms | 156.41 ms |
| 11 s | ANE + CPU | **15.83 ms** | 2.16 ms | 35.14 ms | 1.464 ms | **66.81 ms** |
| 5 s | CPU | 106.03 ms | 2.15 ms | 10.03 ms | 1.433 ms | 123.72 ms |
| 5 s | ANE + CPU | 15.74 ms | 2.15 ms | 10.03 ms | 1.433 ms | 33.41 ms |
| 23 s | CPU | 106.55 ms | 2.16 ms | 66.08 ms | 1.437 ms | 200.06 ms |
| 23 s | ANE + CPU | 15.77 ms | 2.15 ms | 65.85 ms | 1.432 ms | 108.89 ms |

These reproduce the earlier 11-second ANE/CPU result: encode 16.03 ms and whole
transcription 68.37 ms. All clips use the full 30-second encoder context.
The separate `prompt_ms` timer is zero; prompt/setup decoder evaluation appears
in `batchd_ms`, shown as “Prompt batch” above. Its median contains two native
batched calls. Decode times exclude sampling; whole time includes mel, sampling
and orchestration, but excludes compilation, loading and file I/O.
The current native library has been rebuilt and has a different hash from the
historical binary; the MIL, pinned model and private E5RT dylib identities match.
The current library hash and package versions are recorded in the environment receipt.
Runs used an active desktop without pinned clocks, and CPU/ANE backend order
was fixed in this transcription rerun. Cold private-runtime compilation was
occasionally slow and one preliminary transcription initialization timed out;
those setup attempts are excluded from warm measurements.

## Why Linux has not reproduced the result

There are three separate issues:

1. **Extra work in the old dump.** The dense wrapper adds four tasks and about
   2.2 MiB of scratch. The new default removes it by packing the original input
   strides: mel channels occupy 6,016 bytes each; position channels occupy 3,008.
   The Linux loader selects task count, buffers and strides from package metadata.
   The 13% improvement above is measured on Mac; its Linux effect remains to be
   measured.
2. **CPU and transfer boundaries.** The current [`todo.md`](../../todo.md) reports
   Linux readback improving from 5.31 to 1.34 ms, but CPU cross-K/V alone still
   taking about 28 ms. The Mac's 15.83 ms encode timer already includes its ANE
   encoder, transfers and CPU cross-K/V. The Mac run initializes Accelerate BLAS;
   the documented Linux build disables BLAS and uses its native CPU/NEON path.
   Tune the eight cross-K/V matrix products and retain the optimized native
   readback before attributing the gap to ANE execution. The latest Linux raw
   reports are absent on this Mac; those Linux stage numbers are user-provided
   evidence from `todo.md`, not a new paired measurement. Driver scheduling,
   ANE clocks and cache behavior remain unmeasured possible contributors.
3. **Different acceptance criteria.** The original fast Mac benchmark checked
   transcripts and encoder cosine, not every raw decoder logit. It does not
   establish that the fast path can meet the stricter accuracy requirement.

## The strict accuracy gate fails on Mac too

Fresh original-graph output exactly matches all three previous ANE fixtures and
passes HF encoder cosine >= 0.999. An independent test uses the exact same
FP16-rounded mel, a shared HF-generated history and all 51,864 raw logits at
every prefix. Comparing the same HF FP32 CPU decoder with HF encoder features
versus captured ANE features isolates the encoder's contribution:

| Clip | Encoder cosine | Raw vectors | Maximum logit NRMSE | Matching raw argmaxes |
| --- | ---: | ---: | ---: | ---: |
| 11 s | 0.99995589 | 25 | **3.0763%** | 25 / 25 |
| 5 s | 0.99991535 | 8 | **6.6396%** | 8 / 8 |
| 23 s | 0.99971041 | 47 | **2.7802%** | 47 / 47 |

Every clip exceeds the required **0.5%** maximum, despite all 80 token choices
matching. The original fast graph's numerical differences already affect
logits on Mac; replacing the Linux dump alone cannot repair that accuracy gap.
The native Mac CPU and ANE/CPU probes also fail the gate. Their decoder is the
upstream Mac path, which includes padded cross-attention keys and differs from
the corrected Linux CPU path; their larger errors should not be assigned solely
to ANE. The shared HF decoder comparison above avoids that confounder.

Next, run the new 1,779-task package on Linux, compare encoder output against the
exact Mac capture, and profile execute/readback/cross-K/V separately. A bitwise
match would establish replay fidelity, while the full-logit failure would still
require numerical changes to the model implementation. Keep the gate at 0.005
and benchmark any accuracy correction again. This is three constructed JFK
variants, not a diverse speech or word-error-rate evaluation.

The [checkpoint/audio-only profiler](../scripts/benchmark_encoder.py) now reports
dispatch, preparation and readback separately for both compact packages. It
checks the exact Mac outputs by hash without requiring a full HWX or captured
activation bundle. Its input preparation was verified on Mac for all three
cases. Native Linux execution and the latest native cross-K/V profile remain
pending; see the command in [compact encoder](compact-encoder.md).
