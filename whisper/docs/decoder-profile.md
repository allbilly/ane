# Native vocabulary and softmax profile

The missing Mac vocabulary and softmax attribution is now captured on all
three tiny.en clips. Vocabulary projection uses widened FP32 NEON matrix
kernels for both the prompt and individual tokens. Attention uses the sampled
online softmax path; token probability processing calls scalar `expf` and
`logf`. This completes profiling evidence, not a performance optimization.
The original fast ANE graph's full-logit failure remains explicit.

The [profile receipt](../results/macos-decoder-profile-20261008/summary.json),
[targeted assembly](../results/macos-decoder-profile-20261008/selected-assembly.txt)
and representative stack paths
([11 seconds](../results/macos-decoder-profile-20261008/jfk-selected-stacks.txt),
[5 seconds](../results/macos-decoder-profile-20261008/jfk-first-5s-selected-stacks.txt),
[23 seconds](../results/macos-decoder-profile-20261008/jfk-repeat-selected-stacks.txt))
total about 795 KB. Full raw samples, traces and learned data stay in ignored
`whisper/build/macos-decoder-profile-results-fixed-20261008`.

## Numerical preservation and sampling

An isolated shared FP32 preparation names the vocabulary tensor and adds
diagnostic call frames around the existing CPU matrix dispatcher. The two
wrappers differ by one NOP so compiler folding cannot merge their names;
both call the unchanged multiplication routine. The existing log-probability
and probability functions retain their bodies and are marked noninline.
This preparation is for attribution, and does not replace the benchmark build.

Fresh CPU/ANE traces on all three clips reproduce all **160 full native logit
vectors byte for byte**, including histories. Mel and encoder boundaries are
also byte identical: all 18 native trace-file hashes match the existing shared
reference. Its independent 80-vector CPU/HF gate therefore remains passing,
while the original ANE failures remain unchanged and are retained in the
receipt. No numerical gate is weakened.

Each clip has two excluded warmups and 200 transcriptions: **606 transcriptions
in total**. Every transcript, zero-fallback check, full encoder submission and
vocabulary geometry was verified. Sampling begins only after observing the
second warmup's completion. Each process is sampled for ten seconds at a
requested one-millisecond interval. Loaded-image UUIDs and address ranges are
retained per process; targeted LLDB disassembly uses the same native libraries.

## Selected vocabulary path

The fixture's prompt has two tokens; subsequent calls have one. Shapes are
derived from the saved token histories rather than assumed. Each call reports
four CPU workers and F16 weights with F32 activations/output:

| Phase | Matrix dimensions m × n × k | Selected kernel |
| --- | --- | --- |
| Prompt | 2 × 51864 × 384 | `tinyBLAS ... gemm<4, 2, 2>` |
| Token | 1 × 51864 × 384 | `tinyBLAS ... gemm<4, 1, 2>` |

Here m is the number of input tokens, n the vocabulary size, and k the decoder
width. The tied embedding weights occupy 39,831,552 bytes; their row stride is
768 bytes. Each input-token row has a 1,536-byte stride and each output row a
207,456-byte stride. Full strides and all calls are retained in the receipt.

The vocabulary labels are ancestors of the actual sampled tinyBLAS kernels,
so this is separate attribution from generic decoder-layer multiplication.
Assembly shows `fcvtl`/`fcvtl2` widening F16 values and `fmla.4s` FP32 NEON
accumulation. The BLAS backend's minimum batch is 32; these one/two-token
products use the CPU path. The previously sampled cross-K/V Apple BLAS kernel
uses AMX; that evidence is separate from vocabulary projection. See the
[Mac follow-up](macos-followup.md).

## Attention and probability processing

| Operation | Sampled routine | Relevant work |
| --- | --- | --- |
| Decoder attention | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` | FP32 K/V path, `ggml_vec_dot_f32`, online max/sum updates and scalar `expf` |
| Log-probabilities | `whisper_compute_logprobs` | Maximum, scalar exponentials, FP32 sum, `logf`, subtraction |
| Probabilities | `whisper_compute_probs` | Scalar `expf` for unsuppressed logits |

The attention function supports F32 K/V despite its historical name. Its sampled
F32 dot routine and online exponential calls are mapped and disassembled.
The selected flash path uses online normalization; the generic batched
`ggml_vec_soft_max_f32` is not attributed to this run. Probability routines call
`expf` at file offset `0x5b00` and `logf` at `0x5a18` in `libsystem_m.dylib`,
UUID `A3FB340F-3ED5-30F4-8B64-9D5860F64D35`.

For example, the 11-second sample has these inclusive role stack counts:

| Role | Stack count |
| --- | ---: |
| Vocabulary token | 8,890 |
| Vocabulary prompt | 364 |
| Attention | 4,849 |
| Log-probabilities | 266 |
| Probabilities | 145 |

Counts include worker waiting, metadata logging and sampling overhead. They
are neither CPU-cycle shares nor fractions of whole-transcription time.
Repeated frame labels are counted once per role per stack path; the receipt
also records the repeated counts. Representative excerpts repeat aggregate
parent nodes, so their displayed counts must not be added across excerpts.

## Build and limits

The new environment receipt records the actual CMake compiler paths,
`/usr/bin/cc` and `/usr/bin/c++`, and Apple Clang **21.0.0**, plus CPU/BLAS/Whisper
target flags. The earlier runner queried the shell's `clang`, which reported
Homebrew Clang instead of the build compiler; the runner now queries the
recorded CMake compiler. Settings remain Apple BLAS, CPU Accelerate and tiled
kernels enabled, shared FP32 dot precision, CPU host/decoder and four workers.

Thermal checks before/after each sample report no warning. CPU/ANE clocks,
power, exact physical core placement and device timestamps remain unavailable;
blocking runtime time is not a device timestamp. Asahi hardware is inaccessible
from this Mac. No new library/flag ablation or speed/parity claim is made.

Use a new isolated worktree with the pinned shared Mac preparation, then:

```sh
whisper/.venv/bin/python -m whisper.scripts.prepare_decoder_profile \
  --source whisper/vendor/whisper-macos-decoder-profile
```

Build with the shared Apple BLAS settings. Run `whisper.scripts.profile_macos_decoder`
with the build/source, original matched native reference, pinned model,
existing full-encoder payloads/runtime, `--runs 200 --seconds 10` and a new
output directory. It first verifies every native byte before profiling.
`whisper.scripts.summarize_decoder_profile --profile CAPTURE_DIRECTORY
--build BUILD_DIRECTORY --output NEW_DIRECTORY` verifies all artifact hashes
and maps the captured stacks without rerunning hardware.
