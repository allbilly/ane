# Orion GPT-2 on Asahi

Complete GPT-2 124M text generation, ported from `~/Desktop/Orion`. Standard
model weights stay in your Hugging Face cache. ANE coefficients are regenerated
from that checkpoint on the target machine; the dump supplies reference hashes
and packing recipes. Linux execution uses Python, NumPy,
`regex`, `safetensors`, and the `ane` kernel driver; it needs no Apple
frameworks, `anecc`, or Python `libane` binding.

The runner and replay/reference data are about **11 MiB**, versus about
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

**Linux ANE replay has not yet been tested on hardware.** This machine is
running macOS. The port is prepared and checked offline, with real macOS ANE
reference captures; the Asahi run is the remaining hardware acceptance test.
These checks cannot guarantee that the Linux driver executes macOS 27 task
programs correctly. `package.json` records this distinction explicitly.

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

## Orion macOS performance references

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
references. The Python port's macOS fixtures establish numerical parity;
Asahi ANE performance remains unmeasured, and a CPU speedup has not been shown.

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
