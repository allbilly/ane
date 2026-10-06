# macOS follow-up captures

These tools extend the existing validation without changing the retained
Asahi reference fixtures, the default Qwen decoder, or GPT-2's validated
32-token training kit. The initial capture is
`qwen35/local-results/macos-followup-20261005T175748Z/`; its historical receipt
is `qwen35/provenance/m1-macos-followup.json`. The unrestricted rerun is
`qwen35/local-results/macos-unrestricted-20261006T024951Z/`, with receipt
`qwen35/provenance/m1-macos-unrestricted.json`. Large artifacts stay in these
ignored capture directories. `mlx-ane-sd/task.md` remains excluded.

## Subsequent review

The [review receipt](../qwen35/provenance/m1-macos-review.json) retains the
additional loading diagnosis and the source fixes. Direct offline HWX
loading also fails for Qwen and training exports. Minimal/profiling options,
model key `net` and QoS 25 did not fix it; both system controls still load.
Compiling the exact same dense MUL MIL through `_ANEClient` with
`kANEFModelMIL` passes both 64-value numerical cases exactly. This is a
verified private macOS execution path. Following the runtime-scope correction,
our custom HWX dumps are assigned to native Asahi replay; standalone macOS
loading is removed from the pending execution tasks. ANEForge's primary
private E5RT path compiles MIL through the system service, then retains the
program library/function and creates the executable operation. Its documented
custom-HWX signature boundary explains why these two workflows must be
assessed separately; our error 53 alone does not establish its precise cause.
See [problem.md](../problem.md#private-macos-execution-and-historical-hwx-controls-2026-10-06)
and `experimental/probe_static_loading.py` for the repeatable comparison.

Replay validation now requires nonempty, consistent fixtures, unchanged
gates, matching port sizes, finite values and intact payload hashes before
device access. All 62 retained fixture comparisons pass this preflight.
Eleven regression tests cover these guards, stale recurrence inputs, paging
at process exit and lock-timeout receipts.
Qwen reused-input captures also require the same checkpoint and distinct
positions 0/1/7/22; new reports include their input-file hash.

The CPU timing helper now samples immediately before process launch and
after exit, and retains lock-timeout failure receipts. The earlier 13/54
host-gate passes use the historical periodic samples: paging in the final
unsampled interval cannot be ruled out. Their throughput numbers and
classifications are retained rather than recalculated as newly isolated runs.
Compiler-source patching now avoids adding another temporary-path wrapper
on each repeat. Historical archives and capture-time source snapshots remain
unchanged; review sources and evidence are retained separately.

The final comparison CLI built successfully, but its additional repeat
timed out waiting for the other session's ANE lock. That failure receipt is
retained separately from the earlier completed loading matrix and two exact
MIL numerical tests; it is not counted as a hardware pass.

## Unrestricted hardware rerun

The user removed the execution sandbox on 2026-10-06. ANE-only E5RT,
Metal and Core ML hardware routes now execute successfully; the restricted
attempts below remain historical evidence.

The 64-token training capture passed the independent PyTorch loss and
gradient gates. All 196 pretrained weight hashes matched. Three Adam updates
each made 580 actual ANE dispatches, reducing fixed-batch loss from 3.271003
to 0.428595. Loss differed from the independent oracle by 0.000208 and the
minimum selected-gradient cosine was 0.999938. All 27 templates and their
actual FP16 ANE output fixtures exported and passed the existing replay guards.
This remains a fixed-batch smoke test, not a general training-quality result.

All 15 Whisper transcription checks passed: five routes on each of the
5/11/23-second constructed JFK variants. The native ANE encoder fixtures
passed the existing cosine threshold of 0.999 against independent HF output
on the exact rounded mel input. Their NRMSE ranged 0.9393–2.4074%; the
Whisper cosine gate differs from the kernel replay gate. The current copied
trained bundle exported 1,779 tasks. Its MIL and weight hashes identify this
capture independently of the earlier 1,783-task bundle.
The dense 1,783-task wrapper then executed all three fixtures with device
mask 4 and matched the original encoder's FP16 outputs bit for bit. Its
independent HF cosine checks also passed. Actual wrapper outputs and both
comparison references are retained separately.

All four Core ML configurations passed the encoder cosine gate. Five warm
Python predictions per configuration had medians of 31.45 ms CPU, 25.70 ms
CPU/GPU, 18.42 ms CPU/ANE and 18.45 ms all-device. Static preferred-device
counts were 101 CPU operations, 101 GPU operations, or 99 ANE plus two CPU
operations respectively. These describe static placement rather than runtime
utilization, and the timer includes synchronous API I/O. Cold CLI results
and the existing warm-context benchmark have separate timing boundaries.

The unscaled fused Qwen component now executes on ANE. Its 16 state checks
passed 0.5% NRMSE, while its output checks failed with 6.59–35.09% NRMSE.
CPU recomputation from observed ANE state substantially reduces that output
error, and one head's ANE output is entirely zero. Compensated query and
state gains are explicit diagnostic options; they do not change the model
formula or its gate. Checkpointed diagnostics found native SiLU error up to
2.589% against the exact rounded-input CPU expression, while normalized
values remained below 0.248%. The selected graph uses query/state gains of
128 and explicit `z / (1 + exp(-z))`. All 16 cases pass on actual ANE: maximum
state NRMSE 0.294262% and output NRMSE 0.316745%, below the unchanged 0.5%
gate. All four programs exported and passed replay guards. Every failed
attempt and diagnostic remains retained; extra diagnostic outputs can change
compiler fusion and do not replace the two-output selected graph.

The latest kits contain 32 programs, 36 output ports and 62 output-fixture
checks with actual macOS ANE references. The dense Whisper wrapper is also
validated independently against the original encoder. Native Asahi replay
and model-level Linux validation remain required.

Seven static MUL variants compiled with identical task structures, but all
20 constant/pattern checks, including three older controls, failed at model
loading with ANE error 53, underlying status 1, stage 4: "Program load failed
— no memory". This is the runtime's error classification; physical memory
exhaustion has not been established. Missing/empty compiler options remain
negative controls and do not prove recognition of a schema.
The known-good system HWX loaded successfully through the same checker.
A separate dense width-64 MIL MUL produced all 64 constant/patterned outputs
exactly through ANE-only E5RT, while its exported file still failed standalone
loading. These distinguish working MUL execution from exported-file acceptance;
the original port padding alone does not explain the loading failure.

The unrestricted CPU phase completed 18/18 numerical runs with complete
process, swap and thermal observations: all 108 first predictions and 3,456
decode choices matched, with maximum SDOT NRMSE 0.058167%. Five timings passed
the host-activity gates and 13 were affected by desktop activity. The
[Qwen table](../qwen35/README.md#follow-up-validation) aggregates every attempt;
it does not discard affected runs or establish an isolated OS comparison.

## Historical restricted results and limits

Both CPU timing phases completed all 36 numerical runs; eight passed the
host-activity gates and 28 timings were affected. The 1K/2K-context checks
matched all six first predictions and 384 decode choices, and four separate
greedy streams matched all 256 choices. The final BF16 helper rerun matched
the first Mac capture exactly at all arithmetic prefixes and layers.

The offline 64-token training run passed its independent loss/gradient
gates and three updates reduced fixed-batch loss from 3.27151 to 0.42611.
All 27 templates exported. Four fused recurrence programs exported and
their CPU reference state/output comparisons passed the 0.5% gate.
The trained Whisper encoder exported 1,783 tasks and its coefficient bank.

The initial compiler emitted padded Qwen beta/g ports and padded Whisper
mel/position ports. The original captures and explicit guard rejections are
retained. `qwen-fused-dense` repeats per-head beta/g across width 128;
`whisper-dense` uses two contiguous input ports followed by shape-only
reshapes. Whisper input bytes are checked unchanged. These new kits and
`training64-offline` pass every existing replay guard: 32 programs, 36 output
ports and 62 output-fixture checks are prepared. This is preparation only;
all new references are CPU outputs, not observed ANE outputs. The strict
kernel replay gates remain unchanged and still need hardware validation.

Core ML conversion and compiled inference succeeded with `TMPDIR` inside the
writable workspace. Four requested compute configurations passed the
encoder cosine gate: cosine 0.999913, NRMSE 1.32652%, five prediction samples
per configuration around 185 ms. Their static compute plans selected CPU
for all 101 operations; CPU/ANE requests also logged denied sandbox file
extensions. These are CPU-selected results, not an ANE/GPU speed comparison.
Three Core ML CLI transcriptions matched native CPU words on 5/11/23-second
JFK variants. Native Metal aborted at a 2,304,000-byte buffer allocation,
including three diagnostic retries with workspace `TMPDIR`; its cause is
unproven. Direct ANE compilation/loading was denied in this profile; each
failed route's log remains retained.

All seven static MUL variants compiled after the native temporary directory
was moved into the workspace. Every variant has a 504-byte descriptor, one
task and the same task-structure digest as the retained macOS-26 MUL. Both
constant and patterned direct-runtime checks failed at model loading with
`sandbox_extension_issue_file: Operation not permitted`. The three older
controls failed at that same stage. This does not settle compiler option
recognition or the original MUL compatibility issue.
Declared logical MUL inputs and exact CPU expectations are retained even
when loading fails; no actual ANE outputs were produced by these attempts.

## Cross-host BF16 comparison

The independent pinned macOS Uzu oracle matches native BF16 at every one of
the 23 arithmetic tokens, including all 24 residual layers and all logits.
Its final arithmetic logits differ from the saved Asahi oracle. The original
strict fixtures remain unchanged. Comparing both hosts' independent
all-token captures is the next step needed to locate that difference.

On native Asahi, after copying the capture directory and reusing the same
verified Mirai checkpoint:

```sh
qwen35/.venv/bin/python -m qwen35.tools.build_reference \
  --uzu ~/.cache/ane-qwen35/cross-host/uzu \
  --lockfile FOLLOWUP/vendor-build/Cargo.lock
env OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  qwen35/.venv/bin/python -m qwen35.tools.capture_vendor_macos \
  --model MODEL_DIR \
  --oracle ~/.cache/ane-qwen35/cross-host/uzu-reference-cli/target/release/uzu-reference-cli \
  --tag asahi --compare-capture FOLLOWUP/vendor/uzu-macos-all.npz \
  --output NEW_ASAHI_CAPTURE
```

The helper reports the first differing prefix and residual layer between
oracles, plus native-versus-oracle comparisons on the executing host.
The saved Mac binary is for provenance; rebuild for Linux.

## CPU numerical coverage and timing

`qwen35.tools.check_long_context` prepares floating references for four short
chats and exact 1,024/2,048-token prompts, each followed by 64 decode calls.
It separately follows floating and SDOT greedy streams on arithmetic, code,
Chinese and household-advice prompts. All logits are compared while their
histories match. Fixed calls continue beyond EOS for numerical coverage.

The saved `long-context/asahi-replay.json` provides the invocation for
replaying these traces through the accurate Linux ANE backend. This is still
required before claiming long-context ANE parity.

`qwen35.tools.quiet_benchmark` repeats the original six-prompt protocol with
floating/SDOT kernels and 1/2/4 workers. It holds `~/gpu.lock`, rotates the
process order, records host load, external CPU usage, thermal diagnostics
and swapouts, and retains every attempt. Its predeclared host gates flag
affected timings rather than removing numerical results. Two phases on the
same boot do not establish an isolated OS comparison or an optimum worker
count.

When the execution profile prevents process or swap queries, the default
timing helper refuses the preflight. Explicit
`--allow-missing-host-observations` still runs numerical repeats, records each
query failure and flags the timing as affected. It cannot certify quiet
conditions. `QWEN35_CACHE_DIR` routes CPU compilation into an allowed workspace
cache when the normal home-directory cache is read-only.

## Additional hardware captures

The existing Whisper environment contains Torch, Transformers and the
ANEForge dependencies. Core ML conversion and static MUL generation also
need Core ML Tools with its native macOS extensions. This session reused
`/Users/yeren/more-ane-transformers/.venv/bin/python` read-only for those two
tools; `whisper/.venv` does not contain Core ML Tools.

```sh
whisper/.venv/bin/python experimental/capture_training_shape.py \
  --sequence 64 --steps 3 --output NEW_TRAINING_CAPTURE
whisper/.venv/bin/python experimental/capture_qwen_recurrence.py \
  --model MODEL_DIR --output NEW_RECURRENCE_CAPTURE
COREML_PYTHON experimental/probe_whisper_coreml.py \
  --model CACHED_WHISPER_HF_DIR --audio whisper/vendor/whisper.cpp/samples/jfk.wav \
  --output NEW_COREML_CAPTURE
whisper/.venv/bin/python experimental/capture_whisper_followup.py \
  --model CACHED_WHISPER_HF_DIR --output NEW_WHISPER_CAPTURE
```

Coordinate `~/ane.lock` and `~/gpu.lock` in that order while compiling or
executing hardware workloads. The session runner does this automatically.
The 64-token training capture uses its own PyTorch oracle, records all 27
first-use template fixtures and requires 580 ANE dispatches per update.
`--offline` emits the same graph shapes and runs CPU FP32 references with
FP16 program boundaries. Its 580 calls per update are CPU program calls;
they are not hardware submissions. The loss and gradient gates still apply.
The recurrence probe covers layer 0's recurrent update only: four programs
cover 16 heads at arithmetic prefix positions 1, 2, 8 and 23. It captures
both the updated state and mixer output against real FP32 decoder values.
It does not enable a fused full-model decoder.
Its `--offline` mode evaluates exact rounded MIL inputs/constants on CPU,
exports all four programs and identifies the reference kind in the report.
`--inputs PREVIOUS/native-inputs.npz` reuses an existing real decoder capture.
Dense width-128 beta/g ports are now the default. `--no-dense-scalars`
reproduces the original padded scalar port control.
The real captures exposed a difference between ANEForge's generic RMS
epsilon-floor helper and the decoder's epsilon-addition formula. The new
experimental graph uses explicit epsilon addition and scales by 128 so its
FP16 epsilon constant is normal. This passed the CPU reference checks;
hardware results above are judged against the same component gates.

The unrestricted recurrence diagnostics can move an exactly compensated
gain before the output matmul and inside the state update:

```sh
whisper/.venv/bin/python experimental/capture_qwen_recurrence.py \
  --model MODEL_DIR --inputs PREVIOUS/native-inputs.npz --dense-scalars \
  --query-gain 128 --state-gain 128 --silu-mode exp \
  --output NEW_SCALED_RECURRENCE_CAPTURE
```

Both gains must be finite powers of two at least one. The exported state
divides out its state gain, and the matmul output compensates both gains
before RMS normalization. The comparison remains against the original real
FP32 state and output; passing an offline reference does not replace the
actual ANE component gate. These validated options are now the probe's
defaults. `--query-gain 1 --state-gain 1 --silu-mode native` reproduces the
initial failed graph; the production Qwen decoder is unchanged.

Whisper captures use original JFK, its first five seconds, and two copies
separated by one second of silence. This extends duration/segmentation
coverage; it is not a diverse speech corpus or a WER benchmark. Real mel
inputs, positional input and independent HF outputs are saved. ANE outputs
are included only after actual runtime execution; a runtime compile failure
retains CPU reference fixtures and an explicit incomplete status.
Core ML encoder predictions use CPU-only, CPU/GPU, CPU/ANE and all-device
configurations. Preferred-device counts describe static placement, not
runtime utilization. Python encoder prediction timers and cold CLI timers
have different boundaries from the existing warm-context Whisper benchmark.

Every exported program retains the HWX, raw compiler status, task/register
records, coefficient segment hashes and port metadata when the strict parser
accepts its format. A parser rejection remains an explicit limitation.
For native Core ML compilation, create a workspace temporary directory and
set `TMPDIR` to its absolute path before starting Python or the compiler.
In the restricted capture, default native temporary-directory allocation was
denied. Moving permitted temporary output did not remove that profile's ANE
runtime restrictions; the subsequent unrestricted capture is recorded above.

Generate the dense Whisper port wrapper from its real-audio capture with:

```sh
whisper/.venv/bin/python experimental/dense_whisper_capture.py \
  --capture FOLLOWUP/whisper-audio-rerun --output NEW_DENSE_WHISPER_CAPTURE
```

After preparing its replay manifest, validate the new port layout on macOS:

```sh
env PYTHONPATH=/Users/yeren/Desktop/ANEForge ANEFORGE_NO_AUTOBUILD=1 \
  whisper/.venv/bin/python experimental/verify_macos_capture.py \
  --kit NEW_DENSE_WHISPER_CAPTURE --output NEW_MACOS_VALIDATION
```

This check verifies source, weight, HWX and fixture hashes, compiles the
captured MIL with device mask 4, and executes each fixture once while reading
every output. It saves actual FP16 outputs and retains the unchanged kernel
gates: relative L2 below 0.5% and `allclose(rtol=0.01, atol=0.03)`. Keep the
shared hardware locks held during this command. It validates MIL execution
on macOS; native Asahi execution of the exported HWX remains separate.

## Native Asahi fixture replay

`experimental/replay_capture.py prepare` creates a portable manifest from a
capture. It checks the existing guarded GPT-2 loader's coefficient-bank,
FP16 and dense-stride restrictions before declaring a replay candidate.
Unsupported formats or layouts are listed in `asahi-fixtures.json`; their
raw exports are retained. Preparation does not submit work to the ANE.

```sh
qwen35/.venv/bin/python experimental/replay_capture.py prepare \
  --kind training --kit UNRESTRICTED/training64
qwen35/.venv/bin/python experimental/replay_capture.py prepare \
  --kind qwen --kit UNRESTRICTED/qwen-silu-exp-rerun
qwen35/.venv/bin/python experimental/replay_capture.py prepare \
  --kind whisper --kit UNRESTRICTED/whisper-dense
# Execute only on native Asahi with the supported M1 DRM driver:
qwen35/.venv/bin/python experimental/replay_capture.py verify \
  --kit UNRESTRICTED/training64 --output NEW_ASAHI_RESULT.json
```

Repeat `verify` for each supported kit. The helper uses the existing loader
and its unchanged captured-output gates: relative L2 below 0.5%, plus
`allclose(rtol=0.01, atol=0.03)`. Each manifest identifies whether its reference
is an observed Mac ANE output or a CPU reference. Replay of kernel fixtures remains
separate from Linux full-model generation or training validation.

## Static HWX compile options

`experimental/probe_static_options.py` uses a session-owned clone of
`freedomtan/coreml_to_ane_hwx`, records its commit and patch, and routes all
temporary outputs into the new capture directory. It compares baseline
`h13`/`h13g`, missing/empty options files, and the system Espresso network's
actual properties at global and per-network option locations. The properties
are observed evidence; their placement as an options plist is a hypothesis.

`experimental/test_static_hwx.m` retains historical standalone-HWX rejection
controls and can verify 64-value MUL execution through daemon-compiled MIL.
It fills and retains FP16 inputs and checks constant `2 * 3` and nonuniform
signed fractional inputs. ANEForge's private E5RT path is the normal macOS
execution route. Our custom offline HWX exports are replayed through the
guarded Asahi loader; raw macOS loading is not a required follow-up.

The checker also accepts `--load-only` for a known-good system encoder
control. This mode records model loading and attributes, unloads a successful
model, and explicitly performs no inference or numerical validation. Its
`load_pass` status cannot be counted as a MUL numerical pass.
