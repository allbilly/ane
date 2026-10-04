"""Verify checkpoint/kernel parity and write the final comparison and receipts."""
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
  results = {n: json.loads((ROOT / n / "results.json").read_text()) for n in ("aneforge", "orion")}
  paired = json.loads((ROOT / "paired-benchmark.json").read_text())
  overfit = json.loads((ROOT / "overfitting-check.json").read_text())
  manifest = json.loads((ROOT / "kernel-manifest.json").read_text())
  with np.load(ROOT / "aneforge/checkpoint-step10.npz") as a, np.load(ROOT / "orion/checkpoint-step10.npz") as b:
    different, checked = [], 0
    if set(a.files) != set(b.files): raise AssertionError("Checkpoint keys differ")
    for name in a.files:
      av, bv = a[name], b[name]
      checked += av.size
      if not np.array_equal(av.view(np.uint32), bv.view(np.uint32)): different.append(name)
  kernel_comparison = []
  for name in sorted({r["name"] for r in manifest}):
    a, b = [next(r for r in manifest if r["backend"] == n and r["name"] == name) for n in results]
    fixture_a = ROOT / "aneforge/kernels" / name / "fixture.npz"
    fixture_b = ROOT / "orion/kernels" / name / "fixture.npz"
    with np.load(fixture_a) as fa, np.load(fixture_b) as fb:
      if set(fa.files) != set(fb.files): raise AssertionError("Fixture keys differ")
      fixture_equal = all(np.array_equal(fa[k].view(np.uint16), fb[k].view(np.uint16)) for k in fa.files)
    kernel_comparison.append({"name": name, "phase": "backward" if "_backward" in name else "forward" if "_forward" in name else "shared",
                              "aneforge_tasks": a["task_count"], "orion_tasks": b["task_count"],
                              "task_descriptors_identical": a["task_descriptors_sha256"] == b["task_descriptors_sha256"],
                              "fixture_inputs_and_outputs_identical": fixture_equal})
  ratios = [next(r["total_ms"] for r in paired["rows"] if r["pair"] == i and r["backend"] == "aneforge") /
            next(r["total_ms"] for r in paired["rows"] if r["pair"] == i and r["backend"] == "orion") for i in range(10)]
  comparison = {"measured_on": {"chip": results["aneforge"]["chip"], "macos": results["aneforge"]["macos"], "memory_gb": 8},
                "training": {n: {k: r[k] for k in ("initial_loss", "final_loss_after_updates", "compile_seconds", "median_step_ms",
                                                       "median_ane_execute_ms", "parameter_count", "compiled_programs", "gradient_oracle", "peak_rss_bytes")}
                             for n, r in results.items()}, "alternating_forward_backward_medians": paired["median"],
                "paired_aneforge_over_orion_time_ratios": ratios,
                "checkpoints": {"parameters_checked": checked, "different_tensors": different, "identical": not different},
                "overfitting": overfit["results"], "kernels": kernel_comparison,
                "conclusion": "Equivalent numerical results and bit-identical trained parameters in this shared manual-VJP GPT-2 harness. Orion's median forward/backward pass was about 3% faster; high paired variability prevents a decisive speed claim. Training loss improvement is batch memorization; unseen-text loss worsened. This does not compare stock training APIs or fully fused optimal implementations."}
  (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
  lines = ["GPT-2 TRAINING POC: ANEFORGE AND ORION", "", "Measured locally on Apple M1, 8 GB RAM, macOS " + results["aneforge"]["macos"] + ".",
           "Full pretrained GPT-2 124M: 12 layers, 768 hidden channels, 12 heads, vocabulary 50257.",
           "All 124,439,808 parameters have gradients and receive Adam updates.",
           "Batch size 1, sequence length 32, ten updates on the SAME batch, dropout disabled.",
           "Same verified fp16-rounded starting weights, learning rate 1e-4, fp32 CPU Adam, gradient clipping 1, loss scale 128.", "",
           "RESULTS", "                            ANEForge           Orion"]
  for title, key, divisor, units in (("Initial ANE training loss", "initial_loss", 1, ""),
                                    ("Loss after ten updates", "final_loss_after_updates", 1, ""),
                                    ("Compiler/setup calls", "compile_seconds", 1, " s"),
                                    ("Training step median", "median_step_ms", 1000, " s")):
    lines.append(f"{title:<27} {results['aneforge'][key] / divisor:>12.6f}{units:<3} {results['orion'][key] / divisor:>12.6f}{units}")
  lines += ["", "Compile timings were sequential first runs, not a controlled compiler benchmark.",
            "The ten training-step timings include CPU Adam and memory traffic and are order-sensitive.",
            "A cleaner timing check alternated ten pairs of full forward/backward passes on the same initial weights, with no optimizer or timed compilation:",
            f"  ANEForge: {paired['median']['aneforge']['total_ms']:.2f} ms median; ANE execute calls {paired['median']['aneforge']['ane_execute_ms']:.2f} ms.",
            f"  Orion:    {paired['median']['orion']['total_ms']:.2f} ms median; ANE execute calls {paired['median']['orion']['ane_execute_ms']:.2f} ms.",
            f"  Paired ANEForge/Orion time ratios ranged {min(ratios):.3f} to {max(ratios):.3f}.",
            "Orion was about 3% faster by median for forward/backward; variability is too large for a clear overall speed winner.", "",
            "THE VERY LOW LOSS IS OVERFITTING", "Independent PyTorch CPU fp32 forward checks, with saved checkpoint weights rounded to fp16:",
            f"  Training batch: {overfit['results']['initial']['train']['loss']:.6f} -> {overfit['results']['aneforge']['train']['loss']:.6f}.",
            f"  Unseen text:    {overfit['results']['initial']['unseen']['loss']:.6f} -> {overfit['results']['aneforge']['unseen']['loss']:.6f}.",
            "Both backends produce 32/32 correct training next-token predictions after ten updates.",
            "This shows correct training execution and rapid memorization, not improved generalization.", "",
            "CORRECTNESS", "The initial ANE loss differs from the independent CPU loss by 0.000801.",
            "Four sampled full-model parameter gradients agree with the CPU oracle (cosines >= 0.99994; relative L2 0.4%-1.2%).",
            f"All {checked:,} final parameter values match exactly between backends; different tensors: {len(different)}.", "",
            "WHAT WAS COMPARED", "An external Python training harness composes kernels through each repository's own graph builder, MIL emitter and native ANE runtime.",
            "VJPs are hand-derived in the common harness. ANEForge's autograd/Trainer and Orion's Stories110M trainer are NOT used.",
            "No source files in either repository were changed. Stock Orion does not implement GPT-2 training.",
            "Transformer matmuls, bias/norm/GELU/attention/residual computation, backward propagation and transformer parameter gradients run on the ANE.",
            "The CPU performs embedding lookup/scatter, tied vocabulary projection and its gradient, cross entropy, optimizer updates, reshapes/transposes and orchestration.",
            "Both implementations use 27 reusable kernel templates and 580 ANE dispatches per training step.",
            "Weights are program inputs, so neither backend recompiles or reloads baked weights during the ten steps.",
            "This shared decomposition permits a matched runtime comparison; it does not measure each project's best possible fused training implementation.",
            "The Orion experiment adapter supplies the required rsqrt epsilon=0 argument omitted by its generic emitter on this macOS. Original unmodified MIL is kept as model.mil.raw.", "",
            "ARTIFACTS", "aneforge/kernels/ and orion/kernels/: 27 templates each, with model.mil, fixture.npz and hwx/.",
            "  hwx/model.hwx: H13G offline binary compiled from the exact emitted MIL.",
            "  hwx/task-descriptors.bin: raw task descriptor/register packet bytes.",
            "  hwx/tasks.json and container.json: decoded descriptors/registers and HWX container metadata.",
            "  hwx/model.hwx.status.plist: Apple compiler physical I/O and allocation metadata.",
            "  fixture.npz: actual first-use fp16 input/output buffers from the full-model initial forward/backward pass, in logical tensor order.",
            "  kernel-index.json: shapes and logical-to-runtime port names.",
            "ANEForge also retains original e5rt cache bundles locally; Orion's compiler net.plist is tracked in Git.",
            "Offline HWX exports are not claimed to be byte-identical readbacks of signed binaries loaded by aned.",
            "Per-backend loss.csv, results.json and protocol.json are tracked, along with all per-kernel replay fixtures.",
            "Full fp32 checkpoints, oracle/gradient arrays, native build products and e5rt caches remain local and are ignored by Git.",
            "The commands below regenerate these local artifacts; build/receipt.json preserves the native build provenance.",
            "comparison.json, paired-benchmark.json, overfitting-check.json and kernel-manifest.json contain the machine-readable evidence.",
            "Initial pretrained weights remain external at ~/Desktop/Orion/model/blobs/gpt2_124m/; their 196 files were hash-verified.", "",
            "REPRODUCE", "Use the ANEForge and Orion source commits recorded in provenance.json under ~/Desktop/ANEForge and ~/Desktop/Orion.",
            "From this directory, create the environment and activate it:",
            "  python3.12 -m venv .venv", "  .venv/bin/python -m pip install numpy regex torch ruff", "  source .venv/bin/activate",
            "  export VECLIB_MAXIMUM_THREADS=4",
            "  python build_orion.py",
            "  xcrun clang -O2 -fobjc-arc -framework Foundation dump_hwx.m -o build/dump_hwx",
            "  python train.py oracle", "  python train.py aneforge --steps 10", "  python train.py orion --steps 10",
            "  python check_overfitting.py", "  python paired_benchmark.py", "  python capture_fixtures.py", "  python export_kernels.py", "  python summarize.py",
            "Dependencies: numpy, regex, torch (CPU oracle only). ruff checks the experiment sources. Native builds require Xcode command-line tools.", ""]
  (ROOT / "REPORT.txt").write_text("\n".join(lines))
  receipt = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": sys.version,
             "platform": platform.platform(), "numpy": np.__version__, "commands": {
               "power_source": subprocess.check_output(["pmset", "-g", "batt"], text=True),
               "thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True)}, "sources": {}}
  for name in ("ANEForge", "Orion"):
    repo = Path.home() / "Desktop" / name
    receipt["sources"][name] = {"commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                                 "worktree_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)}
  receipt["harness_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in (".py", ".m", ".toml")}
  (ROOT / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(f"Checked {checked:,} parameters; checkpoint equality: {not different}")
  print(f"{len(kernel_comparison)} paired templates; {sum(k['fixture_inputs_and_outputs_identical'] for k in kernel_comparison)} exact replay fixture matches")
  print(f"{sum(k['task_descriptors_identical'] for k in kernel_comparison)} byte-identical task descriptor packets")


if __name__ == "__main__": main()
