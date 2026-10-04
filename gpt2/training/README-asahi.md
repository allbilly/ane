# GPT-2 training on Asahi Linux

`asahi.py` runs the existing GPT-2 124M training harness through the Linux
ANE DRM driver. It reuses the 27 captured H13G kernel templates from ANEForge
or Orion. Matrix weights are runtime inputs, so Adam updates are consumed by
the next forward pass without recompilation. No macOS runtime or compiler is
needed on the Linux machine.

This supports **base M1 (T8103), batch size 1, sequence length 32**. Other
shapes require new compiled templates. The transformer forward pass,
backward pass and parameter gradients run on ANE. Embedding lookup/scatter,
the vocabulary projection, cross entropy and fp32 Adam run on CPU.

## Run

Install and load [the runtime overlay and KMD](../../kmod/README.md), then
prepare the existing inference Python environment and cached GPT-2 weights:

```bash
gpt2/first-run.sh setup
OPENBLAS_NUM_THREADS=1 gpt2/.venv/bin/python gpt2/training/asahi.py verify
OPENBLAS_NUM_THREADS=1 gpt2/.venv/bin/python gpt2/training/asahi.py train \
    --steps 10 --checkpoint
```

The default kernel source is `aneforge`; pass `--kernel-source orion` to use
the Orion captures. `--weights` accepts the verified original Hugging Face
checkpoint or Orion BLOBFILE directory, using the same cache discovery as
inference. The training model starts from fp16-rounded original weights and
keeps its updated parameters and Adam state in fp32.

Every run checks the source HWX hashes, task chains, buffer assignments and
I/O strides, then compares every template against its saved macOS fixture.
Training also checks the initial full-model loss and gradient norm against
the captured macOS run, requires finite gradients for all parameters and
580 ANE submissions per update, and verifies that the final loss decreased.
These checks use captured references; they do not rerun the PyTorch oracle.

Results go to the ignored `gpt2/training/asahi-output/` directory:

- `results.json`: fixture errors, losses, dispatch counts and phase timings.
- `loss.csv`: each training step, including CPU optimizer time.
- `checkpoint-step10.npz`: all 124,439,808 updated parameters, when
  `--steps 10 --checkpoint` is selected. The filename follows the step count.

Use `--output /path/to/run` to retain separate runs. The default is three
updates when `--steps` is omitted. Checkpoints are about 475 MiB. Loading a
trained checkpoint into the inference package requires adapting its weight
verification and coefficient packing; this command does not replace the
original cached inference checkpoint.

## Hardware run on 2026-10-04

Base M1 MacBook Air, Asahi Linux `7.1.13+`, using the runtime device tree
overlay and `/dev/accel/accel0`:

| Check | Result |
| --- | --- |
| Inference reference kernels and generation parity | All 49 kernels passed |
| ANEForge training fixtures | All 27 outputs matched exactly |
| Orion training fixtures | All 27 outputs matched exactly |
| Full-model training | Ten Adam updates, 580 ANE dispatches per update |
| Loss | 4.42788029 → 0.00020708 |
| Initial gradient norm | 41.38332; macOS reference 41.38304 |
| Saved checkpoint | Reloaded on ANE; reproduced the final loss |
| Independent CPU checkpoint evaluation | Loss 0.00020755; 32/32 correct next tokens |
| Median complete update | 1.795 s, including CPU Adam and data transfers |
| Median time inside ANE submission ioctls | 35.44 ms per update |
| Peak process RSS | 2.31 GiB |

The run used a fixed 32-token batch without dropout. Its loss reduction shows
that forward computation, gradients and parameter updates execute; it
measures memorization of this batch. It does not establish improved language
model generalization. The kernel logs contained no new faults or unhandled
interrupts during inference or training.

Local validation files are under `kmod/test-output/gpt2-training/`, with
fixture logs, inference results and the checkpoint reload check beside it.
`gpt2-checkpoint-cpu.json` records a separate NumPy fp32 forward pass with
fp16-rounded checkpoint parameters and no ANE calls.
