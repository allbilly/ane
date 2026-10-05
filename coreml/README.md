# Core ML Qwen3.5 porting notes

This folder records Core ML graph-conversion findings that may inform future
ANE work. It is a reference for the direct-register Linux runtime in this
repository; it does not add Core ML, Espresso, ANEC, or `.mlmodelc` support to
that runtime.

## Why keep this reference

The local [Qwen3.5-0.8B-M implementation](../qwen35/README.md) already has the
hybrid decoder and runs its linear projections on the Linux M1 ANE. Its
recurrent updates and full attention remain on the CPU. The upstream
[ANEMLL Qwen3.5 port](https://github.com/shipstuff/anemll-qwen35/tree/9eb0a09)
documents how a Core ML graph can keep recurrent state explicit and place
whole four-layer super-blocks on ANE. That provides useful compiler and graph
design evidence, though the Core ML artifacts and measurements do not run in
our Linux register-programming path.

The upstream snapshot reviewed here is commit `9eb0a09` (2026-04-13). Its
results are project-reported measurements on a Mac mini M4 Pro. Treat them as
Core ML evidence and optimization hypotheses, not Linux M1 benchmarks or
independent validation.

## State must cross the graph boundary

For a recurrent Qwen3.5 linear-attention layer, expose both state tensors as
graph inputs and outputs:

- GatedDeltaNet state: `[B, value_heads, value_head_dim, key_head_dim]`.
- Depthwise-convolution history: `[B, kernel_size - 1, conv_channels]`.
- Full-attention K/V cache: explicit input/output tensors when the graph owns
  cached attention.

The upstream port found that creating GatedDeltaNet state inside the traced
graph (for example, an internal `torch.zeros` reached through `cache=None`)
could make Core ML place the whole super-block on CPU. Supplying even the
initial zero state as an input avoided that fallback. This is a Core ML
placement behavior; it does not imply that the direct-register driver has the
same fallback rule.

Sizes vary by checkpoint. The upstream 0.8B configuration uses 16 value
heads, 128-wide key/value heads, and 6 `DDD A` super-blocks. Its 9B model uses
32 value heads, 128-wide key/value heads, and 8 super-blocks. The local Mirai
0.8B checkpoint is also a 24-layer `DDD A` model; use its own config and
weights as the authority for tensor dimensions and normalization conventions.

## Graph forms that converted well

- Express the per-token delta-rule update with elementwise multiplies and
  reductions. In the upstream probe, replacing `(state * k).sum(...)` with a
  degenerate matrix multiply lowered poorly and worsened parity.
- Keep the decode recurrence as a static one-token graph. Tracing a long
  prefill by unrolling the recurrence duplicates the update body for every
  input token and can make the MIL graph very large. The upstream plan treats
  CPU/GPU prefill or a scan-style graph as alternatives to an enormous
  unrolled ANE graph.
- Build the causal attention mask explicitly. PyTorch's `is_causal=True` and
  MLX's `mask=None` do not mean the same thing.
- Use `reshape` instead of `view_as` in traced conversion code; the upstream
  notes report that `view_as` was unsupported by its `coremltools` path.
- Keep the source model in FP32 while tracing `layer_norm`, then request FP16
  compute during conversion. Calling `.half()` first caused a conversion
  problem in the upstream setup.
- Use static slices/concats for partial RoPE and verify the exact rotated
  dimensions against the checkpoint config.

These are conversion findings, not guarantees for other Core ML versions,
model shapes, or the Linux driver.

## Precision and placement findings

The upstream port reports that global FP32 compute kept operations on CPU,
while pinning selected operations to FP32 either failed to improve accuracy or
caused placement/NaN problems. Its guidance is to inspect actual per-op device
placement after conversion rather than infer it from the source graph.

The port also reports a checkpoint-specific normalization convention:
some Qwen3.5 norm weights are stored as deltas from one and need a `+1` during
sanitization, while the GatedDeltaNet output norm is excluded. Apply this only
when the source checkpoint's format requires it; the local Mirai loader has
its own validated parameter mapping.

## Super-blocks and context length

The upstream project groups four decoder layers (`DDD A`) into a Core ML
super-block and measures both per-super-block execution and a more fused
single-model path. It reports that fusion improves 0.8B super-block latency
by 39% and 9B latency by 21%. The stated mechanism is fewer host-side
`predict()` boundaries and more scope for Core ML/ANE graph optimization.

It also reports 1.32× higher per-super-block throughput at context 256 than
1024 (2.41 vs. 3.19 ms per super-block, about 24% lower latency), with little
additional gain at 512 and above. This is useful only when the workload fits
the shorter context. The
local Linux path has different submission boundaries and currently keeps
attention on CPU, so neither the fusion gains nor context sweep should be
assumed to transfer. Re-measure them if a compiled graph path is built.

## What to reuse in this repository

1. Preserve recurrent, convolution, and K/V cache state at explicit graph
   boundaries in any future compiler-backed path.
2. Treat static decode and long prefill as separate graph shapes.
3. Record compiler placement and fallback per operation, alongside numerical
   parity; a successful conversion alone does not establish ANE execution.
4. Keep Core ML fusion/context results labeled by runtime, chip, model, and
   context limit. Do not compare them directly with per-projection Linux ANE
   submissions.

## Sources

- [Upstream README and results](https://github.com/shipstuff/anemll-qwen35/blob/9eb0a09/README.md)
- [Upstream Qwen3.5 architecture and conversion plan](https://github.com/shipstuff/anemll-qwen35/blob/9eb0a09/PORT_PLAN.md)
- [Upstream current status](https://github.com/shipstuff/anemll-qwen35/blob/9eb0a09/STATUS.md)
- [Upstream 0.8B fusion/context findings](https://github.com/shipstuff/anemll-qwen35/blob/9eb0a09/notes/small_model_0_8b.md)
- [Local Linux Qwen3.5 implementation and measurements](../qwen35/README.md)
