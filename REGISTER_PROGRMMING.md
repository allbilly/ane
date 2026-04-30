# 11. (⚠️ AI slop) Register Analysis

A structured summary of register differences across all ANE operations.

Add vs Relu (verified empirically, see examples/add.py → examples/relu_from_add.py)

### Data path difference
| | add | relu (raw firmware) | relu (L2-style layout) |
|---|---|---|---|
| Source | TileDMA (two inputs via SrcDMAConfig=0x33881) | TileDMA (single input via SrcDMAConfig=0x33881) | TileDMA via L2 bank stream header (L2Cfg=0x6c013800) |
| Compute | PE/NE ALU (PECfg=0x80000, scales non-zero) | Conv pipeline only (PECfg=0, all scales=0) | Conv pipeline only (PECfg=0) |
| Destination | TileDMA | TileDMA (L2 result staging) | TileDMA |
| Firmware | PE-based | Conv pipeline (relu.hwx) | Conv pipeline (built from segments) |

### Registers that changed (BTSP_BUF offsets, little-endian u32)

**Common section** (all must change together for relu C=1 single-channel config):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x128 | InDim | 0x00010001 | 0x0001004d | Input dimensions |
| 0x130 | ChCfg | 0x2a | 0x22 | Channel config |
| 0x134 | Cin | 0x40 (64) | 0x01 (1) | Input channels |
| 0x138 | Cout | 0x40 (64) | 0x01 (1) | Output channels |
| 0x13c | OutDim | 0x00010001 | 0x0001004d | Output dimensions |
| 0x14c | GroupConvCfg | 0x10001 | 0x14001 | Group conv |
| 0x154 | pad3 | 4 | 0 | Mode select |
| 0x15c | Cfg | 0x33 | 0x04010101 | Config flags |
| 0x160 | TaskInfo | 0 | 0x00100000 | Task info |

**Data source — raw firmware** (relu.hwx uses TileDMA source, same as add; `relu_l2.py` uses an alternative L2-based register layout):
| Offset | Register | add | relu (raw fw) | relu (L2-style) | Notes |
|--------|----------|-----|------|------|-------|
| 0x16c | SrcDMAConfig | 0x33881 | **0x33881** | **0** | Raw fw: TileDMA on; L2-style: off |
| 0x170 | Srcpad0 | 0x33880 | **0x8880** | **0x00500172** | Repurposed in L2-style |
| 0x178 | SrcRowStride | 0x40 | **0xc0** | **0xa0** | Stale in L2-style |
| 0x17c | SrcPlaneStride | 0x40 | **0xc0** | **0xa0** | |
| 0x180 | SrcDepthStride | 0x1000 | **0xc0** | **0xa0** | |
| 0x184 | SrcGroupStride | 0 | 0 | **0xa0** | |
| 0x1a4 | SrcFmt | 0x01002031 | **0x01002031** | **0** | |
| 0x1e0 | L2Cfg | 0 | 0 | **0x6c013800** | L2-style: stream header→TileDMA Src |
| 0x1e4 | SourceCfg | 0x01500172 | 0x00500172 | **0x33881** | |
| 0x1e8 | SourceBase | 0 | 0 | **0x8880** | |
| 0x1f0 | SourceRowStride | 0x420 | 0xa0 | **0xc0** | |

**L2 result staging** (needed by conv pipeline in raw firmware):
| Offset | Register | add | relu (raw fw) | relu (L2-style) | Notes |
|--------|----------|-----|------|------|-------|
| 0x210 | ResultCfg | 0x0050017a | **0x0050017a** | **0** | Raw fw: L2 result enabled |
| 0x214 | ResultBase | 0x860 | **0xa0** | **0** | |
| 0x21c | ConvResultRowStride | 0 | **0** | **0x01002031** | Repurposed in L2-style |

**PE → disabled** (relu runs in conv pipeline, not PE):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x22c | PECfg | 0x80000 | **0** | **Disable PE** |
| 0x230 | BiasScale | 0x3c000000 | **0** | |
| 0x234 | PreScale | 0x3c000000 | **0** | |
| 0x238 | FinalScale | 0x3f800000 | **0** | |

**Destination** (TileDMA, stride changed for C=1):
| Offset | Register | add | relu | Notes |
|--------|----------|-----|------|-------|
| 0x260 | DstRowStride | 0x40 | **0xc0** | 192 bytes |
| 0x264 | DstPlaneStride | 0x40 | **0xc0** | |
| 0x268 | DstDepthStride | 0x1000 | **0xc0** | |
| 0x270 | DstFmt | 0x01002031 | **0x01302031** | Bit 20 set |

**BTSP program code** (instruction bytes changed):
| Offset | add | relu | Notes |
|--------|-----|------|-------|
| 0x01b | 66 49 02 00 | 25 40 02 01 | Program header entry point |
| 0x1b5 | 00 | 88 | Program instruction |
| 0x1b7 | 00 | 0c | |
| 0x1c9-0x1cc | 00 | c8 10 80 | |
| 0x1d0 | 00 | 0c | |
| 0x1d2 | 00 | 11 | |
| 0x1dd | 48 | 3c | |
| 0x225 | 00 | 01 | |

**Key insight**: Unlike add→mul (same BTSP program, 2 register changes), **add→relu requires changing the BTSP firmware program itself + ~25 critical registers**. Relu uses a conv pipeline (not PE) but still sources data via TileDMA — the L2 is used as intermediate result staging, not as input source.

> **Note**: The raw-hex-offset experiment `experimental/test_regs_one_by_one.py` is superseded by the structured register analysis in [§13](#13-structured-register-analysis) below. Results are kept here for reference.

### Experimental results (one-register-at-a-time revert from working relu config)

Each test: start from full relu config, revert ONE register group to add value, check if relu still works.

**Don't-care registers** (relu still works with add's value):
| Register | offset | add | relu | Verdict |
|----------|--------|-----|------|---------|
| `ChCfg`  | 0x130  | 0x2a | 0x22 | **relu works either way** |
| `Cin`    | 0x134  | 64   | 1    | **relu works either way** |

**Wrong output but no ANE crash** (size/dimension mismatch):
| Register | offset | add → relu | Verdict |
|----------|--------|------------|---------|
| `InDim`  | 0x128  | 0x10001 → 0x1004d | FAIL (zeros) |
| `OutDim` | 0x13c  | 0x10001 → 0x1004d | FAIL (zeros) |

**Critical registers** (ANE HANGs when reverted to add value):
`Cout` (0x138), `GroupConvCfg` (0x14c), `pad3` (0x154), `Cfg` (0x15c), `TaskInfo` (0x160),
`L2Cfg` (0x1e0), `SourceCfg` (0x1e4), `SourceBase` (0x1e8), `SourceChStride` (0x1ec), `SourceRowStride` (0x1f0),
`L2pad0-6` (0x1f4-0x20c), `ResultCfg` (0x210), `ResultBase` (0x214), `ConvResultRowStride` (0x21c),
`PECfg` (0x22c), `BiasScale` (0x230), `PreScale` (0x234), `FinalScale` (0x238),
`SrcDMAConfig` (0x16c), `Srcpad0` (0x170), `SrcRow/Plane/Depth/GroupStride` (0x178-0x184),
`Srcpad2-4` (0x18c-0x194), `SrcFmt` (0x1a4), `Srcpad8` (0x1a8),
`DstRowStride` (0x260), `DstPlaneStride` (0x264), `DstDepthStride` (0x268), `DstFmt` (0x270)

### Why add→relu is different from add→mul

| Aspect | add→mul | add→relu |
|--------|---------|----------|
| BTSP program | **Same** firmware | **Different** firmware (program code bytes changed) |
| Data path | TileDMA→PE→TileDMA | **TileDMA→Conv→TileDMA** |
| PE used? | Yes (bit 2 toggles add↔mul) | **No (PECfg=0 disables PE entirely)** |
| Number of inputs | 2 | 1 |
| L2 role | — | Internal pipeline staging (ResultCfg) |
| Minimum register changes | **2** (PECfg, MACCfg) | **~25** plus BTSP program code |

**Note**: The `relu_from_add.py` example uses an alternative register layout where TileDMA source is configured via the L2 bank's stream header (`L2Cfg=0x6c013800`) instead of the direct TileDMA Src registers. Both approaches (`relu_dma.py` raw firmware, `relu_l2.py` L2-style) work standalone — no L2 priming needed.

Relu vs Conv (verified empirically, see examples/relu_from_add.py → examples/conv_from_relu.py)

### Data path comparison

| | relu | conv |
|---|---|---|
| Source | TileDMA (SrcDMAConfig=0x33881) | TileDMA (SrcDMAConfig=0x33881) |
| Kernel weights | None (KernelCfg=0, MACCfg=0) | **3×3 depthwise** (KernelCfg=0x82, MACCfg=0x101c00) |
| Compute | Conv pipeline, no weights (Cfg=0x04010101) | Conv pipeline with weights (Cfg=0x04144405) |
| Destination | TileDMA (DstDMAConfig=0x040000c1) | TileDMA (DstDMAConfig=0xc1) |
| L2 result staging | Enabled (ResultCfg=0x50017a) | Enabled (ResultCfg=0x500172) |
| Channels | 1 | 3 |

### Register differences (relu → conv)

**Same values** (no change needed):
`ChCfg`=0x22, `ConvCfg`=0x5000a021, `pad0`/`pad1`/`pad2`/`pad4`=1/1/0x2041/0, `TileCfg`=1, `DPE`=0,
`PECfg`=0, `BiasScale`=0, `PreScale`=0, `FinalScale`=0, `MatrixVectorBias`=0, `AccBias`=0,
`DstFmt`=0x01302031, `L2pad2-6`=0, `Srcpad2-4`=0, `Srcpad8`=0, `DstBaseAddr`=0, `DstGroupStride`=0, `DstDepthStride`=0xc0

**Changed for multi-channel (C=1→3)**:
| Offset | Register | relu | conv | Notes |
|--------|----------|------|------|-------|
| 0x128 | InDim | 0x1004d | 0x10001 | Input dimensions |
| 0x13c | OutDim | 0x1004d | 0x10001 | Output dimensions |
| 0x134 | Cin | 1 | 3 | **Input channels 1→3** |
| 0x138 | Cout | 1 | 3 | **Output channels 1→3** |

**Data source (both use TileDMA; relu L2-style → conv direct)**:
| Offset | Register | relu (raw fw) | relu (L2-style) | conv | Notes |
|--------|----------|------|------|------|-------|
| 0x16c | SrcDMAConfig | 0x33881 | 0 | 0x33881 | Raw fw uses TileDMA directly |
| 0x170 | Srcpad0 | 0x8880 | 0x00500172 | 0x8880 | |
| 0x178 | SrcRowStride | 0xc0 | 0xa0 | 0x40 | |
| 0x1a4 | SrcFmt | 0x01002031 | 0 | 0x01002031 | |
| 0x1e0 | L2Cfg | 0 | 0x6c013800 | 0 | L2-style uses stream header |
| 0x1e4 | SourceCfg | 0x500172 | 0x33881 | 0x500172 | |
| 0x1e8 | SourceBase | 0 | 0x8880 | 0 | |
| 0x1ec | SourceChannelStride | 0xa0 | 0 | 0x10 | |
| 0x1f0 | SourceRowStride | 0xa0 | 0xc0 | 0x30 | |
| 0x1f4 | L2pad0 | 0xa0 | 0xc0 | 0x30 | |
| 0x1f8 | L2pad1 | 0xa0 | 0xc0 | 0x30 | |

**L2 result path**:
| Offset | Register | relu (raw fw) | relu (L2-style) | conv | Notes |
|--------|----------|------|------|------|-------|
| 0x210 | ResultCfg | 0x50017a | 0 | 0x500172 | Raw fw: L2 result staging active |
| 0x214 | ResultBase | 0xa0 | 0 | 0x30 | |
| 0x218 | ConvResultChannelStride | 0 | 0 | 0x10 | |
| 0x21c | ConvResultRowStride | 0 | 0x01002031 | 0x30 | |

**NE kernel config (only conv uses)**:
| Offset | Register | relu | conv | Notes |
|--------|----------|------|------|-------|
| 0x240 | KernelCfg | 0 | **0x82** | **Kernel 3×3 depthwise** |
| 0x244 | MACCfg | 0 | **0x101c00** | **MAC kernel mode** |
| 0x250 | PostScale | 0 | **0x3c00** | Post-processing scale (=1.0 fp16) |

**Destination config**:
| Offset | Register | relu | conv | Notes |
|--------|----------|------|------|-------|
| 0x258 | DstDMAConfig | 0x040000c1 | 0xc1 | Bit 26 cleared (no L2 flush?) |
| 0x260 | DstRowStride | 0xc0 | 0x40 | Row stride (64 bytes) |
| 0x264 | DstPlaneStride | 0xc0 | 0x40 | Plane stride (64 bytes) |

**BTSP program**: Different firmware (conv loads from `hwx/tinygrad/conv.hwx`, relu has program embedded in hex segments)


### Key Register Map (H13 architecture)

| Script Offset | H13 Bank | Register | Description |
|---|---|---|---|
| 0x128 | Common+0x00 | InDim | Input dimensions (W|H in low/high 16 bits) |
| 0x12c | Common+0x04 | pad0 | Padding mode |
| 0x130 | Common+0x08 | ChCfg | Channel config (in/out format) |
| 0x134 | Common+0x0c | Cin | Input channels |
| 0x138 | Common+0x10 | Cout | Output channels |
| 0x13c | Common+0x14 | OutDim | Output dimensions |
| 0x144 | Common+0x1c | ConvCfg | Conv kernel config (W|H) |
| 0x14c | Common+0x24 | GroupConvCfg | Group conv / depthwise config |
| 0x15c | Common+0x34 | Cfg | General config (activation modes) |
| 0x160 | Common+0x38 | TaskInfo | Task flags |
| 0x16c | TileDMA Src+0x00 | SrcDMAConfig | TileDMA source config |
| 0x1e0 | L2+0x00 | L2Cfg | L2 cache config |
| 0x1e4 | L2+0x04 | SourceCfg | L2 source config (bit 24 = dual input) |
| 0x210 | L2+0x30 | ResultCfg | L2 result config |
| 0x22c | PE+0x00 | PECfg | PE config (bit 2 = mul mode) |
| 0x230 | PE+0x04 | BiasScale | Bias scaling factor |
| 0x240 | NE+0x00 | KernelCfg | NE kernel config |
| 0x244 | NE+0x04 | MACCfg | MAC / ALU mode |
| 0x258 | TileDMA Dst+0x00 | DstDMAConfig | TileDMA dest config |

### Cross-Operation Register Comparison

| Address | Register | Add | Mul | Relu | Conv | GeMM | Concat | Sigmoid |
|---------|----------|-----|-----|------|------|------|--------|---------|
| 0x130 | ChCfg | 0x2a | 0x2a | 0x22 | 0x22 | 0x22 | 0x22 | 0x22 |
| 0x134 | Cin | 64 | 64 | **1** | **3** | **512** | **16** | **1** |
| 0x138 | Cout | 64 | 64 | **1** | **3** | **512** | **16** | **1** |
| 0x144 | ConvCfg | 0x5000a021 | 0x5000a021 | 0x5000a021 | 0x5000a021 | **0x5000b421** | 0x5000a021 | 0x5000a021 |
| 0x14c | GroupConvCfg | 0x10001 | 0x10001 | **0x14001** | 0x10001 | 0x10001 | **0x14001** | **0x14001** |
| 0x15c | Cfg | 0x33 | 0x33 | **0x04010101** | **0x04144405** | **0x00244405** | **0x04211101** | **0x04010101** |
| 0x160 | TaskInfo | 0 | 0 | **0x100000** | **0x100000** | **0x100000** | **0x100000** | **0x100000** |
| 0x16c | SrcDMAConfig | 0x33881 | 0x33881 | **0x33881**ᵃ | **0x33881** | **0x33881** | **0x33881** | **0x33881** |
| 0x1e0 | L2Cfg | 0 | 0 | **0**ᵃ | 0 | 0 | 0 | 0 |
| 0x1e4 | SourceCfg | 0x01500172 | 0x01500172 | **0x00500172**ᵃ | 0x500172 | 0x500172 | **0x172** | 0x500172 |
| 0x210 | ResultCfg | 0x0050017a | 0x0050017a | **0x0050017a**ᵃ | **0x500172** | **0x500172** | **0x17a** | **0x50017a** |
| 0x22c | PECfg | **0x80000** | **0x80004** | **0** | **0** | **0** | **0** | **0** |
| 0x240 | KernelCfg | 0 | 0 | 0ᵃ | **0x82** | **0x82** | **0x82** | **0x82** |
| 0x244 | MACCfg | 0 | **0x30** | 0ᵃ | **0x101c00** | **0x101c00** | **0x101c00** | **0x101c00** |
| 0x258 | DstDMAConfig | 0x040000c1 | 0x040000c1 | **0x040000c1**ᵃ | **0xc1** | **0xc1** | **0xc1** | **0xc1** |
| 0x270 | DstFmt | 0x01002031 | 0x01002031 | **0x01302031**ᵃ | **0x01302031** | **0x01302031** | 0x01002031 | **0x01302031** |

ᵃ Relu values from raw relu.hwx firmware. An alternative L2-style register layout (`relu_l2.py`) uses `L2Cfg=0x6c013800`, `SrcDMAConfig=0`, `ResultCfg=0`.


### Operation Grouping

Operations are grouped by **firmware program** into families. Within a family, ops share the same BTSP firmware and can be converted with few register changes. Cross-family conversion requires different firmware.

| Family | Pipeline | Firmware | Ops | Example Files | Switch Register | Reg Changes | Notes |
|--------|----------|----------|-----|---------------|-----------------|-------------|-------|
| **1 (PE)** | PE elementwise | Bare-metal (no KDMA), `66 49 02 00` | add, mul, max, min, sq | `add.py`, `elementwise.py` | `PECfg[3:2]` (op_mode) | 1 | `add↔mul` also needs `MACCfg[5:4]=0x30` |
| **2a (NE NL)** | Conv+NE nonlinear | KDMA `25 40 02 01` | identity, relu, sigmoid | `relu.py`, `sigmoid.py` | `MACCfg[17:16]` (non_linear_mode) | 1 | sigmoid needs KDMA LUT data |
| **2b (NE comp)** | Conv+NE compute | KDMA, separate firmware | conv, gemm | `conv.py`, `gemm.py` | — | N/A | Different BTSP programs; W8 has compute flags |
| **3 (pass)** | L2-cached conv | Embedded bytes (no KDMA) | relu (L2 style) | `relu_l2.py` | — | N/A | Different arch; no PE/NE needed |
| **4 (tile)** | Multi-tile pass | KDMA | concat | `concat.py` | — | N/A | 2-tile chaining, NE pass-through mode |

#### Family 1 — PE Elementwise (`add`, `mul`, `max`, `min`, `sq`)

Same bare-metal firmware (no KDMA context). PE pipeline (`Cfg=0x33`), NE disabled. Dual-source DMA (W8=`0x00086C66`):

| Op | PECfg | MACCfg | Data Path | Example |
|----|-------|--------|-----------|---------|
| add (default) | `(2<<18)` | 0 | TileDMA→PE→TileDMA | `python add.py` |
| mul | `(2<<18) | (1<<2)` | `(1<<4)|(1<<5)` = 0x30 | TileDMA→PE→TileDMA | `python add.py mul` |
| max | `(2<<18) | (2<<2)` | 0 | TileDMA→PE→TileDMA | `python elementwise.py max` |
| min | `(2<<18) | (3<<2)` | 0 | TileDMA→PE→TileDMA | `python elementwise.py min` |
| sq | `(2<<18) | (4<<2)` | 0 | TileDMA→PE→TileDMA | `python elementwise.py sq` |

Switch: `PECfg[3:2]` (0=add, 1=mul, 2=max, 3=min, 4=sq). Add↔mul also needs `MACCfg[5:4]=0x30`.

#### Family 2a — Conv NE Nonlinear (`identity`, `relu`, `sigmoid`)

Same KDMA-loaded firmware (`25 40 02 01`). Conv pipeline, NE enabled. Single-source DMA (W8=`0x01240025`):

| Op | MACCfg | non_linear_mode | KernelCfg | Data Path | Example |
|----|--------|-----------------|-----------|-----------|---------|
| identity | `0x0010000c` | 0 | `0x80` | TileDMA→Conv+NE→TileDMA | `python relu.py` + MACCfg override |
| relu | `0x0011000c` | 1 | `0x80` | TileDMA→Conv+NE→TileDMA | `python relu.py` |
| sigmoid | `0x0012000c` | 2 | `0x80` | TileDMA→Conv+NE→TileDMA | `python sigmoid.py` |

Switch: `MACCfg[17:16]` (0=identity, 1=relu, 2=sigmoid). Sigmoid needs KDMA LUT data embedded. PE bypassed (all zeros).

#### Family 2b — Conv NE Compute (`conv`, `gemm`)

KDMA-loaded firmware with compute flags. NE enabled with `KernelCfg=0x82`, `MACCfg=0x00101c00`. W8 has compute flags (conv=bit25, gemm=bit26).

#### Standalone

- `relu_l2.py` — L2-cached source, different architecture entirely (no KDMA, no PE/NE)
- `concat.py` — Multi-tile chaining, NE in pass-through mode (same MACCfg as Family 2a identity)

**Detailed experiment results:** see `experimental/expt.md` for complete register classification (expt1: op conversion switches, expt2: bit-level sensitivity, expt3: minimal register set per op).

### Key Patterns

1. **PE vs Conv pipeline**: PECfg=0 disables PE, routing through conv pipeline. All elementwise (add/mul) use PE at 0x80000/0x80004; all others disable PE.

2. **TileDMA source**: All operations use TileDMA source (`SrcDMAConfig=0x33881` for direct, or `L2Cfg=0x6c013800` stream header for indirect). The conv pipeline internally uses L2 for result staging (`ResultCfg` non-zero).

3. **KDMA kernel weights**: KernelCfg=0x82 + MACCfg=0x101c00 = kernel loading pattern (Fmt=FLOAT16, Palettized=off). Relu and add/mul have no kernel. Conv/GeMM/Concat/Sigmoid all load KDMA weights. The weight data in `.ane` files has a 12-byte leading zero header before the first coefficient for channel 0.

4. **Cfg activation encoding**: The `Cfg` register encodes the processing mode:
   - 0x04010101 / 0x4010101 = relu/sigmoid (identity pass-through, C=1)
   - 0x04144405 = conv (1×1 pointwise, C=3)
   - 0x00244405 = gemm (matrix multiply, C=512)
   - 0x04211101 = concat (channel concatenation, C=16)

5. **DstFmt**: Models with output format change (relu/conv/gemm/sigmoid/concat) use 0x01302031 (bit 20 set). Add/mul use 0x01002031 (bit 20 clear). Bit 20 may indicate multi-channel output format variant.

### Family 2 Cross-Operation: sigmoid↔relu via MACCfg bits 16/17

Just as Family 1 (PE-based) has a **1-bit toggle** `PECfg[2]` to switch add↔mul, Family 2 (Conv pipeline) has a **complementary bit pair** `MACCfg[16:17]` to switch relu↔sigmoid:

| Register | Bits | Family 1 (PE) | Family 2 (Conv pipeline) |
|----------|------|---------------|--------------------------|
| `PECfg[2]` | 0x4 | add (0) ↔ mul (1) | — |
| `MACCfg[16]` | 0x10000 | — | relu mode |
| `MACCfg[17]` | 0x20000 | — | sigmoid mode |

Both sigmoid and relu share the same firmware program (entry `25400201`). The MACCfg register selects which processing the NE applies:

| MACCfg value | bit20 | bit17 | bit16 | bits2,3 | Mode |
|---|---|---|---|---|---|
| 0x0012000c | ✓ | ✓ | — | ✓ | Sigmoid (KDMA table lookup) |
| 0x0011000c | ✓ | — | ✓ | ✓ | Relu (pass-through, clamp negative) |
| 0x0002000c | — | ✓ | — | ✓ | Sigmoid-like (reduced precision) |
| 0x0000000c | — | — | — | ✓ | Pass-through (no activation) |
| 0x00000000 | — | — | — | — | **Broken** (garbage: 744.0) |

The relu firmware needs sigmoid KDMA kernel data appended to function as sigmoid. Neither Cfg nor any other register controls this toggle — only MACCfg.

**All experimental scripts** in `experimental/` document the complete test methodology.

### Operational Status

| Operation | anecc | hwx2py | Standalone py | Notes |
|-----------|-------|--------|---------------|-------|
| **Add** | ✓ | ✓ | ✓ `add.py` | PE-based, 2 inputs |
| **Mul** | ✓ | ✓ | ✓ `add.py mul` | Same firmware, 2 reg changes |
| **Relu** | ✓ | ✓ | ✓ `relu.py` (TileDMA src), `relu_l2.py` (L2 cache src), `relu_from_add.py` | Conv pipeline, TileDMA or L2 source, no L2 priming needed |
| **Conv** | ✓ | ✓ | ✓ `conv.py` | 1×1 pointwise, [12,12,12] ✓ |
| **Sigmoid** | ✓ | ✓ | ✓ `sigmoid_from_hwx.py` | sigmoid(3) ≈ 0.9526 ✓ |
| **GeMM** | ✓ | ✓ | ✓ `gemm.py` (inj weights) | Inj 0.5 weights → 128 ✓ |
| **Concat** | ✓ | ✓ | ✓ `concat.py` | 2 inputs, [2.0,...] ✓ |

# 12. Minimal Register Change Analysis

Systematic experimental results from `experimental/test_minimal_regs.py`, `experimental/test_sig_to_relu.py`, `experimental/test_family2_cross.py`.

## 12.1 Minimal Change Matrix

For each pair (base→target), the table shows how many register changes are truly needed vs how many register differences exist in the raw configs. "Needed" means reverting that register to the base value causes the output to change from target to base behavior.

### Same firmware (sigmoid/relu family, entry `25400201`)

| Pair | Total reg diffs | Needed | Don't-care | Minimal change |
|------|----|---------|------------|----------------|
| sigmoid→relu | 1 | 1 | 0 | `MACCfg`: 0x0012000c → 0x0011000c (bit17 off, bit16 on) |
| relu→sigmoid | 1 | 1 | 0 | `MACCfg`: 0x0011000c → 0x0012000c (bit16 off, bit17 on) + KDMA kernel |

### Cross-firmware (different BTSP programs)

| Pair | Base firmware entry | Result |
|------|-------------------|--------|
| conv→relu | `25400203` | **HANG** — different BTSP program |
| conv→sigmoid | `25400203` | **HANG** |
| sigmoid→conv | `25400201` | **HANG** |

The BTSP firmware program is hardcoded with specific data flow operations. Register overrides alone cannot bridge different firmware families.

## 12.2 MACCfg Bit Field (NE Engine Config)

Systematic bit-sweep on sigmoid firmware (input=3.0):

| MACCfg | bit20 | bit17 | bit16 | bits2,3 | Output | Mode |
|--------|-------|-------|-------|--------|--------|------|
| 0x0012000c | 1 | 1 | 0 | 1 | 0.9526 | Sigmoid (table lookup) |
| 0x0011000c | 1 | 0 | 1 | 1 | 3.0000 | Relu (pass-through) |
| 0x0010000c | 1 | 0 | 0 | 1 | 3.0000 | Pass-through (no activation) |
| 0x0002000c | 0 | 1 | 0 | 1 | 0.9995 | Partial sigmoid |
| 0x0001000c | 0 | 0 | 1 | 1 | 3.0000 | Pass-through |
| 0x0000000c | 0 | 0 | 0 | 1 | 3.0000 | Pass-through |
| 0x00000000 | 0 | 0 | 0 | 0 | 744.0 | **Broken** |

**Key findings:**
- `bit20 (0x100000)` = NE core enable (must be set for correct operation)
- `bits2,3 (0x0c)` = ALU/format mode (must be set)
- `bit17 (0x20000)` = sigmoid/KMDA table lookup mode
- `bit16 (0x10000)` = relu pass-through with negative clamping
- `MACCfg=0` breaks the NE engine entirely (garbage 744.0)

## 12.3 Cfg Register: Does NOT Control Op Mode

Cfg was tested with 5 different values on sigmoid firmware. All produced identical sigmoid output:

| Cfg value | Source op | Output | Verdict |
|-----------|-----------|--------|---------|
| 0x04010101 | relu/sigmoid | 0.9526 | Same |
| 0x04144405 | conv | 0.9526 | Same |
| 0x04211101 | concat | 0.9526 | Same |
| 0x00244405 | gemm | 0.9526 | Same |
| 0x00000000 | zero | 0.9526 | Same |

Cfg register controls activation post-processing parameters, not the fundamental op mode. The op mode is determined by MACCfg (within same firmware) or by the BTSP firmware program (cross-firmware).

## 12.4 Comparison: Family 1 vs Family 2 Mode Toggles

| Aspect | Family 1 (PE) | Family 2 (Conv pipeline) |
|--------|---------------|--------------------------|
| Same firmware ops | add ↔ mul | relu ↔ sigmoid |
| Toggle register | PECfg | MACCfg |
| Toggle bits | `PECfg[2]` (0x4) | `MACCfg[16:17]` (0x10000/0x20000) |
| Firmware entry | Same (`66 49 02 00`) | Same (`25 40 02 01`) |
| Other ops | — | conv/gemm/concat = different firmware |

# 13. Structured Register Analysis

Structured analysis using the fully-decoded register infrastructure from `examples/*.py`. Unlike the raw-hex experiments in earlier sections, this analysis uses the named register objects (`reg` class), stream headers, and `build_seg`/`pack_reg` helpers from the example files.

The experiment script `experimental/test_structured_regs.py` runs all 4 phases. Results below require ANE hardware (`/dev/accel/accel0`).

## 13.1 Cross-Operation Register Comparison

Register values extracted from each example's `BTSP_BUF`. Marked values differ from the baseline (add).

| Address | Register | add | relu | sigmoid |
|---------|----------|-----|------|---------|
| 0x128 | InDim | 0x00010001 | **0x0001004d** | **0x0001004d** |
| 0x12c | pad0 | 1 | 1 | 1 |
| 0x130 | ChCfg | 0x2a | **0x22** | **0x22** |
| 0x134 | Cin | 64 | **1** | **1** |
| 0x138 | Cout | 64 | **1** | **1** |
| 0x13c | OutDim | 0x00010001 | **0x0001004d** | **0x0001004d** |
| 0x140 | pad1 | 1 | 1 | 1 |
| 0x144 | ConvCfg | 0x5000a021 | 0x5000a021 | **0x5000a021** |
| 0x148 | pad2 | 0x2041 | 0x2041 | 0x2041 |
| 0x14c | GroupConvCfg | 0x10001 | **0x14001** | **0x14001** |
| 0x150 | TileCfg | 1 | 1 | 1 |
| 0x154 | pad3 | 4 | **0** | **0** |
| 0x158 | pad4 | 0 | 0 | 0 |
| 0x15c | Cfg | 0x33 | **0x04010101** | **0x04010101** |
| 0x160 | TaskInfo | 0 | **0x00100000** | **0x00100000** |
| 0x164 | DPE | 0 | 0 | 0 |
| 0x16c | SrcDMAConfig | 0x33881 | 0x33881 | 0x33881 |
| 0x170 | Srcpad0 | 0x33880 | **0x8880** | **0x8880** |
| 0x174 | SrcBaseAddr | 0 | 0 | 0 |
| 0x178 | SrcRowStride | 0x40 | **0xc0** | **0xc0** |
| 0x17c | SrcPlaneStride | 0x40 | **0xc0** | **0xc0** |
| 0x180 | SrcDepthStride | 0x1000 | **0xc0** | **0xc0** |
| 0x184 | SrcGroupStride | 0 | 0 | 0 |
| 0x188 | Srcpad1 | 0 | 0 | 0 |
| 0x18c | Srcpad2 | 0x40 | **0** | **0** |
| 0x190 | Srcpad3 | 0x40 | **0** | **0** |
| 0x194 | Srcpad4 | 0x1000 | **0** | **0** |
| 0x198 | Srcpad5 | 0 | 0 | 0 |
| 0x19c | Srcpad6 | 0 | 0 | 0 |
| 0x1a0 | Srcpad7 | 0 | 0 | 0 |
| 0x1a4 | SrcFmt | 0x01002031 | 0x01002031 | 0x01002031 |
| 0x1a8 | Srcpad8 | 0x2030 | **0** | **0** |
| 0x1e0 | L2Cfg | 0 | 0 | 0 |
| 0x1e4 | SourceCfg | 0x01500172 | **0x00500172** | **0x00500172** |
| 0x1e8 | SourceBase | 0 | 0 | 0 |
| 0x1ec | SourceChannelStride | 0x10 | **0xa0** | **0xa0** |
| 0x1f0 | SourceRowStride | 0x420 | **0xa0** | **0xa0** |
| 0x1f4 | L2pad0 | 0x400 | **0xa0** | **0xa0** |
| 0x1f8 | L2pad1 | 0x400 | **0xa0** | **0xa0** |
| 0x1fc | L2pad2 | 0x440 | **0** | **0** |
| 0x200 | L2pad3 | 0x10 | **0** | **0** |
| 0x204 | L2pad4 | 0x420 | **0** | **0** |
| 0x208 | L2pad5 | 0x400 | **0** | **0** |
| 0x20c | L2pad6 | 0x400 | **0** | **0** |
| 0x210 | ResultCfg | 0x0050017a | **0x0050017a** | **0x0050017a** |
| 0x214 | ResultBase | 0x860 | **0xa0** | **0xa0** |
| 0x218 | ConvResultChannelStride | 0 | 0 | 0 |
| 0x21c | ConvResultRowStride | 0 | 0 | 0 |
| 0x22c | PECfg | **0x80000** | **0** | **0** |
| 0x230 | BiasScale | **0x3c000000** | **0** | **0** |
| 0x234 | PreScale | **0x3c000000** | **0** | **0** |
| 0x238 | FinalScale | **0x3f800000** | **0** | **0** |
| 0x240 | KernelCfg | 0 | **0x80** | **0x80** |
| 0x244 | MACCfg | 0 | **0x0011000c** | **0x0012000c** |
| 0x248 | MatrixVectorBias | 0 | 0 | 0 |
| 0x24c | AccBias | 0 | 0 | 0 |
| 0x250 | PostScale | 0 | **0x3c00** | **0x3c00** |
| 0x258 | DstDMAConfig | 0x040000c1 | **0x040000c1** | **0x040000c1** |
| 0x25c | DstBaseAddr | 0 | 0 | 0 |
| 0x260 | DstRowStride | 0x40 | **0xc0** | **0xc0** |
| 0x264 | DstPlaneStride | 0x40 | **0xc0** | **0xc0** |
| 0x268 | DstDepthStride | 0x1000 | **0xc0** | **0xc0** |
| 0x26c | DstGroupStride | 0 | 0 | 0 |
| 0x270 | DstFmt | 0x01002031 | **0x01302031** | **0x01302031** |

## 13.2 Register Diff Counts by Pair

| | add | relu | sigmoid |
|---|-----|------|---------|
| **add** | — | 29 | 30 |
| **relu** | 29 | — | 1 |
| **sigmoid** | 30 | 1 | — |

**Same-firmware pair**: relu ↔ sigmoid (1 register diff: `MACCfg` — entry 0x25400201)
**Cross-firmware**: add ↔ relu/sigmoid (29-30 register diffs — different firmware programs)

## 13.3 Register-by-Register Live vs Don't-Care

For the relu ↔ sigmoid pair (same firmware, entry 0x25400201), each of the 1 differing registers tested:

| Register | RelU Value | Sigmoid Value | Revert Result |
|----------|-----------|---------------|---------------|
| MACCfg (0x244) | 0x0011000c | 0x0012000c | **NEEDED** — output reverts to base mode |

**Result**: `MACCfg` bits 16/17 are the sole toggle between relu and sigmoid. No other register changes needed.

## 13.4 MACCfg Comprehensive Bitfield Map

| MACCfg | Description | Output | Verdict |
|--------|-------------|--------|---------|
| 0x0012000c | Sigmoid (bit17, bit20) | 0.9526 | sigmoid |
| 0x0011000c | Relu (bit16, bit20) | 3.0000 | passthru |
| 0x0010000c | Passthru (bit20 only) | 3.0000 | passthru |
| 0x0002000c | Partial sig (bit17 only) | ~0.9995 | partial sigmoid |
| 0x0001000c | Relu-like (bit16 only) | 3.0000 | passthru |
| 0x0000000c | No activation (bits 2-3 only) | 3.0000 | passthru |
| 0x00000000 | All zero | ~744.0 | **BROKEN** |

**Bit map:**

| Bits | Mask | Function |
|------|------|----------|
| [1:0] | 0x3 | ALU thread/core select |
| [3:2] | 0xc | ALU format mode (elementwise fp16) |
| [5:4] | 0x30 | kernel_mode, bias_mode (add=0x00, mul=0x30) |
| [11:8] | 0xf00 | op_mode for PE-based ops (12=relu, 13=exp) |
| [16] | 0x10000 | Relu pass-through (clamp negative to 0) |
| [17] | 0x20000 | Sigmoid table lookup (requires KDMA kernel data) |
| [20] | 0x100000 | NE core enable (**MUST be set**; without it: broken output) |

## 13.5 Cfg Register Analysis

Cfg register was tested with all known values on sigmoid firmware. All produced identical sigmoid output:

| Cfg | Description | Output | Verdict |
|-----|-------------|--------|---------|
| 0x04010101 | Sigmoid default | 0.9526 | sigmoid |
| 0x04144405 | Conv Cfg | 0.9526 | sigmoid |
| 0x04211101 | Concat Cfg | 0.9526 | sigmoid |
| 0x00244405 | GeMM Cfg | 0.9526 | sigmoid |
| 0x00000000 | Zero | 0.9526 | sigmoid |

**Bit map:**

| Bit | Mask | Function |
|-----|------|----------|
| 0 | 0x00000001 | pad0 — padding/layout flag |
| 8 | 0x00000100 | conv_mode — enables conv pipeline processing |
| 16 | 0x00010000 | dst_mode — output format/destination mode |
| 26 | 0x04000000 | enable — master enable for the block |

**Key finding**: Cfg controls **activation post-processing parameters**, not the fundamental operation mode. The op mode is determined by MACCfg (within the same firmware family) or by the BTSP firmware program (cross-firmware).

## 13.6 Operation Families Summary

| Family | Entry | Members | Data Path | Toggle Register |
|--------|-------|---------|-----------|----------------|
| 1 (PE) | 0x66490200 | add/mul/max/min/sq | TileDMA → PE → TileDMA | PECfg[2] + MACCfg[4:5] |
| 2a (Conv) | 0x25400201 | relu/sigmoid | TileDMA → Conv → TileDMA | MACCfg[16:17] |
| 2b (Conv) | 0x25400203 | conv/gemm/concat | TileDMA → Conv+KDMA → TileDMA | Different firmware |

# 14. KernelCfg (0x240): NOT the NE Enable Switch

From expt3, KernelCfg=0x80 (bit 7 set, "NE enable") showed as UNNEEDED for both relu and sigmoid — zeroing it produced identical output. This was surprising since KernelCfg is documented as the NE engine enable.

### Investigation

Tested on a **fresh /dev/accel/accel0** (no prior ANE access in session):

| Test | KernelCfg | Output | Result |
|------|-----------|--------|--------|
| relu baseline | 0x80 | [0, 5, 0, 2] | Correct |
| relu KernelCfg=0 (fresh) | 0x00 | [0, 5, 0, 2] | **Same — still works** |
| sigmoid baseline | 0x80 | [0.007, 0.5, 0.993] | Correct |
| sigmoid KernelCfg=0 (fresh) | 0x00 | [0.007, 0.5, 0.993] | **Same — still works** |
| relu KernelCfg=0 AND MACCfg=0 (fresh) | 0x00 + 0x00 | HANG | MACCfg=0 is the real poison |

### Root Cause

**KernelCfg=0 (NE disabled) does NOT gate the NE pipeline for elementwise nonlinear ops (relu/sigmoid).** The nonlinear activation (relu clamping, sigmoid LUT) is controlled entirely by `MACCfg[17:16]` (`non_linear_mode`). KernelCfg likely gates only NE **compute** modes (matrix multiply for conv/gemm) — the `KernelCfg=0x82` needed by conv/gemm is the compute-enable pattern, not the same as the passthrough-enable pattern `KernelCfg=0x80`.

This explains the expt3 test ordering artifact: in earlier runs, `MACCfg=0` was zeroed first, wedging the device before `KernelCfg` could be tested. On a clean device, `KernelCfg=0` is confirmed UNNEEDED for relu and sigmoid.

### Practical Impact

- **`examples_expt/relu.py` and `examples_expt/sigmoid.py`**: The `KernelCfg` line is commented out in both files (per expt3 results)
- **conv/gemm** still need `KernelCfg=0x82` (their compute mode requires it)
- The actual NE enable for nonlinear ops is `MACCfg[20]` (reserved bit), not `KernelCfg[7]`
**Cross-firmamily register-only conversion is NOT possible** — different BTSP firmware programs encode different data flow operations. The attempt causes ANE HANG.