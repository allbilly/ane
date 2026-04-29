# Register Experiments

## expt1: `test_regs_experiment.py` — Find Op Conversion Switches

### Goal
Find which operations can be converted to other operations with **few register changes**.

Known references:
- **ADD→MUL**: only **2 register changes** (PECfg[2], MACCfg[4:5]) — same BTSP firmware
- **Relu→sigmoid**: only **1 register change** (MACCfg[16:17]) — same BTSP firmware, but needs KDMA kernel data
- **Cross-family** (add→relu, relu→conv): NOT possible with register changes alone — different BTSP firmware programs

The experiment: start from each operation's baseline, substitute register values from other ops, and see which substitutions actually change the operation.

### Method
1. Start from working relu firmware (confirmed: `[-3,5]` → `[0,5]`)
2. For each register, substitute known values from other ops (add, conv, sigmoid, etc.)
3. Record: **DONTCARE** (no change), **FUNCTIONAL** (output changes), or **CRITICAL** (HANG)
4. Special focus on "switch" registers that change operation mode

### Known Op Conversion Pairs (already confirmed)

| Pair | Firmware Family | Reg Changes | Switch Register | Notes |
|------|----------------|-------------|----------------|-------|
| add↔mul | Family 1 (PE) | 2 | PECfg[2] + MACCfg[4:5] | Same BTSP program (`66 49 02 00`) |
| relu↔sigmoid | Family 2a (Conv) | 1 | MACCfg[16:17] | Same BTSP program (`25 40 02 01`); sigmoid needs KDMA LUT |
| add→relu | Cross-family | ~25 + fw | — | Different BTSP programs; regs alone NOT enough |

### MACCfg Sub-fields (from `examples/relu.py`)

The MACCfg register (0x244) is decomposed into named sub-fields:

| Field | Bits | Description |
|-------|------|-------------|
| `op_mode` | [3:0] | ALU operation mode (`12` = elementwise) |
| — | [5:4] | Unlabeled but must be set to `3` (0x0c = op_mode=12) |
| `non_linear_mode` | [17:16] | Nonlinear function select (`0`=none, `1`=relu, `2`=sigmoid) |
| `reserved` | [20] | Always set in working configurations |

Tested on relu firmware (entry 0x25400201), varying only MACCfg:

| op_mode | non_linear_mode | reserved | Output | Meaning |
|---------|-----------------|----------|--------|---------|
| 12 | 1 | 1 | [0, 5] | **RELU** (baseline, clamps negative) |
| 12 | 0 | 0 | [-3, 5] | **ADD-like pass-through** — identity, no clamping |
| 12 | 2 | 1 | HANG | **Sigmoid** — needs KDMA kernel data (not in relu fw) |
| 0 | 0 | 0 | HANG | **Broken** — NE core not configured |

**Key finding**: `op_mode=12` is the elementwise ALU mode (must be set). `non_linear_mode=1` enables relu negative clamping. With `non_linear_mode=0`, the conv pipeline passes data through unchanged — same as add on the PE path.

### Results (87 tests, auto-recovery from HANG)

Input: `[-3.0, 5.0]` → relu expects `[0, 5.0]`

#### FUNCTIONAL — Register Change Alters Output

| Register | Offset | Test Value | Effect |
|----------|--------|------------|--------|
| InDim | 0x128 | 0x10001 (1×1) | pos=0 (only first elem processed) |
| OutDim | 0x13c | 0x10001 (1×1) | pos=0 (only first output) |
| ChCfg | 0x130 | infmt=0, outfmt=0 | ADD-like pass-through (no format conversion) |
| SrcBaseAddr | 0x174 | 0x100 (offset) | pos=0 (reads from wrong addr) |
| DstBaseAddr | 0x25c | 0x100 (offset) | pos=0 (writes to wrong addr) |
| MACCfg | 0x244 | op_mode=12, non_linear_mode=0 | ADD-like pass-through (relu disabled) |
| PostScale | 0x250 | 0x0 | pos=0 (no post-scale = output 0) |
| PostScale | 0x250 | 0xffff | pos=inf (max post-scale overflows) |
| PostScale | 0x250 | 0x3c00 (baseline) | [0, 5] (fp16 1.0, correct) |

#### DONTCARE — Tested Values Produce Identical Output

| Register | Offset | Tested Values | Notes |
|----------|--------|---------------|-------|
| Cin | 0x134 | 64 | Channel count doesn't affect 1-ch pipeline |
| pad2 | 0x148 | 0 | Reserved — ignored by this firmware |
| pad4 | 0x158 | 1 | Reserved — ignored |
| TaskInfo | 0x160 | 0 | Task flags not relevant for single-task |
| DPE | 0x164 | 1 | DPE config unused in conv pipeline |
| SrcRowStride | 0x178 | 0x40 | ±stride from baseline 0xc0 — DMA adapts |
| SrcPlaneStride | 0x17c | 0x40 | Same |
| SrcDepthStride | 0x180 | 0x40, 0x1000 | Plane/depth stride unused for 1-ch |
| SrcGroupStride | 0x184 | 0x40 | Groups not used for single conv |
| Srcpad0–8 | 0x170..0x1a8 | Various | All reserved fields — ignored |
| SrcPadStream | 0x1AC | 0 | Stream padding — ignored |
| SrcFmt | 0x1a4 | 0x01302031 (DstFmt style) | Some format values OK |
| L2Cfg | 0x1e0 | 0x100 | L2 cache config — conv pipeline ignores |
| SourceCfg | 0x1e4 | 0x01500172, 0 | Source config changes tolerated |
| SourceBase | 0x1e8 | 0xa0 | Source base offset tolerated |
| SourceRowStride | 0x1f0 | 0x30, 0 | Row stride changes tolerated |
| L2pad0–8 | 0x1f4..0x224 | Various | All reserved — ignored |
| ResultBase | 0x214 | 0, 0x100 | Result base changes tolerated |
| ConvResultChannelStride | 0x218 | 0x10 | Conv result strides — single channel |
| ConvResultRowStride | 0x21c | 0x30, 0x01002031 | Conv result strides — ignored for 1-ch |
| PECfg | 0x22c | 0x80000, 0x80004 | All PE values tolerated (PE disabled) |
| BiasScale | 0x230 | 0x3c000000 | PE scale — PE disabled |
| PreScale | 0x234 | 0x3c000000 | Same |
| FinalScale | 0x238 | 0x3f800000 | Same |
| KernelCfg | 0x240 | 0, 0x82 | NE kernel config tolerated |
| MatrixVectorBias | 0x248 | 0x3c00 | Bias tolerated (no matrix-vec op) |
| AccBias | 0x24c | 0x3c00 | Acc bias tolerated |
| DstDMAConfig | 0x258 | 0xc1 | DMA config changes tolerated |
| DstRowStride | 0x260 | 0x40 | Stride changes tolerated |
| DstPlaneStride | 0x264 | 0x40 | Same |
| DstDepthStride | 0x268 | 0x40, 0x1000 | Same |
| DstGroupStride | 0x26c | 0x40 | Same |

#### CRITICAL — Register Change Causes HANG

| Register | Offset | Test Value | Notes |
|----------|--------|------------|-------|
| pad0 | 0x12c | 0 (zero) | Padding mode must be 1 |
| Cout | 0x138 | 0x40 (C=64) | Output channel count != input → wedge |
| pad1 | 0x140 | 0 (zero) | Must be 1 |
| ConvCfg | 0x144 | 0 (zero) | Conv config must be valid |
| GroupConvCfg | 0x14c | 0 (zero) | Group config must be valid |
| TileCfg | 0x150 | 0 (zero) | Tile config must be valid |
| pad3 | 0x154 | 4 (add style) | Must be 0 for conv pipeline |
| Cfg | 0x15c | 0x33 (add Cfg) | Pipeline config mismatch |
| SrcDMAConfig | 0x16c | 0 (zero DMA off) | DMA disabled → no data flow |
| SrcDMAConfig | 0x16c | 0 (zero) | Same (last test, safe ordering) |
| SrcFmt | 0x1a4 | 0 (zero Fmt) | Format must be valid |
| SourceChannelStride | 0x1ec | 0x10 (add style) | L2 channel stride mismatch |
| SourceChannelStride | 0x1ec | 0 (zero) | L2 channel stride invalid |
| ResultCfg | 0x210 | 0 (zero) | L2 result config invalid |
| DstFmt | 0x270 | 0 (zero Fmt) | Dest format must be valid |

#### Key Findings

1. **MACCfg `non_linear_mode=1`** is the relu ON/OFF switch — setting to `0` gives ADD-like pass-through
2. **ChCfg `infmt=0`** also produces ADD-like pass-through (data passed through unformatted)
3. **PostScale=0** zeros output — confirms PostScale is a multiplier, not an adder
4. **All PE registers are DONTCARE** on the conv pipeline — PE block is truly bypassed
5. **All stride regs (Src/Dst) are DONTCARE** for 1-channel — stride matters for multi-channel
6. **SourceChannelStride** is the most sensitive L2 register — even `0xa0 → 0x10` causes HANG
7. **Cfg=0x33** (add-style value) causes HANG on conv pipeline — confirms cross-family incompatibility
8. **SrcFmt/DstFmt** = 0 cause HANG — format must always be valid
9. Most pad/reserved registers are truly DONTCARE (except pad0, pad1, pad3)

### Implementation Details

Script: `experimental/test_regs_experiment.py`

Features:
- `reset_ane()` with PM resume retry: after each HANG, closes fd and waits for ANE to recover via kernel PM runtime autosuspend (up to 10 retries × increasing delays 0.3–3s)
- Orders tests from safest (dim/stride) to most dangerous (config regs that may wedge)
- Auto-recovery handles most hangs; hard wedges (L2 stride, DMA config zero) may stall remaining tests

Run:
```bash
python experimental/test_regs_experiment.py        # full experiment
python experimental/test_regs_experiment.py baseline  # verify baseline works
```

Shared infrastructure: `experimental/ane_helpers.py`

---

## expt2: `test_regs_sweep.py` — ±1 Bit-Level Sensitivity Scan

### Goal
For every register, apply ±1 changes to find **which individual bits** are "live" (affect behavior) vs "dead" (dontcare). This reveals the function of undocumented registers and pad/reserved fields.

### Why ±1?
- ±1 toggles the LSB — if the LSB is a live bit, output changes
- If ±1 produces no change, the register value may be:
  - A coarse-grained threshold (only large changes matter)
  - A don't-care field (completely ignored by this firmware)
  - A pointer/address (offset changes matter at higher bits)
- Different from expt1 which tests known values from other ops

### Method
1. Start from working relu firmware baseline
2. For each register, try: `relu_val + 1`, `relu_val - 1` (with wraparound for unsigned)
3. Record: HANG, OK (no change), or behavioral change
4. Build a **bit-sensitivity map**: which registers' LSB changes affect output?

### Signed/Unsigned Handling
All ANE registers are 32-bit unsigned:
- For value=0: try 1 and 0xFFFFFFFF (wraparound = all bits set)
- For value=1: try 0 and 2
- For value>1: try value-1 and value+1

### Expected Outcomes by Register Type

| Register Group | ±1 Expected Effect | What It Reveals |
|---------------|-------------------|-----------------|
| Stride regs (Src/Dst Row/Plane/DepthStride) | Changes element spacing → wrong values read | Confirms stride function |
| Dimension regs (InDim, OutDim) | ±1 changes input/output size | Confirms dimension format |
| Config regs (Cfg, ConvCfg, GroupConvCfg) | May change pipeline mode or HANG | Bit-level encoding of features |
| Pad regs (pad0-4, L2pad0-8, Srcpad1-8) | No change = truly unused; Change = undocumented function | Discovers hidden register function |
| MACCfg | ±1 flips mode bits → changes relu/sigmoid/pass-through | Bit field map |
| Zero regs (DPE, L2pad2-8, Srcpad1-8) | +1 from 0 reveals if they're used at all | Dead vs live detection |

### Implementation Details

Script: `experimental/test_regs_sweep.py`

Uses same infrastructure as expt1 (`experimental/ane_helpers.py` — PM resume reset, BTSP_BUF construction, wedge detection).

Run:
```bash
python experimental/test_regs_sweep.py        # full sweep
python experimental/test_regs_sweep.py baseline  # verify baseline
```

### Register Baseline Table (relu firmware)

| Block | Register | Offset | Value | Description |
|-------|----------|--------|-------|-------------|
| Common | InDim | 0x128 | 0x0001004d | Input W×H |
| Common | pad0 | 0x12c | 1 | Padding mode? |
| Common | ChCfg | 0x130 | 0x22 | Channel config (infmt/outfmt) |
| Common | Cin | 0x134 | 1 | Input channels |
| Common | Cout | 0x138 | 1 | Output channels |
| Common | OutDim | 0x13c | 0x0001004d | Output W×H |
| Common | pad1 | 0x140 | 1 | Unknown |
| Common | ConvCfg | 0x144 | 0x5000a021 | Conv kernel W×H, strides |
| Common | pad2 | 0x148 | 0x2041 | Unknown |
| Common | GroupConvCfg | 0x14c | 0x14001 | Group/depthwise config |
| Common | TileCfg | 0x150 | 1 | Tile config |
| Common | pad3 | 0x154 | 0 | Unknown |
| Common | pad4 | 0x158 | 0 | Unknown |
| Common | Cfg | 0x15c | 0x04010101 | Pipeline config (NOT op mode) |
| Common | TaskInfo | 0x160 | 0x00100000 | Task flags |
| Common | DPE | 0x164 | 0 | DPE config |
| SrcDMA | SrcDMAConfig | 0x16c | 0x33881 | TileDMA source config |
| SrcDMA | Srcpad0 | 0x170 | 0x8880 | Same as SrcDMAConfig with en=0 |
| SrcDMA | SrcBaseAddr | 0x174 | 0 | Source base address |
| SrcDMA | SrcRowStride | 0x178 | 0xc0 | Source row stride (192B) |
| SrcDMA | SrcPlaneStride | 0x17c | 0xc0 | Source plane stride |
| SrcDMA | SrcDepthStride | 0x180 | 0xc0 | Source depth stride |
| SrcDMA | SrcGroupStride | 0x184 | 0 | Source group stride |
| SrcDMA | Srcpad1 | 0x188 | 0 | Unknown |
| SrcDMA | Srcpad2 | 0x18c | 0 | Unknown (add has 0x40 here) |
| SrcDMA | Srcpad3 | 0x190 | 0 | Unknown (add has 0x40) |
| SrcDMA | Srcpad4 | 0x194 | 0 | Unknown (add has 0x1000) |
| SrcDMA | Srcpad5 | 0x198 | 0 | Unknown |
| SrcDMA | Srcpad6 | 0x19c | 0 | Unknown |
| SrcDMA | Srcpad7 | 0x1a0 | 0 | Unknown |
| SrcDMA | SrcFmt | 0x1a4 | 0x01002031 | Source data format |
| SrcDMA | Srcpad8 | 0x1a8 | 0 | Unknown (add has 0x2030) |
| SrcDMA | SrcPadStream | 0x1AC | 0x00000100 | TileDMA Src stream padding |
| L2 | L2Cfg | 0x1e0 | 0 | L2 cache config |
| L2 | SourceCfg | 0x1e4 | 0x00500172 | L2 source config |
| L2 | SourceBase | 0x1e8 | 0 | L2 source base |
| L2 | SourceChannelStride | 0x1ec | 0xa0 | Source channel stride |
| L2 | SourceRowStride | 0x1f0 | 0xa0 | Source row stride |
| L2 | L2pad0 | 0x1f4 | 0xa0 | = SourceRowStride (shadow?) |
| L2 | L2pad1 | 0x1f8 | 0xa0 | = SourceRowStride (shadow?) |
| L2 | L2pad2 | 0x1fc | 0 | Unknown |
| L2 | L2pad3 | 0x200 | 0 | Unknown |
| L2 | L2pad4 | 0x204 | 0 | Unknown |
| L2 | L2pad5 | 0x208 | 0 | Unknown |
| L2 | L2pad6 | 0x20c | 0 | Unknown |
| L2 | ResultCfg | 0x210 | 0x0050017a | L2 result config |
| L2 | ResultBase | 0x214 | 0xa0 | L2 result base |
| L2 | ConvResultChannelStride | 0x218 | 0 | Conv result channel stride |
| L2 | ConvResultRowStride | 0x21c | 0 | Conv result row stride |
| L2 | L2pad7 | 0x220 | 0 | Unknown |
| L2 | L2pad8 | 0x224 | 0 | Unknown |
| PE | PECfg | 0x22c | 0 | PE config (disabled in relu) |
| PE | BiasScale | 0x230 | 0 | Bias/scale (disabled in relu) |
| PE | PreScale | 0x234 | 0 | Pre-scale (disabled in relu) |
| PE | FinalScale | 0x238 | 0 | Final scale (disabled in relu) |
| NE | KernelCfg | 0x240 | 0x80 | NE kernel config (bit 7 = enable) |
| NE | MACCfg | 0x244 | 0x0011000c | NE MAC/ALU mode (op_mode=12, relu) |
| NE | MatrixVectorBias | 0x248 | 0 | Matrix-vector bias |
| NE | AccBias | 0x24c | 0 | Accumulator bias |
| NE | PostScale | 0x250 | 0x3c00 | Post-scale (fp16 1.0) |
| DstDMA | DstDMAConfig | 0x258 | 0x040000c1 | TileDMA destination config |
| DstDMA | DstBaseAddr | 0x25c | 0 | Destination base address |
| DstDMA | DstRowStride | 0x260 | 0xc0 | Dest row stride |
| DstDMA | DstPlaneStride | 0x264 | 0xc0 | Dest plane stride |
| DstDMA | DstDepthStride | 0x268 | 0xc0 | Dest depth stride |
| DstDMA | DstGroupStride | 0x26c | 0 | Dest group stride |
| DstDMA | DstFmt | 0x270 | 0x01302031 | Dest data format |

### Results (84 tests, 1 HANG)

Input: `[-3.0, 5.0]` → relu expects `[0, 5.0]`

#### FUNCTIONAL — ±1 Changes Behavior

| Register | Offset | Baseline | ±1 Test | Effect |
|----------|--------|----------|---------|--------|
| SrcBaseAddr | 0x174 | 0x0 | 0xFFFFFFFF (wrap -1) | pos=0 (all-bits-set base addr) |
| DstBaseAddr | 0x25c | 0x0 | 0xFFFFFFFF (wrap -1) | pos=0 (all-bits-set base addr) |
| MACCfg | 0x244 | op_mode=12, nl=1 | op_mode=13, nl=1 (+1) | pos=0 (op_mode changes behavior) |
| MACCfg | 0x244 | op_mode=12, nl=1 | op_mode=11, nl=1 (-1) | pos=0 (op_mode=11 is different mode) |

#### DONTCARE — ±1 Produces Identical Output (79 tests)

All of the following registers are ±1-insensitive on this firmware:
- **Dimension/stride**: InDim, OutDim, SrcRowStride, SrcPlaneStride, SrcDepthStride, SrcGroupStride, DstRowStride, DstPlaneStride, DstDepthStride, DstGroupStride
- **L2**: SourceChannelStride (+1 only), SourceRowStride, L2pad0–8, ResultBase, ConvResultChannelStride, ConvResultRowStride, SourceBase
- **PE**: PECfg, BiasScale, PreScale, FinalScale
- **NE**: KernelCfg, MatrixVectorBias, AccBias
- **SrcDMA**: SrcDMAConfig, Srcpad0–8, SrcFmt, SrcPadStream
- **DstDMA**: DstDMAConfig, DstFmt
- **All other TILE_DMA**: SrcGroupStride, DstGroupStride

#### HANG — ±1 Causes Hardware Wedge

| Register | Offset | Baseline | ±1 Test |
|----------|--------|----------|---------|
| SourceChannelStride | 0x1ec | 0xa0 | 0x9f (-1) |

**Note**: SourceChannelStride=0x9f causes a hard L2 wedge that even PM runtime resume cannot recover from (requires driver reload). This aligns with expt1 findings where SourceChannelStride=0x10 and 0x0 also caused HANG.

#### Bit-Sensitivity Map

```
Register                  Field(s) Live   Notes
─────────────────────────────────────────────────────
SrcBaseAddr (0x174)       [31:0]          All-bits-set = broken (functional, not bit-level)
DstBaseAddr (0x25c)       [31:0]          Same
MACCfg (0x244)            op_mode[3:0]    ±1 toggles op_mode (12→13, 12→11); non_linear_mode ±1-invariant
All other regs            [none]          ±1 LSB insensitive; dead for this firmware
```

#### Key Findings

1. **Most registers have dead LSBs** — ±1 doesn't change behavior for 79/84 tests
2. **MACCfg `op_mode[3:0]` is live** — ±1 changes op_mode from 12 to 13 or 11, altering behavior. `non_linear_mode[17:16]` and `reserved[20]` are ±1-invariant
3. **SrcBaseAddr/DstBaseAddr** only show effect when all 32 bits are set (0xFFFFFFFF) — offset=1 doesn't change behavior because DMA still reads from valid memory
4. **SourceChannelStride** is extremely sensitive: even 0xa0→0x9f causes hard wedge
5. **All L2pad registers** (0x1f4–0x224) and **Srcpad registers** (0x188–0x1a8) are confirmed dead — truly unused/reserved fields
6. **All PE registers** are dead — confirms PE is fully bypassed on the conv pipeline
7. **KernelCfg, MatrixVectorBias, AccBias** are ±1-insensitive — these need larger changes to affect behavior

---

## expt3: Minimal Register Set per Op

### Goal
For each op in `examples_expt/*.py`, find which registers can be commented out while still producing correct output. Determine the **absolute minimum register set** needed.

### Method (Simplified — Line-by-Line)

**Why this approach:** Each `examples_expt/*.py` file already has one register per line in the BTSP_BUF builder. The file opens its own fd, runs standalone, and prints expected vs actual output. This eliminates all device-state-tracking complexity from the previous `test_min_regs.py` approach — no need to track warm/cold device state, no cross-test pollution, no subprocess management.

**Procedure for each op file:**

1. Copy working `examples/{op}.py` → `examples_expt/{op}.py`
2. For each register line in the BTSP_BUF builder (not stream headers, not TD):
   a. Comment out the line
   b. Run `python3 examples_expt/{op}.py`
   c. If stdout shows correct output → keep commented (UNNEEDED)
   d. If stdout shows wrong output or HANG → uncomment (ESSENTIAL)
3. Skip registers that were already confirmed conditionally dead in expt1 (e.g., `pad2`, `Cin` for 1-ch ops, zero-valued pads)
4. For HANGs, revert immediately and visually verify

**Register lines to test:** Every `(reg.XXX, value)` line in the BTSP_BUF builder. Stream header lines (`reg.XStream, stream_header(...)`) are never commented out (required for task submission infrastructure).

**Skip list (confirmed zero-valued don't-care from expt1):**
- Zero-value registers: `SrcBaseAddr(0)`, `DstBaseAddr(0)`, `SrcGroupStride(0)`, `DstGroupStride(0)`, `DPE(0)`, `MatrixVectorBias(0)`, `AccBias(0)`
- Zero pads: `Srcpad1=0`, `Srcpad2=0`, `Srcpad3=0`, `Srcpad4=0`, `Srcpad5=0`, `Srcpad6=0`, `Srcpad7=0`, `Srcpad8=0`, `L2pad2=0`, `L2pad3=0`, `L2pad4=0`, `L2pad5=0`, `L2pad6=0`, `L2pad7=0`, `L2pad8=0`, `pad3=0`, `pad4=0`
- Single-channel don't-care: `ConvResultChannelStride(0)`, `ConvResultRowStride(0)` (for 1-ch ops)

### Ops to Test (6 total)

| Op | File | Input | Expected Output |
|----|------|-------|-----------------|
| relu | `examples_expt/relu.py` | [-3, 5, -1, 2] (tiled) | [0, 5, 0, 2] |
| sigmoid | `examples_expt/sigmoid.py` | [-5, 0, 5] | [~0.007, 0.5, ~0.993] |
| add | `examples_expt/elementwise.py` | a=3, b=2 (64-ch) | 5 (all channels) |
| mul | `examples_expt/elementwise.py` (MACCfg=0x30) | a=3, b=2 (64-ch) | 6 (all channels) |
| conv | `examples_expt/conv.py` | 3-ch, weights=2.0 | input × 2.0 |
| gemm | `examples_expt/gemm.py` | 512×32, weights=0.5 | input × 0.5 |

### Results

*To be filled in as each register line is tested.*

#### relu.py

| UNNEEDED (6) | Reason |
|---|---|
| W1, W2, W3, W5, W7, W9 | Zero-valued TD words — no function |

All other 42 register blocks are ESSENTIAL (zeroing produces wrong output).

**Warm-device note**: On a device already warmed by a baseline run, 24 registers can be commented (warm UNNEEDED = above 6 + TaskInfo, SrcRowStride, SrcPlaneStride, SrcDepthStride, SrcPadStream, L2Cfg, SourceCfg, SourceBase, SourceRowStride, L2pad0, L2pad1, L2pad4, L2pad5, L2pad6, ResultBase, ConvResultChannelStride, ConvResultRowStride, KernelCfg). For standalone file compatibility, only the 6 cold-boot safe registers are commented.

#### sigmoid.py

| UNNEEDED (5) | Reason |
|---|---|
| W1, W2, W3, W5, W7 | Zero-valued TD words |

W9 is ESSENTIAL for sigmoid (0x21 = KDMA kernel info for LUT transfer). Warm UNNEEDED = 21 registers (same as relu minus W9, ConvResultChannelStride, ConvResultRowStride).

#### elementwise.py

| UNNEEDED (6) | Reason |
|---|---|
| W1, W2, W3, W5, W7, W9 | Zero-valued TD words |

InDim is warm-UNNEEDED but cold-ESSENTIAL. Remaining 49 register blocks are ESSENTIAL.

### Cross-Op Summary

| Register | relu | sigmoid | elementwise |
|----------|------|---------|-------------|
| W1,W2,W3,W5,W7 | UNNEEDED | UNNEEDED | UNNEEDED |
| W9 | UNNEEDED | **ESSENTIAL** | UNNEEDED |
| All others | ESSENTIAL | ESSENTIAL | ESSENTIAL |

**Key finding**: Only zero-valued Task Descriptor words can be safely omitted for cold-boot compatibility. Register configurations are near-complete — skipping any functional register (even strides for 1-channel ops) causes incorrect output on cold device.
