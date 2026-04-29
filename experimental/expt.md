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

### MACCfg Bit Analysis (on relu firmware, conv pipeline)

Tested on relu firmware (entry 0x25400201), varying only MACCfg:

| MACCfg | Bits | Output | Meaning |
|--------|------|--------|---------|
| 0x0011000c | bit16, bits2-3, bit20 | [0, 5] | **RELU** (baseline, clamps negative) |
| 0x0000000c | bits2-3 only | [-3, 5] | **ADD-like pass-through** — no clamping, identity |
| 0x0012000c | bit17, bits2-3, bit20 | HANG | **Sigmoid table lookup** — needs KDMA kernel data (not present in relu firmware) |
| 0x00000000 | none | HANG/wedge | **Broken** — NE core not configured |

**Key finding**: MACCfg bits 2-3 (0x0c) are the ALU format mode (must be set). Bit 16 (0x10000) enables relu negative clamping. Without bit 16, the conv pipeline passes data through unchanged — same as add on the PE path.

### Registers That Change Output (confirmed on ANE)

| Register | Offset | Change | Effect |
|----------|--------|--------|--------|
| InDim | 0x128 | 0x1004d → 0x10001 | Only first element processed, rest = 0 |
| OutDim | 0x13c | 0x1004d → 0x10001 | Only first output position produced, rest = 0 |
| MACCfg | 0x244 | 0x11000c → 0x0000000c | Relu clamping disabled → ADD-like pass-through |
| MACCfg | 0x244 | 0x11000c → 0x00000000 | NE broken → HANG |

### Registers Known to Be Critical (from prior work)

These registers cause HANG if changed from relu values (documented in README §2):
- Cout (0x138), GroupConvCfg (0x14c), pad3 (0x154), Cfg (0x15c), TaskInfo (0x160)
- L2Cfg (0x1e0), SourceCfg (0x1e4), SourceBase (0x1e8)
- ResultCfg (0x210), ResultBase (0x214)
- PECfg (0x22c) — even though PE is "disabled", changing from 0 causes HANG
- SrcDMAConfig (0x16c), Srcpad0 (0x170), SrcRowStride (0x178), etc.
- DstRowStride (0x260), DstFmt (0x270)

**Note**: These are CRITICAL because they affect the entire data path configuration. Changing them doesn't change "op mode" — it breaks the pipeline entirely.

### Next Steps
- Run all remaining register tests with auto-recovery from HANG
- Test Cfg register with all known values (confirmed on sigmoid firmware: Cfg doesn't affect op mode)
- Document which pad/reserved registers are truly don't-care vs functional

### Implementation Details

Script: `experimental/test_regs_experiment.py`

Features:
- `safety_check()` with retry: after each HANG, waits for ANE to recover (5 retries × 0.5s)
- Orders tests from safest (dim/stride) to most dangerous (config regs that may wedge)
- Prints markdown-formatted results

Run:
```bash
python experimental/test_regs_experiment.py        # full experiment
python experimental/test_regs_experiment.py baseline  # verify baseline works
```

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

Based on same framework as expt1 (BTSP_BUF construction, safety_check with retry, wedge detection).

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

### Interpreting Results

After running, results will be categorized as:

1. **DONTCARE** (±1 produces identical correct output) → register is unused by this firmware config
2. **FUNCTIONAL** (±1 changes output values or layout) → register has a live function
3. **HANG** (±1 causes hardware wedge) → register is critical; wrong value breaks the pipeline
4. **MODE CHANGE** (±1 changes operation type, e.g., relu→passthrough) → register is an op-mode switch
