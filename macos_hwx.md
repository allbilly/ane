# macOS HWX Compatibility

## Reference: `examples/add.py` — Complete Register Breakdown

`examples/add.py` is the authoritative reference for ALL register values needed to run ADD and MUL on the ANE. It directly programs every register via `build_seg()` and `BTSP_BUF`, bypassing `anecc` entirely.

### Register Map (H13 architecture, BTSP_BUF offsets)

```
OFFSET  BANK+OFF  NAME            ADD val     MUL val     Description
------  --------  ----            -------     -------     -----------
0x000   TD+0x00   W0 (TID/NID)    0x02000040  0x02000040  tid=0, nid=0x40, eon=1
0x004   TD+0x04   W1              0           0           next_size
0x008   TD+0x08   W2 (exe_cycles) 0x00000422  0x00000422  exe_cycles=1058
0x00c   TD+0x0c   W3              0           0
0x010   TD+0x10   W4              0x00fff86a  0x00fff86a  debug_log_events
0x014   TD+0x14   W5              0           0
0x018   TD+0x18   W6 (flags)      0x30009800  0x30009800  next_priority=38
0x01c   TD+0x1c   W7 (next_ptr)   0           0
0x020   TD+0x20   W8 (base_ene)   0x00008621  0x00008621  rbase0=6,rbe0=1,rbase1=5,rbe1=1,wbase=4,wbe=1
0x024   TD+0x24   W9              0           0
0x028   TD+0x28   KernelDMA       strm(1F800,62) same    stream_header(0x1F800, 62)
---     ---       ---             ---         ---         ---
0x124   TD+0x124  CommonStream    strm(0,16)  same       stream_header(0x00000, 16)
0x128   Common+0  InDim           0x00010001  0x00010001  h=1, w=1
0x12c   Common+4  pad0            1           1
0x130   Common+8  ChCfg           0x2a        0x2a        infmt=fp16, pad0=2, outfmt=fp16
0x134   Common+c  Cin             0x40        0x40        Cin=64
0x138   Common+10 Cout            0x40        0x40        Cout=64
0x13c   Common+14 OutDim          0x00010001  0x00010001  h_out=1, w_out=1
0x140   Common+18 pad1            1           1
0x144   Common+1c ConvCfg         0x5000a021  0x5000a021  kw=1, kh=1, sx=1, sy=1
0x148   Common+20 pad2            0x2041      0x2041      reserved
0x14c   Common+24 GroupConvCfg    0x00010001  0x00010001  groups=1, unicast_cin=1
0x150   Common+28 TileCfg         1           1
0x154   Common+2c pad3            4           4
0x158   Common+30 pad4            0           0
0x15c   Common+34 Cfg             0x33        0x33        PE elementwise mode
0x160   Common+38 TaskInfo        0           0
0x164   Common+3c DPE             0           0
---     ---       ---             ---         ---         ---
0x168   TD+0x168  SrcStream       strm(13800,28) same    stream_header(0x13800, 28)
0x16c   TDMASrc+0 SrcDMAConfig    0x00033881  0x00033881  en=1, cache_hint=8, reuse=8, noreuse=3, dep=3
0x170   TDMASrc+4 Srcpad0         0x00033880  0x00033880  reserved
0x174   TDMASrc+8 SrcBaseAddr     0           0
0x178   TDMASrc+c SrcRowStride    0x40        0x40        64 bytes = 32 fp16
0x17c   TDMASrc+10 SrcPlaneStride 0x40        0x40        same
0x180   TDMASrc+14 SrcDepthStride 0x1000      0x1000      64*32*2 = 0x1000
0x184   TDMASrc+18 SrcGroupStride 0           0
0x188   TDMASrc+1c Srcpad1        0           0
0x18c   TDMASrc+20 Srcpad2        0x40        0x40
0x190   TDMASrc+24 Srcpad3        0x40        0x40
0x194   TDMASrc+28 Srcpad4        0x1000      0x1000
0x198   TDMASrc+2c Srcpad5        0           0
0x19c   TDMASrc+30 Srcpad6        0           0
0x1a0   TDMASrc+34 Srcpad7        0           0
0x1a4   TDMASrc+38 SrcFmt         0x01002031  0x01002031  fmt_mode=1, trunc=3, mem_fmt=2, intrlv=1
0x1a8   TDMASrc+3c Srcpad8        0x00002030  0x00002030  reserved
---     ---       ---             ---         ---         ---
0x1dd   TD+0x1dd  L2Stream        strm(4800,18) same     stream_header(0x04800, 18)
0x1e0   L2+0      L2Cfg           0           0
0x1e4   L2+4      SourceCfg       0x01500172  0x01500172  type=2, alias=both, fmt=1, intrlv=1
0x1e8   L2+8      SourceBase      0           0
0x1ec   L2+c      SourceChanStr   0x10        0x10        16 bytes
0x1f0   L2+10     SourceRowStr    0x420       0x420       1056 bytes (stride=66 in 16B units)
0x1f4   L2+14     L2pad0          0x400       0x400       reserved
0x1f8   L2+18     L2pad1          0x400       0x400       reserved
0x1fc   L2+1c     L2pad2          0x440       0x440       reserved
0x200   L2+20     L2pad3          0x10        0x10        = SourceChannelStride
0x204   L2+24     L2pad4          0x420       0x420       = SourceRowStride
0x208   L2+28     L2pad5          0x400       0x400       reserved
0x20c   L2+2c     L2pad6          0x400       0x400       reserved
0x210   L2+30     ResultCfg       0x0050017a  0x0050017a  type=2, bfrmode=2, alias=both, fmt=1
0x214   L2+34     ResultBase      0x860       0x860       2144 bytes
---     ---       ---             ---         ---         ---
0x229   TD+0x229  PEStream        strm(8800,4) same       stream_header(0x08800, 4)
0x22c   PE+0      PECfg           0x00080000  0x00080004  **bit 2 = 0→ADD, 1→MUL**
0x230   PE+4      BiasScale       0x3c000000  0x3c000000  bias=0, scale=fp16(1.0)
0x234   PE+8      PreScale        0x3c000000  0x3c000000  pre_scale=0
0x238   PE+c      FinalScale      0x3f800000  0x3f800000  fp32(1.0)
---     ---       ---             ---         ---         ---
0x23c   TD+0x23c  NEStream        strm(C800,5) same       stream_header(0x0C800, 5)
0x240   NE+0      KernelCfg       0           0           **No kernel DMA**
0x244   NE+4      MACCfg          0           0x30        **0=ADD, 0x30=MUL**
0x248   NE+8      MatrixVecBias   0           0
0x24c   NE+c      AccBias         0           0
0x250   NE+10     PostScale       0           0
---     ---       ---             ---         ---         ---
0x255   TD+0x255  DstStream       strm(17800,7) same      stream_header(0x17800, 7)
0x258   TDMADst+0 DstDMAConfig    0x040000c1  0x040000c1  en=1, cache_hint=12
0x25c   TDMADst+4 DstBaseAddr     0           0
0x260   TDMADst+8 DstRowStride    0x40        0x40
0x264   TDMADst+c DstPlaneStride  0x40        0x40
0x268   TDMADst+10 DstDepthStride 0x1000      0x1000
0x26c   TDMADst+14 DstGroupStride 0           0
0x270   TDMADst+18 DstFmt         0x01002031  0x01002031
```

**Only 2 register differences between ADD and MUL:**
| Register | Offset | ADD | MUL | Effect |
|----------|--------|-----|-----|--------|
| `PECfg`  | 0x22c  | 0x00080000 | 0x00080004 | PE OpMode bit 2 |
| `MACCfg` | 0x244  | 0x00000000 | 0x00000030 | NE KernelMode=1, BiasMode=1 |

Switching is as simple as: `python add.py` → ADD, `python add.py mul` → MUL.

## macOS-Generated HWX Compatibility

`coreml2hwx` generates `.hwx` files on macOS. macOS 12 produces clean files; macOS 14+ and 26+ produce files with register differences that break elementwise operations.

| Source | macOS Ver | .hwx size | .ane size | tsk_size | Works? | Raw Output | Fixed Via hwx2py |
|--------|-----------|-----------|-----------|----------|--------|------------|------------------|
| macOS 12 VM (M4) | 12.4 | 49152 | 21120 | 628 | ✓ | 6.0 | — |
| macOS 14 VM | 14.x | 49152 | 21120 | 628 | ✗ | 0.0 | ✓ 6.0 |
| macOS 26 M1 | 26.x | 65536 | 20992 | 504 | ✗ | 0.0 | ✓ 6.0 |

### Register-level differences (macOS 12 vs macOS 14 vs macOS 26)

Compared using `python3 hwx_parsing.py hwx/mul_macosNN.hwx -j`:

| Register | macOS 12 | macOS 14 | macOS 26 | Block |
|----------|----------|----------|----------|-------|
| KernelCfg (0xC800) | 0x00000000 | **0x00000080** | **0x00000080** | NE |
| MACCfg (0xC804) | 0x00000000 | **0x00100000** | **0x00100000** | NE |
| CoeffDMAConfig[0..15] | all 0x00 | **all 0x80** | **all 0x80** | KernelDMA |
| CoeffBaseAddr[0..15] | all 0x00 | **all 0x80** | **all 0x80** | KernelDMA |
| CoeffBfrSize[0..15] | all 0x00 | **all 0x40/0x80** | **all 0x40/0x80** | KernelDMA |
| TD byte 0x20 | 0xa549 | **0x6449** | **0x6449** | TD header |

macOS 14 and 26 share the same spurious pattern: 16 coefficient channels configured with garbage values, NE told to load via DMA (KernelCfg=0x80), and wrong MAC mode (MACCfg=0x00100000).

macOS 26 also uses a different task size (504 vs 628), suggesting an updated `coreml2hwx` that generates different tsk_size.

## Running macOS 14/26 HWX Files

### Method 1: Via `hwx2py` with auto-clean (recommended for elementwise)

The `experimental/hwx2py.py` script detects the spurious KernelDMA pattern (`all 16 CoeffDMAConfig == 0x80`), strips all KernelDMA registers, and resets `KernelCfg`/`MACCfg` to 0:

```bash
# macOS 14 mul.hwx → running script
python experimental/hwx2py.py hwx/mul_macos14.hwx -o experimental/mul14_from_hwx.py
python experimental/mul14_from_hwx.py
# → output[0] = 6.0

# macOS 26 mul.hwx → running script  
python experimental/hwx2py.py hwx/mul_macos26_m1.hwx -o experimental/mul26_from_hwx.py
python experimental/mul26_from_hwx.py
# → output[0] = 6.0
```

What `hwx2py --clean` does internally:
1. Detects elementwise op: `ConvCfg K=1×1` and `GroupConvCfg groups=1`
2. Detects spurious KDMA: all `CoeffDMAConfig[0..15] == 0x80`
3. Deletes all KernelDMA registers (`0x1F800` range)
4. If `KernelCfg==0x80` and `MACCfg==0x00100000`: resets both to `0`
5. Re-encodes cleaned register state into CMD_BUF + BTSP

### Method 2: Via `anecc` (works for all ops including weighted)

`anecc` properly handles KernelDMA setup regardless of macOS version:

```bash
anecc hwx/mul_macos14.hwx -o hwx/mul_macos14.ane
python run.py ./hwx/mul_macos14.ane
# → should produce 6.0 (anecc fixes KDMA internally)
```

### Method 3: Manual register patching

For elementwise ops, the fix is just 2 register writes in the CMD_BUF:

```python
# Patch CMD_BUF at specific offsets (H13, tsk_size=628):
CMD_BUF[0x240:0x244] = struct.pack('<I', 0)  # KernelCfg = 0
CMD_BUF[0x244:0x248] = struct.pack('<I', 0)  # MACCfg = 0 (ADD)
# For MUL:
CMD_BUF[0x244:0x248] = struct.pack('<I', 0x30)  # MACCfg = 0x30
```

### Method 4: GitHub Actions (macOS 14 runners)

The `.github/workflows/ane-generation.yml` uses macOS 14 runners. For **weighted ops** (GEMM, Conv, Sigmoid), the KDMA data is real (`CoeffDMAConfig=0x81`) and works correctly through `anecc`. For **elementwise ops** (Add, Mul), the spurious KDMA entries are harmless noise — the weights section is empty, `anecc` strips them, and the output is unaffected.

The only case needing macOS 12 is if you want **visually clean** elementwise `.hwx` files without any KDMA noise.

### Experimental Results

| Input HWX | Method | Output | Status |
|-----------|--------|--------|--------|
| `hwx/mul.hwx` (macOS 12) | hwx2py | 6.0 | ✓ |
| `hwx/mul_macos14.hwx` | hwx2py --clean | 6.0 | ✓ |
| `hwx/mul_macos14.hwx` | Raw (no clean) | 0.0 | ✗ — ANE loads garbage coefficients |
| `hwx/mul_macos26_m1.hwx` | hwx2py --clean | 6.0 | ✓ |
| `hwx/mul_macos26_m1.hwx` | Raw (no clean) | 0.0 | ✗ — same KernelDMA issue |
| `examples/add.py` (hand-written) | Direct | 6.0 | ✓ Reference |
| `examples/add.py mul` | Direct | 6.0 | ✓ Reference |

## Root Cause: macOS 14+ adds spurious KernelDMA coefficient configs

macOS 12 `coreml2hwx` generates minimal hwx files with only essential register writes for elementwise operations. macOS 14+ generates files with **extra KernelDMA coefficient configuration**:

- `NE KernelCfg = 0x80`: tells NE to load coefficients via DMA
- `NE MACCfg = 0x00100000`: sets wrong KernelMode/BiasMode
- `CoeffDMAConfig[0..15] = 0x80`: 16 coefficient channels configured
- `CoeffBaseAddr[0..15] = 0x80`: all pointing to garbage address
- `CoeffBfrSize[0..15] = 0x40/0x80`: garbage buffer sizes

Elementwise add/mul has **no coefficients**, so the ANE computes with zero coefficients → output is 0.0.

## Whisper Encoder/Decoder: Mixed-Op HWX Pipeline

The `.github/workflows/whisper.yml` generates Whisper encoder HWX on macOS 14. The model is a transformer with **mixed operation types** — not just simple elementwise add/mul but a pipeline of different ops chained together:

### Whisper Model Ops Breakdown

| Op | Encoder | Decoder | ANE Support | KDMA needed |
|----|---------|---------|-------------|-------------|
| Conv1d | 2 (strided) | 0 | ✓ (spatial conv) | ✓ weights |
| GEMM (QKV, MLP) | 12 | 18 | ✓ (inner_product) | ✓ weights |
| GELU | 3 | 3 | **?** (may need LUT) | ✓ LUT data |
| LayerNorm | 3 | 3 | **?** (ANE may not support) | — |
| Softmax | 2 | 4 | **?** (cross-attention) | — |
| Elementwise add | 3 | 3 | ✓ | ✗ no weights |
| Elementwise mul | 3 | 3 | ✓ | ✗ no weights |
| Reshape/Transpose | implicit | implicit | ✓ (in firmware) | — |

**Total**: ~30+ operations per forward pass, many chained via task descriptors.

### Does hwx2py work for Whisper? No, not yet.

The current `experimental/hwx2py.py` has multiple gaps for Whisper-scale models:

| Gap | Current behavior | Whisper needs |
|-----|-----------------|---------------|
| **Task chaining** | `parse_hwx_regs()` skips `next_ptr` — only reads first task | Whisper HWX has **multiple chained tasks** connected via next_ptr. Need to parse ALL tasks and emit them sequentially in CMD_BUF |
| **H16/M4 format** | Only H13 (M1) supported | macOS 14+ `coreml2hwx` may generate H16 format for newer HWX. H16 has different packet encoding (masked writes), different register layout, different header |
| **KDMA cleaning heuristic** | `is_spurious_kdma()` strips if ALL 16 CoeffDMAConfig==0x80 | Whisper has **legitimate KDMA** for Conv1d/GEMM (CoeffDMAConfig=0x81). The heuristic must check **per-task**, not globally. Conv1d needs KDMA; elementwise in same pipeline does not |
| **Multi-task BTSP** | Copies first 0x4000 bytes of CMD_BUF | Each task needs its own BTSP, or a combined BTSP covering all tasks. The `make_btsp()` function needs to handle task offsets |
| **Kernel data extraction** | Only reads `__TEXT.__const` — simplistic | Whisper has large weight matrices (GEMM layers). Need to extract all kernel data sections and place them at correct offsets relative to each task's descriptor |
| **Shape metadata** | Hardcoded for 1x1 elementwise | Need to handle arbitrary input/output shapes, multiple tensors, multi-dimensional batch dimensions |
| **Buffer sizing** | `_calc_buffer_size()` uses TileDMA depth/group strides | Whisper has variable-size buffers per task. Buffer size must accommodate all tasks' max requirements |
| **Handle allocation** | Fixed: `[cmd_h, 0,0,0, out_h, src1_h, src2_h]` | Multi-task models need per-task CMD_BUF buffers, input/output buffers for intermediate tensors between tasks |

### What updates hwx2py needs for Whisper

**Phase 1: Multi-task parsing and re-encoding**
```python
# Current: reads one task
regs, ane_data, kernel = parse_hwx_regs(data)

# Needed: read all chained tasks
tasks = parse_all_tasks(data)  # follows next_ptr chain
for task in tasks:
    regs = decode_packets(task.data)
    # Apply per-task KDMA cleaning:
    #   if has_real_kernel(task) → preserve KDMA
    #   if elementwise with spurious 0x80 → strip KDMA
    encode_task(task.header, regs)
# Concatenate all encoded tasks into one CMD_BUF
```

**Phase 2: H16 format support**
```python
# Current: H13 only
packet: count=(hdr>>26)&0x3f, addr=(hdr&0x3ffffff)>>2

# Needed: detect format from header, handle both
if hdr & 0x80000000:  # H16 masked write
    mask = (hdr >> 15) & 0xffff
    word_addr = hdr & 0x7fff
else:
    # H13 or H16 contiguous write
    ...
```

**Phase 3: Kernel data layout**
```python
# Current: dumps ane_data + kernel as one flat buffer
cmdbuf_final = ane_data + kernel

# Needed: place kernel data at correct per-task offsets
# ANE expects: for each task descriptor at offset X,
#   kernel data referenced by CoeffBaseAddr lives at X + task_size + channel_offset
for task in tasks:
    if task.has_kernel:
        offset = task.task_descriptor_offset + task.task_size
        cmdbuf[offset:offset+len(kernel)] = kernel
```

**Phase 4: Multi-buffer IO**
```python
# Current: 1 CMD + 1 OUT + 1 SRC1 + 1 SRC2
handles = [cmd_h, 0,0,0, out_h, src1_h, src2_h] + [0]*25

# Needed: allocate buffer per tile index per task
# Some intermediate outputs become next task's inputs
# Need to track the tensor-to-buffer mapping across tasks
```

### Current workaround: anecc

For now, Whisper HWX must go through `anecc`:
```bash
# Generated by GitHub Actions (whisper.yml), then on Asahi:
anecc whisper-encoder.hwx -o whisper-encoder.ane
anecc whisper-decoder.hwx -o whisper-decoder.ane  
python run.py ./whisper-encoder.ane
```

`anecc` handles multi-task chaining, KDMA layout, and buffer management. `hwx2py` fills the gap for simple single-task models where `anecc` fails (elementwise with spurious KDMA). For complex models, `anecc` is still the reliable path.
