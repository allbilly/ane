# macOS HWX Compatibility

## Background

`coreml2hwx` runs on macOS to convert CoreML models to `.hwx` format, which is then compiled to `.ane` by `anecc`. Different macOS versions produce different `.hwx` files.

## Test Results

| Source | macOS Ver | .hwx size | .ane size | tsk_size | Works? | Output |
|--------|-----------|-----------|-----------|----------|--------|--------|
| macOS 12 VM (M4) | 12.4 | 49152 | 21120 | 628 | ✓ | 6.0 |
| macOS 14 VM | 14.x | 49152 | 21120 | 628 | ✗ | 0.0 |
| macOS 26 M1 | 26.x | 65536 | 20992 | 504 | ✗ | 0.0 |

## Root Cause: macOS 14+ adds KernelDMA coefficient configs

The macOS 12 `coreml2hwx` generates **minimal** hwx files with only the essential register writes for elementwise operations. macOS 14+ generates files with **extra KernelDMA coefficient configuration**.

### CMD_BUF comparison (key differences)

| Offset | macOS 12 | macOS 14 | Register | Effect |
|--------|----------|----------|----------|--------|
| 0x020 | `a5 49` | `64 69` | Common pad2 | Unused |
| 0x030-0x06f | all zeros | `0x80` filled | KernelDMA Coeff DMA configs | Extra coefficient channels |
| 0x240 (KernelCfg) | `0x00` | `0x80` | NE KernelCfg | **KernelDMA enabled** |
| 0x244 (MACCfg) | `0x00000000` | `0x00100000` | NE MACCfg | **Wrong KernelMode/BiasMode** |

### Why macOS 14 fails

The macOS 14+ hwx configures KernelDMA with 16 coefficient channels (all pointing to base address 0x80 with size 0x80), even though **elementwise add/mul has no coefficients**. The NE is told to load coefficients via DMA (`KernelCfg=0x80`), but there's nothing to load and the base addresses are garbage. This causes the ANE to compute with zero coefficients → output is 0.0.

The NE MACCfg value `0x00100000` also sets different kernel/bias modes compared to the clean `0x00000000` from macOS 12, further changing behavior.

### macOS 26 M4

The macOS 26 M1 hwx uses a **different task descriptor format** (tsk_size=504 vs 628, M4/H16 architecture). It has the same extra KernelDMA config problem plus a different register layout.

## Options to Fix

But is the info in `hwx_parsing.py` enough for preprocessing?

**Yes**, `hwx_parsing.py` already has everything needed:
- Full register name tables for both H13 (M1) and H16 (M4+) architectures
- Packet decoder for both H13 and H16 task descriptor formats
- `report_hwx_state_json()` extracts register state to a clean `{addr, val, name}` JSON dict
- The block start addresses and register counts are all known constants

What's missing is only the **reverse encoder** — a function that takes a register-value dict and re-encodes it back into the hwx binary using the same packet format. This is mechanical: iterate sorted register addresses, group contiguous runs, emit header+(data words) packets.

However, a **blanket "zero all KernelDMA" preprocessor is risky** because:
- Convolution ops legitimately need KernelDMA for loading weights
- Only elementwise ops (add/mul with K=1x1, groups=1) should have KernelDMA stripped
- Without macOS 12 reference samples for EVERY op type (conv, pooling, activation, etc.), we can't know which KernelDMA configs are real vs spurious
- The heuristic "all CoeffDMAConfig have identical suspicious value 0x80 → strip" is fragile

### Option B: Fix anecc (recommended, but requires upstream change)

`anecc` is the right place for this logic because it already:
- Knows the op type from the CoreML model metadata
- Knows whether coefficients/weights are present
- Can decide op-by-op: "this is elementwise with no weights → skip KernelDMA"

The fix in `anecc` would be: before writing the cmd buf, if the model is elementwise (no kernel data), zero out KernelDMA registers and reset NE KernelCfg/MACCfg to clean values. This is clean, op-aware, and doesn't need macOS 12 reference data.

### Option C: Use macOS 12 only (not viable)

macOS 12 GitHub runner is discontinued. Running a macOS 12 VM is fragile and inelegant.

## Re-evaluation: Do we need anecc at all?

`min_add.py` proves we don't need `anecc` to run a model on the ANE — we just need:
1. **CMD_BUF** bytes (header + register packets)
2. **BTSP** bytes (bootstrap microcode)
3. **Metadata** (tsk_size, td_size, td_count, handle layout)

All of these can be extracted from any `.hwx` file. `hwx_parsing.py` has:
- ✅ Header parser (TID, NID, sizes, next_ptr)
- ✅ Register stream decoder (both H13/H16 packet formats)
- ✅ Register name tables
- ❌ Missing: CMD_BUF encoder, BTSP extractor, shape metadata extractor

But we already know the missing pieces from our reverse-engineering:
- **CMD_BUF format**: 40-byte header + register packets (packet header = count<<26 | addr, followed by data words) + zero padding
- **BTSP**: same register packets as CMD_BUF, shifted by 1 byte in first segment
- **Packet encoding**: H13 (count in bits 31:26, addr/4 in bits 25:0, packed data) — already implemented in `min_add.py`'s `make_buf` segments
- **Shapes**: extracted from Common registers (InDim, Cin, Cout, OutDim)

## New option: Build the CMD_BUF converter directly

Instead of fixing `anecc`, build a tool that:
1. Parses `.hwx` with `hwx_parsing.py` → register values dict + task header
2. Optionally filters spurious registers (KernelDMA for elementwise ops)
3. Encodes register values → CMD_BUF using the known packet format
4. Synthesizes BTSP from same register data
5. Runs on ANE via `bo_alloc` + `ane_submit` (the `min_add.py` pattern)

This avoids needing `anecc` entirely. The only missing piece is register-value → CMD_BUF encoding, which is mechanical from the known packet format. `hwx_parsing.py` already has `report_hwx_state_json()` for extracting register state; adding an `encode_task()` function that goes back to binary is ~50 lines of Python.

## Implementation plan: hwx2py converter

Build `examples/hwx2py.py` that reads a `.hwx` file and generates a self-contained Python script to run the model on the ANE. Follows the `min_add.py` pattern.

### Phase 1: macOS 12 baseline (H13, tsk_size=628)

Test with `mul.hwx` → generate `mul_from_hwx.py` that produces output 6.0.

```
hwx2py.py hwx/mul.hwx -o examples/mul_from_hwx.py
python examples/mul_from_hwx.py  →  output[0] = 6.0
```

**Steps:**
1. **Parse hwx**: Use `hwx_parsing.py` pipeline:
   - `parse_macho()` to extract ANE data section from the binary
   - H13 format: read task header (offset 0): 8 words → TID, sizes, next_ptr
   - Register stream at offset+40: decode packets (hdr → count, addr, data words)
   - Collect all register values in a `{addr: value}` dict

2. **Extract metadata** from Common registers:
   - InDim=0x000: W, H
   - Cin=0x00c, Cout=0x010: channels
   - OutDim=0x014: output W, H
   - ChCfg=0x008: input/output format
   - tsk_size/task_size: from header

3. **Clean registers** (strip spurious KernelDMA for elementwise):
   - Detect elementwise: ConvCfg K=1×1, GroupConvCfg groups=1
   - If elementwise: zero NE KernelCfg(0xC800), NE MACCfg(0xC804)
   - Zero KernelDMA coeff configs (0x1F808+, H13) if all are suspicious 0x80
   - Set PE OpMode based on mul vs add detection

4. **Encode CMD_BUF** (H13 format):
   ```
   HEADER = struct.pack('<8I', 0x02000000, 0, tsk_size, 0, 0x00fff86a, 0, 0x30009800, 0)
   padding_8 = b'\x00' * 8  # bytes between header and register stream
   
   # Register packets: iterate sorted addrs, group contiguous runs
   for run in contiguous_runs(sorted(regs.keys())):
       count = len(run) - 1
       addr = run[0]  # HW byte address
       hdr = (count << 26) | addr
       packets += struct.pack('<I', hdr)
       packets += struct.pack(f'<{len(run)}I', *[regs[a] for a in run])
   
   CMD_BUF = HEADER + padding_8 + packets
   CMD_BUF += b'\x00' * (0x8000 - len(CMD_BUF))
   ```

5. **Synthesize BTSP**:
   - Same as CMD_BUF but truncated to 0x4000 (or copy from CMD_BUF)

6. **Generate Python output**:
   - Write `bo_alloc()`, `submit()` functions (copy from `min_mul.py`)
   - Embed CMD_BUF and BTSP as bytearray literals
   - Include data buffer setup (inputs = 3.0, 2.0)
   - Include all named register offset constants from `min_add.py`

### Phase 2: macOS 14 (H13, extra KernelDMA)

Test with `mul_macos14.hwx` → with register cleaning, should output 6.0.

**Changes vs Phase 1:**
- Register cleaning is critical: macOS 14 has `KernelCfg=0x80`, `MACCfg=0x00100000`, 16 coefficient channels at 0x80
- Apply cleaning heuristics:
  - If CoeffBaseAddr[0..15] are all identical non-zero values → strip them
  - If KernelCfg has non-zero with no actual kernel → zero it
  - Common header at offset 0x20 will differ (`64 69` vs `a5 49`) — handle automatically since we encode from register values

### Phase 3: macOS 26 M4 (H16, different format)

Test with `mul_macos26_m1.hwx` → with H16 support, should output 6.0.

**Changes vs Phase 1:**
- H16 task header: 10 words (40 bytes), different fields
- H16 register packets: different encoding!
  ```python
  hdr = words[w_idx]; w_idx += 1
  is_masked = (hdr >> 31) & 1
  word_addr = hdr & 0x7fff  # 15 bits
  if not is_masked:  # contiguous write
      num_regs = (hdr >> 15) & 0x3f
      for j in range(num_regs + 1):
          regs[word_addr + j] = data
  else:  # masked write
      mask = (hdr >> 15) & 0xffff
      regs[word_addr] = first_data
      for bit in 0..15: if mask bit set → regs[word_addr + bit + 1] = data
  ```
- CMD_BUF header: different format (tsk_size=504)
- Register layout: different block addresses (H16: Common=0x0, L2=0x4100, PE=0x4500, NE=0x4900, TileDMA_Src=0x4D00, TileDMA_Dst=0x5100, KernelDMA=0x5500)
- Need H16 register names from `get_m4_reg_name()`

### Phase 4: Generalize

- Merge all phases into a single `hwx2py.py` that detects architecture (H13 vs H16) and handles both
- Add command-line options: `--strip-kdma`, `--set-op add|mul`

### Technical details for CMD_BUF encoding

**H13 header** (reconstructed from dump):
```
words[0] = 0x02000000  # TID=0, flags
words[1] = 0x00000000
words[2] = tsk_size    # e.g. 0x274 = 628
words[3] = 0x00000000
words[4] = 0x00fff86a
words[5] = 0x00000000
words[6] = 0x30009800
words[7] = 0x00000000
```

**Register packet encoding** (H13):
```
header = (count << 26) | byte_address  # byte_address = first register's HW addr
data = consecutive register values as u32 words
```

**Contiguous run detection**:
```python
regs = sorted(regs.items())  # (addr, value) pairs
i = 0
while i < len(regs):
    start = regs[i][0]
    run = [regs[i][1]]
    j = i + 1
    while j < len(regs) and regs[j][0] == regs[j-1][0] + 4:
        run.append(regs[j][1])
        j += 1
    # Emit run
    count = len(run) - 1
    hdr = (count << 26) | start
    # ... pack
    i = j
```

**BTSP**: Copy first 0x4000 bytes of CMD_BUF. The BTSP and CMD_BUF share the same register data; `anecc` just replicates it.

### Verification plan

1. `python examples/hwx2py.py hwx/mul.hwx -o /tmp/test.py && python /tmp/test.py` → output 6.0
2. `python examples/hwx2py.py hwx/mul_macos14.hwx -o /tmp/test.py && python /tmp/test.py` → output 6.0
3. `python examples/hwx2py.py hwx/mul_macos26_m1.hwx -o /tmp/test.py && python /tmp/test.py` → output 6.0
4. `python examples/hwx2py.py hwx/sum.hwx -o /tmp/test.py && python /tmp/test.py` → output 5.0
