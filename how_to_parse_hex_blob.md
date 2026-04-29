# How to Parse an ANE Hex Blob

## Workflow

### 1. Extract hex blob from the source file

The hex blob is in a `bytes.fromhex('...')` string. Extract it:

```python
import re, struct
with open('file.py') as f:
    content = f.read()
m = re.search(r"CMD_BUF = bytes\.fromhex\('([^']+)'\)", content)
data = bytes.fromhex(m.group(1))
```

Check `BTSP_BUF` too — it may differ from `CMD_BUF` (sigmoid.py has both as separate blobs).

### 2. Decode all register values at once

Run this decode script:

```python
print("=== Task Descriptor (0x00-0x28) ===")
for off in [0x00, 0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28]:
    print(f"  0x{off:04x}: 0x{struct.unpack_from('<I', data, off)[0]:08x}")

print("=== Common (0x124-0x168) ===")
for off in range(0x124, 0x169, 4):
    print(f"  0x{off:04x}: 0x{struct.unpack_from('<I', data, off)[0]:08x}")

# Then repeat for TileDMA Src (0x168-0x1DC), L2 (0x1DC-0x228),
# PE (0x228-0x23C), NE (0x23C-0x254), Dst (0x254-0x274)
```

Already have this script? Copy-paste the register offsets from `add.py` or `relu.py`'s `reg` class — they're always the same.

### 3. Compare with a reference file

Pick the closest reference (`add.py` for element-wise, `relu.py` for 77x1x1 activation). List the differences to know what changed:

```python
for off in range(0, 0x280, 4):
    ref_val = struct.unpack_from('<I', ref_data, off)[0]
    new_val = struct.unpack_from('<I', data, off)[0]
    if ref_val != new_val:
        print(f"DIFF 0x{off:04x}: ref=0x{ref_val:08x} new=0x{new_val:08x}")
```

### 4. Firmware DMA (0x2C-0x124, 62 words)

This region uses **big-endian** word encoding. Decode with `<I` to get the stored bytes, then convert to big-endian values for `struct.pack('>I', ...)`:

```python
for off in range(0x2C, 0x124, 4):
    le_val = struct.unpack_from('<I', data, off)[0]
    # For struct.pack('>I', ...), use the byte-swapped value:
    be_val = struct.unpack('>I', struct.pack('<I', le_val))[0]
    # Or just add zero bytes: le_val=0x80 → be_val=0x80000000
```

Shortcut: `0x000000XX` (LE) → `0xXX000000` (BE). `0x00000080` → `0x80000000`.

### 5. Build the structured version

Follow the segment layout from the closest reference (`add.py` or `relu.py`):

| Segment | Offset | Size | Description |
|---------|--------|------|-------------|
| Task Desc | 0 | 44 | W0-W9, KernelDMA |
| Firmware DMA | 0x2C | 0xF8 | 62 × BE words |
| Common + Src | 292 | 136/184 | 0x124 to 0x1AC/0x1DC |
| L2 | 476/477 | 57/68 | 0x1DC to 0x214/0x220 |
| PE + NE | 552/553 | 43/44 | 0x228 to 0x253/0x254 |
| Dst | 596/597 | 31/32 | 0x254 to 0x273/0x274 |

Use `make_from_segments`, `build_seg`, `stream_header`, `pack_reg` from the reference.

### 6. Simplify: eliminate CMD_BUF

The structured style only uses `BTSP_BUF`. The first handle entry is `btsp_handle` (same buffer). No separate command buffer needed. `CMD_BUF` and `BTSP_BUF` are usually identical except for W0's `nid` field (which may not matter).

### 7. Verify byte-for-byte

```python
exec(open('new_file.py').read().split("STRIDE")[0])  # load BTSP_BUF
exec(open('old_file.py').read().split("STRIDE")[0])  # load both blobs
for off in range(0, 0x280, 4):
    # compare struct.unpack('<I', new_btsp, off) vs old
```

## Common Register Fields To Decode

- **W0**: `tid=(v>>0)&0xF`, `nid=(v>>16)&0x1FF`, `eon=(v>>25)&1`
- **W8** (base_ene): `rbase0=v&0x1F`, `rbe0=(v>>5)&1`, `rbase1=(v>>6)&0x1F`, `rbe1=(v>>11)&1`, `wbase=(v>>12)&0x3F`, `el0_en=(v>>24)&1`
- **InDim/OutDim**: `h=(v>>16)&0xFFFF`, `w=v&0xFFFF`
- **ChCfg**: `infmt=v&3`, `pad0=(v>>2)&3`, `outfmt=(v>>4)&3`
- **ConvCfg**: `kw=v&0x1F`, `kh=(v>>5)&0x1F`, `sx=(v>>13)&0xF`, `px=(v>>17)&0xF`, `ox=(v>>28)&3`, `oy=(v>>30)&3`
- **SrcFmt/DstFmt**: `fmt_mode=v&0xF`, `truncate=(v>>4)&0xF`, `mem_fmt=(v>>12)&0xF`, `interleave=(v>>24)&0xF`
