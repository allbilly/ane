# macOS 26 HWX Compatibility Problems

## Problem 1: `anecc` assertion failure on macOS 26 HWX

`mul_m4_macos26.hwx` fails with `AssertionError` at `anecc/__init__.py:350`:

```
assert(len(res.nchw) == (src_count + dst_count))
```

**Root cause**: macOS 26's `coreml2hwx` adds an extra `probs/src` intermediate buffer metadata entry to the HWX Mach-O strings section. macOS 12 HWX has 3 stabs (`image`, `image2`, `probs`); macOS 26 HWX has 4 stabs (`image`, `image2`, `probs/src`, `probs`). `anecc` expects `len(nchw) == 3` (2 inputs + 1 output) but gets 4.

**Fix**: In `_anecc_get_nchw()`, filter out stabs whose names contain `/` (like `probs/src`). Real input/output tensor names never use `/`.

```diff
 	nchw_l = []
 	for i,stab in enumerate(stabs):
+		name = stab.split(":t", 1)[0]
+		if "/" in name:
+			logger.debug("STAB%d: %s: skipping (intermediate)" % (i, name))
+			continue
 		nchw = stab.split(":")[1:-1]
```

**Also required**: The original egg (v1.0.9 installed) has a much more robust implementation than the GitHub clone — it includes `_parse_hwx_macho()`, `_try_parse_text_layout()`, `TD_MAGIC_ALT = 0x4401f800`, and smarter buffer scanning. The GitHub clone at `https://github.com/eiln/anecc` is an older, simpler version that lacks these fixes and will also fail on macOS 12 HWX.

| File | stabs | Expected | Result |
|------|-------|----------|--------|
| `mul_m4.hwx` (macOS 12) | 3 | 3 | ✅ Works |
| `mul_m4_macos26.hwx` (macOS 26) | 4 | 3 | ✅ Fixed (filters `probs/src`) |

## Problem 2: `parse.py` default subtype breaks H16-format HWX

macOS 26 generates two HWX variants:
- **H13 format** (`mul_m4_macos26.hwx`): Parses correctly with default subtype=4
- **H16 format** (`mul_h16_macos26.hwx`): Needs explicit `subtype=7`

`load_hwx_data()` defaults to `subtype=4` (H13) and doesn't auto-detect architecture from the binary. H16 files get fed to the H13 parser, producing garbage output.

| File | Subtype default (4) | Subtype=7 |
|------|--------------------|-----------|
| `mul_m4_macos26.hwx` | ✅ Full H13 parse | — |
| `mul_h16_macos26.hwx` | ❌ Garbage (6 regs) | ✅ Full H16 parse |
| `mul_h16_macos26_nodebug.hwx` | ❌ Garbage (6 regs) | ✅ Full H16 parse |

**Spurious KDMA pattern** (same on macOS 14 and 26): `KernelCfg=0x80`, `MACCfg=0x00100000`, 16× `CoeffDMAConfig=0x80`. Fix via `hwx2py --clean` as documented in `macos_hwx.md`.

## System Info

- macOS 26.3 (Sequoia), Apple Clang 17.0.0, Xcode 26.2
- `anecc` v1.0.9
- ANECompiler: MPS dialect v1, SPI v1, validate network v2
- Struct sizes identical to macOS 14 — ABI is stable
- HWX file sizes: macOS 12 = 65536 bytes, macOS 26 = 81920 bytes
