# macOS 14/26 HWX Compatibility Problems

## Current HWX Problem Statement

We are trying to make a simple CoreML-generated elementwise MUL HWX usable both:

- on macOS, through Apple's private HWX runtime path (`run_hwx_with_ane_client`)
- on Asahi/Linux, after conversion to an `.ane` command buffer

The important current finding is that a single system HWX file is enough for the macOS private runner. Copying a known-good system model such as:

```bash
/System/Library/PrivateFrameworks/VideoProcessing.framework/Versions/A/Resources/cnn_frame_enhancer_320p.H13.espresso.hwx
```

to `/tmp` still runs. Removing nearby `/tmp` companion metadata does not stop it. So the direct macOS runner does **not** require a full Espresso bundle at runtime for this path; the HWX itself contains enough program information.

Our generated `/tmp/hwx_output/mul/model.hwx` is different: it is generated successfully from:

```text
/tmp/mul.mlmodel
/tmp/espresso_ir_dump/net.plist
/tmp/espresso_ir_dump/net.precompilation_info
/tmp/espresso_ir_dump/net_aux.json
/tmp/espresso_ir_dump/net.additional.weights
```

but does not behave like the system `.H13.espresso.hwx` under the macOS private runner. This means the problem is now more likely in the **compile inputs/options used to produce the HWX**, not in missing runtime companion files.

Current suspected missing piece:

```text
OptionsFilePath -> net_options.plist
```

We should reconstruct the expected `net_options.plist` schema instead of passing random flags. Candidate fields seen in system-style metadata / notes include:

```text
ane_compiler_batch = 1
anec_flags = SpatialSplitGenericDAG
compress_sparse = 1
per_network_configuration = 1
export_method = Photon-v0.12.1
ModuleCompilationFlags = ...
```

Open experiment:

1. Generate `/tmp/espresso_ir_dump/net_options.plist` with the expected schema.
2. Pass it through `OptionsFilePath` in `coreml_to_ane_hwx/coreml_util.m`.
3. Recompile `/tmp/hwx_output/mul/model.hwx`.
4. Compare against the working system `cnn_frame_enhancer_320p.H13.espresso.hwx` at the structural level: compiler strings, Mach-O sections, TD offset/size/magic, and register blocks.
5. Test the regenerated HWX with `run_hwx_with_ane_client`.

This is separate from `ANE-LM` / `_ANEInMemoryModel`: those APIs compile MIL text to in-memory ANE kernels and do not directly answer the static `.hwx` compile-options problem.

## Current verified artifacts

The local `mul` artifacts show three different compiler generations:

| File | ANECompiler | HWX size | TD offset | TD size | TD magic | Notes |
|------|-------------|----------|-----------|---------|----------|-------|
| `hwx/mul.hwx` | `zin_ane_compiler v5.4.1` | 49152 | `0x4000` | `0x274` | `0xf401f800` | Known-good macOS 12 clean old H13 |
| `hwx/mul_macos14.hwx` | `zin_ane_compiler v7.6.4` | 49152 | `0x4000` | `0x274` | `0xf401f800` | Old H13 layout plus spurious KDMA/NE |
| `hwx/mul_macos26_m1.hwx` | `zin_ane_compiler v9.509.0` | 65536 | `0x8000` | `0x1f8` | `0x4401f800` | Compact alternate H13 TD plus spurious KDMA/NE |
| `hwx/mul_macos26_h13.hwx` | `zin_ane_compiler v9.509.0` | 65536 | `0x8000` | `0x1f8` | `0x4401f800` | Newly generated; differs from `mul_macos26_m1.hwx` only by embedded output path |

All four HWX files are CPU subtype `4`, i.e. H13/A14/M1 format. The macOS 26 files are still H13, but use a compact alternate task descriptor.

Compiler version can be checked with:

```bash
strings -a hwx/mul_macos14.hwx | rg -i "ANEC v|zin_ane_compiler|ModuleVersion|ModuleBundleName"
```

## Problem 1: `anecc` assertion failure on macOS 26 HWX

`mul_m4_macos26.hwx` fails with `AssertionError` at `anecc/__init__.py:350`:

```
assert(len(res.nchw) == (src_count + dst_count))
```

**Root cause**: Some macOS 26 `coreml2hwx` outputs add an extra `probs/src` intermediate buffer metadata entry to the HWX Mach-O strings section. macOS 12 HWX has 3 stabs (`image`, `image2`, `probs`); affected macOS 26 HWX has 4 stabs (`image`, `image2`, `probs/src`, `probs`). `anecc` expects `len(nchw) == 3` (2 inputs + 1 output) but gets 4.

Note: the currently regenerated local files `hwx/mul_macos26_m1.hwx` and `hwx/mul_macos26_h13.hwx` only contain 3 stabs (`image`, `image2`, `probs`), so this bug is not triggered by those exact files. The filter is still the correct defensive fix for affected macOS 26 artifacts.

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

**Also required for compact macOS 26 H13 TDs**: `anecc` must handle `TD_MAGIC_ALT = 0x4401f800` and `td_size = tsk_size = 0x1f8`. The older/simple GitHub clone assumes the old H13 `0xf401f800` / `0x274` layout and fails before it reaches NCHW validation.

| File | stabs | Expected | Result |
|------|-------|----------|--------|
| `mul_m4.hwx` (macOS 12) | 3 | 3 | ✅ Works |
| `mul_m4_macos26.hwx` (macOS 26) | 4 | 3 | ✅ Fixed (filters `probs/src`) |

## Problem 2: macOS 14/26 spurious KDMA/NE state for elementwise MUL

`hwx/mul.hwx` and `hwx/mul_macos14.hwx` use the same old H13 TD layout and the same functional PE elementwise MUL path. Decoded with the offsets from `examples/elementwise.py`, the key functional fields are identical:

| TD offset | Field | Value |
|-----------|-------|-------|
| `0x22c` | `PECfg` | `0x00080004` (`OpMode=1`, MUL) |
| `0x128` | `InDim` | `0x00010001` |
| `0x134` | `Cin` | `0x40` |
| `0x138` | `Cout` | `0x40` |
| `0x178` | `SrcRowStride` | `0x40` |
| `0x260` | `DstRowStride` | `0x40` |

The differences are header/noise plus spurious KDMA/NE fields:

| TD offset | Field | `mul.hwx` | `mul_macos14.hwx` |
|-----------|-------|-----------|-------------------|
| `0x008` | `W2/ExeCycles` | `0x00000422` | `0x0000042a` |
| `0x020` | `W8/base_ene` | `0x000249a5` | `0x00026964` |
| `0x034..0x070` | `CoeffDMAConfig[0..15]` | `0` | `0x80` |
| `0x0b4..0x0f0` | `CoeffBfrSize[0..15]` | `0` | `0x40` |
| `0x1ac` | `SrcPadStream/pad9` | `0` | `0x100` |
| `0x240` | `KernelCfg` | `0` | `0x80` |
| `0x244` | `MACCfg` | `0` | `0x00100000` |

`hwx/mul.hwx` has `MACCfg=0`; the MUL operation is encoded by `PECfg OpMode=1`. `examples/elementwise.py mul` additionally patches `MACCfg=0x30`, but that is not present in the raw `hwx/mul.hwx`.

The previously documented statement that compiled `.ane` files differ by "only 2 bytes" is not correct for raw files. Actual local comparison:

| Comparison | Result |
|------------|--------|
| `hwx/mul.ane` vs `hwx/mul_macos14.ane` | same size, 46 differing bytes |
| `hwx/mul.ane` vs `hwx/mul_macos26_h13.ane` | macOS 26 `.ane` is 128 bytes smaller; 203 differing bytes in shared prefix |
| `hwx/mul_macos26_m4.ane` vs `hwx/mul_macos26_h13.ane` | byte-identical |

The practical fix for elementwise `mul_macos14` is not "2 bytes"; it is cleaning the spurious KDMA/NE register state:

```text
KernelCfg = 0
MACCfg = 0
CoeffDMAConfig[0..15] = 0
CoeffBfrSize[0..15] = 0
```

For Asahi conversion, these spurious registers can matter because they become part of the emitted `.ane` command buffer unless the converter normalizes them. The raw generated `.ane` files are not currently a "2-byte difference" case; local comparisons show dozens or hundreds of byte differences depending on which macOS-generated HWX is used.

The logical reason output becomes `0.0` is that the macOS 14/26 elementwise HWX advertises coefficient/kernel DMA state even though elementwise MUL should not need a coefficient load. The hardware/runtime can then execute the PE MUL path with bogus NE/KDMA state, effectively feeding/using invalid coefficient-related state and producing zero instead of `2.0 * 3.0 = 6.0`.

## How to test `mul_macos14.hwx` on Asahi

On an Asahi machine with `/dev/accel/accel0` and the ANE KMD installed:

1. Convert the raw macOS 14 HWX with `anecc`:

```bash
anecc hwx/mul_macos14.hwx -o hwx/mul_macos14.ane
python run.py hwx/mul_macos14.ane
```

Expected result if raw spurious KDMA/NE is harmless on that stack:

```text
6.0
```

Likely failure mode if the hardware honors the bogus KDMA state:

```text
0.0
```

2. Test the direct-register reference:

```bash
python examples/elementwise.py mul
```

Expected:

```text
6.0
```

3. Generate and run a cleaned command buffer from the macOS 14 HWX:

```bash
python experimental/hwx2py.py hwx/mul_macos14.hwx --clean -o /tmp/mul14_clean.py
python /tmp/mul14_clean.py
```

Expected:

```text
output[0] = 6.0
```

If raw `mul_macos14.ane` fails but `examples/elementwise.py mul` and the cleaned `hwx2py` script pass, the incompatibility is isolated to the spurious KDMA/NE fields rather than shape, tiling, L2, PE, or TileDMA setup.

## Problem 3: `parse.py` default subtype breaks H16-format HWX

macOS 26 generates two HWX variants:
- **H13 format** (`mul_m4_macos26.hwx`): Parses correctly with default subtype=4
- **H16 format** (`mul_h16_macos26.hwx`): Needs explicit `subtype=7`

`load_hwx_data()` defaults to `subtype=4` (H13) and doesn't auto-detect architecture from the binary. H16 files get fed to the H13 parser, producing garbage output.

| File | Subtype default (4) | Subtype=7 |
|------|--------------------|-----------|
| `mul_m4_macos26.hwx` | ✅ Full H13 parse | — |
| `mul_h16_macos26.hwx` | ❌ Garbage (6 regs) | ✅ Full H16 parse |
| `mul_h16_macos26_nodebug.hwx` | ❌ Garbage (6 regs) | ✅ Full H16 parse |

**Spurious KDMA pattern** (same on macOS 14 and 26 for elementwise): `KernelCfg=0x80`, `MACCfg=0x00100000`, 16× `CoeffDMAConfig=0x80`. Fix via `hwx2py --clean` or by normalizing the TD as documented above.

## System Info

- macOS 26.3 (Sequoia), Apple Clang 17.0.0, Xcode 26.2
- `anecc` v1.0.9
- ANECompiler: MPS dialect v1, SPI v1, validate network v2
- Struct sizes identical to macOS 14 — ABI is stable
- Current local `mul` HWX sizes:
  - `hwx/mul.hwx` macOS 12 clean H13: 49152 bytes
  - `hwx/mul_macos14.hwx`: 49152 bytes
  - `hwx/mul_macos26_m1.hwx`: 65536 bytes
  - `hwx/mul_macos26_h13.hwx`: 65536 bytes
- The earlier note "HWX file sizes: macOS 12 = 65536 bytes, macOS 26 = 81920 bytes" is not true for the current local `mul` files.



› Here's the complete detailed breakdown:
  What changed in anecc
  Single change: _anecc_get_nchw() in anecc/__init__.py:167
  # Before (line 165-166):
  nchw_l = []
  for i,stab in enumerate(stabs):
      nchw = stab.split(":")[1:-1]
  # After:
  nchw_l = []
  for i,stab in enumerate(stabs):
      name = stab.split(":t", 1)[0]
      if "/" in name:
          logger.debug("STAB%d: %s: skipping (intermediate)" % (i, name))
          continue
      nchw = stab.split(":")[1:-1]
  macOS 12 stabs: image, image2, probs → 3
  macOS 26 stabs: image, image2, probs/src, probs → 4
  probs/src is an intermediate source-alias buffer that macOS 26's coreml2hwx adds. It contains / in its name, unlike real IO buffers. Filtering it out restores len(nchw)
  = 3 = src_count(2) + dst_count(1).
  How to generate and compare
  # 1. Generate .ane from macOS 12 HWX
  anecc mul_m4.hwx -o /tmp/macos12.ane
  # → anecc::info: found input 1/2: (1, 64, 1, 1)
  # → anecc::info: found input 2/2: (1, 64, 1, 1)
  # → anecc::info: found output 1/1: (1, 64, 1, 1)
  # 2. Generate .ane from macOS 26 HWX (needs the fix above)
  anecc mul_m4_macos26.hwx -o /tmp/macos26.ane
  # → same output
  # 3. Compare
  ls -la /tmp/macos12.ane /tmp/macos26.ane
  # both 20992 bytes
  diff <(xxd /tmp/macos12.ane) <(xxd /tmp/macos26.ane)
  # output: only 1 line differs
  # 258c258
  # < 00001010: 6af8 ff00 0000 0000 0098 0030 0000 0000
  # ---
  # > 00001010: fffb ff00 0000 0000 0098 0030 0000 0000
  What differs
  At file offset 0x1010 (payload offset 0x10 — the W4 register = debug_log_events in the TD header):
  File  Value   Register
  macOS 12 .ane 0x00fff86a      debug_log_events=0xfff86a
  macOS 26 .ane 0x00fffbff      debug_log_events=0xfffbff
  Everything else (header, NCHW metadata, stride configs, tile layout, kernel weights) is byte-identical. This is a cosmetic compiler difference — the debug event mask
  doesn't affect computation. , is it true for the current *.ane, or i need patch anecc
