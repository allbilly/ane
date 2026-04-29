"""Structured register analysis using examples/*.py BTSP_BUF directly.

Phases:
  1. Cross-operation register diff (no HW needed)
  2. Register-by-register live vs don't-care (HW needed)
  3. MACCfg comprehensive bitfield sweep (HW needed)
  4. Cfg register analysis (HW needed)

Run:  python experimental/test_structured_regs.py
"""

from fcntl import ioctl
import os, mmap, ctypes, struct, sys, importlib.util
import numpy as np

ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]
class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32),
                ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
                ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]
def _IOWR(nr, size): return (3 << 30) | (0x64 << 8) | (size << 16) | nr
DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def bo_alloc(fd, s):
    b = drm_ane_bo_init(handle=0, pad=0, size=s, offset=0); ioctl(fd, DRM_IOCTL_ANE_BO_INIT, b)
    return b.handle, mmap.mmap(fd, s, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=b.offset)

def submit(fd, tsk_sz, td_c, td_sz, handles, btsp):
    r = drm_ane_submit(tsk_size=tsk_sz, td_count=td_c, td_size=td_sz, btsp_handle=btsp, pad=0)
    for i in range(ANE_TILE_COUNT): r.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, r)

def pack_reg(buf, offset, value):
    struct.pack_into('<I', buf, offset, value)

def read_reg(buf, offset):
    return struct.unpack_from('<I', buf, offset)[0]

# ── Load BTSP_BUF from example modules ─────────────────────────────────────

def load_example(name, path, argv_override=None):
    spec = importlib.util.spec_from_file_location(name.replace('.','_'), path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = argv_override or [path]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod

def load_all_examples():
    bufs = {}
    print("Loading BTSP_BUF from examples...")
    for name, path, argv in [
        ("add", "examples/add.py", ["add.py"]),
        ("elementwise", "examples/elementwise.py", ["elementwise.py", "add"]),
        ("relu", "examples/relu.py", ["relu.py"]),
        ("sigmoid", "examples/sigmoid.py", ["sigmoid.py"]),
    ]:
        try:
            mod = load_example(name, path, argv)
            bufs[name] = mod.BTSP_BUF
            print(f"  {name}: {len(bufs[name])} bytes")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    return bufs

# ── Register offsets ────────────────────────────────────────────────────────
reg_offsets = {
    "W0":0x00,"W1":0x04,"W2":0x08,"W3":0x0c,"W4":0x10,"W5":0x14,"W6":0x18,"W7":0x1c,"W8":0x20,"W9":0x24,
    "KernelDMA":0x28,
    "CommonStream":0x124,"SrcStream":0x168,"L2Stream":0x1DC,"PEStream":0x228,"NEStream":0x23C,"DstStream":0x254,
    "InDim":0x128,"pad0":0x12c,"ChCfg":0x130,"Cin":0x134,"Cout":0x138,
    "OutDim":0x13c,"pad1":0x140,"ConvCfg":0x144,"pad2":0x148,
    "GroupConvCfg":0x14c,"TileCfg":0x150,"pad3":0x154,"pad4":0x158,"Cfg":0x15c,
    "TaskInfo":0x160,"DPE":0x164,
    "L2Cfg":0x1e0,"SourceCfg":0x1e4,"SourceBase":0x1e8,
    "SourceChannelStride":0x1ec,"SourceRowStride":0x1f0,
    "L2pad0":0x1f4,"L2pad1":0x1f8,"L2pad2":0x1fc,
    "L2pad3":0x200,"L2pad4":0x204,"L2pad5":0x208,"L2pad6":0x20c,
    "ResultCfg":0x210,"ResultBase":0x214,
    "ConvResultChannelStride":0x218,"ConvResultRowStride":0x21c,
    "L2pad7":0x220,"L2pad8":0x224,
    "PECfg":0x22c,"BiasScale":0x230,"PreScale":0x234,"FinalScale":0x238,
    "KernelCfg":0x240,"MACCfg":0x244,"MatrixVectorBias":0x248,"AccBias":0x24c,"PostScale":0x250,
    "SrcDMAConfig":0x16c,"Srcpad0":0x170,"SrcBaseAddr":0x174,
    "SrcRowStride":0x178,"SrcPlaneStride":0x17c,"SrcDepthStride":0x180,
    "SrcGroupStride":0x184,"Srcpad1":0x188,"Srcpad2":0x18c,"Srcpad3":0x190,"Srcpad4":0x194,
    "Srcpad5":0x198,"Srcpad6":0x19c,"Srcpad7":0x1a0,"SrcFmt":0x1a4,"Srcpad8":0x1a8,"SrcPadStream":0x1AC,
    "DstDMAConfig":0x258,"DstBaseAddr":0x25c,"DstRowStride":0x260,
    "DstPlaneStride":0x264,"DstDepthStride":0x268,"DstGroupStride":0x26c,"DstFmt":0x270,
}
off_to_name = {v:k for k,v in reg_offsets.items()}
# Only register offsets (exclude stream headers and task descriptor)
reg_offsets_only = {k:v for k,v in reg_offsets.items()
                    if 0x128 <= v <= 0x270 and k not in
                    ["CommonStream","SrcStream","L2Stream","PEStream","NEStream","DstStream"]}

def extract_vals(buf):
    return {name: read_reg(buf, off) for name, off in reg_offsets_only.items()}

def fmt_val(v):
    if v == 0: return "0"
    return f"**0x{v:08x}**"

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Cross-operation register diff
# ═════════════════════════════════════════════════════════════════════════════

def phase1(bufs):
    print("=" * 60)
    print("PHASE 1: CROSS-OPERATION REGISTER DIFF")
    print("=" * 60)
    print()

    all_vals = {name: extract_vals(buf) for name, buf in bufs.items()}
    ALL_OPS = list(bufs.keys())

    # Cross-op comparison table
    print("### Cross-Operation Register Comparison")
    print()
    h = "| Register | Address |"
    s = "|----------|---------|"
    for op in ALL_OPS:
        h += f" {op} |"
        s += f"{'-'*(len(op)+2)}|"
    print(h)
    print(s)
    for rname in reg_offsets_only:
        off = reg_offsets_only[rname]
        row = f"| {rname} | 0x{off:04x} |"
        for op in ALL_OPS:
            v = all_vals[op].get(rname, 0)
            row += f" {fmt_val(v)} |"
        print(row)
    print()

    # Diff counts
    print("### Register diff counts by pair")
    print()
    h2 = "| |"
    for op in ALL_OPS:
        h2 += f" {op} |"
    print(h2)
    sep = "|"
    for _ in ALL_OPS:
        sep += " --- |"
    print(sep)
    for op1 in ALL_OPS:
        row = f"| {op1} |"
        for op2 in ALL_OPS:
            if op1 == op2:
                row += " — |"
                continue
            d1, d2 = all_vals[op1], all_vals[op2]
            diffs = sum(1 for k in d1 if d1[k] != d2.get(k))
            row += f" {diffs} |"
        print(row)
    print()

    # Family detection
    # Known from the BTSP program header (offset 0x1c in CMD_BUF, not in BTSP_BUF)
    # Add/mul: 0x66490200, relu/sigmoid: 0x25400201
    # For now, we detect by checking MACCfg/PECfg patterns
    print("### Detected families (by register pattern)")
    print()
    for op in ALL_OPS:
        v = all_vals[op]
        pec = v.get("PECfg", 0)
        mac = v.get("MACCfg", 0)
        cfg = v.get("Cfg", 0)
        if pec != 0:
            family = "Family 1 (PE-based: add/mul/max/min/sq)"
        elif mac & 0x10000 or mac & 0x20000:
            family = "Family 2a (conv pipeline: relu/sigmoid)"
        else:
            family = "Unknown"
        print(f"  {op}: PECfg=0x{pec:08x} MACCfg=0x{mac:08x} -> {family}")
    print()

    return all_vals

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Register-by-register live vs don't-care
#  Tests relu<->sigmoid (same firmware family, known to be MACCfg toggle)
# ═════════════════════════════════════════════════════════════════════════════

def run_and_check(fd, buf, input_vals, expect_relu=True):
    """Run a BTSP_BUF on ANE, return (out0, out1) or None on hang."""
    btsp_h, btsp_m = bo_alloc(fd, 16384)
    btsp_m.write(bytes(buf))
    btsp_m.close()
    out_h, out_m = bo_alloc(fd, 16384)
    src_h, src_m = bo_alloc(fd, 16384)
    inp = np.zeros(8192, dtype=np.float16)
    for i, v in enumerate(input_vals):
        inp[i] = np.float16(v)
    src_m.write(inp.tobytes())
    try:
        submit(fd, 0x274, 1, 0x274,
               [btsp_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=4).copy()
        return float(out[0]), float(out[1])
    except Exception:
        return None
    finally:
        btsp_m.close(); out_m.close(); src_m.close()

def phase2(bufs, all_vals):
    print("=" * 60)
    print("PHASE 2: REGISTER-BY-REGISTER LIVE VS DON'T-CARE")
    print("=" * 60)
    print()

    # relu <-> sigmoid: same firmware family, only MACCfg differs
    if "relu" not in bufs or "sigmoid" not in bufs:
        print("Need relu and sigmoid examples loaded. Skipping.")
        return

    base_buf = bufs["relu"]
    base_vals = all_vals["relu"]
    target_vals = all_vals["sigmoid"]

    diffs = {k: (base_vals[k], target_vals[k])
             for k in base_vals if base_vals[k] != target_vals.get(k)}

    print(f"relu -> sigmoid: {len(diffs)} register differences")
    for k, (bv, tv) in diffs.items():
        print(f"  {k} (0x{reg_offsets_only[k]:04x}): 0x{bv:08x} -> 0x{tv:08x}")
    print()

    # Test: start from relu config, apply ALL target regs, then revert one at a time
    print("Testing: start from relu config, override to sigmoid regs, revert one-by-one")
    print("Input: 3.0 (should give sigmoid ~0.9526 if override works)")
    print()

    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    try:
        # Verify baseline: relu firmware with all relu regs
        r = run_and_check(fd, base_buf, [-3.0, 5.0], expect_relu=True)
        if r is None:
            print("  relu baseline: HANG (unexpected)")
        else:
            print(f"  relu baseline: [{r[0]:.4f}, {r[1]:.4f}] (expect [0, 5])")

        # Test: relu firmware + sigmoid MACCfg
        test_buf = bytearray(base_buf)
        for k, (bv, tv) in diffs.items():
            pack_reg(test_buf, reg_offsets_only[k], tv)
        r = run_and_check(fd, test_buf, [3.0], expect_relu=False)
        if r is None:
            print(f"  relu fw + all sigmoid regs: HANG")
        else:
            print(f"  relu fw + all sigmoid regs: [{r[0]:.4f}] (expect ~0.9526)")

        # Revert MACCfg to relu value
        test_buf2 = bytearray(base_buf)
        for k, (bv, tv) in diffs.items():
            if k == "MACCfg":
                pack_reg(test_buf2, reg_offsets_only[k], bv)
            else:
                pack_reg(test_buf2, reg_offsets_only[k], tv)
        r = run_and_check(fd, test_buf2, [3.0], expect_relu=True)
        if r is None:
            print(f"  sigmoid regs but MACCfg=relu: HANG")
        else:
            print(f"  sigmoid regs but MACCfg=relu: [{r[0]:.4f}] (expect ~3.0 -> relu)")
    finally:
        os.close(fd)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3: MACCfg comprehensive bitfield sweep
# ═════════════════════════════════════════════════════════════════════════════

def phase3(bufs):
    print("=" * 60)
    print("PHASE 3: MACCfg COMPREHENSIVE BITFIELD SWEEP")
    print("=" * 60)
    print()

    if "sigmoid" not in bufs:
        print("Need sigmoid example. Skipping.")
        return

    base_buf = bufs["sigmoid"]

    # Use the sigmoid MACCfg from the actual buffer
    sig_maccfg = read_reg(base_buf, 0x244)
    print(f"Sigmoid default MACCfg: 0x{sig_maccfg:08x}")
    print()

    test_configs = [
        ("Sigmoid baseline",             sig_maccfg),
        ("Relu (bit16=1)",               0x0011000c),
        ("Passthru (bit20=1)",           0x0010000c),
        ("Partial sig (bit17=1)",        0x0002000c),
        ("No activation (0x0c)",         0x0000000c),
        ("All zeros",                    0x00000000),
        ("Exp like (op_mode=13)",        0x0011000d),
        ("Bit18",                        0x0015000c),
        ("Bit19",                        0x0019000c),
        ("Bit21",                        0x0031000c),
        ("Bit22",                        0x0051000c),
        ("Bit23",                        0x0091000c),
        ("Bits21-23",                    0x00f1000c),
        ("op_mode=1 (bit8)",             0x00110800),
        ("op_mode=2 (bit9)",             0x00111000),
        ("op_mode=3 (bits8+9)",          0x00111800),
        ("op_mode=4 (bit10)",            0x00112000),
        ("op_mode=5 (bits8+10)",         0x00112800),
    ]

    print(f"| MACCfg | Description | Output | Verdict |")
    print(f"|--------|-------------|--------|---------|")

    results = []
    for desc, mcfg in test_configs:
        fd = os.open("/dev/accel/accel0", os.O_RDWR)
        try:
            buf = bytearray(base_buf)
            pack_reg(buf, 0x244, mcfg)
            r = run_and_check(fd, buf, [3.0])
            if r is None:
                print(f"| 0x{mcfg:08x} | {desc} | HANG | HANG |")
                results.append((mcfg, None, "HANG"))
            else:
                val = r[0]
                if abs(val - 0.9526) < 0.01:
                    verd = "sigmoid"
                elif abs(val - 3.0) < 0.01:
                    verd = "passthru"
                elif abs(val) < 0.01:
                    verd = "clamp(0)"
                elif val < -100 or val > 1000:
                    verd = "BROKEN"
                else:
                    verd = f"other({val:.4f})"
                print(f"| 0x{mcfg:08x} | {desc} | {val:.4f} | {verd} |")
                results.append((mcfg, val, verd))
        finally:
            os.close(fd)

    print()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Cfg register analysis
# ═════════════════════════════════════════════════════════════════════════════

def phase4(bufs):
    print("=" * 60)
    print("PHASE 4: Cfg REGISTER ANALYSIS")
    print("=" * 60)
    print()

    if "sigmoid" not in bufs:
        print("Need sigmoid example. Skipping.")
        return

    base_buf = bufs["sigmoid"]
    sig_maccfg = read_reg(base_buf, 0x244)

    print("Using sigmoid firmware + MACCfg, varying only Cfg.")
    print("Input: 3.0 (expected sigmoid ~0.9526 if Cfg doesn't affect op mode)")
    print()

    cfg_tests = [
        ("Sigmoid default",              0x04010101),
        ("Conv Cfg",                     0x04144405),
        ("Concat Cfg",                   0x04211101),
        ("GeMM Cfg",                     0x00244405),
        ("Zero",                         0x00000000),
        ("Only enable bit26",            0x04000000),
        ("Only conv_mode bit8",          0x00000100),
        ("Only dst_mode bit16",          0x00010000),
        ("Enable+conv",                  0x04000100),
        ("Enable+dst",                   0x04010000),
        ("Enable+conv+dst",              0x04010100),
    ]

    print(f"| Cfg | Description | Output | Verdict |")
    print(f"|-----|-------------|--------|---------|")

    for desc, cfg_val in cfg_tests:
        fd = os.open("/dev/accel/accel0", os.O_RDWR)
        try:
            buf = bytearray(base_buf)
            pack_reg(buf, 0x15c, cfg_val)
            r = run_and_check(fd, buf, [3.0])
            if r is None:
                print(f"| 0x{cfg_val:08x} | {desc} | HANG | HANG |")
            else:
                val = r[0]
                if abs(val - 0.9526) < 0.01:
                    verd = "sigmoid"
                elif abs(val - 3.0) < 0.01:
                    verd = "passthru"
                elif val < -100 or val > 10000:
                    verd = "BROKEN"
                else:
                    verd = f"other({val:.4f})"
                print(f"| 0x{cfg_val:08x} | {desc} | {val:.4f} | {verd} |")
        finally:
            os.close(fd)

    print()

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STRUCTURED REGISTER ANALYSIS")
    print("Using BTSP_BUF loaded directly from examples/*.py")
    print("=" * 60)
    print()

    if not os.path.exists("/dev/accel/accel0"):
        print("ERROR: /dev/accel/accel0 not found - need ANE hardware.")
        sys.exit(1)

    bufs = load_all_examples()
    print()

    if not bufs:
        print("No examples could be loaded.")
        sys.exit(1)

    all_vals = phase1(bufs)
    print()

    phase2(bufs, all_vals)
    print()

    phase3(bufs)
    print()

    phase4(bufs)
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    if "relu" in all_vals and "sigmoid" in all_vals:
        rv = all_vals["relu"]
        sv = all_vals["sigmoid"]
        diffs = {k: (rv[k], sv[k]) for k in rv if rv[k] != sv.get(k)}
        print(f"Same-firmware pair: relu <-> sigmoid ({len(diffs)} register diff)")
        for k, (bv, tv) in diffs.items():
            print(f"  {k}: 0x{bv:08x} <-> 0x{tv:08x}")
    print()

if __name__ == "__main__":
    main()
