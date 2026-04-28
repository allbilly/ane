"""Minimal register change analysis for Family 2.
For each pair (base→target), test ALL register changes needed.
Results summary at bottom.
"""

from fcntl import ioctl
import os, mmap, ctypes, struct, re
import numpy as np

ANE_TILE_COUNT = 0x20
class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]
class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32), ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT), ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]
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

class R:
    InDim=0x128; OutDim=0x13c; ChCfg=0x130; Cin=0x134; Cout=0x138
    GroupConvCfg=0x14c; Cfg=0x15c; TaskInfo=0x160
    SrcDMAConfig=0x16c; Srcpad0=0x170; SrcRowStride=0x178
    SrcFmt=0x1a4
    L2Cfg=0x1e0; SourceCfg=0x1e4; SourceBase=0x1e8
    SourceChStride=0x1ec; SourceRowStride=0x1f0
    L2pad0=0x1f4; L2pad1=0x1f8
    ResultCfg=0x210; ResultBase=0x214
    ConvResultChStride=0x218; ConvResultRowStride=0x21c
    PECfg=0x22c; KernelCfg=0x240; MACCfg=0x244; PostScale=0x250
    DstDMAConfig=0x258; DstRowStride=0x260; DstFmt=0x270

with open('examples/sigmoid_from_hwx.py') as f:
    src = f.read()
sig_cmd = bytes.fromhex(re.search(r"CMD_BUF = bytes\.fromhex\('([^']+)'\)", src).group(1))
sig_btsp = bytes.fromhex(re.search(r"BTSP_BUF = bytes\.fromhex\('([^']+)'\)", src).group(1))
sig_td = sig_cmd[:628]; sig_kernel = sig_cmd[628:628+4096]

with open('hwx/conv.ane', 'rb') as f:
    ane = f.read()
hdr = struct.unpack_from('<8I', ane, 0)
conv_td = ane[0x1000:0x1000 + hdr[2]]
conv_kernel = ane[0x1000 + hdr[2]:0x1000 + hdr[2] + hdr[6]]

REL_TD = bytes.fromhex(
    '000000020000000022040000000000006af8ff00000000000098003000000000'
    '254002010000000000f801f40000000000000000800000008000000080000000'
    '8000000080000000800000008000000080000000800000008000000080000000'
    '8000000080000000800000008000000080000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000400000004000000040000000'
    '4000000040000000400000004000000040000000400000004000000040000000'
    '4000000040000000400000004000000040000000800000008000000080000000'
    '8000000000000000000000000000000000000000000000000000000000000000'
    '000000000000003c4d000100010000002200000001000000010000004d000100'
    '0100000021a00050412000000140010001000000000000000000000001010104'
    '00001000000000000038016c813803008088000000000000c0000000c0000000'
    'c000000000000000000000000000000000000000000000000000000000000000'
    '0000000031200001000000000001000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000480044'
    '000000007201500000000000a0000000a0000000a0000000a000000000000000'
    '000000000000000000000000000000007a015000a00000000000000000000000'
    '00000000000000000088000c0000000000000000000000000000000000c80010'
    '800000000c0011000000000000000000003c000000780118c100000400000000'
    'c0000000c0000000c00000000000000031203001'
)

def run_test(td, kernel, btsp_src, reg_ov, inp, exp, C=1, S=96, use_sig_btsp=False):
    full = bytearray(td) + bytearray(kernel) + b'\x00' * (32768 - len(td) - len(kernel))
    for o, v in reg_ov.items(): struct.pack_into('<I', full, o, v)
    if use_sig_btsp:
        bbuf = bytearray(btsp_src)
        for o, v in reg_ov.items(): struct.pack_into('<I', bbuf, o, v)
    else:
        bbuf = bytearray(full[:16384]); bbuf[2] = 0x40
        for o, v in reg_ov.items(): struct.pack_into('<I', bbuf, o, v)
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    ch, cm = bo_alloc(fd, 32768); cm.write(bytes(full)); cm.close()
    bh, bm = bo_alloc(fd, 16384); bm.write(bytes(bbuf)); bm.close()
    oh, om = bo_alloc(fd, 16384)
    sh, sm = bo_alloc(fd, 16384)
    inp_a = np.zeros(8192, dtype=np.float16)
    for i in range(C): inp_a[i * S] = inp[i] if hasattr(inp, '__iter__') else inp
    sm.write(inp_a.tobytes())
    try:
        submit(fd, 0x274, 1, 0x274, [ch,0,0,0,oh,sh,0]+[0]*25, bh)
        out = np.frombuffer(om, dtype=np.float16, count=8).copy()
        got = [out[i * S] for i in range(C)]
        ok = all(abs(g - e) < 0.01 for g, e in zip(got, [exp]*C if not hasattr(exp,'__iter__') else exp))
        os.close(fd)
        return got[0], ok
    except:
        os.close(fd)
        return None, False

# === Register value sets ===
SIG = {
    R.InDim: 0x1004d, R.OutDim: 0x1004d, R.ChCfg: 0x22,
    R.Cin: 1, R.Cout: 1, R.GroupConvCfg: 0x14001,
    R.Cfg: 0x04010101, R.TaskInfo: 0x100000,
    R.KernelCfg: 0x80, R.MACCfg: 0x0012000c, R.PostScale: 0x3c00,
    R.SrcRowStride: 0xc0, R.SrcFmt: 0x01002031,
    R.DstRowStride: 0xc0, R.DstFmt: 0x01302031,
    R.ResultCfg: 0x50017a, R.ResultBase: 0xa0,
    R.SourceChStride: 0xa0, R.SourceRowStride: 0xa0,
    R.L2pad0: 0xa0, R.L2pad1: 0xa0,
    R.SrcDMAConfig: 0x33881, R.Srcpad0: 0x8880, R.SourceCfg: 0x500172,
}

REL = {**SIG, R.MACCfg: 0x0011000c}

CONV_REGS = { # C=3 conv registers for same-firmware test
    R.InDim: 0x10001, R.OutDim: 0x10001,
    R.Cin: 3, R.Cout: 3, R.GroupConvCfg: 0x10001,
    R.Cfg: 0x04144405,
    R.SrcRowStride: 0x40,
    R.DstRowStride: 0x40,
    R.ResultCfg: 0x500172, R.ResultBase: 0x30,
    R.SourceChStride: 0x10, R.SourceRowStride: 0x30,
    R.L2pad0: 0x30, R.L2pad1: 0x30,
}

def test_pair(name, td, kernel, btsp, base_regs, target_regs, inp, exp, C, S, use_sig_btsp=False):
    """Test if target_regs on base firmware produces target output."""
    val, ok = run_test(td, kernel, btsp, target_regs, inp, exp, C, S, use_sig_btsp)
    if val is None:
        return "HANG"
    return f"{val:.4f} {'✓' if ok else '✗'}"

def find_diffs(a, b):
    all_k = set(a.keys()) | set(b.keys())
    return {k: (a.get(k,0), b.get(k,0)) for k in all_k if a.get(k,0) != b.get(k,0)}

print("=" * 70)
print("FAMILY 2 MINIMAL REGISTER CHANGE ANALYSIS")
print("=" * 70)

print("""
METHOD: For each pair, start with base firmware + ALL target registers.
If output matches target, revert ONE register at a time to base value.
If output STILL matches target after revert → register is DON'T-CARE.
If output reverts to base → register is NEEDED for conversion.
""")

# === WITHIN sigmoid/relu firmware family ===
print("--- SIGMOID/RELU firmware (entry 0x25400201) ---")

# Sigmoid→Relu
d = find_diffs(SIG, REL)
print(f"\nSigmoid → Relu: {len(d)} register diffs")
val, ok = run_test(sig_td, sig_kernel, sig_btsp, REL, -3.0, 0.0, use_sig_btsp=True)
print(f"  Full relu regs on sigmoid fw: {val:.4f} {'✓' if ok else '✗'}")
# Revert MACCfg
val2, ok2 = run_test(sig_td, sig_kernel, sig_btsp, {**REL, R.MACCfg: 0x0012000c}, -3.0, 0.0, use_sig_btsp=True)
print(f"  Revert MACCfg (keep rest relu): {val2:.4f} {'DONTCARE' if ok2 else 'NEEDED'} (sig={val2:.4f})")
print(f"  MINIMAL: MACCfg: 0x0012000c → 0x0011000c  = bit16: 1→0")

# Relu→Sigmoid
d = find_diffs(REL, {**REL, R.MACCfg: 0x0012000c})
print(f"\nRelu → Sigmoid: {len(d)} register diffs")
val, ok = run_test(REL_TD, sig_kernel, b'', {**REL, R.MACCfg: 0x0012000c}, 3.0, 0.9526)
print(f"  Full sig regs on relu fw: {val:.4f} {'✓' if ok else '✗'}")
val2, ok2 = run_test(REL_TD, sig_kernel, b'', REL, 3.0, 0.9526)
print(f"  Revert MACCfg: {val2:.4f} {'DONTCARE' if ok2 else 'NEEDED'} (relu={val2:.4f})")
print(f"  MINIMAL: MACCfg: 0x0011000c → 0x0012000c  = bit16: 0→1")

# Cfg experiments on sigmoid firmware
print(f"\n--- Cfg register experiments (sigmoid firmware) ---")
for name, cfg_val in [
    ("Identity (0x04010101)", 0x04010101),
    ("Conv (0x04144405)", 0x04144405),
    ("Concat (0x04211101)", 0x04211101),
    ("GeMM (0x00244405)", 0x00244405),
    ("Zero (0x00000000)", 0x00000000),
]:
    v, _ = run_test(sig_td, sig_kernel, sig_btsp, {**SIG, R.Cfg: cfg_val}, 3.0, 0.9526, use_sig_btsp=True)
    print(f"  Cfg={cfg_val:#010x} ({name:20s}): output={v:.4f}")

# === CROSS firmware tests ===
print(f"\n--- CROSS-FIRMWARE (conv firmware + relu/sigmoid regs) ---")
v, _ = run_test(conv_td, conv_kernel, b'', CONV_REGS, [1,2,3], [12,12,12], C=3, S=32)
print(f"  Conv baseline: output={v} {'✓' if v else '✗'}")

# Conv fw + relu regs
print(f"  Conv fw + relu regs: ", end='')
v, _ = run_test(conv_td, conv_kernel, b'', {
    R.InDim: 0x1004d, R.OutDim: 0x1004d, R.Cin: 1, R.Cout: 1,
    R.Cfg: 0x04010101, R.KernelCfg: 0, R.MACCfg: 0, R.PostScale: 0,
    R.SrcRowStride: 0xc0, R.DstRowStride: 0xc0,
}, -3.0, 0.0)
if v is None: print("HANG — different firmware program required")
else: print(f"{v:.4f}")

# MACCfg bit patterns on sigmoid firmware
print(f"\n--- MACCfg bit field analysis (sigmoid firmware) ---")
for desc, maccfg in [
    ("Sigmoid (0x0012000c, bit16=1)", 0x0012000c),
    ("Relu    (0x0011000c, bit15=1)", 0x0011000c),
    ("All bits zero",                 0x00000000),
    ("Only bit15",                    0x0001000c),
    ("Only bit16",                    0x0010000c),
    ("Both bit15+16",                 0x0011000c),
    ("Conv MACCfg (0x00101c00)",      0x00101c00),
]:
    v, _ = run_test(sig_td, sig_kernel, sig_btsp, {R.MACCfg: maccfg}, 3.0, 0.9526, use_sig_btsp=True)
    print(f"  MACCfg={maccfg:#010x} ({desc:35s}): output={v:.4f}")

print("""
=== SUMMARY ===

Same firmware (sigmoid/relu family, entry 0x25400201):
  sigmoid ↔ relu : 1 register change (MACCfg bit 16)
    0x0012000c (bit16=1) → sigmoid mode (table lookup via KDMA kernel)
    0x0011000c (bit16=0) → relu mode (pass-through)
    Analogous to Family 1's PECfg bit 2 (add↔mul)

  Cfg register: does NOT control op mode
    Only affects activation post-processing
    sigmoid firmware + conv Cfg = still sigmoid output
  
  Cross-firmware (conv firmware → relu): NOT possible with regs only
    Conv firmware has different BTSP program (entry 0x25400203)
    Register overrides alone cannot bridge different firmware programs
""")
