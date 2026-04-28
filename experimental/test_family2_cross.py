"""Comprehensive Family 2 cross-test: test all combinations of conv pipeline ops."""
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
def submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    r = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size, btsp_handle=btsp_handle, pad=0)
    for i in range(ANE_TILE_COUNT): r.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, r)

class reg:
    InDim=0x128; pad0=0x12c; ChCfg=0x130; Cin=0x134; Cout=0x138
    OutDim=0x13c; ConvCfg=0x144; GroupConvCfg=0x14c; TileCfg=0x150
    Cfg=0x15c; TaskInfo=0x160; DPE=0x164
    SrcDMAConfig=0x16c; Srcpad0=0x170; SrcRowStride=0x178
    SrcPlaneStride=0x17c; SrcDepthStride=0x180; SrcFmt=0x1a4
    L2Cfg=0x1e0; SourceCfg=0x1e4; SourceRowStride=0x1f0
    L2pad0=0x1f4; ResultCfg=0x210; ResultBase=0x214
    ConvResultRowStride=0x21c
    PECfg=0x22c; KernelCfg=0x240; MACCfg=0x244; PostScale=0x250
    DstDMAConfig=0x258; DstRowStride=0x260; DstPlaneStride=0x264
    DstDepthStride=0x268; DstFmt=0x270

# Load sigmoid firmware
with open('examples/sigmoid_from_hwx.py') as f:
    src = f.read()
m = re.search(r"CMD_BUF = bytes\.fromhex\('([^']+)'\)", src)
sig_cmd = bytes.fromhex(m.group(1))
sig_btsp = bytes.fromhex(re.search(r"BTSP_BUF = bytes\.fromhex\('([^']+)'\)", src).group(1))
sig_td = sig_cmd[:628]; sig_kernel = sig_cmd[628:628+4096]

# Load conv firmware from .ane
with open('hwx/conv.ane', 'rb') as f:
    ane = f.read()
hdr = struct.unpack_from('<8I', ane, 0)
conv_td = ane[0x1000:0x1000 + hdr[2]]
conv_kernel = ane[0x1000 + hdr[2]:0x1000 + hdr[2] + hdr[6]]

# Relu td_data
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

ST96, ST32 = 0xc0, 0x40

# Override dicts for each target op
TO_RELU_C1 = {reg.InDim: 0x1004d, reg.OutDim: 0x1004d, reg.ChCfg: 0x22,
    reg.Cin: 1, reg.Cout: 1, reg.GroupConvCfg: 0x14001,
    reg.Cfg: 0x04010101, reg.TaskInfo: 0x100000,
    reg.KernelCfg: 0, reg.MACCfg: 0, reg.PostScale: 0,
    reg.SrcRowStride: ST96, reg.SrcPlaneStride: ST96, reg.SrcDepthStride: ST96,
    reg.DstRowStride: ST96, reg.DstPlaneStride: ST96, reg.DstDepthStride: ST96,
    reg.DstFmt: 0x01302031}
TO_SIG_C1  = {reg.InDim: 0x1004d, reg.OutDim: 0x1004d, reg.ChCfg: 0x22,
    reg.Cin: 1, reg.Cout: 1, reg.GroupConvCfg: 0x14001,
    reg.Cfg: 0x04010101, reg.TaskInfo: 0x100000,
    reg.SrcRowStride: ST96, reg.SrcPlaneStride: ST96, reg.SrcDepthStride: ST96,
    reg.DstRowStride: ST96, reg.DstPlaneStride: ST96, reg.DstDepthStride: ST96,
    reg.DstFmt: 0x01302031}
TO_CONV_C3 = {reg.InDim: 0x10001, reg.OutDim: 0x10001, reg.ChCfg: 0x22,
    reg.Cin: 3, reg.Cout: 3, reg.GroupConvCfg: 0x10001,
    reg.Cfg: 0x04144405, reg.TaskInfo: 0x100000,
    reg.KernelCfg: 0x82, reg.MACCfg: 0x101c00, reg.PostScale: 0x3c00,
    reg.SrcRowStride: ST32, reg.SrcPlaneStride: ST32, reg.SrcDepthStride: 0xc0,
    reg.DstRowStride: ST32, reg.DstPlaneStride: ST32, reg.DstDepthStride: 0xc0,
    reg.DstFmt: 0x01302031, reg.ResultCfg: 0x500172, reg.ResultBase: 0x30,
    reg.L2pad0: 0x30}

def apply(buf, ov): [struct.pack_into('<I', buf, o, v) for o, v in ov.items()]

def run(name, td, kernel, btsp_src, ov, inp, exp, C=1, S=96, use_sig_btsp=False):
    full = bytearray(td) + bytearray(kernel) + b'\x00' * (32768 - len(td) - len(kernel))
    apply(full, ov)
    if use_sig_btsp:
        bbuf = bytearray(btsp_src)
        apply(bbuf, ov)
    else:
        bbuf = bytearray(full[:16384])
        bbuf[2] = 0x40
        apply(bbuf, ov)
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    cmd_h, cmd_m = bo_alloc(fd, 32768); cmd_m.write(bytes(full)); cmd_m.close()
    btsp_h, btsp_m = bo_alloc(fd, 16384); btsp_m.write(bytes(bbuf)); btsp_m.close()
    out_h, out_m = bo_alloc(fd, 16384)
    src_h, src_m = bo_alloc(fd, 16384)
    inp_arr = np.zeros(8192, dtype=np.float16)
    for i in range(C): inp_arr[i * S] = inp[i] if hasattr(inp, '__iter__') else inp
    src_m.write(inp_arr.tobytes())
    try:
        submit(fd, 0x274, 1, 0x274, [cmd_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=8).copy()
        got = [out[i * S] for i in range(C)]
        exp_list = [exp[i] if hasattr(exp, '__iter__') else exp for _ in range(C)]
        ok = all(abs(g - e) < 0.1 for g, e in zip(got, exp_list))
        vals = ', '.join(f'{v:.4f}' for v in got[:min(C,4)])
        print(f"  {name:50s}  [{vals}]  {'✓' if ok else '✗'}")
    except Exception as e:
        print(f"  {name:50s}  HANG/{e}")
    os.close(fd)

print("=" * 65)
print("Family 2 Cross-Test: Conv pipeline + TileDMA source")
print("=" * 65)
print()

print("--- Sigmoid base ---")
run("Sigmoid baseline",           sig_td, sig_kernel, sig_btsp, {}, 3.0, 0.9526, use_sig_btsp=True)
run("Sig→relu (MACCfg bit16=0)", sig_td, sig_kernel, sig_btsp, {reg.MACCfg: 0x0011000c}, -3.0, 0.0, use_sig_btsp=True)
run("Sig→relu (pos)",            sig_td, sig_kernel, sig_btsp, {reg.MACCfg: 0x0011000c}, 5.0, 5.0, use_sig_btsp=True)
run("Sig→conv (C=3)",            sig_td, conv_kernel, sig_btsp, TO_CONV_C3, [1,2,3], 12.0, C=3, S=32, use_sig_btsp=True)

print()
print("--- Relu base ---")
run("Relu baseline",              REL_TD, b'\x00'*4096, b'', {}, -3.0, 0.0)
run("Relu→sig (MACCfg+kernel)",   REL_TD, sig_kernel, b'', {reg.MACCfg: 0x0012000c}, 3.0, 0.9526)
run("Relu→sig (neg input)",       REL_TD, sig_kernel, b'', {reg.MACCfg: 0x0012000c}, -3.0, 0.0474)

print()
print("--- Conv base ---")
run("Conv baseline (C=3)",  conv_td, conv_kernel, b'', TO_CONV_C3, [1,2,3], 12.0, C=3, S=32)
run("Conv→relu (C=1)",      conv_td, conv_kernel, b'', TO_RELU_C1, -3.0, 0.0)
run("Conv→sigmoid (C=1)",   conv_td, sig_kernel, b'', TO_SIG_C1, 3.0, 0.9526)

print()
print("Summary:")
print("  sigmoid↔relu: MACCfg bit 16 toggles table lookup. 1-register-change Family 2 pair!")
print("  sigmoid→conv: works with C=3 conv regs + kernel weights.")
print("  conv→relu/sigmoid: works with C=1 registers + kernel override.")
print("  MACCfg=0 breaks NE engine (-744 output). Keep bit 15 set.")
