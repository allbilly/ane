"""Test: sigmoid↔relu via MACCfg bit 16 flip (Family 2 equivalent of add↔mul).
Sigmoid MACCfg=0x0012000c (bit16=1, table lookup).
Relu   MACCfg=0x0011000c (bit16=0, pass-through)."""

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

# Load sigmoid firmware from source
with open('examples/sigmoid_from_hwx.py') as f:
    src = f.read()
m = re.search(r"CMD_BUF = bytes\.fromhex\('([^']+)'\)", src)
sig_cmd = bytes.fromhex(m.group(1))
m2 = re.search(r"BTSP_BUF = bytes\.fromhex\('([^']+)'\)", src)
sig_btsp = bytes.fromhex(m2.group(1))

# Relu td_data (from relu.py)
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
REL_KERNEL = b'\x00' * 4096

reg_MACCfg = 0x244

def run_sig(name, cmd_buf, btsp_buf, maccfg, test_in, expect, C=1, S=96):
    cbuf = bytearray(cmd_buf)
    bbuf = bytearray(btsp_buf)
    struct.pack_into('<I', cbuf, reg_MACCfg, maccfg)
    struct.pack_into('<I', bbuf, reg_MACCfg, maccfg)

    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    cmd_h, cmd_m = bo_alloc(fd, 32768); cmd_m.write(bytes(cbuf)); cmd_m.close()
    btsp_h, btsp_m = bo_alloc(fd, 16384); btsp_m.write(bytes(bbuf)); btsp_m.close()
    out_h, out_m = bo_alloc(fd, 16384)
    src_h, src_m = bo_alloc(fd, 16384)
    inp = np.zeros(8192, dtype=np.float16)
    for i in range(C): inp[i * S] = test_in[i] if hasattr(test_in, '__iter__') else test_in
    src_m.write(inp.tobytes())
    try:
        ret = submit(fd, 0x274, 1, 0x274, [cmd_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=8).copy()
        got = [out[i * S] for i in range(C)]
        exp_list = [expect[i] if hasattr(expect, '__iter__') else expect for _ in range(C)]
        ok = all(abs(g - e) < 0.1 for g, e in zip(got, exp_list))
        vals = ', '.join(f'{v:.4f}' for v in got)
        print(f"  {name:45s}  [{vals}]  {'✓' if ok else '✗'} (expected {[f'{e:.4f}' for e in exp_list]})")
    except Exception as e:
        print(f"  {name:45s}  HANG/{e}")
    os.close(fd)

def run_relu(name, td, kernel, maccfg, test_in, expect, C=1, S=96):
    full = bytearray(td) + bytearray(kernel)
    full += b'\x00' * (32768 - len(full))
    btsp = bytearray(full[:16384])
    btsp[2] = 0x40
    struct.pack_into('<I', full, reg_MACCfg, maccfg)
    struct.pack_into('<I', btsp, reg_MACCfg, maccfg)

    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    cmd_h, cmd_m = bo_alloc(fd, 32768); cmd_m.write(bytes(full)); cmd_m.close()
    btsp_h, btsp_m = bo_alloc(fd, 16384); btsp_m.write(bytes(btsp)); btsp_m.close()
    out_h, out_m = bo_alloc(fd, 16384)
    src_h, src_m = bo_alloc(fd, 16384)
    inp = np.zeros(8192, dtype=np.float16)
    for i in range(C): inp[i * S] = test_in[i] if hasattr(test_in, '__iter__') else test_in
    src_m.write(inp.tobytes())
    try:
        ret = submit(fd, 0x274, 1, 0x274, [cmd_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=8).copy()
        got = [out[i * S] for i in range(C)]
        exp_list = [expect[i] if hasattr(expect, '__iter__') else expect for _ in range(C)]
        ok = all(abs(g - e) < 0.1 for g, e in zip(got, exp_list))
        vals = ', '.join(f'{v:.4f}' for v in got)
        print(f"  {name:45s}  [{vals}]  {'✓' if ok else '✗'} (expected {[f'{e:.4f}' for e in exp_list]})")
    except Exception as e:
        print(f"  {name:45s}  HANG/{e}")
    os.close(fd)

print("=== Sigmoid ↔ Relu via MACCfg bit 16 ===")
print()

# Extract sigmoid td_data and kernel from full cmd_buf
sig_td = sig_cmd[:628]
sig_kernel = sig_cmd[628:628+4096]

# Run tests using proper submit (td_size=0x274)
run_sig("Sigmoid baseline (full fw)", sig_cmd, sig_btsp, 0x0012000c, 3.0, 0.9526)
run_sig("Sig→relu (MACCfg bit16=0)", sig_cmd, sig_btsp, 0x0011000c, -3.0, 0.0)
run_sig("Sig→relu (pos input)",      sig_cmd, sig_btsp, 0x0011000c, 5.0, 5.0)
run_sig("Sig→relu (MACCfg=0)",       sig_cmd, sig_btsp, 0x00000000, -3.0, 0.0)
print()

# Relu firmware tests (using relu td_data)
run_relu("Relu→sig (MACCfg+kernel)",  REL_TD, sig_kernel, 0x0012000c, 3.0, 0.9526)
run_relu("Relu→sig (neg input)",       REL_TD, sig_kernel, 0x0012000c, -3.0, 0.0474)
print()

run_relu("Relu baseline",             REL_TD, REL_KERNEL, 0x0011000c, -3.0, 0.0)
run_relu("Relu baseline (pos)",       REL_TD, REL_KERNEL, 0x0011000c, 5.0, 5.0)
print()

run_sig("Sig MACCfg, no kernel",      sig_cmd, sig_btsp, 0x0012000c, 3.0, 3.0)

print()
print("Summary: MACCfg bit 16 (0x00100000) toggles table lookup mode.")
print("bit16=0 → pass-through (relu-like). bit16=1 → table lookup (sigmoid-like).")
