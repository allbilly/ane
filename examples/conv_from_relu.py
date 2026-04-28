from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hwx_parsing import parse_macho

ANE_TILE_COUNT = 0x20

class reg:
    InDim, pad0, ChCfg, Cin, Cout = 0x128, 0x12c, 0x130, 0x134, 0x138
    OutDim, pad1, ConvCfg, pad2 = 0x13c, 0x140, 0x144, 0x148
    GroupConvCfg, TileCfg, pad3, pad4, Cfg = 0x14c, 0x150, 0x154, 0x158, 0x15c
    TaskInfo, DPE = 0x160, 0x164
    L2Cfg, SourceCfg, SourceBase = 0x1e0, 0x1e4, 0x1e8
    SourceChannelStride, SourceRowStride = 0x1ec, 0x1f0
    L2pad0, L2pad1, L2pad2 = 0x1f4, 0x1f8, 0x1fc
    L2pad3, L2pad4, L2pad5, L2pad6 = 0x200, 0x204, 0x208, 0x20c
    ResultCfg, ResultBase = 0x210, 0x214
    ConvResultChannelStride, ConvResultRowStride = 0x218, 0x21c
    PECfg, BiasScale, PreScale, FinalScale = 0x22c, 0x230, 0x234, 0x238
    KernelCfg, MACCfg, MatrixVectorBias, AccBias, PostScale = 0x240, 0x244, 0x248, 0x24c, 0x250
    SrcDMAConfig, Srcpad0, SrcBaseAddr = 0x16c, 0x170, 0x174
    SrcRowStride, SrcPlaneStride, SrcDepthStride = 0x178, 0x17c, 0x180
    SrcGroupStride, Srcpad2, Srcpad3, Srcpad4 = 0x184, 0x18c, 0x190, 0x194
    SrcFmt, Srcpad8 = 0x1a4, 0x1a8
    DstDMAConfig, DstBaseAddr, DstRowStride = 0x258, 0x25c, 0x260
    DstPlaneStride, DstDepthStride, DstGroupStride, DstFmt = 0x264, 0x268, 0x26c, 0x270

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
        ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64),
    ]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [
        ("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32),
        ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
        ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
    ]

def _IOWR(nr, size):
    return (3 << 30) | (0x64 << 8) | (size << 16) | nr

DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def allocate_buffer(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit_task(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    req = drm_ane_submit(
        tsk_size=tsk_size, td_count=td_count, td_size=td_size,
        btsp_handle=btsp_handle, pad=0,
    )
    for i in range(ANE_TILE_COUNT):
        req.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, req)

# --- Conv register values ---
commands = {
    reg.InDim: 0x10001, reg.pad0: 1, reg.ChCfg: 0x22, reg.Cin: 3, reg.Cout: 3,
    reg.OutDim: 0x10001, reg.pad1: 1, reg.ConvCfg: 0x5000a021, reg.pad2: 0x2041,
    reg.GroupConvCfg: 0x10001, reg.TileCfg: 1, reg.pad3: 0, reg.pad4: 0, reg.Cfg: 0x4144405,
    reg.TaskInfo: 0x100000, reg.DPE: 0,
    reg.L2Cfg: 0, reg.SourceCfg: 0x500172, reg.SourceBase: 0,
    reg.SourceChannelStride: 0x10, reg.SourceRowStride: 0x30,
    reg.L2pad0: 0x30, reg.L2pad1: 0x30, reg.L2pad2: 0,
    reg.L2pad3: 0, reg.L2pad4: 0, reg.L2pad5: 0, reg.L2pad6: 0,
    reg.ResultCfg: 0x500172, reg.ResultBase: 0x30,
    reg.ConvResultChannelStride: 0x10, reg.ConvResultRowStride: 0x30,
    reg.PECfg: 0, reg.BiasScale: 0, reg.PreScale: 0, reg.FinalScale: 0,
    reg.KernelCfg: 0x82, reg.MACCfg: 0x101c00, reg.MatrixVectorBias: 0, reg.AccBias: 0, reg.PostScale: 0x3c00,
    reg.SrcDMAConfig: 0x33881, reg.Srcpad0: 0x8880, reg.SrcBaseAddr: 0,
    reg.SrcRowStride: 0x40, reg.SrcPlaneStride: 0x40, reg.SrcDepthStride: 0xc0,
    reg.SrcGroupStride: 0, reg.Srcpad2: 0, reg.Srcpad3: 0, reg.Srcpad4: 0,
    reg.SrcFmt: 0x1002031, reg.Srcpad8: 0,
    reg.DstDMAConfig: 0xc1, reg.DstBaseAddr: 0, reg.DstRowStride: 0x40,
    reg.DstPlaneStride: 0x40, reg.DstDepthStride: 0xc0, reg.DstGroupStride: 0,
    reg.DstFmt: 0x1302031,
}

# --- Build CMD_BUF from conv.hwx firmware + kernel weights ---
with open('hwx/tinygrad/conv.hwx', 'rb') as f:
    data = f.read()
ane_data = parse_macho(data)

kernel_hex = ('00000000000000000000000000400040004000000'
    '0000000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000004000400040000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000400040004000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000')
kernel = bytes.fromhex(kernel_hex)

CMD_BUF = bytearray(ane_data + kernel)
CMD_BUF += b'\x00' * (32768 - len(CMD_BUF))

BTSP_BUF = bytearray(CMD_BUF[:0x4000])
BTSP_BUF[2] = 0x40

for offset, value in commands.items():
    struct.pack_into('<I', CMD_BUF, offset, value)
    struct.pack_into('<I', BTSP_BUF, offset, value)

# --- Run ---
BUF_SIZE = 16384
STRIDE = 32
C = 3

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_handle, cmd_map = allocate_buffer(fd, 32768)
cmd_map.write(bytes(CMD_BUF)); cmd_map.close()
out_handle, out_map = allocate_buffer(fd, BUF_SIZE)
src1_handle, src1_map = allocate_buffer(fd, BUF_SIZE)
src1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)

# Test input: 3 channels with values 1.0, 2.0, 3.0
vals = [np.float16(1.0), np.float16(2.0), np.float16(3.0)]
for i in range(C):
    src1[i * STRIDE] = vals[i]
src1_map.write(src1.tobytes()); src1_map.close()

btsp_handle, btsp_map = allocate_buffer(fd, BUF_SIZE)
btsp_map.write(bytes(BTSP_BUF)); btsp_map.close()

handles = [cmd_handle, 0, 0, 0, out_handle, src1_handle, 0] + [0] * 25
ret = submit_task(
    fd=fd, tsk_size=0x274, td_count=1, td_size=0x274,
    handles=handles, btsp_handle=btsp_handle,
)
print(f"submit returned: {ret}")
output_all = np.frombuffer(out_map, dtype=np.float16).copy(); out_map.close()
non_zero = np.where(output_all != 0)[0]
print(f"Total output values: {len(output_all)}, non-zero count: {len(non_zero)}")
if len(non_zero) > 0:
    print(f"Non-zero indices: {non_zero[:20]}")
    print(f"Non-zero values: {output_all[non_zero[:20]]}")
print("output[:64] =", output_all[:64])
os.close(fd)
