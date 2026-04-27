from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys

ANE_TILE_COUNT = 0x20
fd = os.open("/dev/accel/accel0", os.O_RDWR)

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

commands = {
    reg.InDim: 0x1004d, reg.pad0: 1, reg.ChCfg: 0x22, reg.Cin: 1, reg.Cout: 1,
    reg.OutDim: 0x1004d, reg.pad1: 1, reg.ConvCfg: 0x5000a021, reg.pad2: 0x2041,
    reg.GroupConvCfg: 0x14001, reg.TileCfg: 1, reg.pad3: 0, reg.pad4: 0, reg.Cfg: 0x04010101,
    reg.TaskInfo: 0x00100000, reg.DPE: 0,

    reg.L2Cfg: 0x6c013800, reg.SourceCfg: 0x33881, reg.SourceBase: 0x8880,
    reg.SourceChannelStride: 0, reg.SourceRowStride: 0xc0,
    reg.L2pad0: 0xc0, reg.L2pad1: 0xc0, reg.L2pad2: 0,
    reg.L2pad3: 0, reg.L2pad4: 0, reg.L2pad5: 0, reg.L2pad6: 0,
    reg.ResultCfg: 0, reg.ResultBase: 0,
    reg.ConvResultChannelStride: 0, reg.ConvResultRowStride: 0x01002031,

    reg.PECfg: 0, reg.BiasScale: 0, reg.PreScale: 0,
    reg.FinalScale: 0,

    reg.KernelCfg: 0, reg.MACCfg: 0, reg.MatrixVectorBias: 0, reg.AccBias: 0, reg.PostScale: 0,

    reg.SrcDMAConfig: 0, reg.Srcpad0: 0x00500172, reg.SrcBaseAddr: 0,
    reg.SrcRowStride: 0xa0, reg.SrcPlaneStride: 0xa0, reg.SrcDepthStride: 0xa0,
    reg.SrcGroupStride: 0xa0, reg.Srcpad2: 0, reg.Srcpad3: 0, reg.Srcpad4: 0,
    reg.SrcFmt: 0, reg.Srcpad8: 0,

    reg.DstDMAConfig: 0x040000c1, reg.DstBaseAddr: 0, reg.DstRowStride: 0xc0,
    reg.DstPlaneStride: 0xc0, reg.DstDepthStride: 0xc0, reg.DstGroupStride: 0,
    reg.DstFmt: 0x01302031,
}

def make_from_segments(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset + length] = data
    return buf

BTSP_BUF = make_from_segments(0x4000, [
    (2, 42, bytes.fromhex('40020000000022040000000000006af8ff00000000000098003000000000254002010000000000f801f4')),
    (295, 131, bytes.fromhex('3c4d000100010000002200000001000000010000004d0001000100000021a00050412000000140010001000000000000000000000001010104000010000000000000480044000000007201500000000000a0000000a0000000a0000000a000000000000000000000000000000000000000000000007a015000a0000000000000000000')),
    (426, 51, bytes.fromhex('000000000000000000000088000c0000000000000000000000000000000000c80010800000000c001100000000000000000000')),
    (477, 57, bytes.fromhex('3c00000038016c813803008088000000000000c0000000c0000000c00000000000000000000000000000000000000000000000000000000000')),
    (534, 19, bytes.fromhex('00000000000031200001000000000001000000')),
    (553, 23, bytes.fromhex('0000000000000000000000000000000000000000000000')),
    (597, 31, bytes.fromhex('780118c100000400000000c0000000c0000000c00000000000000031203001')),
])
for offset, value in commands.items():
    struct.pack_into('<I', BTSP_BUF, offset, value)

input_a = np.zeros(8192, dtype=np.float16); input_a[:2017:96] = 3.0
input_a[1::96] = 5.0

out_handle, out_map = allocate_buffer(fd, 0x4000)
src1_handle, src1_map = allocate_buffer(fd, 0x4000)
btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)

src1_map.write(input_a.tobytes())
btsp_map.write(BTSP_BUF)

ret = submit_task(
    fd=fd,
    tsk_size=0x274,
    td_count=1,
    td_size=0x274,
    handles=[btsp_handle, 0, 0, 0, out_handle, src1_handle, 0] + [0] * 25,
    btsp_handle=btsp_handle,
)
os.close(fd)

output = np.frombuffer(out_map, dtype=np.float16, count=64).copy()
print("output = ", output)
print("expected relu = ", np.maximum(0, input_a[:64]))
