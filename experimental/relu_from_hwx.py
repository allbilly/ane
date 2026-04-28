#!/usr/bin/env python3
from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

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

commands = {
        reg.InDim: 0x1004d,
        reg.pad0: 0x1,
        reg.ChCfg: 0x22,
        reg.Cin: 0x1,
        reg.Cout: 0x1,
        reg.OutDim: 0x1004d,
        reg.pad1: 0x1,
        reg.ConvCfg: 0x5000a021,
        reg.pad2: 0x2041,
        reg.GroupConvCfg: 0x14001,
        reg.TileCfg: 0x1,
        reg.pad3: 0x0,
        reg.pad4: 0x0,
        reg.Cfg: 0x4010101,
        reg.TaskInfo: 0x100000,
        reg.DPE: 0x0,
        reg.L2Cfg: 0x0,
        reg.SourceCfg: 0x500172,
        reg.SourceBase: 0x0,
        reg.SourceChannelStride: 0xa0,
        reg.SourceRowStride: 0xa0,
        reg.L2pad0: 0xa0,
        reg.L2pad1: 0xa0,
        reg.L2pad2: 0x0,
        reg.L2pad3: 0x0,
        reg.L2pad4: 0x0,
        reg.L2pad5: 0x0,
        reg.L2pad6: 0x0,
        reg.ResultCfg: 0x50017a,
        reg.ResultBase: 0xa0,
        reg.ConvResultChannelStride: 0x0,
        reg.ConvResultRowStride: 0x0,
        reg.PECfg: 0x0,
        reg.BiasScale: 0x0,
        reg.PreScale: 0x0,
        reg.FinalScale: 0x0,
        reg.KernelCfg: 0x80,
        reg.MACCfg: 0x11000c,
        reg.MatrixVectorBias: 0x0,
        reg.AccBias: 0x0,
        reg.PostScale: 0x3c00,
        reg.SrcDMAConfig: 0x33881,
        reg.Srcpad0: 0x8880,
        reg.SrcBaseAddr: 0x0,
        reg.SrcRowStride: 0xc0,
        reg.SrcPlaneStride: 0xc0,
        reg.SrcDepthStride: 0xc0,
        reg.SrcGroupStride: 0x0,
        reg.Srcpad2: 0x0,
        reg.Srcpad3: 0x0,
        reg.Srcpad4: 0x0,
        reg.SrcFmt: 0x1002031,
        reg.Srcpad8: 0x0,
        reg.DstDMAConfig: 0x40000c1,
        reg.DstBaseAddr: 0x0,
        reg.DstRowStride: 0xc0,
        reg.DstPlaneStride: 0xc0,
        reg.DstDepthStride: 0xc0,
        reg.DstGroupStride: 0x0,
        reg.DstFmt: 0x1302031,
}

def make_from_segments(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset + length] = data
    return buf

CMD_BUF = make_from_segments(32768, [
        (0, 628, bytes.fromhex(
            '000000020000000022040000000000006af8ff00000000000098003000000000254002010000000000f801f400000000000000000000000000000000' +
            '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000' +
            '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000' +
            '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000' +
            '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003c4d000100' +
            '010000002200000001000000010000004d0001000100000021a000504120000001400100010000000000000000000000010101040000100000000000' +
            '00480044000000007201500000000000a0000000a0000000a0000000a000000000000000000000000000000000000000000000007a015000a0000000' +
            '000000000000000000000000000000000088000c0000000000000000000000000000000000c80010800000000c0011000000000000000000003c0000' +
            '0038016c813803008088000000000000c0000000c0000000c00000000000000000000000000000000000000000000000000000000000000000000000' +
            '312000010000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000780118' +
            'c100000400000000c0000000c0000000c00000000000000031203001'))
])
BTSP_BUF = bytearray(CMD_BUF[:0x4000])
BTSP_BUF[2] = 0x40

for offset, value in commands.items():
    struct.pack_into('<I', CMD_BUF, offset, value)
    struct.pack_into('<I', BTSP_BUF, offset, value)

BUF_SIZE = 16384
STRIDE = 96
C = 1

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

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_handle, cmd_map = allocate_buffer(fd, 32768)
cmd_map.write(bytes(CMD_BUF)); cmd_map.close()
out_handle, out_map = allocate_buffer(fd, BUF_SIZE)
src1_handle, src1_map = allocate_buffer(fd, BUF_SIZE)
src1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
src1[:C * STRIDE:STRIDE] = np.float16(3.0)
src1_map.write(src1.tobytes()); src1_map.close()
if False:
    src2_handle, src2_map = allocate_buffer(fd, BUF_SIZE)
    src2 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
    src2[:C * STRIDE:STRIDE] = np.float16(2.0)
    src2_map.write(src2.tobytes()); src2_map.close()
btsp_handle, btsp_map = allocate_buffer(fd, BUF_SIZE)
btsp_map.write(bytes(BTSP_BUF)); btsp_map.close()

handles = [cmd_handle, 0, 0, 0, out_handle, src1_handle, src2_handle if False else 0] + [0] * 25
ret = submit_task(
    fd=fd,
    tsk_size=0x274,
    td_count=1,
    td_size=0x274,
    handles=handles,
    btsp_handle=btsp_handle,
)
output = np.frombuffer(out_map, dtype=np.float16, count=64).copy(); out_map.close()
print("output =", output)
os.close(fd)
