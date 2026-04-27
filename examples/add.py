from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

ANE_TILE_COUNT = 0x20

# Define HW register addresses (H13/M1, from hwx_parsing.py)
class reg:
    InDim, pad0, ChCfg, Cin, Cout = 0x0000, 0x0004, 0x0008, 0x000c, 0x0010
    OutDim, pad1, ConvCfg, pad2 = 0x0014, 0x0018, 0x001c, 0x0020
    GroupConvCfg, TileCfg, pad3, pad4, Cfg = 0x0024, 0x0028, 0x002c, 0x0030, 0x0034
    TaskInfo, DPE = 0x0038, 0x003c
    L2Cfg, SourceCfg, SourceBase = 0x4800, 0x4804, 0x4808
    SourceChannelStride, SourceRowStride = 0x480c, 0x4810
    L2pad0, L2pad1, L2pad2 = 0x4814, 0x4818, 0x481c
    L2pad3, L2pad4, L2pad5, L2pad6 = 0x4820, 0x4824, 0x4828, 0x482c
    ResultCfg, ResultBase = 0x4830, 0x4834
    ConvResultChannelStride, ConvResultRowStride = 0x4838, 0x483c
    PECfg, BiasScale, PreScale, FinalScale = 0x8800, 0x8804, 0x8808, 0x880c
    KernelCfg, MACCfg, MatrixVectorBias, AccBias, PostScale = 0xc800, 0xc804, 0xc808, 0xc80c, 0xc810
    SrcDMAConfig, Srcpad0, SrcBaseAddr = 0x13800, 0x13804, 0x13808
    SrcRowStride, SrcPlaneStride, SrcDepthStride = 0x1380c, 0x13810, 0x13814
    SrcGroupStride, Srcpad2, Srcpad3, Srcpad4 = 0x13818, 0x13820, 0x13824, 0x13828
    SrcFmt, Srcpad8 = 0x13838, 0x1383c
    DstDMAConfig, DstBaseAddr, DstRowStride = 0x17800, 0x17804, 0x17808
    DstPlaneStride, DstDepthStride, DstGroupStride, DstFmt = 0x1780c, 0x17810, 0x17814, 0x17818

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

# Registers for ADD
CMD_OFST = {
    reg.InDim: 0x128, reg.pad0: 0x12c, reg.ChCfg: 0x130, reg.Cin: 0x134, reg.Cout: 0x138,
    reg.OutDim: 0x13c, reg.pad1: 0x140, reg.ConvCfg: 0x144, reg.pad2: 0x148, reg.GroupConvCfg: 0x14c,
    reg.TileCfg: 0x150, reg.pad3: 0x154, reg.pad4: 0x158, reg.Cfg: 0x15c, reg.TaskInfo: 0x160,
    reg.DPE: 0x164,
    reg.L2Cfg: 0x1e0, reg.SourceCfg: 0x1e4, reg.SourceBase: 0x1e8, reg.SourceChannelStride: 0x1ec, reg.SourceRowStride: 0x1f0,
    reg.L2pad0: 0x1f4, reg.L2pad1: 0x1f8, reg.L2pad2: 0x1fc, reg.L2pad3: 0x200, reg.L2pad4: 0x204,
    reg.L2pad5: 0x208, reg.L2pad6: 0x20c, reg.ResultCfg: 0x210, reg.ResultBase: 0x214, reg.ConvResultChannelStride: 0x218,
    reg.ConvResultRowStride: 0x21c,
    reg.PECfg: 0x22c, reg.BiasScale: 0x230, reg.PreScale: 0x234, reg.FinalScale: 0x238,
    reg.KernelCfg: 0x240, reg.MACCfg: 0x244, reg.MatrixVectorBias: 0x248, reg.AccBias: 0x24c, reg.PostScale: 0x250,
    reg.SrcDMAConfig: 0x16c, reg.Srcpad0: 0x170, reg.SrcBaseAddr: 0x174, reg.SrcRowStride: 0x178, reg.SrcPlaneStride: 0x17c,
    reg.SrcDepthStride: 0x180, reg.SrcGroupStride: 0x184, reg.Srcpad2: 0x18c, reg.Srcpad3: 0x190, reg.Srcpad4: 0x194,
    reg.SrcFmt: 0x1a4, reg.Srcpad8: 0x1a8,
    reg.DstDMAConfig: 0x258, reg.DstBaseAddr: 0x25c, reg.DstRowStride: 0x260, reg.DstPlaneStride: 0x264, reg.DstDepthStride: 0x268,
    reg.DstGroupStride: 0x26c, reg.DstFmt: 0x270,
}


def build_buffers():
    commands = {
        reg.InDim: 0x10001, reg.pad0: 1, reg.ChCfg: 0x2a, reg.Cin: 0x40, reg.Cout: 0x40,
        reg.OutDim: 0x10001, reg.pad1: 1, reg.ConvCfg: 0x5000a021, reg.pad2: 0x2041,
        reg.GroupConvCfg: 0x10001, reg.TileCfg: 1, reg.pad3: 4, reg.pad4: 0, reg.Cfg: 0x33,
        reg.TaskInfo: 0, reg.DPE: 0,
        reg.L2Cfg: 0, reg.SourceCfg: 0x01500172, reg.SourceBase: 0,
        reg.SourceChannelStride: 0x10, reg.SourceRowStride: 0x420,
        reg.L2pad0: 0x400, reg.L2pad1: 0x400, reg.L2pad2: 0x440,
        reg.L2pad3: 0x10, reg.L2pad4: 0x420, reg.L2pad5: 0x400, reg.L2pad6: 0x400,
        reg.ResultCfg: 0x0050017a, reg.ResultBase: 0x860,
        reg.ConvResultChannelStride: 0, reg.ConvResultRowStride: 0,
        reg.PECfg: 0x80000, reg.BiasScale: 0x3c000000, reg.PreScale: 0x3c000000,
        reg.FinalScale: 0x3f800000,
        reg.KernelCfg: 0, reg.MACCfg: 0, reg.MatrixVectorBias: 0, reg.AccBias: 0, reg.PostScale: 0,
        reg.SrcDMAConfig: 0x33881, reg.Srcpad0: 0x33880, reg.SrcBaseAddr: 0,
        reg.SrcRowStride: 0x40, reg.SrcPlaneStride: 0x40, reg.SrcDepthStride: 0x1000,
        reg.SrcGroupStride: 0, reg.Srcpad2: 0x40, reg.Srcpad3: 0x40, reg.Srcpad4: 0x1000,
        reg.SrcFmt: 0x01002031, reg.Srcpad8: 0x2030,
        reg.DstDMAConfig: 0x040000c1, reg.DstBaseAddr: 0, reg.DstRowStride: 0x40,
        reg.DstPlaneStride: 0x40, reg.DstDepthStride: 0x1000, reg.DstGroupStride: 0,
        reg.DstFmt: 0x01002031,
    }

    def make_from_segments(size, segments):
        buf = bytearray(size)
        for offset, length, data in segments:
            buf[offset:offset + length] = data
        return buf

    btsp_buf = make_from_segments(0x4000, [
        (2, 42, bytes.fromhex('40020000000022040000000000006af8ff00000000000098003000000000664902000000000000f801f4')),
        (295, 131, bytes.fromhex('3c01000100010000002a0000004000000040000000010001000100000021a0005041200000010001000100000004000000000000003300000000000000000000000038016c8138030080380300000000004000000040000000001000000000000000000000400000004000000000100000000000000000000000000000312000013020')),
        (477, 57, bytes.fromhex('4800440000000072015001000000001000000020040000000400000004000040040000100000002004000000040000000400007a0150006008')),
        (553, 23, bytes.fromhex('88000c000008000000003c0000003c0000803f00c80010')),
        (597, 31, bytes.fromhex('780118c1000004000000004000000040000000001000000000000031200001')),
    ])
    for hw_addr, value in commands.items():
        offset = CMD_OFST.get(hw_addr)
        struct.pack_into('<I', btsp_buf, offset, value) if offset is not None else None
    return btsp_buf

BTSP_BUF = build_buffers()

input_a = np.zeros(8192, dtype=np.float16); input_a[:2017:32] = 3.0
input_b = np.zeros(8192, dtype=np.float16); input_b[:2017:32] = 2.0

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_handle, cmd_map = allocate_buffer(fd, 0x8000)
out_handle, out_map = allocate_buffer(fd, 0x4000)
src1_handle, src1_map = allocate_buffer(fd, 0x4000)
src1_map.write(input_a.tobytes()); src1_map.close()
src2_handle, src2_map = allocate_buffer(fd, 0x4000)
src2_map.write(input_b.tobytes()); src2_map.close()
btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)
btsp_map.write(BTSP_BUF); btsp_map.close()

handles = [cmd_handle, 0, 0, 0, out_handle, src1_handle, src2_handle] + [0] * 25
ret = submit_task(fd, 0x274, 1, 0x274, handles, btsp_handle)

result = np.frombuffer(out_map, dtype=np.float16, count=64).copy(); out_map.close()
print("output = ", result)
os.close(fd)
