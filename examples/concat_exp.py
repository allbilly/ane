from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys

STRIDE = 32
CHANNELS = 0x4000
BUF_SIZE = 0x4000
ANE_TILE_COUNT = 0x20

class reg:
    W0, W1, W2 = 0x00, 0x04, 0x08
    W3, W4, W5, W6 = 0x0c, 0x10, 0x14, 0x18
    W7, W8, W9 = 0x1c, 0x20, 0x24
    KernelDMA = 0x28

    CommonStream = 0x124
    SrcStream = 0x168
    L2Stream = 0x1DC
    PEStream = 0x228
    NEStream = 0x23C
    DstStream = 0x254

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
    SrcGroupStride, Srcpad1 = 0x184, 0x188
    Srcpad2, Srcpad3, Srcpad4 = 0x18c, 0x190, 0x194
    Srcpad5, Srcpad6, Srcpad7 = 0x198, 0x19c, 0x1a0
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

def make_from_segments(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset + length] = data
    return buf

def stream_header(hw_addr, num_words):
    return ((num_words - 1) << 26) | hw_addr

def build_seg(seg_off, seg_len, word_packs):
    max_off = max(boff for boff, _ in word_packs) if word_packs else 0
    tmp = bytearray(max(max_off + 4, seg_off + seg_len + 4))
    for boff, val in word_packs:
        pack_reg(tmp, boff, val)
    return bytes(tmp[seg_off:seg_off + seg_len])

def pack_reg(buf, offset, value):
    struct.pack_into('<I', buf, offset, value)

DMA_END = 0x80000000
DMA_BUF = 0x40000000

kernel_extra = bytes.fromhex(
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '000000000000000000000000010000030000000022040000000000006a000000'
    '000000000098003000000000254002050000000000f801f40000000000000000'
    '8000000080000000800000008000000080000000800000008000000080000000'
    '8000000080000000800000008000000080000000800000008000000080000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '4000000040000000400000004000000040000000400000004000000040000000'
    '4000000040000000400000004000000040000000400000004000000040000000'
    '8000000080000000800000008000000000000000000000000000000000000000'
    '000000000000000000000000000000000000003c010001000100000022000000'
    '1000000010000000010001000100000021a00050412000000140010001000000'
    '00000000000000000111210400001000000000000038016c8138030080880000'
    '0000000040000000400000000004000000000000000000000000000000000000'
    '0000000000000000000000000000000031200001000000000001000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000004800440000000072010000000000001000000000010000'
    '000100000001000000000000000000000000000000000000000000007a010000'
    '00010000000000000000000000000000000000000088000c0000000000000000'
    '000000000000000000c80010800000000c0010000000000000000000003c0000'
    '00780118c1000004000010004000000040000000000410000000000031203001'
)

BTSP_BUF = make_from_segments(0x4000, [
    (0, 44, build_seg(0, 44, [
        (reg.W0, 0),
        (reg.W1, 0x009c0000),
        (reg.W2, 0x400),
        (reg.W3, 0),
        (reg.W4, 0x00000068),
        (reg.W5, 0),
        (reg.W6,  # flags: next_priority=38
            (38 << 10) |
            (3 << 28)),
        (reg.W7, 0x00000300),
        (reg.W8, 0x05824026),
        (reg.W9, 0),
        (reg.KernelDMA, 0xf401f800),
    ])),

    (0x2C, 0xF8, struct.pack('>' + 'I' * 62,
        *([0]*2 + [DMA_END]*16 + [0]*16
          + [DMA_BUF]*16 + [DMA_END]*4 + [0]*8))),

    (0x274, len(kernel_extra), kernel_extra),

    (292, 140, build_seg(0x124, 140, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim, (1 << 16) | 1),
        (reg.pad0, 1),
        (reg.ChCfg, 0x22),
        (reg.Cin, CHANNELS),
        (reg.Cout, CHANNELS),
        (reg.OutDim, (1 << 16) | 1),
        (reg.pad1, 1),
        (reg.ConvCfg, 0x5000a021),
        (reg.pad2, 0x2041),
        (reg.GroupConvCfg, (0x4001) | (1 << 16)),
        (reg.TileCfg, 1),
        (reg.pad3, 0),
        (reg.pad4, 0),
        (reg.Cfg, 0x04211101),
        (reg.TaskInfo, 0x00100000),
        (reg.DPE, 0),
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig, 0x00033881),
        (reg.Srcpad0, 0x00008880),
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, STRIDE * 2),
        (reg.SrcPlaneStride, STRIDE * 2),
        (reg.SrcDepthStride, 0x00100000),
        (reg.SrcGroupStride, 0),
        (reg.Srcpad1, 0),
        (reg.Srcpad2, 0),
        (reg.Srcpad3, 0),
        (reg.Srcpad4, 0),
        (reg.Srcpad5, 0),
        (reg.Srcpad6, 0),
        (reg.Srcpad7, 0),
        (reg.SrcFmt, 0x01002031),
        (reg.Srcpad8, 0),
        (0x1AC, 0x00000100),
    ])),

    (476, 68, build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg, 0x00500172),
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0x10),
        (reg.SourceRowStride, 0x40000),
        (reg.L2pad0, 0x40000),
        (reg.L2pad1, 0x40000),
        (reg.L2pad2, 0),
        (reg.L2pad3, 0),
        (reg.L2pad4, 0),
        (reg.L2pad5, 0),
        (reg.L2pad6, 0),
        (reg.ResultCfg, 0x0050017a),
        (reg.ResultBase, 0x40000),
        (reg.ConvResultChannelStride, 0),
        (reg.ConvResultRowStride, 0),
    ])),

    (552, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.PECfg, 0),
        (reg.BiasScale, 0),
        (reg.PreScale, 0),
        (reg.FinalScale, 0),
        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, 0x80),
        (reg.MACCfg, 0x0010000c),
        (reg.MatrixVectorBias, 0),
        (reg.AccBias, 0),
        (reg.PostScale, 0x3c00),
    ])),

    (596, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig, 0x040000c1),
        (reg.DstBaseAddr, 0),
        (reg.DstRowStride, STRIDE * 2),
        (reg.DstPlaneStride, STRIDE * 2),
        (reg.DstDepthStride, 0x00100400),
        (reg.DstGroupStride, 0),
        (reg.DstFmt, 0x01302031),
    ])),
])

if len(sys.argv) > 1 and sys.argv[1] == "exp":
    pack_reg(BTSP_BUF, reg.KernelCfg, 0x80 | 0x200)
    pack_reg(BTSP_BUF, reg.MACCfg, 0x0010000c | 1)

CMD_BUF = bytearray(BTSP_BUF)
CMD_BUF += b'\x00' * (32768 - len(CMD_BUF))
BTSP_BUF[2] = 0x40

C1 = 16
C2 = 16384

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_handle, cmd_map = allocate_buffer(fd, len(CMD_BUF))
cmd_map.write(bytes(CMD_BUF)); cmd_map.close()
out_handle, out_map = allocate_buffer(fd, BUF_SIZE)
src1_handle, src1_map = allocate_buffer(fd, BUF_SIZE)
src1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
src1[:C1 * STRIDE:STRIDE] = np.float16(3.0)
src1_map.write(src1.tobytes()); src1_map.close()
src2_handle, src2_map = allocate_buffer(fd, BUF_SIZE)
src2 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
src2[:C2 * 1:1] = np.float16(2.0)
src2_map.write(src2.tobytes()); src2_map.close()
btsp_handle, btsp_map = allocate_buffer(fd, BUF_SIZE)
btsp_map.write(bytes(BTSP_BUF)); btsp_map.close()

handles = [cmd_handle, 0, 0, 0, out_handle, src1_handle, src2_handle] + [0] * 25
ret = submit_task(fd, 0x274, 1, 0x274, handles, btsp_handle)
print(f"submit returned: {ret}")
out = np.frombuffer(out_map, dtype=np.float16).copy(); out_map.close()
non_zero = np.where(out != 0)[0]
print(f"Total: {len(out)}, non-zero: {len(non_zero)}")
if len(non_zero) > 0:
    print(f"Non-zero values: {out[non_zero[:20]]}")
os.close(fd)
