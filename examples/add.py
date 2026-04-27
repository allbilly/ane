from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys

ANE_TILE_COUNT = 0x20
fd = os.open("/dev/accel/accel0", os.O_RDWR)

class reg:  # register offset 
    # --- Common (0x0000) ---
    InDim, pad0, ChCfg, Cin, Cout = 0x128, 0x12c, 0x130, 0x134, 0x138
    OutDim, pad1, ConvCfg, pad2 = 0x13c, 0x140, 0x144, 0x148
    GroupConvCfg, TileCfg, pad3, pad4, Cfg = 0x14c, 0x150, 0x154, 0x158, 0x15c
    TaskInfo, DPE = 0x160, 0x164
    
    # --- L2 (0x4800) ---
    L2Cfg, SourceCfg, SourceBase = 0x1e0, 0x1e4, 0x1e8
    SourceChannelStride, SourceRowStride = 0x1ec, 0x1f0
    L2pad0, L2pad1, L2pad2 = 0x1f4, 0x1f8, 0x1fc
    L2pad3, L2pad4, L2pad5, L2pad6 = 0x200, 0x204, 0x208, 0x20c
    ResultCfg, ResultBase = 0x210, 0x214
    ConvResultChannelStride, ConvResultRowStride = 0x218, 0x21c

    # --- PE (0x8800) ---
    PECfg, BiasScale, PreScale, FinalScale = 0x22c, 0x230, 0x234, 0x238

    # --- NE (0xC800) ---
    KernelCfg, MACCfg, MatrixVectorBias, AccBias, PostScale = 0x240, 0x244, 0x248, 0x24c, 0x250
    
    # --- TileDMA Src (0x13800) ---
    SrcDMAConfig, Srcpad0, SrcBaseAddr = 0x16c, 0x170, 0x174
    SrcRowStride, SrcPlaneStride, SrcDepthStride = 0x178, 0x17c, 0x180
    SrcGroupStride, Srcpad2, Srcpad3, Srcpad4 = 0x184, 0x18c, 0x190, 0x194
    SrcFmt, Srcpad8 = 0x1a4, 0x1a8

    # --- TileDMA Dst (0x17800) ---
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

class H13TaskHeader:
    def __init__(self, tid=0, nid=0x40, exe_cycles=1058, next_ptr=0):
        self.words = [
            tid | (nid << 16) | (1 << 25),  # w0: tid[15:0], nid[23:16], bit25
            0,                                # w1: next_size in upper 16 bits
            exe_cycles,                       # w2
            0, 0x00fff86a, 0, 0x30009800,    # w3-w6
            next_ptr,                         # w7
            0x00024966, 0,                    # w8-w9
        ]

    def pack_into(self, buf, offset=0):
        for i, w in enumerate(self.words):
            struct.pack_into('<I', buf, offset + i * 4, w)

commands = {
    # --- Common (0x0000) ---
    reg.InDim: 0x10001, reg.pad0: 1, reg.ChCfg: 0x2a, reg.Cin: 0x40, reg.Cout: 0x40,
    reg.OutDim: 0x10001, reg.pad1: 1, reg.ConvCfg: 0x5000a021, reg.pad2: 0x2041,
    reg.GroupConvCfg: 0x10001, reg.TileCfg: 1, reg.pad3: 4, reg.pad4: 0, reg.Cfg: 0x33,
    reg.TaskInfo: 0, reg.DPE: 0,
    
    # --- L2 (0x4800) ---
    reg.L2Cfg: 0, reg.SourceCfg: 0x01500172, reg.SourceBase: 0,
    reg.SourceChannelStride: 0x10, reg.SourceRowStride: 0x420,
    reg.L2pad0: 0x400, reg.L2pad1: 0x400, reg.L2pad2: 0x440,
    reg.L2pad3: 0x10, reg.L2pad4: 0x420, reg.L2pad5: 0x400, reg.L2pad6: 0x400,
    reg.ResultCfg: 0x0050017a, reg.ResultBase: 0x860,
    reg.ConvResultChannelStride: 0, reg.ConvResultRowStride: 0,

    # --- PE (0x8800) ---
    reg.PECfg: 0x80000, reg.BiasScale: 0x3c000000, reg.PreScale: 0x3c000000,
    reg.FinalScale: 0x3f800000,
    
    # --- NE (0xC800) ---
    reg.KernelCfg: 0, reg.MACCfg: 0, reg.MatrixVectorBias: 0, reg.AccBias: 0, reg.PostScale: 0,
    
    # --- TileDMA Src (0x13800) ---
    reg.SrcDMAConfig: 0x33881, reg.Srcpad0: 0x33880, reg.SrcBaseAddr: 0,
    reg.SrcRowStride: 0x40, reg.SrcPlaneStride: 0x40, reg.SrcDepthStride: 0x1000,
    reg.SrcGroupStride: 0, reg.Srcpad2: 0x40, reg.Srcpad3: 0x40, reg.Srcpad4: 0x1000,
    reg.SrcFmt: 0x01002031, reg.Srcpad8: 0x2030,
    
    # --- TileDMA Dst (0x17800) ---
    reg.DstDMAConfig: 0x040000c1, reg.DstBaseAddr: 0, reg.DstRowStride: 0x40,
    reg.DstPlaneStride: 0x40, reg.DstDepthStride: 0x1000, reg.DstGroupStride: 0,
    reg.DstFmt: 0x01002031,
}

def make_from_segments(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset + length] = data
    return buf

hdr = H13TaskHeader(tid=0, nid=0x40, exe_cycles=1058)
hdr_seg = bytearray(44)
hdr.pack_into(hdr_seg, 0)
struct.pack_into('<I', hdr_seg, 0x28, (61 << 26) | 0x1F800)  # KernelDMA stream header

BTSP_BUF = make_from_segments(0x4000, [
    (2, 42, bytes(hdr_seg[2:44])),
    (295, 131, bytes.fromhex('3c01000100010000002a0000004000000040000000010001000100000021a0005041200000010001000100000004000000000000003300000000000000000000000038016c8138030080380300000000004000000040000000001000000000000000000000400000004000000000100000000000000000000000000000312000013020')),
    (477, 57, bytes.fromhex('4800440000000072015001000000001000000020040000000400000004000040040000100000002004000000040000000400007a0150006008')),
    (553, 23, bytes.fromhex('88000c000008000000003c0000003c0000803f00c80010')),
    (597, 31, bytes.fromhex('780118c1000004000000004000000040000000001000000000000031200001')),
])
if len(sys.argv)>1 and sys.argv[1]=="mul":
    commands[reg.PECfg] = (0x80000 & ~0x04) | 0x04
    commands[reg.MACCfg] = 0x30
for offset, value in commands.items():
    struct.pack_into('<I', BTSP_BUF, offset, value)

input_a = np.zeros(8192, dtype=np.float16); input_a[:2017:32] = 3.0
input_b = np.zeros(8192, dtype=np.float16); input_b[:2017:32] = 2.0

out_handle, out_map = allocate_buffer(fd, 0x4000)
src1_handle, src1_map = allocate_buffer(fd, 0x4000)
src2_handle, src2_map = allocate_buffer(fd, 0x4000)
btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)

src1_map.write(input_a.tobytes())
src2_map.write(input_b.tobytes())
btsp_map.write(BTSP_BUF)

ret = submit_task(
    fd=fd,
    tsk_size=0x274,
    td_count=1,
    td_size=0x274,
    handles=[btsp_handle, 0, 0, 0, out_handle, src1_handle, src2_handle] + [0] * 25,
    btsp_handle=btsp_handle,
)
os.close(fd)

output = np.frombuffer(out_map, dtype=np.float16, count=64).copy()
print("output = ", output)