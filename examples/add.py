from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys

ANE_TILE_COUNT = 0x20
fd = os.open("/dev/accel/accel0", os.O_RDWR)

class reg:  # register offset 
    # --- Task Descriptor (0x0000) ---
    W0, W1, W2 = 0x00, 0x04, 0x08
    W3, W4, W5, W6 = 0x0c, 0x10, 0x14, 0x18
    W7, W8, W9 = 0x1c, 0x20, 0x24
    KernelDMA = 0x28

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
    SrcGroupStride, Srcpad1 = 0x184, 0x188
    Srcpad2, Srcpad3, Srcpad4 = 0x18c, 0x190, 0x194
    Srcpad5, Srcpad6, Srcpad7 = 0x198, 0x19c, 0x1a0
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
        struct.pack_into('<I', tmp, boff, val)
    return bytes(tmp[seg_off:seg_off + seg_len])

def pack_reg(buf, offset, value):
    struct.pack_into('<I', buf, offset, value)

BTSP_BUF = make_from_segments(0x4000, [
    (0, 44, build_seg(0, 44, [
        (reg.W0, 0x02400000),
        (reg.W1, 0),
        (reg.W2, 1058),
        (reg.W3, 0), 
        (reg.W4, 0x00fff86a),
        (reg.W5, 0), 
        (reg.W6, 0x30009800),
        (reg.W7, 0),
        (reg.W8, 0x00024966), (reg.W9, 0),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),
    (292, 136, build_seg(0x124, 136, [
        # Common HEADER
        (0x124, stream_header(0x00000, 16)),
        (reg.InDim, 0x10001), 
        (reg.OutDim, 0x10001),
        (reg.ChCfg, 0x2a),
        (reg.Cin, 0x40), 
        (reg.Cout, 0x40), 
        (reg.pad0, 1), 
        (reg.pad1, 1), 
        (reg.pad2, 0x2041),
        (reg.pad3, 4),
        (reg.pad4, 0), 
        (reg.ConvCfg, 0x5000a021), 
        (reg.GroupConvCfg, 0x10001), 
        (reg.TileCfg, 1), 
        (reg.Cfg, 0x33), 
        (reg.TaskInfo, 0), 
        (reg.DPE, 0),

        # TileDMA Src HEADER
        (0x168, stream_header(0x13800, 28)),                          
        (reg.SrcDMAConfig, 0x33881), 
        (reg.Srcpad0, 0x33880), 
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, 0x40), 
        (reg.SrcPlaneStride, 0x40), 
        (reg.SrcDepthStride, 0x1000),
        (reg.SrcGroupStride, 0), 
        (reg.Srcpad1, 0),
        (reg.Srcpad2, 0x40), 
        (reg.Srcpad3, 0x40), 
        (reg.Srcpad4, 0x1000),
        (reg.Srcpad5, 0), 
        (reg.Srcpad6, 0), 
        (reg.Srcpad7, 0),
        (reg.Srcpad8, 0x2030),
        (reg.SrcFmt, 0x01002031), 
    ])),
    (477, 57, build_seg(0x1DD, 57, [
        # L2 HEADER
        (0x1DC, stream_header(0x04800, 18)),                     
        (reg.L2Cfg, 0), 
        (reg.SourceCfg, 0x01500172), 
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0x10), 
        (reg.SourceRowStride, 0x420),
        (reg.L2pad0, 0x400), 
        (reg.L2pad1, 0x400), 
        (reg.L2pad2, 0x440),
        (reg.L2pad3, 0x10), 
        (reg.L2pad4, 0x420), 
        (reg.L2pad5, 0x400),
        (reg.L2pad6, 0x400), 
        (reg.ResultCfg, 0x0050017a), 
        (reg.ResultBase, 0x860),
    ])),
    (553, 43, build_seg(0x229, 43, [
        # PE HEADER
        (0x228, stream_header(0x08800, 4)),
        (reg.PECfg, 0x80000),
        (reg.BiasScale, 0x3c000000),
        (reg.PreScale, 0x3c000000),
        (reg.FinalScale, 0x3f800000),

        # NE HEADER
        (0x23C, stream_header(0x0C800, 5)),
        (reg.KernelCfg, 0),
        (reg.MACCfg, 0),
        (reg.MatrixVectorBias, 0),
        (reg.AccBias, 0),
        (reg.PostScale, 0),
    ])),
    (597, 31, build_seg(0x255, 31, [
        # TileDMA Dst HEADER
        (0x254, stream_header(0x17800, 7)),                       
        (reg.DstDMAConfig, 0x040000c1), 
        (reg.DstBaseAddr, 0), 
        (reg.DstRowStride, 0x40),
        (reg.DstPlaneStride, 0x40), 
        (reg.DstDepthStride, 0x1000), 
        (reg.DstGroupStride, 0),
        (reg.DstFmt, 0x01002031),
    ])),
])

if len(sys.argv) > 1 and sys.argv[1] == "mul":
    pack_reg(BTSP_BUF, reg.PECfg, (0x80000 & ~0x04) | 0x04)
    pack_reg(BTSP_BUF, reg.MACCfg, 0x30)

input_a = np.zeros(8192, dtype=np.float16)
input_b = np.zeros(8192, dtype=np.float16)
input_a[:0x40 * 32:32] = 3.0
input_b[:0x40 * 32:32] = 2.0

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