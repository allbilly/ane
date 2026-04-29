from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

ST = 0xc0  # buffer stride in bytes (192 = 96 float16 elements)
ANE_TILE_COUNT = 0x20
fd = os.open("/dev/accel/accel0", os.O_RDWR)

class reg:
    # --- Task Descriptor (0x0000) ---
    W0, W1, W2 = 0x00, 0x04, 0x08
    W3, W4, W5, W6 = 0x0c, 0x10, 0x14, 0x18
    W7, W8, W9 = 0x1c, 0x20, 0x24
    KernelDMA = 0x28

    # --- Stream Headers ---
    CommonStream = 0x124
    SrcStream = 0x168
    L2Stream = 0x1DC
    PEStream = 0x228
    NEStream = 0x23C
    DstStream = 0x254

    # --- Common (0x0124) ---
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
        pack_reg(tmp, boff, val)
    return bytes(tmp[seg_off:seg_off + seg_len])

def pack_reg(buf, offset, value):
    struct.pack_into('<I', buf, offset, value)

BTSP_BUF = make_from_segments(0x4000, [
    # Task Descriptor
    (0, 44, build_seg(0, 44, [
        (reg.W0,
            (0 << 0) |
            (0x40 << 16) |
            (1 << 25)),
        (reg.W1, 0),
        (reg.W2, 1058),
        (reg.W3, 0),
        (reg.W4, 0x00fff86a),
        (reg.W5, 0),
        (reg.W6,
            (38 << 10) |
            (3 << 28)),
        (reg.W7, 0),
        (reg.W8,
            (5) |
            (1 << 5) |
            (0 << 6) |
            (0 << 11) |
            (36 << 12) |
            (1 << 24)),
        (reg.W9, 0x21),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),

    # Firmware DMA context
    (0x2C, 0xF8, struct.pack('>' + 'I' * 62,
        *([0x40000000, 0] + [0x81000000] + [0x80000000]*15 + [0]*16
          + [0x80000000] + [0x40000000]*15 + [0x80000000]*4 + [0]*8))),

    # Common + TileDMA Src
    (292, 184, build_seg(0x124, 184, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim,
            (1 << 16) | 77),
        (reg.OutDim,
            (1 << 16) | 77),
        (reg.ChCfg,
            (2) | (2 << 4)),
        (reg.Cin, 1),
        (reg.Cout, 1),
        (reg.pad0, 1),
        (reg.pad1, 1),
        (reg.pad2, 0x2041),
        (reg.pad3, 0),
        (reg.pad4, 0),
        (reg.ConvCfg,
            (1) |
            (1 << 5) |
            (5 << 13) |
            (0 << 17) |
            (1 << 28) |
            (1 << 30)),
        (reg.GroupConvCfg,
            (0x4001) | (1 << 16)),
        (reg.TileCfg, 1),
        (reg.Cfg,
            (1 << 0) |
            (1 << 8) |
            (1 << 16) |
            (1 << 26)),
        (reg.TaskInfo, 0x100000),
        (reg.DPE, 0),

        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig,
            (1) |
            (8 << 4) |
            (8 << 8) |
            (3 << 12) |
            (3 << 16)),
        (reg.Srcpad0, 0x8880),
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, ST),
        (reg.SrcPlaneStride, ST),
        (reg.SrcDepthStride, ST),
        (reg.SrcGroupStride, 0),
        (reg.Srcpad1, 0),
        (reg.Srcpad2, 0),
        (reg.Srcpad3, 0),
        (reg.Srcpad4, 0),
        (reg.Srcpad5, 0),
        (reg.Srcpad6, 0),
        (reg.Srcpad7, 0),
        (reg.Srcpad8, 0),
        (reg.SrcFmt,
            (1) |
            (3 << 4) |
            (2 << 12) |
            (1 << 24)),
        (0x1AC, 0x00000100),
    ])),

    # L2
    (476, 68, build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg, 0x00500172),
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0xa0),
        (reg.SourceRowStride, 0xa0),
        (reg.L2pad0, 0xa0),
        (reg.L2pad1, 0xa0),
        (reg.L2pad2, 0),
        (reg.L2pad3, 0),
        (reg.L2pad4, 0),
        (reg.L2pad5, 0),
        (reg.L2pad6, 0),
        (reg.ResultCfg, 0x0050017a),
        (reg.ResultBase, 0xa0),
        (reg.ConvResultChannelStride, 0),
        (reg.ConvResultRowStride, 0),
    ])),

    # PE + NE
    (552, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),

        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, 0x80),
        (reg.MACCfg, 0x0012000c),
        (reg.MatrixVectorBias, 0),
        (reg.AccBias, 0),
        (reg.PostScale, 0x3c00),
    ])),

    # Sigmoid LUT data (loaded via firmware DMA)
    (0x274, 228, bytes.fromhex(
        '000000000000000000000000f8c829480000003c7f0d87107713261610192b1c'
        'da1ea0219b248127122adb2ca12fd6314e340a360038fb38d9398a3a0c3b653b'
        '9f3bc43bdb3be93bf23bf83bfb3bfd3bfe3bff3bff3bac090c173e16e63b0100'
        '0100000000000000000000000000000000000000000000000000000000000000'
        '000000000000000000000000f8c829480000003c7f0d87107713261610192b1c'
        'da1ea0219b248127122adb2ca12fd6314e340a360038fb38d9398a3a0c3b653b'
        '9f3bc43bdb3be93bf23bf83bfb3bfd3bfe3bff3bff3bac090c173e16e63b0100'
        '01000000'
    )),

    # TileDMA Dst
    (596, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig,
            (1) |
            (12 << 4) |
            (1 << 26)),
        (reg.DstBaseAddr, 0),
        (reg.DstRowStride, ST),
        (reg.DstPlaneStride, ST),
        (reg.DstDepthStride, ST),
        (reg.DstGroupStride, 0),
        (reg.DstFmt,
            (1) |
            (3 << 4) |
            (2 << 12) |
            (3 << 20) |
            (1 << 24)),
    ])),
])

STRIDE = 96
C = 1
input_a = np.zeros(8192, dtype=np.float16)
input_a[:C * STRIDE:STRIDE] = np.float16(3.0)

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

out_arr = np.frombuffer(out_map, dtype=np.float16, count=64).copy()
print(f"output[0] = {out_arr[0]}")
