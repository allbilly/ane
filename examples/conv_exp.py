from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

STRIDE = 32
C = 3
ANE_TILE_COUNT = 0x20
fd = os.open("/dev/accel/accel0", os.O_RDWR)

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

def pack_le_words(words):
    return struct.pack('<' + 'I' * len(words), *words)

TD_BUF = make_from_segments(0x4000, [
    # ── Task Descriptor ──────────────────────────────────────────────
    (0, 44, build_seg(0, 44, [
        (reg.W0,  # tid=0, nid=0, eon=1
            (0 << 0) |      # tid=0
            (0 << 16) |     # nid=0
            (1 << 25)),     # eon=1
        (reg.W1, 0),  # next_size
        (reg.W2, 1058),  # exe_cycles
        (reg.W3, 0),
        (reg.W4, 0x00fff86a),  # debug_log_events
        (reg.W5, 0),
        (reg.W6,  # flags: next_priority=38
            (38 << 10) |
            (3 << 28)),
        (reg.W7, 0),  # next_ptr
        (reg.W8,  # base_ene: rbase0=5, rbe0=1, wbase=36, el0_en=1
            (5) |          # rbase0=5
            (1 << 5) |     # rbe0=1
            (0 << 6) |     # rbase1=0
            (0 << 11) |    # rbe1=0
            (36 << 12) |   # wbase=36
            (1 << 24) |    # el0_en=1
            (1 << 25)),    # conv flag
        (reg.W9, 0x21),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),

    # ── Firmware DMA context ─────────────────────────────────────────
    (0x2C, 0xF8, pack_le_words(
        [0x40, 0] + [0x81]*3 + [0x80]*13
        + [0, 0x40, 0x80] + [0]*13
        + [0x40]*16 + [0x80]*4 + [0]*8
    )),

    # ── Common + TileDMA Src ─────────────────────────────────────────
    (292, 184, build_seg(0x124, 184, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim,  # h_in=1, w_in=1
            (1 << 16) | 1),
        (reg.pad0, 1),
        (reg.ChCfg,  # infmt=fp16, outfmt=fp16
            (2) | (2 << 4)),
        (reg.Cin, C),
        (reg.Cout, C),
        (reg.OutDim,  # h_out=1, w_out=1
            (1 << 16) | 1),
        (reg.pad1, 1),
        (reg.ConvCfg, 0x5000a021),  # conv config: kw=1, kh=1, sx=5
        (reg.pad2, 0x2041),  # reserved
        (reg.GroupConvCfg, 0x00010001),
        (reg.TileCfg, 1),
        (reg.pad3, 0),
        (reg.pad4, 0),
        (reg.Cfg, 0x04144405),  # conv pipeline config
        (reg.TaskInfo, 0x100000),
        (reg.DPE, 0),

        # TileDMA Src HEADER
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig,  # en=1, cache_hint=8, reuse=8, noreuse=3, dep=3
            (1) |          # en=1
            (8 << 4) |     # cache_hint=8
            (8 << 8) |     # reuse=8
            (3 << 12) |    # noreuse=3
            (3 << 16)),    # dep=3
        (reg.Srcpad0, 0x8880),  # reserved
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, 0x40),
        (reg.SrcPlaneStride, 0x40),
        (reg.SrcDepthStride, 0xc0),
        (reg.SrcGroupStride, 0),
        (reg.Srcpad1, 0),
        (reg.Srcpad2, 0),
        (reg.Srcpad3, 0),
        (reg.Srcpad4, 0),
        (reg.Srcpad5, 0),
        (reg.Srcpad6, 0),
        (reg.Srcpad7, 0),
        (reg.Srcpad8, 0),
        (reg.SrcFmt,  # source data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (2 << 12) |    # mem_fmt=2
            (1 << 24)),    # interleave=1
        (0x1AC, 0x00000100),  # TileDMA Src stream padding
    ])),

    # ── L2 ───────────────────────────────────────────────────────────
    (476, 76, build_seg(0x1DC, 76, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg, 0x00500172),  # L2 source config
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0x10),
        (reg.SourceRowStride, 0x30),
        (reg.L2pad0, 0x30),  # reserved
        (reg.L2pad1, 0x30),  # reserved
        (reg.L2pad2, 0),
        (reg.L2pad3, 0),
        (reg.L2pad4, 0),
        (reg.L2pad5, 0),
        (reg.L2pad6, 0),
        (reg.ResultCfg, 0x00500172),  # L2 result config
        (reg.ResultBase, 0x30),
        (reg.ConvResultChannelStride, 0x10),
        (reg.ConvResultRowStride, 0x30),
        (0x220, 0x30),  # reserved
        (0x224, 0x30),  # reserved
    ])),

    # ── PE + NE ──────────────────────────────────────────────────────
    (552, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.PECfg, 0),
        (reg.BiasScale, 0),
        (reg.PreScale, 0),
        (reg.FinalScale, 0),

        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, 0x82),  # NE enable + exp mode
        (reg.MACCfg, 0x00101c00),  # conv exp MAC op
        (reg.MatrixVectorBias, 0),
        (reg.AccBias, 0),
        (reg.PostScale, 0x3c00),  # fp16(1.0)
    ])),

    # ── TileDMA Dst ──────────────────────────────────────────────────
    (596, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig,  # en=1, cache_hint=12
            (1) |          # en=1
            (12 << 4)),    # cache_hint=12
        (reg.DstBaseAddr, 0),
        (reg.DstRowStride, 0x40),
        (reg.DstPlaneStride, 0x40),
        (reg.DstDepthStride, 0xc0),
        (reg.DstGroupStride, 0),
        (reg.DstFmt,  # destination data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (2 << 12) |    # mem_fmt=2
            (3 << 20) |    # bank=3
            (1 << 24)),    # interleave=1
    ])),
])

# Kernel data: 3x3 1x1 conv with 0.5 weights on diagonal
kernel_hex = (
    '0000000000000000000000000040004000400000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000040004000400000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000040004000400000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
)
kernel = bytes.fromhex(kernel_hex)

CMD_BUF = bytearray(TD_BUF)
CMD_BUF.extend(kernel)
CMD_BUF.extend(b'\x00' * (32768 - len(CMD_BUF)))
BTSP_BUF = bytearray(TD_BUF[:0x4000])
BTSP_BUF[2] = 0x40

BUF_SIZE = 16384

out_handle, out_map = allocate_buffer(fd, BUF_SIZE)
src1_handle, src1_map = allocate_buffer(fd, BUF_SIZE)
cmd_handle, cmd_map = allocate_buffer(fd, 32768)
cmd_map.write(bytes(CMD_BUF)); cmd_map.close()
btsp_handle, btsp_map = allocate_buffer(fd, BUF_SIZE)
btsp_map.write(bytes(BTSP_BUF)); btsp_map.close()

src1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
src1[0] = np.float16(1.0)
src1[STRIDE] = np.float16(2.0)
src1[STRIDE * 2] = np.float16(3.0)
src1_map.write(src1.tobytes()); src1_map.close()

ret = submit_task(
    fd=fd, tsk_size=0x274, td_count=1, td_size=0x274,
    handles=[cmd_handle, 0, 0, 0, out_handle, src1_handle, 0] + [0] * 25,
    btsp_handle=btsp_handle,
)
os.close(fd)

output_all = np.frombuffer(out_map, dtype=np.float16).copy(); out_map.close()
non_zero = np.where(output_all != 0)[0]
print(f"Total output values: {len(output_all)}, non-zero count: {len(non_zero)}")
if len(non_zero) > 0:
    print(f"Non-zero indices: {non_zero[:20]}")
    print(f"Non-zero values: {output_all[non_zero[:20]]}")
print("output[:64] =", output_all[:64])
