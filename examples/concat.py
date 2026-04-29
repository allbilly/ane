from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys

STRIDE = 32
CHANNELS = 0x4000
BUF_SIZE = 0x4000
HALF_ONE = 0x3C00
DMA_EOL = 0x80000000
DMA_ACTIVE = 0x40000000
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
    SrcPadStream = 0x1AC

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
    # ═══════════════════════════════════════════════════════════════════
    # Tile 1: processes CHANNELS=0x4000-wide src (src1 planar, src2 via L2)
    # ═══════════════════════════════════════════════════════════════════

    # ── Tile 1: Task Descriptor ──────────────────────────────────────
    (0, 44, build_seg(0, 44, [
        (reg.W0,  # tid=0, nid=0x40 (64), eon=0
            (0 << 0) |      # tid=0
            (0x40 << 16)    # nid=64
        ),
        (reg.W1, 0x009c0000),  # next_size
        (reg.W2, 1024),  # exe_cycles
        (reg.W3, 0),
        (reg.W4, 0x68),  # debug_log_events
        (reg.W5, 0),
        (reg.W6,  # flags: next_priority=38
            (38 << 10) |    # next_priority=38
            (3 << 28)       # pad bits
        ),
        (reg.W7, 0x300),  # next_ptr → tile 2 at offset 0x300
        (reg.W8, 0x05824026),  # base_ene: rbase0=6, rbe0=1, wbase=0, wbe=1, el0_en=1, +reserved
        (reg.W9, 0),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),

    # ── Tile 1: Firmware DMA context ─────────────────────────────────
    (0x2C, 0xF8, struct.pack('>' + 'I' * 62,
        *([0]*2 + [DMA_EOL]*16 + [0]*16 + [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8))),

    # ═══════════════════════════════════════════════════════════════════
    # Tile 2: processes 16-wide src2, writes to DstBaseAddr=0x100000
    # ═══════════════════════════════════════════════════════════════════

    # ── Tile 2: Task Descriptor ──────────────────────────────────────
    (0x300, 44, build_seg(0, 44, [
        (reg.W0,  # tid=1, nid=0x100, eon=1
            (1 << 0) |      # tid=1
            (0x100 << 16) | # nid=256
            (1 << 25)       # eon=1
        ),
        (reg.W1, 0),  # next_size
        (reg.W2, 1058),  # exe_cycles
        (reg.W3, 0),
        (reg.W4, 0x6a),  # debug_log_events
        (reg.W5, 0),
        (reg.W6,  # flags: next_priority=38
            (38 << 10) |    # next_priority=38
            (3 << 28)       # pad bits
        ),
        (reg.W7, 0),  # next_ptr (no successor)
        (reg.W8, 0x05024025),  # base_ene: rbase0=5, rbe0=1, wbase=0, wbe=1, el0_en=1, +reserved
        (reg.W9, 0),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),

    # ── Tile 1: Firmware DMA context ─────────────────────────────────
    (0x2C, 0xF8, struct.pack('>' + 'I' * 62,
        *([0]*2 + [DMA_EOL]*16 + [0]*16 + [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8))),

    # ═══════════════════════════════════════════════════════════════════
    # Tile 2: processes 16-wide src2, writes to DstBaseAddr=0x100000
    # ═══════════════════════════════════════════════════════════════════

    # ── Tile 2: Task Descriptor ──────────────────────────────────────
    (0x300, 44, build_seg(0, 44, [
        (reg.W0,  # tid=1, nid=0x100 (256), eon=1
            (1 << 0) |      # tid=1
            (0x100 << 16) | # nid=256
            (1 << 25)       # eon=1
        ),
        (reg.W1, 0),  # next_size
        (reg.W2, 1058),  # exe_cycles
        (reg.W3, 0),
        (reg.W4, 0x6a),  # debug_log_events
        (reg.W5, 0),
        (reg.W6,  # flags: next_priority=38
            (38 << 10) |    # next_priority=38
            (3 << 28)       # pad bits
        ),
        (reg.W7, 0),  # next_ptr (no successor)
        (reg.W8,  # base_ene: rbase0=5, rbe0=1, wbase=36, wbe=1, el0_en=1
            (5) |          # rbase0=5
            (1 << 5) |     # rbe0=1
            (36 << 12) |   # wbase=36
            (1 << 17) |    # wbe=1
            (1 << 24)      # el0_en=1
        ),
        (reg.W9, 0),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),

    # ── Tile 2: Firmware DMA context ─────────────────────────────────
    (0x32C, 0xF8, struct.pack('>' + 'I' * 62,
        *([0]*2 + [DMA_EOL]*16 + [0]*16 + [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8))),

    # ── Tile 2: Common + TileDMA Src ────────────────────────────────
    (0x424, 140, build_seg(0x124, 140, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim,  # h_in=1, w_in=1
            (1 << 16) | 1),
        (reg.pad0, 1),
        (reg.ChCfg,  # infmt=fp16, outfmt=fp16
            (2) |          # infmt=fp16
            (2 << 4)),     # outfmt=fp16
        (reg.Cin, 0x10),   # 16 channels
        (reg.Cout, 0x10),  # 16 channels
        (reg.OutDim,  # h_out=1, w_out=1
            (1 << 16) | 1),
        (reg.pad1, 1),
        (reg.ConvCfg,  # kw=1, kh=1, sx=5, ox=1, oy=1
            (1) |          # kw=1
            (1 << 5) |     # kh=1
            (5 << 13) |    # sx=5
            (1 << 28) |    # ox=1
            (1 << 30)),    # oy=1
        (reg.pad2, 0x2041),  # reserved
        (reg.GroupConvCfg,  # num_groups=1, unicast_en=1, unicast_cin=1
            (1) |          # num_groups=1
            (1 << 14) |    # unicast_en=1
            (1 << 16)),    # unicast_cin=1
        (reg.TileCfg, 1),
        (reg.Cfg,  # conv pipeline: pad0=1, conv_mode=1, dst_mode=1, enable=1
            (1 << 0) |      # pad0=1
            (1 << 8) |      # conv_mode=1
            (1 << 12) |     # reserved
            (1 << 16) |     # dst_mode=1
            (1 << 21) |     # reserved
            (1 << 26)),     # enable=1
        (reg.TaskInfo, (1 << 20)),  # enable
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig,  # en=1, cache_hint=8, reuse=8, noreuse=3, dep=3
            (1) |          # en=1
            (8 << 4) |     # cache_hint=8
            (8 << 8) |     # cache_hint_reuse=8
            (3 << 12) |    # cache_hint_noreuse=3
            (3 << 16)),    # dep_mode=3
        (reg.Srcpad0, 0x8880),  # TileDMA Src pad, same as SrcDMAConfig with en=0
        (reg.SrcRowStride, 0x40),    # 64 bytes
        (reg.SrcPlaneStride, 0x40),  # same
        (reg.SrcDepthStride, 0x400), # 16 * 64 = 1024 = Cin * SrcRowStride
        (reg.SrcFmt,  # source data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (2 << 12) |    # mem_fmt=2
            (1 << 24)),    # interleave=1
        (reg.SrcPadStream, 0x100),  # TileDMA Src stream padding
    ])),

    # ── Tile 2: L2 ──────────────────────────────────────────────────
    (0x4DC, 68, build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg,  # L2 source: type=2, alias_conv=both, fmt=0, interleave=1
            (2) |          # type=2
            (1 << 4) |     # alias_conv_src=1
            (1 << 5) |     # alias_conv_rslt=1
            (1 << 8)),     # interleave=1
        (reg.SourceChannelStride, 0x10),   # 16 bytes
        (reg.SourceRowStride, 0x100),      # 256 bytes
        (reg.L2pad0, 0x100),   # reserved (= SourceRowStride)
        (reg.L2pad1, 0x100),   # reserved (= SourceRowStride)
        (reg.ResultCfg,  # L2 result: type=2, bfrmode=2, alias_conv=both, fmt=0
            (2) |          # type=2
            (2 << 2) |     # bfrmode=2
            (1 << 4) |     # alias_conv_src=1
            (1 << 5) |     # alias_conv_rslt=1
            (1 << 8)),     # interleave=1
        (reg.ResultBase, 0x100),   # L2 result base: 256 bytes
    ])),

    # ── Tile 2: PE + NE ─────────────────────────────────────────────
    (0x528, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, (1 << 7)),  # NE enable
        (reg.MACCfg,  # NE op_mode=12 (elementwise pass-through)
            (12) |          # op_mode=12
            (1 << 20)),     # reserved
        (reg.PostScale, HALF_ONE),  # fp16(1.0) post-scale
    ])),

    # ── Tile 2: TileDMA Dst ─────────────────────────────────────────
    (0x554, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig,  # en=1, cache_hint=12
            (1) |          # en=1
            (12 << 4) |    # cache_hint=12
            (1 << 26)),    # reserved
        (reg.DstBaseAddr, 0x100000),  # separate output region for tile 2
        (reg.DstRowStride, 0x40),
        (reg.DstPlaneStride, 0x40),
        (reg.DstDepthStride, 0x100400),
        (reg.DstFmt,  # destination data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (1 << 13) |    # bank_split=1
            (3 << 20) |    # bank=3
            (1 << 24)),    # interleave=1
    ])),

    # ── Tile 1: Common + TileDMA Src ────────────────────────────────
    (292, 140, build_seg(0x124, 140, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim,  # h_in=1, w_in=1
            (1 << 16) | 1),
        (reg.pad0, 1),
        (reg.ChCfg,  # infmt=fp16, outfmt=fp16
            (2) |          # infmt=fp16
            (2 << 4)),     # outfmt=fp16
        (reg.Cin, CHANNELS),
        (reg.Cout, CHANNELS),
        (reg.OutDim,  # h_out=1, w_out=1
            (1 << 16) | 1),
        (reg.pad1, 1),
        (reg.ConvCfg,  # kw=1, kh=1, sx=5, ox=1, oy=1
            (1) |          # kw=1
            (1 << 5) |     # kh=1
            (5 << 13) |    # sx=5
            (1 << 28) |    # ox=1
            (1 << 30)),    # oy=1
        (reg.pad2, 0x2041),  # reserved
        (reg.GroupConvCfg,  # num_groups=1, unicast_en=1, unicast_cin=1
            (1) |          # num_groups=1
            (1 << 14) |    # unicast_en=1
            (1 << 16)),    # unicast_cin=1
        (reg.TileCfg, 1),
        (reg.Cfg,  # conv pipeline: pad0=1, conv_mode=1, dst_mode=1, enable=1
            (1 << 0) |      # pad0=1
            (1 << 8) |      # conv_mode=1
            (1 << 12) |     # reserved
            (1 << 16) |     # dst_mode=1
            (1 << 21) |     # reserved
            (1 << 26)),     # enable=1
        (reg.TaskInfo, (1 << 20)),  # enable
        (reg.DPE, 0),
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig,  # en=1, cache_hint=8, reuse=8, noreuse=3, dep=3
            (1) |          # en=1
            (8 << 4) |     # cache_hint=8
            (8 << 8) |     # cache_hint_reuse=8
            (3 << 12) |    # cache_hint_noreuse=3
            (3 << 16)),    # dep_mode=3
        (reg.Srcpad0, 0x8880),  # TileDMA Src pad, same as SrcDMAConfig with en=0
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, STRIDE * 2),                # 64 bytes
        (reg.SrcPlaneStride, STRIDE * 2),              # same
        (reg.SrcDepthStride, CHANNELS * STRIDE * 2),   # 0x4000 * 64 = 0x100000
        (reg.SrcGroupStride, 0),
        (reg.Srcpad1, 0),
        (reg.Srcpad2, 0),
        (reg.Srcpad3, 0),
        (reg.Srcpad4, 0),
        (reg.Srcpad5, 0),
        (reg.Srcpad6, 0),
        (reg.Srcpad7, 0),
        (reg.SrcFmt,  # source data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (2 << 12) |    # mem_fmt=2
            (1 << 24)),    # interleave=1
        (reg.Srcpad8, 0),
        (reg.SrcPadStream, 0x100),  # TileDMA Src stream padding
    ])),

    # ── Tile 1: L2 ───────────────────────────────────────────────────
    (476, 68, build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg,  # L2 source: type=2, alias=both, fmt=0, interleave=1
            (2) |          # type=2
            (1 << 4) |     # alias_conv_src=1
            (1 << 5) |     # alias_conv_rslt=1
            (1 << 8) |     # interleave=1
            (1 << 20) |    # alias_planar_src=1
            (1 << 22)),    # alias_planar_rslt=1
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0x10),    # 16 bytes
        (reg.SourceRowStride, 0x40000),     # 262144 bytes (strided for CHANNELS)
        (reg.L2pad0, 0x40000),   # reserved (= SourceRowStride)
        (reg.L2pad1, 0x40000),   # reserved (= SourceRowStride)
        (reg.L2pad2, 0),
        (reg.L2pad3, 0),
        (reg.L2pad4, 0),
        (reg.L2pad5, 0),
        (reg.L2pad6, 0),
        (reg.ResultCfg,  # L2 result: type=2, bfrmode=2, alias=both, interleave=1
            (2) |          # type=2
            (2 << 2) |     # bfrmode=2
            (1 << 4) |     # alias_conv_src=1
            (1 << 5) |     # alias_conv_rslt=1
            (1 << 8) |     # interleave=1
            (1 << 20) |    # alias_planar_src=1
            (1 << 22)),    # alias_planar_rslt=1
        (reg.ResultBase, 0x40000),  # L2 result base: 262144 bytes
        (reg.ConvResultChannelStride, 0),
        (reg.ConvResultRowStride, 0),
    ])),

    # ── Tile 1: PE + NE ──────────────────────────────────────────────
    (552, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.PECfg, 0),  # PE bypassed (no elementwise op)
        (reg.BiasScale, 0),
        (reg.PreScale, 0),
        (reg.FinalScale, 0),
        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, (1 << 7)),  # NE enable
        (reg.MACCfg,  # NE op_mode=12 (elementwise pass-through)
            (12) |          # op_mode=12
            (1 << 20)),     # reserved
        (reg.MatrixVectorBias, 0),
        (reg.AccBias, 0),
        (reg.PostScale, HALF_ONE),  # fp16(1.0) post-scale
    ])),

    # ── Tile 1: TileDMA Dst ──────────────────────────────────────────
    (596, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig,  # en=1, cache_hint=12
            (1) |          # en=1
            (12 << 4) |    # cache_hint=12
            (1 << 26)),    # reserved
        (reg.DstBaseAddr, 0),  # output at buffer start
        (reg.DstRowStride, STRIDE * 2),
        (reg.DstPlaneStride, STRIDE * 2),
        (reg.DstDepthStride, 0x100400),
        (reg.DstGroupStride, 0),
        (reg.DstFmt,  # destination data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (1 << 13) |    # bank_split=1
            (3 << 20) |    # bank=3
            (1 << 24)),    # interleave=1
    ])),
])

if len(sys.argv) > 1 and sys.argv[1] == "exp":
    pack_reg(BTSP_BUF, reg.KernelCfg, (1 << 7) | (1 << 9))
    pack_reg(BTSP_BUF, reg.MACCfg, 0x0010000c | 1)

C1 = 16
C2 = 16384

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

handles = [btsp_handle, 0, 0, 0, out_handle, src1_handle, src2_handle] + [0] * 25
ret = submit_task(fd, 0x274, 1, 0x274, handles, btsp_handle)
print(f"submit returned: {ret}")
out = np.frombuffer(out_map, dtype=np.float16).copy(); out_map.close()
non_zero = np.where(out != 0)[0]
print(f"Total: {len(out)}, non-zero: {len(non_zero)}")
if len(non_zero) > 0:
    print(f"Non-zero values: {out[non_zero[:20]]}")
os.close(fd)
