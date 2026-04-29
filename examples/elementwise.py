from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np
import sys

STRIDE = 32    # element stride between channels (SrcRowStride / sizeof(float16))
CHANNELS = 0x40  # Cin = Cout = 64
HALF_ONE = 0x3C00  # float16(1.0) encoding
ANE_TILE_COUNT = 0x20
fd = os.open("/dev/accel/accel0", os.O_RDWR)

class reg:  # register offset
    # --- Task Descriptor (0x0000) ---
    W0, W1, W2 = 0x00, 0x04, 0x08
    W3, W4, W5, W6 = 0x0c, 0x10, 0x14, 0x18
    W7, W8, W9 = 0x1c, 0x20, 0x24
    KernelDMA = 0x28

    # --- Stream Headers ---
    CommonStream = 0x124  # stream_header(0x00000, 16)
    SrcStream = 0x168     # stream_header(0x13800, 28)
    L2Stream = 0x1DC      # stream_header(0x04800, 18)
    PEStream = 0x228      # stream_header(0x08800, 4)
    NEStream = 0x23C      # stream_header(0x0C800, 5)
    DstStream = 0x254     # stream_header(0x17800, 7)

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
    SrcPadStream = 0x1AC    # TileDMA Src stream padding

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
    # ── Task Descriptor ──────────────────────────────────────────────
    (0, 44, build_seg(0, 44, [
        (reg.W0,  # tid=0, nid=0x40, eon=1
            (0 << 0) |      # tid=0
            (0x40 << 16) |  # nid=64
            (1 << 25)),     # eon=1
        (reg.W1, 0),  # next_size
        (reg.W2, 1058),  # exe_cycles
        (reg.W3, 0),
        (reg.W4,  # debug_log_events
            (0xFFF86A)       # event mask [23:0], pad=0
        ),
        (reg.W5, 0),
        (reg.W6,  # flags: next_priority=38
            (38 << 10) |    # next_priority=38
            (3 << 28)),     # pad bits
        (reg.W7, 0),  # next_ptr
        (reg.W8,  # base_ene: rbase0=6, rbe0=1, rbase1=5, rbe1=1, wbase=4, wbe=1
            (6) |          # rbase0=6
            (1 << 5) |     # rbe0=1
            (5 << 6) |     # rbase1=5
            (1 << 11) |    # rbe1=1
            (4 << 12) |    # wbase=4
            (1 << 17)),    # wbe=1
        (reg.W9, 0),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),

    # ── Common + TileDMA Src ─────────────────────────────────────────
    (292, 184, build_seg(0x124, 184, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim,  # h_in=1, w_in=1
            (1 << 16) | 1),
        (reg.OutDim,  # h_out=1, w_out=1
            (1 << 16) | 1),
        (reg.ChCfg,  # infmt=fp16, pad0=2, outfmt=fp16
            (2) |          # infmt=fp16
            (2 << 2) |     # pad0=2
            (2 << 4)),     # outfmt=fp16
        (reg.Cin, CHANNELS),
        (reg.Cout, CHANNELS),
        (reg.pad0, 1),
        (reg.pad1, 1),
        (reg.pad2, 0x2041),  # reserved
        (reg.pad3, 4),
        (reg.pad4, 0),
        (reg.ConvCfg,  # kw=1, kh=1, sx=1, sy=1, ox=1, oy=1
            (1) |          # kw=1
            (1 << 5) |     # kh=1
            (1 << 13) |    # sx=1
            (1 << 15) |    # sy=1
            (1 << 28) |    # ox=1
            (1 << 30)),    # oy=1
        (reg.GroupConvCfg,  # num_groups=1, unicast_cin=1
            (1) |          # num_groups=1
            (1 << 16)),    # unicast_cin=1
        (reg.TileCfg, 1),
        (reg.Cfg,  # PE elementwise pipeline (add/mul): 0x33
            (3 << 0) |  # pad0=3
            (0 << 2) |  # small_src_mode=0
            (6 << 3)),  # pad1=6
        (reg.TaskInfo, 0),
        (reg.DPE, 0),

        # TileDMA Src HEADER
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig,  # en=1, cache_hint=8, reuse=8, noreuse=3, dep=3
            (1) |          # en=1
            (8 << 4) |     # cache_hint=8
            (8 << 8) |     # cache_hint_reuse=8
            (3 << 12) |    # cache_hint_noreuse=3
            (3 << 16)),    # dep_mode=3
        (reg.Srcpad0, 0x33880),  # TileDMA Src pad2: same value as SrcDMAConfig with en=0
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, STRIDE * 2),                # 64 bytes = 32 float16
        (reg.SrcPlaneStride, STRIDE * 2),              # same
        (reg.SrcDepthStride, CHANNELS * STRIDE * 2),   # 64*32*2 = 0x1000
        (reg.SrcGroupStride, 0),
        (reg.Srcpad1, 0),
        (reg.Srcpad2, STRIDE * 2),                     # 64
        (reg.Srcpad3, STRIDE * 2),                     # 64
        (reg.Srcpad4, CHANNELS * STRIDE * 2),          # 0x1000
        (reg.Srcpad5, 0),
        (reg.Srcpad6, 0),
        (reg.Srcpad7, 0),
        (reg.Srcpad8, 0x2030),  # reserved
        (reg.SrcFmt,  # source data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (2 << 12) |    # mem_fmt=2
            (1 << 24)),    # interleave=1
    ])),

    # ── L2 ───────────────────────────────────────────────────────────
    (476, 68, build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg,  # L2 source config: type=2, fmt=1, alias=both
            (2) |          # type=2
            (1 << 4) |     # alias_conv_src=1
            (1 << 5) |     # alias_conv_rslt=1
            (1 << 6) |     # fmt=1
            (1 << 8) |     # interleave=1
            (1 << 20) |    # alias_planar_src=1
            (1 << 22) |    # alias_planar_rslt=1
            (1 << 24)),    # reserved bit
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0x10),   # 16 bytes
        (reg.SourceRowStride, 0x420),      # 1056 bytes (stride=66 in 16B units)
        (reg.L2pad0, 0x400),   # reserved
        (reg.L2pad1, 0x400),   # reserved
        (reg.L2pad2, 0x440),   # reserved
        (reg.L2pad3, 0x10),    # = SourceChannelStride
        (reg.L2pad4, 0x420),   # = SourceRowStride
        (reg.L2pad5, 0x400),   # reserved
        (reg.L2pad6, 0x400),   # reserved
        (reg.ResultCfg,  # L2 result config: type=2, bfrmode=2, alias=both
            (2) |          # type=2
            (2 << 2) |     # bfrmode=2
            (1 << 4) |     # alias_conv_src=1
            (1 << 5) |     # alias_conv_rslt=1
            (1 << 6) |     # fmt=1
            (1 << 8) |     # interleave=1
            (1 << 20) |    # alias_planar_src=1
            (1 << 22)),    # alias_planar_rslt=1
        (reg.ResultBase, 0x860),  # L2 result base: 2144 bytes (addr=134 in 16B units)
    ])),

    # ── PE + NE ──────────────────────────────────────────────────────
    (552, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.PECfg, (2 << 18)),  # second_source=2 (L2 result); add mode
        (reg.BiasScale, (HALF_ONE << 16)),  # bias=0, scale=fp16(1.0)
        (reg.PreScale, (HALF_ONE << 16)),   # pre_scale=0
        (reg.FinalScale, 0x3f800000),     # fp32(1.0)

        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, 0),
        (reg.MACCfg, 0),  # add mode; 0x30 = mul mode
        (reg.MatrixVectorBias, 0),
        (reg.AccBias, 0),
        (reg.PostScale, 0),
    ])),

    # ── TileDMA Dst ──────────────────────────────────────────────────
    (596, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig,  # en=1, cache_hint=12
            (1) |          # en=1
            (12 << 4) |    # cache_hint=12
            (1 << 26)),    # reserved bit
        (reg.DstBaseAddr, 0),
        (reg.DstRowStride, STRIDE * 2),
        (reg.DstPlaneStride, STRIDE * 2),
        (reg.DstDepthStride, CHANNELS * STRIDE * 2),
        (reg.DstGroupStride, 0),
        (reg.DstFmt,  # destination data format
            (1) |          # fmt_mode=1
            (3 << 4) |     # truncate=3
            (2 << 12) |    # mem_fmt=2
            (1 << 24)),    # interleave=1
    ])),
])

# PE op_mode dispatch: second_source=2 (L2 result), op_mode selects operation
#   op_mode=0 → a+b       add       (default)
#   op_mode=1 → a*b       mul
#   op_mode=2 → max(a,b)  max
#   op_mode=3 → min(a,b)  min
#   op_mode=4 → (a+b)^2   sq
op_modes = {"add": 0, "mul": 1, "max": 2, "min": 3, "sq": 4}
mode = sys.argv[1] if len(sys.argv) > 1 else "add"

if mode in op_modes:
    pack_reg(BTSP_BUF, reg.PECfg, (2 << 18) | (op_modes[mode] << 2))
    if mode == "mul":
        pack_reg(BTSP_BUF, reg.MACCfg, (1 << 4) | (1 << 5))
else:
    print(f"Unknown mode '{mode}'. Options: {list(op_modes.keys())}")
    sys.exit(1)

input_a = np.zeros(8192, dtype=np.float16)
input_b = np.zeros(8192, dtype=np.float16)
input_a[:CHANNELS * STRIDE:STRIDE] = 3.0
input_b[:CHANNELS * STRIDE:STRIDE] = 2.0

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

output = np.frombuffer(out_map, dtype=np.float16, count=CHANNELS * STRIDE).reshape(CHANNELS, STRIDE)[:, 0].copy()
print("output =", output)
ops = {"add": lambda a, b: a + b, "mul": lambda a, b: a * b,
       "max": lambda a, b: np.maximum(a, b), "min": lambda a, b: np.minimum(a, b),
       "sq": lambda a, b: (a + b) ** 2}
expected = ops[mode](input_a[:CHANNELS * STRIDE], input_b[:CHANNELS * STRIDE]).reshape(CHANNELS, STRIDE)[:, 0]
print(f"expected {mode} =", expected)
