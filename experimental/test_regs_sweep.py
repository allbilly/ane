"""expt2: ±1 Bit-Level Sensitivity Scan on relu firmware.
For every register, apply relu_val + 1 and relu_val - 1 to find which bits
are "live" (affect behavior) vs "dead" (dontcare).

Usage:
  python experimental/test_regs_sweep.py
  python experimental/test_regs_sweep.py safe     # only safe regs
  python experimental/test_regs_sweep.py baseline  # verify baseline
"""

from fcntl import ioctl
import os, mmap, ctypes, struct, sys
import numpy as np

STRIDE = 96
CHANNELS = 1
W = 77
ST = STRIDE * 2
HALF_ONE = 0x3C00
DMA_EOL = 0x80000000
DMA_ACTIVE = 0x40000000
ANE_TILE_COUNT = 0x20

class reg:
    W0=0x00;W1=0x04;W2=0x08;W3=0x0c;W4=0x10;W5=0x14;W6=0x18;W7=0x1c;W8=0x20;W9=0x24;KernelDMA=0x28
    CommonStream=0x124;SrcStream=0x168;L2Stream=0x1DC;PEStream=0x228;NEStream=0x23C;DstStream=0x254
    InDim=0x128;pad0=0x12c;ChCfg=0x130;Cin=0x134;Cout=0x138
    OutDim=0x13c;pad1=0x140;ConvCfg=0x144;pad2=0x148
    GroupConvCfg=0x14c;TileCfg=0x150;pad3=0x154;pad4=0x158;Cfg=0x15c
    TaskInfo=0x160;DPE=0x164
    L2Cfg=0x1e0;SourceCfg=0x1e4;SourceBase=0x1e8
    SourceChannelStride=0x1ec;SourceRowStride=0x1f0
    L2pad0=0x1f4;L2pad1=0x1f8;L2pad2=0x1fc
    L2pad3=0x200;L2pad4=0x204;L2pad5=0x208;L2pad6=0x20c
    ResultCfg=0x210;ResultBase=0x214
    ConvResultChannelStride=0x218;ConvResultRowStride=0x21c
    L2pad7=0x220;L2pad8=0x224
    PECfg=0x22c;BiasScale=0x230;PreScale=0x234;FinalScale=0x238
    KernelCfg=0x240;MACCfg=0x244;MatrixVectorBias=0x248;AccBias=0x24c;PostScale=0x250
    SrcDMAConfig=0x16c;Srcpad0=0x170;SrcBaseAddr=0x174
    SrcRowStride=0x178;SrcPlaneStride=0x17c;SrcDepthStride=0x180
    SrcGroupStride=0x184;Srcpad1=0x188;Srcpad2=0x18c;Srcpad3=0x190;Srcpad4=0x194
    Srcpad5=0x198;Srcpad6=0x19c;Srcpad7=0x1a0;SrcFmt=0x1a4;Srcpad8=0x1a8;SrcPadStream=0x1AC
    DstDMAConfig=0x258;DstBaseAddr=0x25c;DstRowStride=0x260
    DstPlaneStride=0x264;DstDepthStride=0x268;DstGroupStride=0x26c;DstFmt=0x270

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]
class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32),
                ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
                ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]
def _IOWR(nr, size): return (3 << 30) | (0x64 << 8) | (size << 16) | nr
DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def allocate_buffer(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit_task(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    req = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size,
                         btsp_handle=btsp_handle, pad=0)
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
    (0, 44, build_seg(0, 44, [
        (reg.W0, (0<<0)|(0x40<<16)|(1<<25)),
        (reg.W1, 0), (reg.W2, 1058), (reg.W3, 0),
        (reg.W4, 0xFFF86A), (reg.W5, 0),
        (reg.W6, (38<<10)|(3<<28)),
        (reg.W7, 0),
        (reg.W8, (5)|(1<<5)|(4<<12)|(1<<17)|(1<<24)),
        (reg.W9, 0), (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])),
    (0x2C, 0xF8, struct.pack('>'+'I'*62,
        *([0]*2+[DMA_EOL]*16+[0]*16+[DMA_ACTIVE]*16+[DMA_EOL]*4+[0]*8))),
    (292, 184, build_seg(0x124, 184, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim, (1<<16)|W), (reg.OutDim, (1<<16)|W),
        (reg.ChCfg, (2)|(2<<4)),
        (reg.Cin, CHANNELS), (reg.Cout, CHANNELS),
        (reg.pad0, 1), (reg.pad1, 1), (reg.pad2, 0x2041),
        (reg.pad3, 0), (reg.pad4, 0),
        (reg.ConvCfg, (1)|(1<<5)|(1<<13)|(1<<15)|(1<<28)|(1<<30)),
        (reg.GroupConvCfg, (1)|(1<<14)|(1<<16)),
        (reg.TileCfg, 1),
        (reg.Cfg, (1<<0)|(1<<8)|(1<<16)|(1<<26)),
        (reg.TaskInfo, (1<<20)), (reg.DPE, 0),
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig, (1)|(8<<4)|(8<<8)|(3<<12)|(3<<16)),
        (reg.Srcpad0, 0x8880), (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, ST), (reg.SrcPlaneStride, ST),
        (reg.SrcDepthStride, ST), (reg.SrcGroupStride, 0),
        (reg.Srcpad1, 0), (reg.Srcpad2, 0), (reg.Srcpad3, 0),
        (reg.Srcpad4, 0), (reg.Srcpad5, 0), (reg.Srcpad6, 0),
        (reg.Srcpad7, 0), (reg.Srcpad8, 0),
        (reg.SrcFmt, (1)|(3<<4)|(2<<12)|(1<<24)),
        (reg.SrcPadStream, 0x00000100),
    ])),
    (476, 68, build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.L2Cfg, 0),
        (reg.SourceCfg, (2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
        (reg.SourceBase, 0),
        (reg.SourceChannelStride, 0xa0), (reg.SourceRowStride, 0xa0),
        (reg.L2pad0, 0xa0), (reg.L2pad1, 0xa0), (reg.L2pad2, 0),
        (reg.L2pad3, 0), (reg.L2pad4, 0), (reg.L2pad5, 0), (reg.L2pad6, 0),
        (reg.ResultCfg, (2)|(2<<2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<20)|(1<<22)),
        (reg.ResultBase, 0xa0),
        (reg.ConvResultChannelStride, 0), (reg.ConvResultRowStride, 0),
    ])),
    (552, 44, build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, (1<<7)),
        (reg.MACCfg, (12)|(1<<16)|(1<<20)),
        (reg.MatrixVectorBias, 0), (reg.AccBias, 0),
        (reg.PostScale, HALF_ONE),
    ])),
    (596, 32, build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig, (1)|(12<<4)|(1<<26)),
        (reg.DstBaseAddr, 0),
        (reg.DstRowStride, ST), (reg.DstPlaneStride, ST),
        (reg.DstDepthStride, ST), (reg.DstGroupStride, 0),
        (reg.DstFmt, (1)|(3<<4)|(2<<12)|(1<<13)|(3<<20)|(1<<24)),
    ])),
])

# ── Test runner with wedge detection ────────────────────────────────────────

ane_wedged = False

def safety_check():
    global ane_wedged
    if ane_wedged:
        return False
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    try:
        out_h, out_m = allocate_buffer(fd, 0x4000)
        src_h, src_m = allocate_buffer(fd, 0x4000)
        btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
        inp = np.zeros(8192, dtype=np.float16)
        inp[0] = np.float16(-3.0)
        inp[1] = np.float16(5.0)
        src_m.write(inp.tobytes())
        btsp_m.write(bytes(BTSP_BUF))
        submit_task(fd, 0x274, 1, 0x274,
            [btsp_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=4).copy()
        ok = abs(float(out[0])) < 0.01 and abs(float(out[1]) - 5.0) < 0.01
        if not ok:
            ane_wedged = True
            return False
        return True
    except Exception:
        ane_wedged = True
        return False
    finally:
        os.close(fd)

def run_one(buf, inputs):
    if ane_wedged:
        return None, None
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    try:
        out_h, out_m = allocate_buffer(fd, 0x4000)
        src_h, src_m = allocate_buffer(fd, 0x4000)
        btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
        inp = np.zeros(8192, dtype=np.float16)
        for i, v in enumerate(inputs):
            inp[i] = np.float16(v)
        src_m.write(inp.tobytes())
        btsp_m.write(bytes(buf))
        submit_task(fd, 0x274, 1, 0x274,
            [btsp_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=4).copy()
        return float(out[0]), float(out[1])
    except Exception:
        return None, None
    finally:
        os.close(fd)

# ── Register values (relu baseline) ─────────────────────────────────────────

# All relu registers with their baseline values
# Format: (block, name, offset, relu_value, [optional extra tests])
# The ±1 tests are generated automatically
ALL_REGS = [
    # SAFE: dimension + stride regs (changing these just changes data layout, no hang risk)
    ("Common", "InDim", 0x128, 0x0001004d),
    ("Common", "OutDim", 0x13c, 0x0001004d),
    ("SrcDMA", "SrcRowStride", 0x178, 0xc0),
    ("SrcDMA", "SrcPlaneStride", 0x17c, 0xc0),
    ("SrcDMA", "SrcDepthStride", 0x180, 0xc0),
    ("SrcDMA", "SrcGroupStride", 0x184, 0),
    ("SrcDMA", "SrcBaseAddr", 0x174, 0),
    ("DstDMA", "DstRowStride", 0x260, 0xc0),
    ("DstDMA", "DstPlaneStride", 0x264, 0xc0),
    ("DstDMA", "DstDepthStride", 0x268, 0xc0),
    ("DstDMA", "DstGroupStride", 0x26c, 0),
    ("DstDMA", "DstBaseAddr", 0x25c, 0),
    ("L2", "SourceChannelStride", 0x1ec, 0xa0),
    ("L2", "SourceRowStride", 0x1f0, 0xa0),
    ("L2", "L2pad0", 0x1f4, 0xa0),
    ("L2", "L2pad1", 0x1f8, 0xa0),
    ("L2", "L2pad2", 0x1fc, 0),
    ("L2", "L2pad3", 0x200, 0),
    ("L2", "L2pad4", 0x204, 0),
    ("L2", "L2pad5", 0x208, 0),
    ("L2", "L2pad6", 0x20c, 0),
    ("L2", "L2pad7", 0x220, 0),
    ("L2", "L2pad8", 0x224, 0),
    ("L2", "ResultBase", 0x214, 0xa0),
    ("L2", "ConvResultChannelStride", 0x218, 0),
    ("L2", "ConvResultRowStride", 0x21c, 0),
    ("L2", "SourceBase", 0x1e8, 0),

    # PE block (all zero in relu)
    ("PE", "PECfg", 0x22c, 0),
    ("PE", "BiasScale", 0x230, 0),
    ("PE", "PreScale", 0x234, 0),
    ("PE", "FinalScale", 0x238, 0),
    ("PE", "PEStream", 0x228, 0x12001000),  # stream header

    # NE block
    ("NE", "KernelCfg", 0x240, 0x80),
    ("NE", "MACCfg", 0x244, 0x0011000c),
    ("NE", "MatrixVectorBias", 0x248, 0),
    ("NE", "AccBias", 0x24c, 0),
    ("NE", "PostScale", 0x250, 0x3c00),

    # SrcDMA
    ("SrcDMA", "SrcDMAConfig", 0x16c, 0x33881),
    ("SrcDMA", "Srcpad0", 0x170, 0x8880),
    ("SrcDMA", "Srcpad1", 0x188, 0),
    ("SrcDMA", "Srcpad2", 0x18c, 0),
    ("SrcDMA", "Srcpad3", 0x190, 0),
    ("SrcDMA", "Srcpad4", 0x194, 0),
    ("SrcDMA", "Srcpad5", 0x198, 0),
    ("SrcDMA", "Srcpad6", 0x19c, 0),
    ("SrcDMA", "Srcpad7", 0x1a0, 0),
    ("SrcDMA", "SrcFmt", 0x1a4, 0x01002031),
    ("SrcDMA", "Srcpad8", 0x1a8, 0),
    ("SrcDMA", "SrcPadStream", 0x1AC, 0x00000100),

    # L2 config regs
    ("L2", "L2Cfg", 0x1e0, 0),
    ("L2", "SourceCfg", 0x1e4, 0x00500172),
    ("L2", "ResultCfg", 0x210, 0x0050017a),

    # DstDMA
    ("DstDMA", "DstDMAConfig", 0x258, 0x040000c1),
    ("DstDMA", "DstFmt", 0x270, 0x01302031),

    # Common block
    ("Common", "ChCfg", 0x130, 0x22),
    ("Common", "Cin", 0x134, 1),
    ("Common", "Cout", 0x138, 1),
    ("Common", "ConvCfg", 0x144, 0x5000a021),
    ("Common", "pad0", 0x12c, 1),
    ("Common", "pad1", 0x140, 1),
    ("Common", "pad2", 0x148, 0x2041),
    ("Common", "pad3", 0x154, 0),
    ("Common", "pad4", 0x158, 0),
    ("Common", "Cfg", 0x15c, 0x04010101),
    ("Common", "GroupConvCfg", 0x14c, 0x14001),
    ("Common", "TileCfg", 0x150, 1),
    ("Common", "TaskInfo", 0x160, 0x00100000),
    ("Common", "DPE", 0x164, 0),
]

def gen_tests(rv):
    """Generate ±1 test values for a register based on its baseline."""
    tests = []
    # +1
    tests.append((rv + 1, f"relu+1"))
    # -1 (if non-zero, do value-1; if zero, try all-1s aka -1 wrap)
    if rv == 0:
        tests.append((0xFFFFFFFF, f"relu-1(wrap)"))
        tests.append((1, f"0+1"))
    elif rv == 1:
        tests.append((0, f"relu-1"))
    else:
        tests.append((rv - 1, f"relu-1"))
    return tests

def run_sweep():
    global ane_wedged
    print("=" * 100)
    print("expt2: ±1 BIT-LEVEL SENSITIVITY SCAN (relu firmware)")
    print("For each register, apply relu_val ± 1 and observe effect.")
    print("Input: [-3.0, 5.0]  Expected relu: [0, 5.0]")
    print("=" * 100)
    print()

    total = 0
    dontcare = 0
    functional = []
    hangs = []

    print(f"{'Block':<10} {'Register':<25} {'ReluVal':>10} {'TestVal':>10} {'Desc':<20} {'out[0]':>8} {'out[1]':>8} {'Result':<15}")
    print("-" * 120)

    for block, rname, off, rv in ALL_REGS:
        if ane_wedged:
            print(f"  [ANE WEDGED — remaining tests skipped]")
            break

        # Baseline check: verify relu value works
        base_buf = bytearray(BTSP_BUF)
        pack_reg(base_buf, off, rv)
        v0, v1 = run_one(base_buf, [-3.0, 5.0])
        if v0 is None or abs(v0) >= 0.01 or abs(v1 - 5.0) >= 0.01:
            print(f"{block:<10} {rname:<25} {rv:>10x} {'BASELINE':>10} {'CRITICAL':<20} {'FAIL':>8} {'':>8} {'BASELINE BROKEN':<15}")
            continue

        tests = gen_tests(rv)
        for test_val, desc in tests:
            if ane_wedged:
                break
            total += 1
            buf = bytearray(BTSP_BUF)
            pack_reg(buf, off, test_val)
            v0, v1 = run_one(buf, [-3.0, 5.0])

            if v0 is None:
                result_str = "HANG"
                v0s = "HANG"
                v1s = ""
                hangs.append((rname, off, test_val, desc))
            elif abs(v0) < 0.01 and abs(v1 - 5.0) < 0.01:
                result_str = "OK (dontcare)"
                v0s = f"{v0:.4f}"
                v1s = f"{v1:.4f}"
                dontcare += 1
            elif abs(v0 + 3.0) < 0.01 and abs(v1 - 5.0) < 0.01:
                result_str = "ADD-like"
                v0s = f"{v0:.4f}"
                v1s = f"{v1:.4f}"
                functional.append((rname, off, rv, test_val, desc, f"ADD-like({v0:.4f},{v1:.4f})"))
            elif abs(v1 - 5.0) < 0.01:
                result_str = f"neg={v0:.4f}"
                v0s = f"{v0:.4f}"
                v1s = f"{v1:.4f}"
                functional.append((rname, off, rv, test_val, desc, f"neg_affected({v0:.4f})"))
            elif abs(v0) < 0.01:
                result_str = f"pos={v1:.4f}"
                v0s = f"{v0:.4f}"
                v1s = f"{v1:.4f}"
                functional.append((rname, off, rv, test_val, desc, f"pos_affected({v1:.4f})"))
            else:
                result_str = f"both={v0:.4f},{v1:.4f}"
                v0s = f"{v0:.4f}"
                v1s = f"{v1:.4f}"
                functional.append((rname, off, rv, test_val, desc, f"both_affected({v0:.4f},{v1:.4f})"))

            print(f"{block:<10} {rname:<25} {rv:>10x} {test_val:>10x} {desc:<20} {v0s:>8} {v1s:>8} {result_str:<15}")

            # After HANG, safety check
            if v0 is None:
                ok = safety_check()
                if not ok:
                    print(f"  >>> ANE WEDGED. Reload: sudo rmmod apple_ane && sudo modprobe apple_ane <<<")
                    break

        if ane_wedged:
            break

    # Summary
    print()
    print("=" * 120)
    print("SUMMARY: BIT-LEVEL SENSITIVITY")
    print("=" * 120)
    print()

    print(f"Total tests: {total}")
    print(f"Don't-care (no effect): {dontcare}")
    print(f"Functional (changed behavior): {len(functional)}")
    print(f"Hangs: {len(hangs)}")
    print()

    if functional:
        print("### Registers where ±1 changes behavior:")
        for rname, off, rv, tv, desc, effect in functional:
            print(f"  {rname:25s} (0x{off:04x}): baseline=0x{rv:x} test=0x{tv:x} ({desc}) -> {effect}")

    if hangs:
        print()
        print("### Registers where ±1 causes HANG:")
        for rname, off, tv, desc in hangs:
            print(f"  ** {rname:25s} (0x{off:04x}): test=0x{tv:x} ({desc}) -> HANG")

def run_baseline():
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    out_h, out_m = allocate_buffer(fd, 0x4000)
    src_h, src_m = allocate_buffer(fd, 0x4000)
    btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
    inp = np.zeros(8192, dtype=np.float16)
    inp[0] = -3.0; inp[1] = 5.0
    src_m.write(inp.tobytes())
    btsp_m.write(bytes(BTSP_BUF))
    submit_task(fd, 0x274, 1, 0x274,
        [btsp_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
    os.close(fd)
    out = np.frombuffer(out_m, dtype=np.float16, count=4).copy()
    print(f"Baseline: [{float(out[0]):.4f}, {float(out[1]):.4f}]  expect [0, 5]")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        run_baseline()
    else:
        run_sweep()
