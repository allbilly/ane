from fcntl import ioctl
import os, mmap, ctypes, struct, re
import numpy as np

ANE_TILE_COUNT = 0x20

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

relu_cmds = {
    reg.InDim: 0x1004d, reg.pad0: 1, reg.ChCfg: 0x22, reg.Cin: 1, reg.Cout: 1,
    reg.OutDim: 0x1004d, reg.pad1: 1, reg.ConvCfg: 0x5000a021, reg.pad2: 0x2041,
    reg.GroupConvCfg: 0x14001, reg.TileCfg: 1, reg.pad3: 0, reg.pad4: 0, reg.Cfg: 0x04010101,
    reg.TaskInfo: 0x00100000, reg.DPE: 0,
    reg.L2Cfg: 0x6c013800, reg.SourceCfg: 0x33881, reg.SourceBase: 0x8880,
    reg.SourceChannelStride: 0, reg.SourceRowStride: 0xc0,
    reg.L2pad0: 0xc0, reg.L2pad1: 0xc0, reg.L2pad2: 0,
    reg.L2pad3: 0, reg.L2pad4: 0, reg.L2pad5: 0, reg.L2pad6: 0,
    reg.ResultCfg: 0, reg.ResultBase: 0,
    reg.ConvResultChannelStride: 0, reg.ConvResultRowStride: 0x01002031,
    reg.PECfg: 0, reg.BiasScale: 0, reg.PreScale: 0, reg.FinalScale: 0,
    reg.KernelCfg: 0, reg.MACCfg: 0, reg.MatrixVectorBias: 0, reg.AccBias: 0, reg.PostScale: 0,
    reg.SrcDMAConfig: 0, reg.Srcpad0: 0x00500172, reg.SrcBaseAddr: 0,
    reg.SrcRowStride: 0xa0, reg.SrcPlaneStride: 0xa0, reg.SrcDepthStride: 0xa0,
    reg.SrcGroupStride: 0xa0, reg.Srcpad2: 0, reg.Srcpad3: 0, reg.Srcpad4: 0,
    reg.SrcFmt: 0, reg.Srcpad8: 0,
    reg.DstDMAConfig: 0x040000c1, reg.DstBaseAddr: 0, reg.DstRowStride: 0xc0,
    reg.DstPlaneStride: 0xc0, reg.DstDepthStride: 0xc0, reg.DstGroupStride: 0,
    reg.DstFmt: 0x01302031,
}

add_cmds = {
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

# Load relu's BTSP program from reference
with open('/home/asahi/allbilly_ane/examples/relu_from_hwx.py') as f:
    content = f.read()
m = re.search(r"BTSP_BUF = bytearray\(bytes\.fromhex\('([^']+)'\)\)", content) or \
    re.search(r"BTSP_BUF = bytes\.fromhex\('([^']+)'\)", content)
ref_btsp = bytes.fromhex(m.group(1))

# Build add's BTSP program
add_btsp_segments = [
    (2, 42, bytes.fromhex('40020000000022040000000000006af8ff00000000000098003000000000664902000000000000f801f4')),
    (295, 131, bytes.fromhex('3c01000100010000002a0000004000000040000000010001000100000021a0005041200000010001000100000004000000000000003300000000000000000000000038016c8138030080380300000000004000000040000000001000000000000000000000400000004000000000100000000000000000000000000000312000013020')),
    (477, 57, bytes.fromhex('4800440000000072015001000000001000000020040000000400000004000040040000100000002004000000040000000400007a0150006008')),
    (553, 23, bytes.fromhex('88000c000008000000003c0000003c0000803f00c80010')),
    (597, 31, bytes.fromhex('780118c1000004000000004000000040000000001000000000000031200001')),
]
add_btsp = bytearray(0x4000)
for off, ln, data in add_btsp_segments:
    add_btsp[off:off+ln] = data

def run_test(fd, commands, label, btsp_pgm):
    buf = bytearray(btsp_pgm)
    for offset, value in commands.items():
        struct.pack_into('<I', buf, offset, value)

    # Use neg and pos test inputs
    input_a = np.zeros(8192, dtype=np.float16)
    input_a[0] = -3.0
    input_a[1] = 5.0

    out_handle, out_map = allocate_buffer(fd, 0x4000)
    src1_handle, src1_map = allocate_buffer(fd, 0x4000)
    btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)

    src1_map.write(input_a.tobytes())
    btsp_map.write(buf)

    try:
        ret = submit_task(
            fd=fd, tsk_size=0x274, td_count=1, td_size=0x274,
            handles=[btsp_handle, 0, 0, 0, out_handle, src1_handle, 0] + [0] * 25,
            btsp_handle=btsp_handle,
        )
    except Exception as e:
        return None, f"HANG"

    output = np.frombuffer(out_map, dtype=np.float16, count=64).copy()
    out_map.close()
    return True, output

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

print("Starting add→relu register exploration")
print("Each test: open/close fd fresh to avoid ANE state bleed")
print()

# Test groups: each test opens its own fd
def test_one(name, commands, btsp_pgm, expected_relu=True):
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    try:
        ok, out = run_test(fd, commands, name, btsp_pgm)
        if ok is None:
            tag = "HANG"
            out0, out1 = "?", "?"
        else:
            neg_clamped = abs(float(out[0])) < 0.01
            pos_passed = abs(float(out[1]) - 5.0) < 0.01
            out0 = f"{float(out[0]):.1f}"
            out1 = f"{float(out[1]):.1f}"
            if expected_relu:
                tag = "OK  " if (neg_clamped and pos_passed) else "FAIL"
            else:
                tag = "ADD " if (abs(float(out[0]) + 3.0) < 0.01 and pos_passed) else f"OTHER"
        return tag, out0, out1
    finally:
        os.close(fd)

def reg_name(off):
    names = {v: k for k, v in vars(reg).items() if isinstance(v, int)}
    return names.get(off, f"0x{off:04x}")

# Phase 1: Add+Relu btsp program compare
print("--- Phase 1: Baseline ---")
t, o0, o1 = test_one("add reg + add pgm", add_cmds, add_btsp)
print(f"  add pgm + add regs: {t} out=[{o0}, {o1}]")
t, o0, o1 = test_one("add reg + relu pgm", add_cmds, ref_btsp)
print(f"  relu pgm + add regs: {t} out=[{o0}, {o1}]")

# Phase 2: Start from ADD config with ADD program, apply ONE relu register change at a time
print("\n--- Phase 2: Start from ADD (add pgm + add regs), apply ONE relu reg change ---")
print(f"{'Test':<55} {'Result':>6} {'out[0]':>6} {'out[1]':>6}")
print("-"*75)

# Register diffs ordered by likelihood of being the "relu switch"
changes = [
    ("Cfg=0x04010101", [(reg.Cfg, 0x04010101)]),
    ("GroupConvCfg=0x14001", [(reg.GroupConvCfg, 0x14001)]),
    ("TaskInfo=0x00100000", [(reg.TaskInfo, 0x00100000)]),
    ("PECfg=0 (PE off)", [(reg.PECfg, 0)]),
    ("BiasScale=0", [(reg.BiasScale, 0)]),
    ("PreScale=0", [(reg.PreScale, 0)]),
    ("FinalScale=0", [(reg.FinalScale, 0)]),
    ("L2Cfg=0x6c013800", [(reg.L2Cfg, 0x6c013800)]),
    ("SourceCfg=0x33881", [(reg.SourceCfg, 0x33881)]),
    ("SourceBase=0x8880", [(reg.SourceBase, 0x8880)]),
    ("SrcDMAConfig=0", [(reg.SrcDMAConfig, 0)]),
    ("Srcpad0=0x00500172", [(reg.Srcpad0, 0x00500172)]),
    ("ResultCfg=0", [(reg.ResultCfg, 0)]),
    ("ResultBase=0", [(reg.ResultBase, 0)]),
    ("pad3=0", [(reg.pad3, 0)]),
    ("Cout=1", [(reg.Cout, 1)]),
    ("Cin=1", [(reg.Cin, 1)]),
    ("ChCfg=0x22", [(reg.ChCfg, 0x22)]),
    ("InDim=0x1004d", [(reg.InDim, 0x1004d)]),
    ("OutDim=0x1004d", [(reg.OutDim, 0x1004d)]),
    ("DstRowStride=0xc0", [(reg.DstRowStride, 0xc0)]),
    ("DstFmt=0x01302031", [(reg.DstFmt, 0x01302031)]),
]

for name, chgs in changes:
    cmds = dict(add_cmds)
    for off, val in chgs:
        cmds[off] = val
    tag, o0, o1 = test_one(name, cmds, add_btsp)
    print(f"{name:<55} {tag:>6} {o0:>6} {o1:>6}")

# Phase 3: Now that we know which single regs have effect, try combinations
print("\n--- Phase 3: Binary search for minimal add→relu change (add pgm) ---")

# Key candidates from phase 2 that might affect behavior
# Need to try combinations
candidates = [
    ("PECfg=0", ["PECfg=0 (PE off)"]),
    ("PECfg=0+PreScale=0", ["PECfg=0 (PE off)", "PreScale=0"]),
    ("PECfg=0+FinalScale=0", ["PECfg=0 (PE off)", "FinalScale=0"]),
    ("PECfg=0+BiasScale=0", ["PECfg=0 (PE off)", "BiasScale=0"]),
    ("PECfg=0+all scales=0", ["PECfg=0 (PE off)", "BiasScale=0", "PreScale=0", "FinalScale=0"]),
    ("Cfg=0x04010101", ["Cfg=0x04010101"]),
    ("GroupConvCfg=0x14001", ["GroupConvCfg=0x14001"]),
    ("TaskInfo=0x00100000", ["TaskInfo=0x00100000"]),
    ("Cfg+GroupConvCfg", ["Cfg=0x04010101", "GroupConvCfg=0x14001"]),
    ("Cfg+GroupConvCfg+TaskInfo", ["Cfg=0x04010101", "GroupConvCfg=0x14001", "TaskInfo=0x00100000"]),
    ("PECfg=0+Cfg", ["PECfg=0 (PE off)", "Cfg=0x04010101"]),
    ("PECfg=0+GroupConvCfg", ["PECfg=0 (PE off)", "GroupConvCfg=0x14001"]),
    ("PECfg=0+TaskInfo", ["PECfg=0 (PE off)", "TaskInfo=0x00100000"]),
    ("Cfg+GroupConvCfg+TaskInfo+L2Cfg+SourceCfg", ["Cfg=0x04010101", "GroupConvCfg=0x14001", "TaskInfo=0x00100000", "L2Cfg=0x6c013800", "SourceCfg=0x33881"]),
]

def lookup_val(chg_name):
    for n, chgs in changes:
        if n == chg_name:
            return chgs
    return []

for name, chg_names in candidates:
    cmds = dict(add_cmds)
    for cn in chg_names:
        for off, val in lookup_val(cn):
            cmds[off] = val
    tag, o0, o1 = test_one(name, cmds, add_btsp)
    print(f"{name:<55} {tag:>6} {o0:>6} {o1:>6}")

# Phase 4: Try with relu btsp program + minimal add changes
print("\n--- Phase 4: Relu PGM + add regs + one relu change ---")
for name, chgs in changes[:15]:
    cmds = dict(add_cmds)
    for off, val in chgs:
        cmds[off] = val
    tag, o0, o1 = test_one(name, cmds, ref_btsp)
    print(f"{name:<55} {tag:>6} {o0:>6} {o1:>6}")
