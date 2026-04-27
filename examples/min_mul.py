from fcntl import ioctl
import os, mmap, ctypes
import numpy as np

ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]

class drm_ane_bo_free(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [
        ("tsk_size", ctypes.c_uint64),
        ("td_count", ctypes.c_uint32),
        ("td_size", ctypes.c_uint32),
        ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
        ("btsp_handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

def _IOWR(nr, size):
    return (3 << 30) | (0x64 << 8) | (size << 16) | nr

DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_BO_FREE = _IOWR(0x42, ctypes.sizeof(drm_ane_bo_free))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

# Register offset constants — CMD_BUF byte offsets (H13/M1, from hwx_parsing)
# --- Common (0x0000) ---
InDim=0x128; pad0=0x12c; ChCfg=0x130; Cin=0x134; Cout=0x138; OutDim=0x13c
pad1=0x140; ConvCfg=0x144; pad2=0x148; GroupConvCfg=0x14c; TileCfg=0x150
pad3=0x154; pad4=0x158; Cfg=0x15c; TaskInfo=0x160; DPE=0x164
# --- TileDMA Src (0x13800) ---
SrcDMAConfig=0x16c; Srcpad0=0x170; SrcBaseAddr=0x174; SrcRowStride=0x178
SrcPlaneStride=0x17c; SrcDepthStride=0x180; SrcGroupStride=0x184; Srcpad1=0x188
Srcpad2=0x18c; Srcpad3=0x190; Srcpad4=0x194; Srcpad5=0x198; Srcpad6=0x19c
Srcpad7=0x1a0; SrcFmt=0x1a4; Srcpad8=0x1a8; Srcpad9=0x1ac; Srcpad10=0x1b0
Srcpad11=0x1b4; Srcpad12=0x1b8; PixelOffset0=0x1bc; PixelOffset1=0x1c0
PixelOffset2=0x1c4; PixelOffset3=0x1c8
# --- L2 (0x4800) ---
L2Cfg=0x1e0; SourceCfg=0x1e4; SourceBase=0x1e8; SourceChannelStride=0x1ec
SourceRowStride=0x1f0; L2pad0=0x1f4; L2pad1=0x1f8; L2pad2=0x1fc; L2pad3=0x200
L2pad4=0x204; L2pad5=0x208; L2pad6=0x20c; ResultCfg=0x210; ResultBase=0x214
ConvResultChannelStride=0x218; ConvResultRowStride=0x21c
# --- PE (0x8800) ---
PECfg=0x22c; BiasScale=0x230; PreScale=0x234; FinalScale=0x238
# --- NE (0xC800) ---
KernelCfg=0x240; MACCfg=0x244; MatrixVectorBias=0x248; AccBias=0x24c; PostScale=0x250
# --- TileDMA Dst (0x17800) ---
DstDMAConfig=0x258; DstBaseAddr=0x25c; DstRowStride=0x260; DstPlaneStride=0x264
DstDepthStride=0x268; DstGroupStride=0x26c; DstFmt=0x270

def make_buf(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset+length] = data
    return buf

CMD_BUF = make_buf(0x8000, [
    (3, 41, b'\x02\x00\x00\x00\x00"\x04\x00\x00\x00\x00\x00\x00j\xf8\xff\x00\x00\x00\x00\x00\x00\x98\x000\x00\x00\x00\x00fI\x02\x00\x00\x00\x00\x00\x00\xf8\x01\xf4'),
    (295, 131, b'<\x01\x00\x01\x00\x01\x00\x00\x00*\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x01\x00\x01\x00\x01\x00\x00\x00!\xa0\x00PA \x00\x00\x01\x00\x01\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x003\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x008\x01l\x818\x03\x00\x808\x03\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x001 \x00\x010 '),
    (477, 57, b'H\x00D\x00\x00\x00\x00r\x01P\x01\x00\x00\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00@\x04\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00z\x01P\x00`\x08'),
    (553, 23, b'\x88\x00\x0c\x00\x00\x08\x00\x00\x00\x00<\x00\x00\x00<\x00\x00\x80?\x00\xc8\x00\x10'),
    (597, 31, b'x\x01\x18\xc1\x00\x00\x04\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x001 \x00\x01'),
])

BTSP_BUF = make_buf(0x4000, [
    (2, 42, b'@\x02\x00\x00\x00\x00"\x04\x00\x00\x00\x00\x00\x00j\xf8\xff\x00\x00\x00\x00\x00\x00\x98\x000\x00\x00\x00\x00fI\x02\x00\x00\x00\x00\x00\x00\xf8\x01\xf4'),
    (295, 131, b'<\x01\x00\x01\x00\x01\x00\x00\x00*\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x01\x00\x01\x00\x01\x00\x00\x00!\xa0\x00PA \x00\x00\x01\x00\x01\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x003\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x008\x01l\x818\x03\x00\x808\x03\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x001 \x00\x010 '),
    (477, 57, b'H\x00D\x00\x00\x00\x00r\x01P\x01\x00\x00\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00@\x04\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00z\x01P\x00`\x08'),
    (553, 23, b'\x88\x00\x0c\x00\x00\x08\x00\x00\x00\x00<\x00\x00\x00<\x00\x00\x80?\x00\xc8\x00\x10'),
    (597, 31, b'x\x01\x18\xc1\x00\x00\x04\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x001 \x00\x01'),
])
CMD_BUF[MACCfg] = 0x30; CMD_BUF[PECfg] = (CMD_BUF[PECfg] & ~0x04) | 0x04
BTSP_BUF[MACCfg] = 0x30; BTSP_BUF[PECfg] = (BTSP_BUF[PECfg] & ~0x04) | 0x04

_s1 = np.zeros(8192, dtype=np.float16)
_s1[:2017:32] = 3.0
SRC1 = _s1.tobytes()
_s2 = np.zeros(8192, dtype=np.float16)
_s2[:2017:32] = 2.0
SRC2 = _s2.tobytes()

def bo_alloc(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    print(f"Memory mapped at offset={bo.offset:#x}")

    return bo.handle, buf

def submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    submit_struct = drm_ane_submit(
        tsk_size=tsk_size, 
        td_count=td_count, 
        td_size=td_size, 
        btsp_handle=btsp_handle, 
        pad=0)
    for i in range(ANE_TILE_COUNT):
        submit_struct.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, submit_struct)

fd = os.open("/dev/accel/accel0", os.O_RDWR)

cmd_h, cmd_buf = bo_alloc(fd, 0x8000)
cmd_buf.write(CMD_BUF)
print(f"CMD: handle={cmd_h}")

out_h, out_buf = bo_alloc(fd, 0x4000)
src1_h, src1_buf = bo_alloc(fd, 0x4000)
src1_buf.write(SRC1)
src2_h, src2_buf = bo_alloc(fd, 0x4000)
src2_buf.write(SRC2)
btsp_h, btsp_buf = bo_alloc(fd, 0x4000)
btsp_buf.write(BTSP_BUF)

handles = [cmd_h, 0, 0, 0, out_h, src1_h, src2_h] + [0] * 25

ret = submit(fd, 0x274, 1, 0x274, handles, btsp_h)
print(f"SUBMIT ret={ret}")

for label, buf in [("bdx4(out)", out_buf), ("bdx5(src1)", src1_buf), ("bdx6(src2)", src2_buf)]:
    arr = np.frombuffer(buf, dtype=np.float16, count=64).copy()
    print(f"{label}: {arr[:8]}")

os.close(fd)
