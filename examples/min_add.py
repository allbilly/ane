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

def make_buf(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset+length] = data
    return bytes(buf)

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