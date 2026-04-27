from fcntl import ioctl
import os
import mmap
import ctypes
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


def _DRM_IOWR(nr, size):
    return (0x3 << 30) | (size << 16) | (0x64 << 8) | nr


DRM_IOCTL_ANE_BO_INIT = _DRM_IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_BO_FREE = _DRM_IOWR(0x42, ctypes.sizeof(drm_ane_bo_free))
DRM_IOCTL_ANE_SUBMIT = _DRM_IOWR(0x43, ctypes.sizeof(drm_ane_submit))


# sum.ane layout from dump: output=bdx4, src0=bdx5(3.0), src1=bdx6(2.0) => 3+2=5
OUT_BUF = b''  # bdx 4: initially zeros, overwritten with result
_s1 = np.zeros(8192, dtype=np.float16)
_s1[:2017:32] = 3.0
SRC1 = _s1.tobytes()              # bdx 5: 3.0
_s2 = np.zeros(8192, dtype=np.float16)
_s2[:2017:32] = 2.0
SRC2 = _s2.tobytes()              # bdx 6: 2.0


# Command buffer non-zero segments (offset, len, data)
CMD_SEGMENTS = [
    (3, 41, b'\x02\x00\x00\x00\x00"\x04\x00\x00\x00\x00\x00\x00j\xf8\xff\x00\x00\x00\x00\x00\x00\x98\x000\x00\x00\x00\x00fI\x02\x00\x00\x00\x00\x00\x00\xf8\x01\xf4'),
    (295, 131, b'<\x01\x00\x01\x00\x01\x00\x00\x00*\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x01\x00\x01\x00\x01\x00\x00\x00!\xa0\x00PA \x00\x00\x01\x00\x01\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x003\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x008\x01l\x818\x03\x00\x808\x03\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x001 \x00\x010 '),
    (477, 57, b'H\x00D\x00\x00\x00\x00r\x01P\x01\x00\x00\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00@\x04\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00z\x01P\x00`\x08'),
    (553, 23, b'\x88\x00\x0c\x00\x00\x08\x00\x00\x00\x00<\x00\x00\x00<\x00\x00\x80?\x00\xc8\x00\x10'),
    (597, 31, b'x\x01\x18\xc1\x00\x00\x04\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x001 \x00\x01'),
]


# BTSP microcode non-zero segments
BTSP_SEGMENTS = [
    (2, 42, b'@\x02\x00\x00\x00\x00"\x04\x00\x00\x00\x00\x00\x00j\xf8\xff\x00\x00\x00\x00\x00\x00\x98\x000\x00\x00\x00\x00fI\x02\x00\x00\x00\x00\x00\x00\xf8\x01\xf4'),
    (295, 131, b'<\x01\x00\x01\x00\x01\x00\x00\x00*\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x01\x00\x01\x00\x01\x00\x00\x00!\xa0\x00PA \x00\x00\x01\x00\x01\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x003\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x008\x01l\x818\x03\x00\x808\x03\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x001 \x00\x010 '),
    (477, 57, b'H\x00D\x00\x00\x00\x00r\x01P\x01\x00\x00\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00@\x04\x00\x00\x10\x00\x00\x00 \x04\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00z\x01P\x00`\x08'),
    (553, 23, b'\x88\x00\x0c\x00\x00\x08\x00\x00\x00\x00<\x00\x00\x00<\x00\x00\x80?\x00\xc8\x00\x10'),
    (597, 31, b'x\x01\x18\xc1\x00\x00\x04\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x001 \x00\x01'),
]


def make_buf(size, segments):
    buf = bytearray(size)
    for offset, length, data in segments:
        buf[offset:offset+length] = data
    return bytes(buf)


CMD_BUF = make_buf(0x8000, CMD_SEGMENTS)
BTSP_BUF = make_buf(0x4000, BTSP_SEGMENTS)

MODEL_TSK_SIZE = 0x274
MODEL_TD_SIZE = 0x274
MODEL_TD_COUNT = 1


def ane_submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    submit_args = drm_ane_submit(
        tsk_size=tsk_size, td_count=td_count, td_size=td_size,
        btsp_handle=btsp_handle, pad=0,
    )
    for i in range(ANE_TILE_COUNT):
        submit_args.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, submit_args)


fd = os.open("/dev/accel/accel0", os.O_RDWR)
print(fd)

try:
    bo = drm_ane_bo_init(handle=0, pad=0, size=0x8000, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, 0x8000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    buf.write(CMD_BUF)
    buf.close()
    cmd_handle = bo.handle
    print(f"CMD buf: handle={cmd_handle}, offset=0x{bo.offset:x}")

    data_offsets = {}
    data_handles = []
    for idx, inp in enumerate([OUT_BUF, SRC1, SRC2], start=2):
        bo = drm_ane_bo_init(handle=0, pad=0, size=0x4000, offset=0)
        ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
        data_handles.append(bo.handle)
        data_offsets[idx] = bo.offset
        buf = mmap.mmap(fd, 0x4000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
        if inp:
            buf.write(inp)
        buf.close()
        print(f"Data buf {idx}: handle={bo.handle}, offset=0x{bo.offset:x}")
    dst_offset = data_offsets[2]

    bo_btsp = drm_ane_bo_init(handle=0, pad=0, size=0x4000, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo_btsp)
    buf = mmap.mmap(fd, 0x4000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo_btsp.offset)
    buf.write(BTSP_BUF)
    buf.close()
    print(f"BTSP buf: handle={bo_btsp.handle}, offset=0x{bo_btsp.offset:x}")

    handles = [cmd_handle, 0, 0, 0] + data_handles + [0] * 25
    print(f"handles={handles[:8]}... btsp_handle={bo_btsp.handle}")

    ret = ane_submit(fd, MODEL_TSK_SIZE, MODEL_TD_COUNT, MODEL_TD_SIZE, handles, bo_btsp.handle)
    print(f"SUCCESS ret={ret}")

    for idx, bdx in [(2, 4), (3, 5), (4, 6)]:
        buf = mmap.mmap(fd, 0x4000, mmap.MAP_SHARED, mmap.PROT_READ, offset=data_offsets[idx])
        arr = np.frombuffer(buf, dtype=np.float16, count=64).copy()
        buf.close()
        print(f"bdx {bdx} (handle {data_handles[idx-2]}): first 8 = {arr[:8]}")
except OSError as e:
    print(f"FAILED - {e}")

os.close(fd)
