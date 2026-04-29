import os, time
from fcntl import ioctl
import mmap, ctypes, struct
import numpy as np

ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32),
                ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
                ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]

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

def run_one_raw(buf, inputs):
    try:
        fd = os.open("/dev/accel/accel0", os.O_RDWR)
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

_ANE_WEDGED = False

def is_wedged():
    return _ANE_WEDGED

def set_wedged(w):
    global _ANE_WEDGED
    _ANE_WEDGED = w

def try_baseline(btsp_buf):
    try:
        fd = os.open("/dev/accel/accel0", os.O_RDWR)
        out_h, out_m = allocate_buffer(fd, 0x4000)
        src_h, src_m = allocate_buffer(fd, 0x4000)
        btsp_h, btsp_m = allocate_buffer(fd, 0x4000)
        inp = np.zeros(8192, dtype=np.float16)
        inp[0] = np.float16(-3.0)
        inp[1] = np.float16(5.0)
        src_m.write(inp.tobytes())
        btsp_m.write(bytes(btsp_buf))
        submit_task(fd, 0x274, 1, 0x274,
            [btsp_h, 0, 0, 0, out_h, src_h, 0] + [0]*25, btsp_h)
        out = np.frombuffer(out_m, dtype=np.float16, count=4).copy()
        os.close(fd)
        ok = abs(float(out[0])) < 0.01 and abs(float(out[1]) - 5.0) < 0.01
        return ok
    except Exception:
        try: os.close(fd)
        except: pass
        return False

def reset_ane(btsp_buf, max_retries=10):
    global _ANE_WEDGED

    delays = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0]
    for attempt in range(min(max_retries, len(delays))):
        time.sleep(delays[attempt])
        ok = try_baseline(btsp_buf)
        if ok:
            _ANE_WEDGED = False
            return True

    _ANE_WEDGED = True
    return False
