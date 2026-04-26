from fcntl import ioctl
import os
import mmap
import ctypes
import struct

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
    # Fixed: DRM uses non-standard encoding where _IOWR direction = 0x3
    # Standard Linux: (direction << 30) | (size << 16) | (type << 8) | nr
    # DRM encoding: direction=0x3, type='d'=0x64
    return (0x3 << 30) | (size << 16) | (0x64 << 8) | nr

DRM_IOCTL_ANE_BO_INIT = _DRM_IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_BO_FREE = _DRM_IOWR(0x42, ctypes.sizeof(drm_ane_bo_free))
DRM_IOCTL_ANE_SUBMIT  = _DRM_IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def bo_allocate(fd, size):
    """Allocate a buffer object and map it to userspace."""
    bo_init = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo_init)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo_init.offset)
    return buf, bo_init.handle, bo_init.offset

def bo_free(fd, handle):
    """Free a buffer object."""
    if handle:
        bo_free_args = drm_ane_bo_free(handle=handle, pad=0)
        ioctl(fd, DRM_IOCTL_ANE_BO_FREE, bo_free_args)

def ane_submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    """Submit a task to the ANE device."""
    submit_args = drm_ane_submit(
        tsk_size=tsk_size,
        td_count=td_count,
        td_size=td_size,
        btsp_handle=btsp_handle,
        pad=0,
    )
    for i in range(ANE_TILE_COUNT):
        submit_args.handles[i] = handles[i] if i < len(handles) else 0
    
    # Debug output
    print(f"DEBUG SUBMIT: tsk_size=0x{tsk_size:x}, td_count={td_count}, td_size=0x{td_size:x}, btsp_handle={btsp_handle}")
    print(f"DEBUG SUBMIT: handles={handles[:8]}... (showing first 8)")
    print(f"DEBUG SUBMIT: struct size={ctypes.sizeof(submit_args)}")
    
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, submit_args)

def load_anec_metadata(path):
    """Load anec metadata from model file."""
    with open(path, 'rb') as f:
        header = f.read(0x800)
        offset = 0
        size = struct.unpack('<Q', header[offset:offset+8])[0]
        offset += 8
        td_size = struct.unpack('<I', header[offset:offset+4])[0]
        offset += 4
        td_count = struct.unpack('<I', header[offset:offset+4])[0]
        offset += 4
        tsk_size = struct.unpack('<Q', header[offset:offset+8])[0]
        offset += 8
        krn_size = struct.unpack('<Q', header[offset:offset+8])[0]
        offset += 8
        src_count = struct.unpack('<I', header[offset:offset+4])[0]
        offset += 4
        dst_count = struct.unpack('<I', header[offset:offset+4])[0]
        
        return {
            'size': size,
            'td_size': td_size,
            'td_count': td_count,
            'tsk_size': tsk_size,
            'krn_size': krn_size,
            'src_count': src_count,
            'dst_count': dst_count,
        }

fd = os.open("/dev/accel/accel0", os.O_RDWR)
print(fd)

model_path = "/home/asahi/allbilly_ane/hwx/sum.ane"
print(f"Loading model: {model_path}")
anec = load_anec_metadata(model_path)
print(f"  size=0x{anec['size']:x}, td_size=0x{anec['td_size']:x}, td_count={anec['td_count']}")
print(f"  tsk_size=0x{anec['tsk_size']:x}, src_count={anec['src_count']}, dst_count={anec['dst_count']}")

try:
    # Allocate command buffer with size 0x8000 (32768 bytes)
    bo_init_args = drm_ane_bo_init(handle=0, pad=0, size=0x8000, offset=0)
    print("DRM_IOCTL_ANE_BO_INIT:", hex(DRM_IOCTL_ANE_BO_INIT))

    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo_init_args)
    print(f"DRM_IOCTL_ANE_BO_INIT: SUCCESS")
    print(f"  handle={bo_init_args.handle}, offset=0x{bo_init_args.offset:x}")
    
    # Map the command buffer and load from dump
    cmd_buf = mmap.mmap(fd, 0x8000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo_init_args.offset)
    
    # Load the command buffer content from the working dump
    with open('/tmp/ane_bo_00.bin', 'rb') as f:
        dump_data = f.read()
        cmd_buf.write(dump_data)
        print(f"Loaded {len(dump_data)} bytes from ane_bo_00.bin into command buffer")
    
    cmd_buf.close()
except OSError as e:
    print(f"DRM_IOCTL_ANE_BO_INIT: FAILED - {e}")

# Try matching the dump output exactly
try:
    # From dump files:
    # Handle 1: size 32768 (0x8000) - command buffer (already allocated above)
    # Handles 2, 3, 4: size 16384 (0x4000) - data buffers
    # Handle 5: size 16384 (0x4000) - btsp buffer
    # handles=[1, 0, 0, 0, 2, 3, 4, 0, ...]
    # btsp_handle=5
    
    # Handle 1 is already allocated above with size 0x8000
    # Now allocate handles 2, 3, 4 for data buffers with size 0x4000 and load from dumps
    bo_handles = [1]  # Handle 1 is the command buffer
    dump_files = ['/tmp/ane_bo_04.bin', '/tmp/ane_bo_05.bin', '/tmp/ane_bo_06.bin']
    for i, dump_file in enumerate(dump_files, start=2):
        bo_init_args = drm_ane_bo_init(handle=0, pad=0, size=0x4000, offset=0)
        ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo_init_args)
        bo_handles.append(bo_init_args.handle)
        print(f"Allocated data buffer {i}: handle={bo_init_args.handle}, size=0x4000, offset=0x{bo_init_args.offset:x}")
        
        # Load data from dump file
        buf = mmap.mmap(fd, 0x4000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo_init_args.offset)
        with open(dump_file, 'rb') as f:
            dump_data = f.read()
            buf.write(dump_data)
            print(f"  Loaded {len(dump_data)} bytes from {dump_file}")
        buf.close()
    
    # Allocate handle 5 as btsp buffer with size 0x4000
    bo_init_args = drm_ane_bo_init(handle=0, pad=0, size=0x4000, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo_init_args)
    btsp_handle = bo_init_args.handle
    print(f"Allocated btsp buffer: handle={btsp_handle}, size=0x4000, offset=0x{bo_init_args.offset:x}")
    
    # Load btsp buffer from dump
    buf = mmap.mmap(fd, 0x4000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo_init_args.offset)
    with open('/tmp/ane_btsp.bin', 'rb') as f:
        dump_data = f.read()
        buf.write(dump_data)
        print(f"  Loaded {len(dump_data)} bytes from /tmp/ane_btsp.bin")
    buf.close()
    
    # Build handles array exactly like dump: [1, 0, 0, 0, 2, 3, 4, 0, ...]
    handles = [1, 0, 0, 0, 2, 3, 4] + [0] * 25
    print(f"Using handles: {handles[:8]}...")
    print(f"Using btsp_handle: {btsp_handle}")
    
    ret = ane_submit(
        fd,
        tsk_size=anec['tsk_size'],
        td_count=anec['td_count'],
        td_size=anec['td_size'],
        handles=handles,
        btsp_handle=btsp_handle
    )
    print(f"DRM_IOCTL_ANE_SUBMIT: SUCCESS")
    print(f"  ret={ret}")
except OSError as e:
    print(f"DRM_IOCTL_ANE_SUBMIT: FAILED - {e}")

os.close(fd)
