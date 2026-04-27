#!/usr/bin/env python3
"""
hwx2py: Convert .hwx ANE model files to self-contained Python scripts.

Reads a .hwx file, parses register state, optionally cleans spurious
KernelDMA configs (macOS 14+), and generates a Python script following
the min_mul.py pattern that can run the model on the ANE.

Usage:
    python examples/hwx2py.py hwx/mul.hwx -o examples/mul_from_hwx.py
    python examples/mul_from_hwx.py   # on ANE hardware -> output 6.0
"""

import struct
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hwx_parsing import (
    parse_macho, HWX_MAGIC, get_instruction_set_version,
    get_m1_reg_name,
    H13_COMMON_START, H13_L2_START, H13_PE_START, H13_NE_START,
    H13_TILEDMA_SRC_START, H13_TILEDMA_DST_START, H13_KERNELDMA_START,
)


H13_HEADER_WORDS = [0x02000000, 0, 0, 0, 0x00fff86a, 0, 0x30009800, 0]


def parse_hwx_regs(data):
    ane_data = parse_macho(data)
    if not ane_data:
        magic = struct.unpack_from('<I', data, 0)[0]
        if magic == HWX_MAGIC:
            ane_data = data[16:]
        else:
            for o in range(0, min(len(data), 0x1000) - 40, 4):
                h = struct.unpack_from('<I', data, o)[0]
                tid = h & 0xffff
                if 0 < tid < 0x1000:
                    ane_data = data[o:]
                    break

    if not ane_data:
        raise ValueError('Could not identify HWX format')

    total_len = len(ane_data)
    regs = {}
    offset = 0

    while offset + 32 <= total_len:
        h = struct.unpack_from('<8I', ane_data, offset)
        tid = h[0] & 0xffff
        if tid == 0 and h[1] == 0 and h[2] == 0:
            break
        next_ptr = h[7]

        start_off = offset + 40
        end_off = next_ptr if next_ptr > start_off and next_ptr < total_len else total_len
        num_words = (end_off - start_off) // 4

        if num_words > 0:
            words = struct.unpack_from(f'<{num_words}I', ane_data, start_off)
            w_idx = 0
            while w_idx < num_words:
                hdr = words[w_idx]
                w_idx += 1
                if hdr == 0:
                    continue
                count = (hdr >> 26) & 0x3f
                addr_bits = hdr & 0x3ffffff
                addr = addr_bits >> 2
                for i in range(count + 1):
                    if w_idx >= num_words:
                        break
                    if addr < 0x8000:
                        regs[(addr + i) * 4] = words[w_idx]
                    w_idx += 1

        if next_ptr == 0 or next_ptr <= offset:
            break
        offset = next_ptr

    return regs, ane_data


def extract_h13_header(ane_data):
    return ane_data[:40]


def encode_regs(regs, hdr_bytes=None):
    if hdr_bytes is None:
        hdr_bytes = (b'\x00\x00\x00\x02\x00\x00\x00\x00\x22\x04\x00\x00\x00\x00\x00\x00'
                     b'\x6a\xf8\xff\x00\x00\x00\x00\x00\x00\x98\x00\x30\x00\x00\x00\x00'
                     b'\x66\x49\x02\x00\x00\x00\x00\x00\x00\xf8\x01\xf4')
    else:
        hdr_part = hdr_bytes[:40]
        kdma_pkt = b'\x00\xf8\x01\xf4'
        hdr_bytes = hdr_part + kdma_pkt

    HDR = hdr_bytes + b'\x00' * (292 - len(hdr_bytes))

    addrs = sorted(regs.keys())
    i = 0
    pkts = bytearray()
    while i < len(addrs):
        start = addrs[i]
        vals = []
        while i < len(addrs) and addrs[i] == start + len(vals) * 4:
            vals.append(regs[addrs[i]])
            i += 1
        pkts += struct.pack('<I', ((len(vals) - 1) << 26) | start)
        pkts += struct.pack(f'<{len(vals)}I', *vals)

    return bytes(HDR + pkts).ljust(0x8000, b'\x00')


def detect_elementwise(regs):
    conv = regs.get(H13_COMMON_START + 0x1c, 0)
    kw = conv & 0x1f
    kh = (conv >> 5) & 0x1f
    groups = regs.get(H13_COMMON_START + 0x24, 0) & 0x1fff
    return kw == 1 and kh == 1 and groups == 1


def is_spurious_kdma(regs):
    coeff_configs = [regs.get(H13_KERNELDMA_START + 8 + i * 4, 0) for i in range(16)]
    non_zero = [c for c in coeff_configs if c != 0]
    if not non_zero:
        return False
    return all(c == 0x80 for c in non_zero)

def clean_regs(regs, force=False):
    regs = dict(regs)
    cleaned = False

    if not is_spurious_kdma(regs) and not force:
        return regs, False

    for addr in list(regs.keys()):
        if H13_KERNELDMA_START <= addr < H13_KERNELDMA_START + 0x800:
            del regs[addr]

    kernel_cfg = regs.get(H13_NE_START + 0, 0)
    mac_cfg = regs.get(H13_NE_START + 4, 0)
    if kernel_cfg == 0x80 and mac_cfg == 0x00100000:
        regs[H13_NE_START + 0] = 0
        regs[H13_NE_START + 4] = 0

    cleaned = True

    return regs, cleaned


def get_shape_info(regs):
    w_in = regs.get(H13_COMMON_START + 0, 0) & 0x7fff
    h_in = (regs.get(H13_COMMON_START + 0, 0) >> 16) & 0x7fff
    cin = regs.get(H13_COMMON_START + 0xc, 0) & 0x1ffff
    cout = regs.get(H13_COMMON_START + 0x10, 0) & 0x1ffff
    w_out = regs.get(H13_COMMON_START + 0x14, 0) & 0x7fff
    h_out = (regs.get(H13_COMMON_START + 0x14, 0) >> 16) & 0x7fff

    pe_cfg = regs.get(H13_PE_START + 0, 0)
    op_mode = (pe_cfg >> 2) & 7
    op_name = {0: 'add', 1: 'mul', 2: 'min', 3: 'max'}.get(op_mode, f'op{op_mode}')

    chcfg = regs.get(H13_COMMON_START + 8, 0)
    infmt = {0: 'int8', 1: 'uint8', 2: 'float16'}.get(chcfg & 3, 'unknown')
    outfmt = {0: 'int8', 1: 'uint8', 2: 'float16'}.get((chcfg >> 4) & 3, 'unknown')

    group_stride = regs.get(H13_TILEDMA_DST_START + 0x14, 0)
    # GroupStride zero means add, non-zero means mul
    return {
        'w_in': w_in, 'h_in': h_in, 'cin': cin,
        'w_out': w_out, 'h_out': h_out, 'cout': cout,
        'op_name': op_name,
        'infmt': infmt, 'outfmt': outfmt,
    }


def make_btsp(cmdbuf):
    btsp = bytearray(cmdbuf[:0x4000])
    btsp[2] = 0x40
    return bytes(btsp)

def _calc_buffer_size(regs):
    tile_w = 32
    cin = regs.get(H13_COMMON_START + 0xc, 64) & 0x1ffff
    cout = regs.get(H13_COMMON_START + 0x10, 64) & 0x1ffff
    src_depth = regs.get(H13_TILEDMA_SRC_START + 0x14, 0x1000) & 0x3ffffff
    dst_depth = regs.get(H13_TILEDMA_DST_START + 0x10, 0x1000) & 0x3ffffff
    src_grp = regs.get(H13_TILEDMA_SRC_START + 0x18, 0x1000) & 0x3ffffff
    dst_grp = regs.get(H13_TILEDMA_DST_START + 0x14, 0x1000) & 0x3ffffff
    depth_groups = max((cin + tile_w - 1) // tile_w, (cout + tile_w - 1) // tile_w)
    src_total = depth_groups * src_grp
    dst_total = depth_groups * dst_grp
    return max(src_total, dst_total, 0x4000)

def generate_script(regs, ane_data, output_path):
    hdr_bytes = extract_h13_header(ane_data)
    cmdbuf = encode_regs(regs, hdr_bytes)
    btsp = make_btsp(cmdbuf)
    tsk_size = 0x274
    shape = get_shape_info(regs)
    buf_size = _calc_buffer_size(regs)

    cmdbuf_hex = cmdbuf.hex()
    btsp_hex = btsp.hex()

    n_fp16 = buf_size // 2
    src_plane_stride = regs.get(H13_TILEDMA_SRC_START + 0x10, 0x40) & 0x3ffffff
    dst_plane_stride = regs.get(H13_TILEDMA_DST_START + 0x10, 0x1000) & 0x3ffffff
    tile_stride = min(src_plane_stride, dst_plane_stride) // 2

    fname = os.path.basename(output_path)
    script = f'''#!/usr/bin/env python3
# Auto-generated by hwx2py from {fname}
# Shape: {shape['w_in']}x{shape['h_in']}x{shape['cin']} ({shape['infmt']}) -> {shape['w_out']}x{shape['h_out']}x{shape['cout']} ({shape['outfmt']})
# Op: {shape['op_name']}  tsk_size: 0x{tsk_size:x}

from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

BUF_SIZE = {buf_size}
ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32), ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT), ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]

def _IOWR(nr, size):
    return (3 << 30) | (0x64 << 8) | (size << 16) | nr

DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

CMD_BUF = bytes.fromhex('{cmdbuf_hex}')
BTSP_BUF = bytes.fromhex('{btsp_hex}')

STRIDE = {tile_stride}
C = {shape['cin']}
_s1 = np.zeros({n_fp16}, dtype=np.float16)
_s1[:C * STRIDE:STRIDE] = np.float16(3.0)
SRC1 = _s1.tobytes()
_s2 = np.zeros({n_fp16}, dtype=np.float16)
_s2[:C * STRIDE:STRIDE] = np.float16(2.0)
SRC2 = _s2.tobytes()

def bo_alloc(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    s = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size, btsp_handle=btsp_handle, pad=0)
    for i in range(ANE_TILE_COUNT):
        s.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, s)

fd = os.open("/dev/accel/accel0", os.O_RDWR)
cmd_h, cmd_buf = bo_alloc(fd, 0x8000)
cmd_buf.write(CMD_BUF)
out_h, out_buf = bo_alloc(fd, BUF_SIZE)
src1_h, src1_buf = bo_alloc(fd, BUF_SIZE)
src1_buf.write(SRC1)
src2_h, src2_buf = bo_alloc(fd, BUF_SIZE)
src2_buf.write(SRC2)
btsp_h, btsp_buf = bo_alloc(fd, BUF_SIZE)
btsp_buf.write(BTSP_BUF)
handles = [cmd_h, 0, 0, 0, out_h, src1_h, src2_h] + [0] * 25
ret = submit(fd, 0x{tsk_size:x}, 1, 0x{tsk_size:x}, handles, btsp_h)
print(f"SUBMIT ret={{ret}}")
out_arr = np.frombuffer(out_buf, dtype=np.float16, count=64).copy()
print(f"output[0] = {{out_arr[0]}}")
os.close(fd)
'''

    return script


def _parse_ane(path):
    with open(path, 'rb') as f:
        raw = f.read()
    hdr = struct.unpack_from('<8I', raw, 0)
    td_size = hdr[2]
    krn_size = hdr[6]
    cmdbuf = raw[0x1000:0x1000 + td_size]
    kernel = raw[0x1000 + td_size:0x1000 + td_size + krn_size]
    regs = {}
    start_off = 40
    num_words = (td_size - start_off) // 4
    if num_words > 0:
        words = struct.unpack_from(f'<{num_words}I', cmdbuf, start_off)
        w_idx = 0
        while w_idx < num_words:
            hdr_w = words[w_idx]; w_idx += 1
            if hdr_w == 0: continue
            count = (hdr_w >> 26) & 0x3f
            addr_bits = hdr_w & 0x3ffffff
            addr = addr_bits >> 2
            for i in range(count + 1):
                if w_idx >= num_words: break
                if addr < 0x8000: regs[(addr + i) * 4] = words[w_idx]
                w_idx += 1
    return regs, cmdbuf, kernel, td_size, krn_size


def generate_script_ane(regs, cmdbuf, kernel, td_size, krn_size, output_path):
    cmdbuf_hex = cmdbuf.hex()
    kernel_hex = kernel.hex()
    shape = get_shape_info(regs)

    fname = os.path.basename(output_path)
    buf_size = max(_calc_buffer_size(regs), krn_size)
    buf_size = max(buf_size, 0x4000)

    script = f'''#!/usr/bin/env python3
# Auto-generated by hwx2py from {fname}
# Shape: {shape['w_in']}x{shape['h_in']}x{shape['cin']} ({shape['infmt']}) -> {shape['w_out']}x{shape['h_out']}x{shape['cout']} ({shape['outfmt']})
# Op: {shape['op_name']}

from fcntl import ioctl
import os, mmap, ctypes, struct
import numpy as np

BUF_SIZE = {buf_size}
KRNDATA = bytes.fromhex('{kernel_hex}')
CMD_BUF = bytes.fromhex('{cmdbuf_hex}')
ANE_TILE_COUNT = 0x20

class drm_ane_bo_init(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class drm_ane_submit(ctypes.Structure):
    _fields_ = [("tsk_size", ctypes.c_uint64), ("td_count", ctypes.c_uint32), ("td_size", ctypes.c_uint32), ("handles", ctypes.c_uint32 * ANE_TILE_COUNT), ("btsp_handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]

def _IOWR(nr, size):
    return (3 << 30) | (0x64 << 8) | (size << 16) | nr

DRM_IOCTL_ANE_BO_INIT = _IOWR(0x41, ctypes.sizeof(drm_ane_bo_init))
DRM_IOCTL_ANE_SUBMIT = _IOWR(0x43, ctypes.sizeof(drm_ane_submit))

def bo_alloc(fd, size):
    bo = drm_ane_bo_init(handle=0, pad=0, size=size, offset=0)
    ioctl(fd, DRM_IOCTL_ANE_BO_INIT, bo)
    buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return bo.handle, buf

def submit(fd, tsk_size, td_count, td_size, handles, btsp_handle):
    s = drm_ane_submit(tsk_size=tsk_size, td_count=td_count, td_size=td_size, btsp_handle=btsp_handle, pad=0)
    for i in range(ANE_TILE_COUNT):
        s.handles[i] = handles[i] if i < len(handles) else 0
    return ioctl(fd, DRM_IOCTL_ANE_SUBMIT, s)

fd = os.open("/dev/accel/accel0", os.O_RDWR)

cmd_h, cmd_buf = bo_alloc(fd, 0x8000)
cmd_buf.write(CMD_BUF)
btsp = CMD_BUF[:0x4000]
btsp_h, btsp_buf = bo_alloc(fd, BUF_SIZE)
btsp_buf.write(btsp)

krn_h, krn_buf = bo_alloc(fd, BUF_SIZE)
krn_buf.write(KRNDATA)
out_h, out_buf = bo_alloc(fd, BUF_SIZE)

STRIDE = min({shape.get('plane_stride', 64)} // 2, 32)
CIN = {shape['cin']}
_s1 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
_s1[:CIN * STRIDE:STRIDE] = np.float16(3.0)
src1_h, src1_buf = bo_alloc(fd, BUF_SIZE)
src1_buf.write(_s1.tobytes())

handles = [cmd_h, krn_h] + [0] * 2 + [out_h, src1_h]
if {1 if shape.get('op_name') in ('add', 'mul') else 0}:
    _s2 = np.zeros(BUF_SIZE // 2, dtype=np.float16)
    _s2[:CIN * STRIDE:STRIDE] = np.float16(2.0)
    src2_h, src2_buf = bo_alloc(fd, BUF_SIZE)
    src2_buf.write(_s2.tobytes())
    handles += [src2_h]
handles += [0] * (32 - len(handles))

ret = submit(fd, 0x{td_size:x}, 1, 0x{td_size:x}, handles, btsp_h)
print(f"SUBMIT ret={{ret}}")
out_arr = np.frombuffer(out_buf, dtype=np.float16, count={shape['cout']}).copy()
print(f"output[:8] = {{out_arr[:8]}}")
os.close(fd)
'''

    return script


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert .hwx/.ane to self-contained Python ANE script')
    parser.add_argument('input', help='Input .hwx or .ane file')
    parser.add_argument('-o', '--output', required=True, help='Output .py file')
    parser.add_argument('--no-clean', action='store_true', help='Do not strip spurious KernelDMA registers (hwx only)')
    parser.add_argument('--force-clean', action='store_true', help='Force stripping KernelDMA even for non-elementwise')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print register dump')
    parser.add_argument('--use-anecc', action='store_true', help='Use anecc to convert .hwx first')
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()

    if ext == '.ane' or args.use_anecc:
        if ext != '.ane':
            import subprocess, tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.ane', delete=False)
            tmp.close()
            ret = subprocess.run(['anecc', args.input, '-o', tmp.name], capture_output=True)
            if ret.returncode != 0:
                print(f'anecc failed: {ret.stderr.decode()}')
                os.unlink(tmp.name)
                return
            ane_path = tmp.name
        else:
            ane_path = args.input

        regs, cmdbuf, kernel, td_size, krn_size = _parse_ane(ane_path)

        if args.verbose:
            from hwx_parsing import parse_hwx
            print(f'Input: {args.input}')
            print(f'td_size={td_size} krn_size={krn_size}')
            print(f'Register count: {len(regs)}')
            parse_hwx(cmdbuf, subtype=4)

        if args.use_anecc and ext != '.ane':
            os.unlink(ane_path)

        script = generate_script_ane(regs, cmdbuf, kernel, td_size, krn_size, args.output)
    else:
        with open(args.input, 'rb') as f:
            data = f.read()
        regs, ane_data = parse_hwx_regs(data)

        if args.verbose:
            from hwx_parsing import parse_hwx
            print(f'Input: {args.input}')
            print(f'ANE data size: {len(ane_data)} (0x{len(ane_data):x})')
            print(f'Register count: {len(regs)}')
            parse_hwx(ane_data, subtype=4)

        if not args.no_clean:
            regs, cleaned = clean_regs(regs, force=args.force_clean)
            if cleaned:
                print(f'Cleaned spurious KernelDMA registers (elementwise={detect_elementwise(regs)})')
            else:
                print('No spurious registers to clean')

        script = generate_script(regs, ane_data, args.output)

    with open(args.output, 'w') as f:
        f.write(script)
    os.chmod(args.output, 0o755)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
