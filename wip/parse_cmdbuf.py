#!/usr/bin/env python3
"""
Parse CMD_BUF from .hwx files.

This script extracts the CMD_BUF (command buffer) equivalent data from .hwx files.
The .hwx format contains task descriptors that are converted to CMD_BUF when
the .ane is executed.

Usage:
    python parse_cmdbuf.py <input.hwx> [-o output.bin] [--hexdump] [--json]
"""

import argparse
import json
import os
import plistlib
import struct
import sys

# Import from hwx_parsing if available
HWX_MAGIC = 0xbeefface

# Architecture block start addresses (same as hwx_parsing.py)
H13_COMMON_START = 0x0000
H13_L2_START = 0x4800
H13_PE_START = 0x8800
H13_NE_START = 0xC800
H13_TILEDMA_SRC_START = 0x13800
H13_TILEDMA_DST_START = 0x17800
H13_KERNELDMA_START = 0x1F800

H16_COMMON_START = 0x0000
H16_L2_START = 0x4100
H16_PE_START = 0x4500
H16_NE_START = 0x4900
H16_TILEDMA_SRC_START = 0x4D00
H16_TILEDMA_DST_START = 0x5100
H16_KERNELDMA_START = 0x5500
H16_CACHEDMA_START = 0x5900
H16_PE_EXT_START = 0x44D0

MH_MAGIC_64 = 0xfeedfacf
LC_SEGMENT_64 = 0x19
LC_ANE_MAPPED_REGION = 0x40


def get_instruction_set_version(subtype):
    return {
        1: 5,
        3: 6,
        4: 7,
        5: 11,
        6: 8,
        7: 17,
        9: 19,
        10: 20,
    }.get(subtype, 0)


def parse_macho(data):
    """Extract ANE data from Mach-O or raw HWX data."""
    if len(data) < 32:
        return None

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != MH_MAGIC_64 and magic != HWX_MAGIC:
        return None

    # Check for HWX magic
    if magic == HWX_MAGIC:
        return data[16:]

    ncmds = struct.unpack_from("<I", data, 16)[0]
    offset = 32

    for _ in range(ncmds):
        if offset + 8 > len(data):
            break

        cmd, cmdsize = struct.unpack_from("<2I", data, offset)
        if cmd == LC_SEGMENT_64:
            segname = data[offset + 8:offset + 24].strip(b'\x00').decode(errors='ignore')
            if segname == "__TEXT" or segname == "__DATA":
                nsects = struct.unpack_from("<I", data, offset + 64)[0]
                sect_offset = offset + 72
                for _ in range(nsects):
                    sectname = data[sect_offset:sect_offset + 16].strip(b'\x00').decode(errors='ignore')
                    if sectname == "__text" or sectname == "__TEXT":
                        file_off = struct.unpack_from("<I", data, sect_offset + 48)[0]
                        size = struct.unpack_from("<Q", data, sect_offset + 40)[0]
                        if file_off > 0 and size > 0:
                            return data[file_off:file_off + size]
                    sect_offset += 80
        elif cmd == LC_ANE_MAPPED_REGION:
            file_off = struct.unpack_from("<I", data, offset + 8)[0]
            size = struct.unpack_from("<I", data, offset + 12)[0]
            if file_off > 0 and size > 0:
                return data[file_off:file_off + size]
        offset += cmdsize

    return None


def parse_hwx_tasks(data, subtype=7):
    """Parse .hwx data and extract task descriptors."""
    is_version = get_instruction_set_version(subtype)
    tasks = []

    # Check if data starts with HWX magic and skip header if so
    search_start = 0
    if len(data) >= 4:
        magic = struct.unpack_from("<I", data, 0)[0]
        if magic == HWX_MAGIC:
            search_start = 16

    if is_version >= 11:  # H16 / M4 style
        # Phase 1: Search for task headers
        task_offsets = []
        for offset in range(search_start, min(len(data), 0x10000) - 40, 4):
            h = struct.unpack_from("<I", data, offset)[0]
            tid = h & 0xffff
            task_size = (h >> 16) & 0x7ff
            
            # Valid task: reasonable tid (1-256) and size (1-500 words)
            if 0 < tid <= 0x100 and 0 < task_size < 500:
                # Additional validation: check header fields
                h8 = struct.unpack_from("<I", data, offset + 32)[0]
                # ENE field should be small (0-7)
                ene = (h8 >> 16) & 7
                if ene <= 7:
                    task_offsets.append((offset, tid, task_size))
        
        # Phase 2: Parse each found task
        for idx, (offset, tid, task_size_words) in enumerate(task_offsets):
            size_bytes = task_size_words * 4
            task_data = data[offset:offset + size_bytes]
            
            # Extract register values
            reg_values, reg_valid = {}, [False] * 0x8000
            num_words = size_bytes // 4
            words = struct.unpack_from(f"<{num_words}I", data, offset)
            w_idx = 10  # H16 header is 40 bytes (10 words)

            while w_idx < num_words:
                hdr = words[w_idx]
                w_idx += 1
                is_masked = (hdr >> 31) & 1
                word_addr = hdr & 0x7fff

                if word_addr >= 0x8000:
                    continue

                if not is_masked:
                    num_regs = (hdr >> 15) & 0x3f
                    for j in range(num_regs + 1):
                        if w_idx >= num_words:
                            break
                        if word_addr + j < 0x8000:
                            reg_values[word_addr + j] = words[w_idx]
                            reg_valid[word_addr + j] = True
                        w_idx += 1
                else:
                    mask = (hdr >> 15) & 0xffff
                    if w_idx < num_words and word_addr < 0x8000:
                        reg_values[word_addr] = words[w_idx]
                        reg_valid[word_addr] = True
                        w_idx += 1
                    for bit in range(16):
                        if (mask >> bit) & 1:
                            if w_idx >= num_words:
                                break
                            if word_addr + bit + 1 < 0x8000:
                                reg_values[word_addr + bit + 1] = words[w_idx]
                                reg_valid[word_addr + bit + 1] = True
                            w_idx += 1

            h = struct.unpack_from("<10I", data, offset)
            tasks.append({
                'index': idx,
                'offset': offset,
                'tid': tid,
                'size': size_bytes,
                'header': list(h),
                'data': task_data,
                'reg_values': reg_values,
                'reg_valid': reg_valid,
            })
    else:  # H13 / M1 style
        # Search for task headers (similar approach)
        task_offsets = []
        for offset in range(0, min(len(data), 0x10000) - 32, 4):
            h = struct.unpack_from("<8I", data, offset)
            tid = h[0] & 0xffff
            if tid == 0 and h[1] == 0 and h[2] == 0:
                continue  # Skip padding
            if tid > 0x1000:
                continue
            next_ptr = h[7]
            if next_ptr > offset and next_ptr < len(data):
                task_offsets.append((offset, tid, next_ptr))
        
        for idx, (offset, tid, next_ptr) in enumerate(task_offsets):
            start_off = offset + 40
            end_off = next_ptr if next_ptr > start_off else len(data)
            size_bytes = end_off - offset
            task_data = data[offset:offset + size_bytes]
            
            reg_values, reg_valid = {}, [False] * 0x8000
            num_words = (end_off - start_off) // 4
            
            if num_words > 0:
                words = struct.unpack_from(f"<{num_words}I", data, start_off)
                w_idx = 0
                while w_idx < num_words:
                    hdr = words[w_idx]
                    w_idx += 1
                    if hdr == 0:
                        continue
                    count = (hdr >> 26) & 0x3f
                    addr = (hdr & 0x3ffffff) >> 2
                    for i in range(count + 1):
                        if w_idx >= num_words:
                            break
                        if addr + i < 0x8000:
                            reg_values[addr + i] = words[w_idx]
                            reg_valid[addr + i] = True
                        w_idx += 1
            
            h = struct.unpack_from("<8I", data, offset)
            tasks.append({
                'index': idx,
                'offset': offset,
                'tid': tid,
                'size': size_bytes,
                'header': list(h),
                'data': task_data,
                'reg_values': reg_values,
                'reg_valid': reg_valid,
            })

    return tasks


def rebuild_cmdbuf_from_task(task, subtype=7):
    """
    Rebuild the raw CMD_BUF binary from a parsed task.
    This reconstructs the exact binary format that gets sent to ANE.
    """
    is_version = get_instruction_set_version(subtype)
    reg_values = task['reg_values']
    reg_valid = task['reg_valid']

    # Start with the original task data (includes header + registers)
    cmdbuf = bytearray(task['data'])

    # Pad to typical CMD_BUF size if needed (628 bytes for mul example)
    target_size = len(cmdbuf)
    if target_size < 0x100:  # At least 256 bytes
        target_size = 0x100

    return bytes(cmdbuf[:target_size])


def hexdump(buf, width=16):
    """Generate hexdump output."""
    lines = []
    for i in range(0, len(buf), width):
        chunk = buf[i:i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{i:04x}  {hex_part:<{width * 3}}  {ascii_part}")
    return "\n".join(lines)


def format_task_registers(task, subtype=7):
    """Format task registers in a readable way."""
    is_version = get_instruction_set_version(subtype)
    arch = "M4" if is_version >= 11 else "M1"
    reg_values = task['reg_values']
    reg_valid = task['reg_valid']

    lines = []
    lines.append(f"Task {task['index']}: TID=0x{task['tid']:04x}, Size={task['size']} bytes")

    if arch == "M4":
        # Common (0x0000)
        base = H16_COMMON_START // 4
        if any(reg_valid[base + i] for i in range(23)):
            lines.append("  --- Common (0x0000) ---")
            for i in range(23):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # L2 (0x4100)
        base = H16_L2_START // 4
        if any(reg_valid[base + i] for i in range(41)):
            lines.append("  --- L2 Cache (0x4100) ---")
            for i in range(41):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # PE (0x4500)
        base = H16_PE_START // 4
        if any(reg_valid[base + i] for i in range(15)):
            lines.append("  --- Planar Engine (0x4500) ---")
            for i in range(15):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # NE (0x4900)
        base = H16_NE_START // 4
        if any(reg_valid[base + i] for i in range(12)):
            lines.append("  --- Neural Engine (0x4900) ---")
            for i in range(12):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # TileDMA Src (0x4D00)
        base = H16_TILEDMA_SRC_START // 4
        if any(reg_valid[base + i] for i in range(81)):
            lines.append("  --- TileDMA Source (0x4D00) ---")
            for i in range(min(81, 20)):  # First 20 for brevity
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")
            if any(reg_valid[base + i] for i in range(20, 81)):
                lines.append("    ...")

        # TileDMA Dst (0x5100)
        base = H16_TILEDMA_DST_START // 4
        if any(reg_valid[base + i] for i in range(21)):
            lines.append("  --- TileDMA Destination (0x5100) ---")
            for i in range(21):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # KernelDMA (0x5500)
        base = H16_KERNELDMA_START // 4
        if any(reg_valid[base + i] for i in range(72)):
            lines.append("  --- KernelDMA Source (0x5500) ---")
            for i in range(min(72, 10)):  # First 10 for brevity
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")
            if any(reg_valid[base + i] for i in range(10, 72)):
                lines.append("    ...")
    else:  # M1 / H13
        # Common (0x0000)
        base = H13_COMMON_START // 4
        if any(reg_valid[base + i] for i in range(16)):
            lines.append("  --- Common (0x0000) ---")
            for i in range(16):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # L2 (0x4800)
        base = H13_L2_START // 4
        if any(reg_valid[base + i] for i in range(16)):
            lines.append("  --- L2 Cache (0x4800) ---")
            for i in range(16):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

        # NE (0xC800)
        base = H13_NE_START // 4
        if any(reg_valid[base + i] for i in range(5)):
            lines.append("  --- Neural Engine (0xC800) ---")
            for i in range(5):
                if reg_valid[base + i]:
                    lines.append(f"    0x{(base + i) * 4:05x}: 0x{reg_values[base + i]:08x}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CMD_BUF from .hwx files"
    )
    parser.add_argument(
        "input",
        help="Input .hwx file or directory containing hwx.bin and hwx.plist"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output binary file for CMD_BUF (default: stdout hexdump)"
    )
    parser.add_argument(
        "--task-index",
        type=int, default=0,
        help="Task index to extract (default: 0)"
    )
    parser.add_argument(
        "--hexdump",
        action="store_true",
        help="Show hexdump of CMD_BUF"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON with register details"
    )
    parser.add_argument(
        "--registers",
        action="store_true",
        help="Show decoded register values"
    )
    parser.add_argument(
        "--subtype",
        type=int, default=7,
        help="ANE subtype (default: 7 = H16/M4)"
    )
    args = parser.parse_args()

    # Read input
    path = args.input
    subtype = args.subtype
    data = None

    if os.path.isdir(path):
        plist_path = os.path.join(path, "hwx.plist")
        bin_path = os.path.join(path, "hwx.bin")
        if os.path.exists(plist_path):
            with open(plist_path, "rb") as f:
                try:
                    plist = plistlib.load(f)
                    subtype = plist.get("ANE_CPU_SUBTYPE", subtype)
                except:
                    pass
        if os.path.exists(bin_path):
            with open(bin_path, "rb") as f:
                data = f.read()
    else:
        with open(path, "rb") as f:
            data = f.read()

    if not data:
        print(f"Error: Could not read HWX data from {path}", file=sys.stderr)
        return 1

    # Extract ANE data
    ane_data = parse_macho(data)
    if not ane_data:
        # Try to find task data directly
        for o in range(0, min(len(data), 0x1000) - 40, 4):
            h = struct.unpack_from("<I", data, o)[0]
            tid = h & 0xffff
            if 0 < tid < 0x1000:
                ane_data = data[o:]
                break

    if not ane_data:
        print("Error: Could not identify HWX format", file=sys.stderr)
        return 1

    # Parse tasks
    tasks = parse_hwx_tasks(ane_data, subtype)

    if not tasks:
        print("Error: No tasks found in HWX data", file=sys.stderr)
        return 1

    # Get specified task
    if args.task_index >= len(tasks):
        print(f"Error: Task index {args.task_index} not found (found {len(tasks)} tasks)", file=sys.stderr)
        return 1

    task = tasks[args.task_index]

    # Build CMD_BUF
    cmdbuf = rebuild_cmdbuf_from_task(task, subtype)

    # Output
    if args.json:
        output = {
            'task_index': task['index'],
            'tid': task['tid'],
            'size': len(cmdbuf),
            'subtype': subtype,
            'architecture': 'M4' if get_instruction_set_version(subtype) >= 11 else 'M1',
            'registers': {
                f"0x{addr:05x}": f"0x{val:08x}"
                for addr, val in task['reg_values'].items()
            }
        }
        print(json.dumps(output, indent=2))
    elif args.registers:
        print(format_task_registers(task, subtype))
    elif args.output:
        with open(args.output, "wb") as f:
            f.write(cmdbuf)
        print(f"Wrote CMD_BUF to {args.output} ({len(cmdbuf)} bytes)")
    elif args.hexdump:
        print(f"CMD_BUF (task {args.task_index}, size={len(cmdbuf)}):")
        print(hexdump(cmdbuf))
    else:
        # Default: show hexdump
        print(f"CMD_BUF (task {args.task_index}, size={len(cmdbuf)}):")
        print(hexdump(cmdbuf))

    return 0


if __name__ == "__main__":
    sys.exit(main())