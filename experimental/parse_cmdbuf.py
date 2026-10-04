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
from contextlib import redirect_stdout
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hwx_parsing import (
    decode_regs, get_arch_name,
    iter_hwx_tasks, load_hwx_data, parse_macho,
)


def parse_hwx_tasks(data, subtype=7):
    """Extract tasks using the shared generation-specific HWX reader."""
    section = parse_macho(data, subtype)
    if section is not None:
        data = section
    tasks = []
    for index, task in enumerate(iter_hwx_tasks(data, subtype)):
        state = task['state']
        offset, size = task['offset'], task['size']
        tasks.append({
            'index': index,
            'offset': offset,
            'tid': task['tid'],
            'size': size,
            'header': list(task['header']),
            'data': data[offset:offset + size],
            'reg_values': {word: state.values[word] for word, valid in enumerate(state.valid) if valid},
            'reg_valid': state.valid,
        })
    return tasks


def rebuild_cmdbuf_from_task(task, subtype=7):
    """
    Rebuild the raw CMD_BUF binary from a parsed task.
    This reconstructs the exact binary format that gets sent to ANE.
    """
    return bytes(task['data'])


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
    """Use the same register names and field decoders as the parser CLI."""
    output = io.StringIO()
    with redirect_stdout(output):
        print(f"Task {task['index']}: TID=0x{task['tid']:04x}, Size={task['size']} bytes")
        decode_regs(task['reg_values'], task['reg_valid'], subtype)
    return output.getvalue().rstrip()


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
        type=int, default=None,
        help="ANE subtype override (default: read container/plist metadata, or H13 for raw streams)"
    )
    args = parser.parse_args()

    try:
        ane_data, subtype = load_hwx_data(args.input, args.subtype)
        if not ane_data:
            raise ValueError('could not identify HWX command stream')
        tasks = parse_hwx_tasks(ane_data, subtype)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not tasks:
        print("Error: No tasks found in HWX data", file=sys.stderr)
        return 1

    # Get specified task
    if not 0 <= args.task_index < len(tasks):
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
            'architecture': get_arch_name(subtype),
            'registers': {
                f"0x{addr*4:05x}": f"0x{val:08x}"
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
