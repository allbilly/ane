"""Strict H13G reader and replay relocation for the Orion GPT-2 dump."""
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_container(data):
    require(len(data) >= 32, "truncated HWX header")
    magic, cpu, subtype, _, ncmds, sizeofcmds, _, _ = struct.unpack_from("<8I", data)
    require((magic, cpu, subtype) == (0xBEEFFACE, 128, 4), "expected H13G/M1 HWX")
    end = 32 + sizeofcmds
    require(end <= len(data), "truncated load commands")
    segments, sections, threads, symbols = [], [], [], {}
    symtab = None
    offset = 32
    for _ in range(ncmds):
        require(offset + 8 <= end, "missing load command")
        cmd, size = struct.unpack_from("<2I", data, offset)
        require(size >= 8 and offset + size <= end, "invalid load command size")
        if cmd == 0x19:
            require(size >= 72, "truncated segment")
            name = data[offset + 8:offset + 24].split(b"\0")[0].decode()
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<4Q", data, offset + 24)
            count = struct.unpack_from("<I", data, offset + 64)[0]
            require(72 + 80 * count <= size, "truncated sections")
            require(fileoff + filesize <= len(data), "segment outside HWX")
            segment = dict(name=name, vmaddr=vmaddr, vmsize=vmsize,
                           fileoff=fileoff, filesize=filesize)
            segments.append(segment)
            for index in range(count):
                pos = offset + 72 + index * 80
                sname = data[pos:pos + 16].split(b"\0")[0].decode()
                addr, length, soff = struct.unpack_from("<QQI", data, pos + 32)
                sections.append(dict(name=sname, segment=name, addr=addr, size=length, offset=soff))
        elif cmd == 4 and size >= 0x820 and struct.unpack_from("<I", data, offset + 8)[0] == 1:
            bars = list(struct.unpack_from("<32Q", data, offset + 16))
            entry, size_minus_one, count = struct.unpack_from("<QII", data, offset + 0x810)
            threads.append(dict(bars=bars, entry=entry, td_size=(size_minus_one + 1) * 4, td_count=count))
        elif cmd == 2:
            require(size >= 24, "truncated symbol command")
            symtab = struct.unpack_from("<4I", data, offset + 8)
        offset += size
    require(offset == end and len(threads) == 1, "expected one H13 entry point")
    if symtab:
        symoff, count, stroff, strsize = symtab
        require(symoff + 16 * count <= len(data) and stroff + strsize <= len(data), "invalid symbols")
        for index in range(count):
            string, kind, section, _, value = struct.unpack_from("<IBBHQ", data, symoff + index * 16)
            require(string < strsize, "invalid string index")
            tail = data.find(b"\0", stroff + string, stroff + strsize)
            require(tail >= 0, "unterminated symbol")
            if kind == 0xF:
                symbols[data[stroff + string:tail].decode()] = dict(addr=value, section=section)
    return dict(segments=segments, sections=sections, thread=threads[0], symbols=symbols)


def parse_tasks(text, first_size, expected_count):
    """Follow NextPtr and NextSize; skip the optional 44-byte extended header."""
    tasks, seen = [], set()
    offset, size = 0, first_size
    while True:
        require(offset not in seen and len(tasks) < expected_count, "cyclic/excess task chain")
        seen.add(offset)
        require(size >= 40 and size % 4 == 0 and offset + size <= len(text), "invalid task bounds")
        header = struct.unpack_from("<10I", text, offset)
        require((header[0] & 0xFFFF) == len(tasks), "nonsequential task IDs")
        cursor = offset + (44 if header[6] & (1 << 24) else 40)
        require(cursor <= offset + size, "truncated extended header")
        registers = {}
        while cursor < offset + size:
            word = struct.unpack_from("<I", text, cursor)[0]
            cursor += 4
            if not word:
                continue
            count, address = (word >> 26) + 1, word & 0x3FFFFFF
            require(address % 4 == 0 and address + 4 * count <= 0x20000, "invalid register range")
            require(cursor + 4 * count <= offset + size, "truncated register packet")
            for index, value in enumerate(struct.unpack_from(f"<{count}I", text, cursor)):
                registers[address + 4 * index] = value
            cursor += 4 * count
        tasks.append(dict(offset=offset, size=size, header=list(header), registers=registers))
        pointer = header[7]
        if not pointer:
            break
        require(pointer >= offset + size and pointer % 4 == 0, "overlapping/misaligned tasks")
        offset, size = pointer, (((header[1] >> 16) & 0x1FF) + 1) * 4
    require(len(tasks) == expected_count, "task count mismatch")
    require(bool(tasks[-1]["header"][0] & (1 << 25)), "last task missing end-of-network")
    return tasks


def relocate(text, tasks, bank_map):
    """Remap BAR selectors only. Preserve packets, dependencies, and NextPtr."""
    result = bytearray(text)
    for task in tasks:
        for delta in (32, 36):
            word = struct.unpack_from("<I", text, task["offset"] + delta)[0]
            for shift in (0, 6, 12, 18):
                if word & (1 << (shift + 5)):
                    old = (word >> shift) & 31
                    require(old in bank_map, f"unmapped active BAR {old}")
                    word = (word & ~(31 << shift)) | (bank_map[old] << shift)
            struct.pack_into("<I", result, task["offset"] + delta, word)
    return bytes(result)
