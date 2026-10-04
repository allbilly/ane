"""Parser regressions checked against real HWX files and independent layouts."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import plistlib
import struct
import subprocess
import sys
import tempfile
import unittest

import hwx_parsing as parser
from experimental import parse_cmdbuf
from gpt2.hwx import parse_container, parse_tasks


ROOT = Path(__file__).resolve().parents[1]


def words(values):
    return struct.pack(f'<{len(values)}I', *values)


def h13_task(packets, *, tid=0, pointer=0, next_size=0, extended=False):
    header = [tid | (1 << 25), next_size << 16, 1, 0, 0, 0,
              (1 << 24) if extended else 0, pointer, 0, 0]
    return words(header + ([0xDEADBEEF] if extended else []) + packets)


def dense_task(subtype, packets, tid=0):
    header = [0] * (8 if subtype in (5, 6) else 9)
    header[0] = tid | ((len(header) + len(packets)) << 16)
    header[2] = 1
    return words(header + packets)


def container(stream, subtype):
    streams = stream if isinstance(stream, list) else [stream]
    command_size = 72 + 80 * len(streams)
    offset = 32 + command_size
    size = sum(map(len, streams))
    header = words([parser.HWX_MAGIC, 128, subtype, 2, 1, command_size, 0, 0])
    segment = struct.pack('<II16s4Q4I', 0x19, command_size, b'__TEXT',
                          0, size, offset, size, 7, 5, len(streams), 0)
    sections = []
    for stream in streams:
        sections.append(struct.pack('<16s16s2Q8I', b'__text', b'__TEXT',
                                    0, len(stream), offset, 2, 0, 0, 0, 0, 0, 0))
        offset += len(stream)
    return header + segment + b''.join(sections) + b''.join(streams)


def capture(function, *args, **kwargs):
    output = io.StringIO()
    with redirect_stdout(output):
        result = function(*args, **kwargs)
    return output.getvalue(), result


def register_state(subtype, registers):
    state = parser.HwxState(subtype, parser.get_instruction_set_version(subtype))
    for address, value in registers.items():
        index = address // 4
        state.values[index] = state.first_values[index] = value
        state.valid[index] = state.first_written[index] = True
    return state


class HwxParserTests(unittest.TestCase):
    def test_real_h13_corpus_matches_strict_gpt2_reader(self):
        paths = sorted((ROOT / 'hwx').rglob('*.hwx'))
        paths += sorted((ROOT / 'gpt2/training').rglob('*.hwx'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                stream, subtype = parser.load_hwx_data(path)
                self.assertEqual(subtype, 4)
                thread = parse_container(path.read_bytes())['thread']
                expected = parse_tasks(stream, thread['td_size'], thread['td_count'])
                tasks = parser.iter_hwx_tasks(stream, subtype)
                for new, old in zip(tasks, expected, strict=True):
                    state = new['state']
                    actual = {word * 4: state.values[word]
                              for word, valid in enumerate(state.valid) if valid}
                    self.assertEqual(actual, old['registers'])

    def test_short_h13_header_and_extended_header(self):
        for extended in (False, True):
            with self.subTest(extended=extended):
                data = h13_task([0x4800, 0x12345678], extended=extended)
                task, = parser.iter_hwx_tasks(data, 4)
                self.assertEqual(task['state'].values[0x4800 // 4], 0x12345678)
                self.assertEqual(sum(task['state'].valid), 1)

    def test_h13_chain_uses_next_pointer_and_next_size(self):
        second = h13_task([0x8800, 7], tid=1)
        first = h13_task([0x4800, 9], pointer=64, next_size=len(second) // 4 - 1)
        data = first.ljust(64, b'\0') + second + words([0x13800, 0xBAD])
        tasks = list(parser.iter_hwx_tasks(data, 4, first_task_size=len(first)))
        self.assertEqual([task['tid'] for task in tasks], [0, 1])
        self.assertEqual([task['size'] for task in tasks], [48, 48])
        self.assertEqual(tasks[1]['state'].values[0x8800 // 4], 7)
        self.assertFalse(tasks[1]['state'].valid[0x13800 // 4])

    def test_dense_headers_masked_writes_and_first_values(self):
        for subtype in (5, 6, 7, 9, 10, 11):
            with self.subTest(subtype=subtype):
                address = 0x10
                mask = (1 << 0) | (1 << 15)
                packets = [address | (1 << 15), 0xAA, 0xBB,
                           0x80000000 | (mask << 15) | address, 1, 2, 3]
                task, = parser.iter_hwx_tasks(dense_task(subtype, packets), subtype)
                state = task['state']
                self.assertEqual(state.values[address], 1)
                self.assertEqual(state.values[address + 1], 2)
                self.assertEqual(state.values[address + 16], 3)
                self.assertEqual(state.first_values[address], 0xAA)
                self.assertEqual(state.first_values[address + 1], 0xBB)
                self.assertEqual(sum(state.valid), 3)

    def test_generation_specific_names_and_formats(self):
        self.assertEqual(parser.get_ch_fmt_name(0), 'UINT8')
        self.assertEqual(parser.get_ch_fmt_name(1), 'INT8')
        self.assertEqual(parser.get_ch_fmt_name(4), 'E4M3')
        self.assertEqual(parser.get_kernel_fmt_name(4), 'INT4')
        self.assertIsNotNone(parser.get_reg_name(0x0500, 5))
        self.assertIsNone(parser.get_reg_name(0x4100, 5))
        self.assertIsNotNone(parser.get_reg_name(0x4100, 6))
        self.assertIsNone(parser.get_reg_name(0x0500, 6))
        self.assertEqual(parser.get_instruction_set_version(11), 24)

    def test_pe_condition_mask_does_not_include_reduction_bit(self):
        for subtype in (7, 9, 10):
            with self.subTest(subtype=subtype):
                state = register_state(subtype, {0x4500: (3 << 6) | (1 << 9)})
                output, _ = capture(parser.report_hwx_state, state, False)
                self.assertIn('Cond=3 (NotEqual)', output)
                self.assertIn('RedIdx=1', output)
        self.assertEqual([parser.get_pe_condition_name_v17(i) for i in range(8)],
                         ['None', 'Less', 'Greater', 'NotEqual', 'Equal',
                          'LessEqual', 'GreaterEqual', 'Abs'])

    def test_tiledma_unsigned_fields_and_destination_shift_mask(self):
        src_fmt = (7 << 16) | (15 << 28)
        dst_fmt = 3 | (1 << 8) | (1 << 11) | (3 << 12)
        state = register_state(7, {0x4D00: 1, 0x4D00 + 26 * 4: src_fmt,
                                   0x5100 + 14 * 4: dst_fmt})
        src_output, _ = capture(parser.print_tiledmasrc_h16, state)
        dst_output, _ = capture(parser.print_tiledmadst_h16, state)
        self.assertIn('OffCh=7 CmpVec=15', src_output)
        self.assertIn('Shift=1 -> FLOAT32', dst_output)

    def test_cachedma_masks_ignore_reserved_bits(self):
        state = register_state(7, {
            0x5900 + 6 * 4: (0x123456 << 7) | 0xC000007F,
            0x5900 + 7 * 4: (0x456 << 17) | 0xF001FFFF,
        })
        output, _ = capture(parser.print_cachedma_h16, state)
        self.assertIn('DSID_Size=0x123456', output)
        self.assertIn('Footprint: Arg2=0x456', output)

    def test_json_is_one_document_for_multiple_dense_tasks(self):
        first = dense_task(7, [0, 123])
        second = dense_task(7, [0x1040, 456], tid=1)
        stream = first.ljust((len(first) + 15) & ~15, b'\0') + second
        output, result = capture(parser.parse_hwx, stream, 7, True)
        self.assertEqual(json.loads(output), result)
        self.assertEqual(len(result['tasks']), 2)
        self.assertEqual(result['tasks'][1]['registers'][0]['addr'], '0x04100')

    def test_load_subtype_from_container_plist_and_override(self):
        stream = dense_task(5, [0x0500 // 4, 42])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'hwx.bin').write_bytes(stream)
            (root / 'hwx.plist').write_bytes(plistlib.dumps({'ANE_CPU_SUBTYPE': 5}))
            self.assertEqual(parser.load_hwx_data(root), (stream, 5))
            self.assertEqual(parser.load_hwx_data(root, 6), (stream, 6))
            (root / 'hwx.bin').write_bytes(container(stream, 6))
            self.assertEqual(parser.load_hwx_data(root), (stream, 6))
            self.assertEqual(parser.load_hwx_data(root, 5), (stream, 5))

    def test_both_clis_autodetect_and_preserve_json(self):
        stream = dense_task(6, [0x4100 // 4, 42])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.hwx'
            path.write_bytes(container(stream, 6))
            for entry in ('parse.py', 'hwx_parsing.py'):
                for arguments in ([], ['6'], ['-s', '6'], ['--container']):
                    with self.subTest(entry=entry, arguments=arguments):
                        command = [sys.executable, str(ROOT / entry), str(path), '-j'] + arguments
                        result = subprocess.run(command, check=True, text=True, capture_output=True)
                        report = json.loads(result.stdout)
                        self.assertEqual(report['tasks'][0]['subtype'], 6)
                        self.assertEqual(report['tasks'][0]['registers'][0]['name'],
                                         parser.get_reg_name(0x4100, 6))

    def test_h19_reports_raw_registers(self):
        stream = dense_task(11, [0x1040, 42])
        output, result = capture(parser.parse_hwx, stream, 11, True)
        task, = json.loads(output)['tasks']
        self.assertEqual(task['arch'], 'H19')
        self.assertEqual(task['registers'], [{'addr': '0x04100', 'val': '0x0000002a'}])
        output, _ = capture(parser.parse_hwx, stream, 11)
        self.assertIn('not yet verified', output)
        self.assertIn('0x04100: 0x0000002a', output)

    def test_cli_json_includes_all_executable_sections(self):
        streams = [dense_task(7, [0x1040, 42]), dense_task(7, [0x1140, 99])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'multi.hwx'
            path.write_bytes(container(streams, 7))
            command = [sys.executable, str(ROOT / 'parse.py'), str(path), '-j']
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            tasks = json.loads(result.stdout)['tasks']
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0]['registers'][0]['val'], '0x0000002a')
            self.assertEqual(tasks[1]['registers'][0]['val'], '0x00000063')

    def test_reject_invalid_packet_task_and_container_bounds(self):
        cases = [
            (dense_task(7, [0x1040 | (1 << 15), 1]), 7),
            (dense_task(6, [0x80000000 | (1 << 15) | 0x1040, 1]), 6),
            (dense_task(7, [0, 1])[:-4], 7),
            (h13_task([0x4800, 1], pointer=4), 4),
            (h13_task([0x4801, 1]), 4),
            (h13_task([0x4800 | (1 << 26), 1]), 4),
        ]
        for data, subtype in cases:
            with self.subTest(data=data.hex(), subtype=subtype):
                with self.assertRaises(ValueError):
                    list(parser.iter_hwx_tasks(data, subtype))
        for field, value in ((32 + 4, 0), (32 + 72 + 48, 0xFFFF)):
            data = bytearray(container(h13_task([0x4800, 1]), 4))
            struct.pack_into('<I', data, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parser.parse_macho(data)
        data = bytearray(container(h13_task([0x4800, 1]), 4))
        data[184:184] = words([0x40, 0])
        struct.pack_into('<2I', data, 16, 2, 160)
        struct.pack_into('<Q', data, 32 + 40, 192)
        struct.pack_into('<I', data, 32 + 72 + 48, 192)
        with self.assertRaises(ValueError):
            parser.parse_macho(data)

    def test_raw_stream_and_fallback_preserve_first_task(self):
        raw = h13_task([0x4800, 123])
        self.assertEqual(parser.find_task_stream(raw, 4), raw)
        self.assertEqual(parser.find_task_stream(b'arbitrary prefix' + raw, 4), raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.bin'
            path.write_bytes(raw)
            self.assertEqual(parser.load_hwx_data(path), (raw, 4))

    def test_cmdbuf_parser_shares_dense_and_h13_task_reader(self):
        for subtype, data in ((6, dense_task(6, [0x1040, 42])),
                              (4, h13_task([0x4800, 42]))):
            with self.subTest(subtype=subtype):
                task, = parse_cmdbuf.parse_hwx_tasks(data, subtype)
                self.assertEqual(task['tid'], 0)
                self.assertEqual(task['data'], data)
                self.assertEqual(parse_cmdbuf.rebuild_cmdbuf_from_task(task, subtype), data)
                self.assertIn('HW Block Register State', parse_cmdbuf.format_task_registers(task, subtype))


if __name__ == '__main__':
    unittest.main()
