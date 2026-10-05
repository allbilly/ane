import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import external_weights
from bpe import Tokenizer
from checks import compare, cpu_parity, integrity
from hwx import parse_tasks, relocate
from model import CPUKernels, GPT2
from external_weights import find_weights, verify_weights, blob_bytes, load_weights
import replay
from packing import PackedAssets, reconstruct, matrix_bytes


class ExternalWeightsTests(unittest.TestCase):
    def package(self, root):
        package = root / "package"
        package.mkdir()
        (package / "model-checksums.json").write_text(json.dumps(
            {f"tensor{i}.bin": "unused for safetensors validation" for i in range(196)}))
        return package

    def test_concurrent_downloads_publish_one_valid_checkpoint(self):
        checkpoint = b"test checkpoint data" * 4096
        expected = hashlib.sha256(checkpoint).hexdigest()
        barrier = threading.Barrier(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, cache = self.package(root), root / "cache"
            class Response(io.BytesIO):
                waited = False
                def read(response, size=-1):
                    if not response.waited:
                        response.waited = True
                        barrier.wait(timeout=5)
                        # Both writers must have their own open partial file.
                        self.assertEqual(len(list(cache.glob("*.download"))), 2)
                        barrier.wait(timeout=5)
                    return super().read(size)
            with patch.object(external_weights, "HF_SHA256", expected), \
                 patch.object(external_weights.urllib.request, "urlopen",
                              side_effect=lambda *a, **kw: Response(checkpoint)), \
                 ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(external_weights.setup, package, output=cache,
                                           progress=lambda message: None) for _ in range(2)]
                paths = [future.result(timeout=10) for future in futures]
            self.assertEqual(paths, [cache.resolve() / "model.safetensors"] * 2)
            self.assertEqual(paths[0].read_bytes(), checkpoint)
            self.assertEqual(list(cache.glob("*.download")), [])

    def test_failed_download_preserves_previous_file_and_cleans_partial(self):
        expected = hashlib.sha256(b"expected checkpoint").hexdigest()
        class Interrupted(io.BytesIO):
            calls = 0
            def read(response, size=-1):
                response.calls += 1
                if response.calls == 2:
                    raise OSError("connection interrupted")
                return super().read(size)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, cache = self.package(root), root / "cache"
            cache.mkdir()
            source = cache / "model.safetensors"
            source.write_bytes(b"previous checkpoint")
            for response, error in ((io.BytesIO(b"wrong checkpoint"), ValueError),
                                    (Interrupted(b"partial checkpoint"), OSError)):
                with self.subTest(error=error.__name__), \
                     patch.object(external_weights, "HF_SHA256", expected), \
                     patch.object(external_weights.urllib.request, "urlopen", return_value=response):
                    with self.assertRaises(error):
                        external_weights.setup(package, output=cache, progress=lambda message: None)
                    self.assertEqual(source.read_bytes(), b"previous checkpoint")
                    self.assertEqual(list(cache.glob("*.download")), [])


class PackageTests(unittest.TestCase):
    def test_integrity(self):
        self.assertGreater(integrity(ROOT), 300)

    def test_checksum_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weight.bin").write_bytes(b"corrupt")
            (root / "checksums.json").write_text(json.dumps({"weight.bin": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                integrity(root)

    def test_all_task_packets_preserved_and_bank_relocation_reversible(self):
        names = json.loads((ROOT / "package.json").read_text())["kernels"]
        count = extended = 0
        for name in names:
            root = ROOT / "kernels" / name
            meta = json.loads((root / "meta.json").read_text())
            program = (ROOT / meta["program"]).read_bytes()
            original = json.loads(gzip.decompress((root / "registers.json.gz").read_bytes()))["tasks"]
            tasks = parse_tasks(program, meta["td_size"], meta["td_count"])
            reverse = {int(new): int(old) for old, new in meta["bank_map"].items()}
            restored = parse_tasks(relocate(program, tasks, reverse), meta["td_size"], meta["td_count"])
            for actual, expected in zip(restored, original):
                self.assertEqual(actual["header"], expected["header"])
                self.assertEqual(actual["offset"], expected["offset"])
                self.assertEqual(actual["size"], expected["size"])
                self.assertEqual({str(k): v for k, v in actual["registers"].items()}, expected["registers"])
            count += len(tasks)
            extended += meta["extended_headers"]
        self.assertEqual(count, 1574)
        self.assertGreater(extended, 0)

    def test_truncated_and_cyclic_tasks_rejected(self):
        meta = json.loads((ROOT / "kernels/decode_proj_L0/meta.json").read_text())
        data = (ROOT / meta["program"]).read_bytes()
        with self.assertRaises(ValueError):
            parse_tasks(data[:60], meta["td_size"], meta["td_count"])
        corrupted = bytearray(data)
        struct.pack_into("<I", corrupted, 0x1C, 4)
        with self.assertRaises(ValueError):
            parse_tasks(corrupted, meta["td_size"], meta["td_count"])

    def test_compiled_ffn_weights_deduplicated(self):
        for layer in range(12):
            a = json.loads((ROOT / f"kernels/decode_ffn_L{layer}/meta.json").read_text())
            b = json.loads((ROOT / f"kernels/prefill_ffn_L{layer}/meta.json").read_text())
            self.assertEqual(a["weights"], b["weights"])

    def test_numerical_check_rejects_zero_output_and_wrong_shape(self):
        expected = np.arange(10, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            compare(np.zeros_like(expected), expected, "zeros")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare(expected[:1], expected, "shape")


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer(ROOT / "tokenizer")
        selected = find_weights()
        if selected is None:
            raise unittest.SkipTest("set GPT2_WEIGHTS to run generation tests without downloading weights")
        cls.weights = load_weights(selected)
        verify_weights(selected, ROOT, weights=cls.weights)
        if not getattr(cls.weights, "reference", True):
            raise unittest.SkipTest("original-model fixtures require reference-equivalent weights")

    def model(self):
        return GPT2(self.weights, CPUKernels(self.weights))

    def test_unicode_and_hash_merge_roundtrip(self):
        self.assertEqual(self.tokenizer.encode("Hello world"), [15496, 995])
        self.assertEqual(self.tokenizer.encode("##"), [2235])
        for text in ("香港 🧠 Asahi", "I'm\n\tready", "   ", "##### foo###", "é e\u0301"):
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_logits_against_independent_orion_cpu(self):
        self.assertEqual(len(cpu_parity(ROOT, self.model(), self.tokenizer)), 3)

    def test_hf_fp16_tensors_match_original_orion_blobs(self):
        if not hasattr(self.weights, "tensors"):
            self.skipTest("external BLOBFILE backend selected")
        expected = json.loads((ROOT / "model-checksums.json").read_text())
        actual = {name + ".bin": hashlib.sha256(blob_bytes(value)).hexdigest()
                  for name, value in self.weights.tensors.items()}
        self.assertEqual(actual, expected)

    def test_greedy_generation_and_reset(self):
        model = self.model()
        tokens = self.tokenizer.encode("Hello world")
        expected = [11, 314, 1101, 407, 1654, 644, 284, 910, 13, 198, 198, 1, 40, 1101, 7926, 11, 475, 314]
        self.assertEqual(list(model.generate(tokens, 18)), expected)
        self.assertEqual(list(model.generate(tokens, 4)), expected[:4])
        self.assertEqual(list(model.generate(tokens, 0)), [])

    def test_seeded_sampling(self):
        tokens = self.tokenizer.encode("Hello world")
        model = self.model()
        a = list(model.generate(tokens, 3, temperature=0.7, top_k=10, seed=123))
        self.assertEqual(a, list(model.generate(tokens, 3, temperature=0.7, top_k=10, seed=123)))

    def test_context_limits(self):
        model = self.model()
        with self.assertRaises(ValueError):
            list(model.generate([], 3))
        with self.assertRaises(ValueError):
            list(model.generate([1] * 1024, 2))
        model.position = 1024
        with self.assertRaises(ValueError):
            model.step(1)
        with self.assertRaises(ValueError):
            model.step(50257)

    def test_all_49_kernels_reconstructed_against_captured_hashes(self):
        names = json.loads((ROOT / "package.json").read_text())["kernels"]
        self.assertEqual(len(names), 49)
        payloads = 0
        for name in names:
            meta = json.loads((ROOT / "kernels" / name / "meta.json").read_text())
            for field in ("weights", "constants", "program"):
                recipe = meta["packing"][field]
                actual = reconstruct(ROOT, self.weights, recipe)
                self.assertEqual(hashlib.sha256(actual).hexdigest(), recipe["sha256"])
                if "template" in recipe:
                    template = (ROOT / recipe["template"]).read_bytes()
                    for op in recipe["operations"]:
                        self.assertEqual(template[op["offset"]:op["offset"] + op["size"]], bytes(op["size"]))
                payloads += 1
        self.assertEqual(payloads, 147)

    def test_corrupted_packed_cache_is_rebuilt_from_hf(self):
        meta = json.loads((ROOT / "kernels/decode_proj_L0/meta.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            assets = PackedAssets(ROOT, self.weights, Path(directory))
            expected = assets.payload(meta, "weights")
            path = Path(directory) / (meta["packing"]["weights"]["sha256"] + ".bin")
            path.write_bytes(bytes(len(expected)))
            self.assertEqual(assets.payload(meta, "weights"), expected)
            self.assertEqual(path.read_bytes(), expected)

    def test_wrong_tensor_rejected_before_cache_write(self):
        meta = json.loads((ROOT / "kernels/decode_proj_L0/meta.json").read_text())
        weights = self.weights
        class WrongWeights:
            def get(self, name, shape):
                value = weights.get(name, shape).copy()
                if name == "layer0/wq":
                    value[0, 0] += 1
                return value
        with tempfile.TemporaryDirectory() as directory:
            assets = PackedAssets(ROOT, WrongWeights(), Path(directory))
            with self.assertRaisesRegex(ValueError, "differs from captured reference"):
                assets.payload(meta, "weights")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_only_weight_free_templates_remain(self):
        referenced = set()
        for path in (ROOT / "kernels").glob("*/meta.json"):
            meta = json.loads(path.read_text())
            for recipe in meta["packing"].values():
                if "template" in recipe:
                    referenced.add(recipe["template"])
            self.assertFalse((ROOT / meta["weights"]).exists())
        actual = {str(p.relative_to(ROOT)) for p in (ROOT / "objects").glob("*.bin")}
        self.assertEqual(actual, referenced)

    def test_matrix_engine_schedule_with_asymmetric_input(self):
        # Small deterministic input catches swapped input/output axes, engine
        # order, bias placement, and mixed 32/16 tail tiles independently of HF.
        matrix = np.arange(768 * 3, dtype=np.float32).reshape(768, 3)
        bias = -np.arange(768, dtype=np.float32)
        class InputWeights:
            def get(self, name, shape):
                return matrix if name == "matrix" else bias
        data = matrix_bytes(InputWeights(), dict(matrix="matrix", bias="bias", shape=(768, 3), tiles=[32, 16]))
        # One engine occupies align64(32*4*2 + 16*4*2) = 384 bytes.
        engine1 = np.frombuffer(data[384:768], dtype="<f2")
        self.assertTrue(np.array_equal(engine1[:32], bias[32:64]))
        self.assertTrue(np.array_equal(engine1[32:128], matrix[32:64].T.ravel()))
        self.assertTrue(np.array_equal(engine1[128:144], bias[528:544]))
        self.assertTrue(np.array_equal(engine1[144:192], matrix[528:544].T.ravel()))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = find_weights()
        if selected is None:
            raise unittest.SkipTest("external GPT-2 weights required for reconstructed replay tests")
        cls.weights = load_weights(selected)
        verify_weights(selected, ROOT, weights=cls.weights)

    def assets(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return PackedAssets(ROOT, self.weights, Path(directory.name))

    def test_abi_layout(self):
        self.assertEqual(replay.ctypes.sizeof(replay.BOInit), 24)
        self.assertEqual(replay.ctypes.sizeof(replay.BOFree), 8)
        self.assertEqual(replay.ctypes.sizeof(replay.Submit), 152)
        self.assertEqual(replay.INIT, 0xC0186441)
        self.assertEqual(replay.SUBMIT, 0xC0986443)

    def test_project_buffers_weights_bootstrap_and_cleanup(self):
        allocated, requests = [], []
        class FakeBuffer:
            def __init__(self, fd, size):
                self.size = (size + 0x3FFF) & ~0x3FFF
                self.handle = len(allocated) + 1
                self.map = bytearray(self.size)
                self.closed = False
                allocated.append(self)
            def write(self, data, offset=0):
                if offset + len(data) > self.size:
                    raise ValueError("overflow")
                self.map[offset:offset + len(data)] = data
            def close(self):
                self.closed = True
        class FakeDevice:
            root, fd = ROOT, 123
            assets = self.assets()
        fixture = ROOT / "fixtures/decode_proj_L0"
        def submit(fd, number, request):
            self.assertEqual(number, replay.SUBMIT)
            self.assertEqual((request.tsk_size, request.td_count, request.td_size), (16384, 18, 504))
            self.assertEqual(request.handles[1], 0)
            requests.append(request)
            for bank, name in ((4, "k16"), (5, "q16"), (6, "v16")):
                allocated[request.handles[bank] - 1].write((fixture / (name + ".bin")).read_bytes())
        with patch.object(replay, "Buffer", FakeBuffer), patch.object(replay, "ioctl", submit):
            kernel = replay.Kernel(FakeDevice(), "decode_proj_L0")
            meta = kernel.meta
            cmd = kernel.buffers[0].map
            weights = FakeDevice.assets.payload(meta, "weights")
            self.assertEqual(cmd[16384:16384 + len(weights)], weights)
            self.assertEqual((struct.unpack_from("<I", kernel.bootstrap.map)[0] >> 16) & 255, 64)
            x = np.fromfile(fixture / "input.bin", dtype="<f2").reshape(768, 32)
            result = kernel.run(x)
            self.assertEqual(kernel.buffers[7].map[:49152], x.tobytes())
            self.assertEqual(set(result), {"k16", "q16", "v16"})
            vectors = kernel.vector(x[:, 0])
            written_input = np.frombuffer(kernel.buffers[7].map, dtype="<f2", count=768 * 32).reshape(768, 32)
            self.assertTrue(np.array_equal(written_input[:, 0], x[:, 0]))
            self.assertTrue(np.all(written_input[:, 1:] == 0))
            for bank, name in ((4, "k16"), (5, "q16"), (6, "v16")):
                self.assertEqual(vectors[name].shape, (768, 1))
                self.assertTrue(np.array_equal(vectors[name], result[name][:, :1]))
                # Returned vectors must remain valid after the device buffer
                # is overwritten by another submission or released.
                kernel.buffers[bank].write(bytes(49152))
                self.assertTrue(np.array_equal(vectors[name], result[name][:, :1]))
            kernel.close()
            kernel.close()
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(buffer.closed for buffer in allocated))

    def test_allocation_failure_cleans_earlier_buffers(self):
        allocations = []
        class FailingBuffer:
            def __init__(self, fd, size):
                if len(allocations) == 2:
                    raise OSError("allocation failed")
                allocations.append(self)
                self.closed = False
            def write(self, data, offset=0):
                pass
            def close(self):
                self.closed = True
        class FakeDevice:
            root, fd = ROOT, 123
            assets = self.assets()
        with patch.object(replay, "Buffer", FailingBuffer):
            with self.assertRaises(OSError):
                replay.Kernel(FakeDevice(), "decode_proj_L0")
        self.assertTrue(all(buffer.closed for buffer in allocations))


if __name__ == "__main__":
    unittest.main()
