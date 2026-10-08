"""Check sampler accounting and preservation of native diagnostic evidence."""
import json
from pathlib import Path
import unittest

from whisper.scripts.summarize_decoder_profile import summarize


class DecoderProfileTests(unittest.TestCase):
    def test_repeated_role_frames_count_each_sample_once(self):
        text = """Call graph:
    10 main  (in main) + 0  [0x3000]
    + 10 whisper_compute_probs  (in libwhisper) + 0  [0x1020]
    +   8 whisper_compute_probs  (in libwhisper) + 4  [0x1024]
    +   ! 5 expf  (in libmath) + 0  [0x2004]
Total number in stack:
Binary Images:
0x1000 - 0x1fff libwhisper (0) <AAAA-BBBB> /libwhisper
0x2000 - 0x2fff  libmath (0) <CCCC-DDDD> /libmath
0x3000 - 0x3fff main (0) <EEEE-FFFF> /main
"""
        role = summarize(text)["roles"]["whisper_compute_probs"]
        self.assertEqual(role["inclusive_stack_count"],10)
        self.assertEqual(role["repeated_frame_count"],8)
        self.assertEqual(sum(r["self_stack_count"] for r in role["leaves"]),10)
        self.assertEqual(role["self_stack_count"],5)
        self.assertEqual(next(r["file_offset"] for r in role["leaves"] if r["symbol"] == "expf"),"0x4")
        with self.assertRaises(ValueError):
            summarize(text.replace("+   8 whisper", "+   11 whisper"))

    def test_all_native_vectors_and_profile_roles_are_retained(self):
        path = Path(__file__).resolve().parents[1]/"whisper/results/macos-decoder-profile-20261008"
        report = json.loads((path/"summary.json").read_text())
        self.assertEqual(report["full_logit_gate_nrmse"],.005)
        self.assertTrue(report["numerical_failures"])
        self.assertEqual(sum(r["decoder_calls"] for r in report["boundaries"]),160)
        counts = {"jfk":25,"jfk-first-5s":8,"jfk-repeat":47}
        for clip in report["correctness"]:
            self.assertEqual(len(clip["hf_logit_checks"]),counts[clip["audio"]])
            for vector in clip["hf_logit_checks"]:
                self.assertLess(vector["cpu"]["nrmse"],.005)
                self.assertTrue(vector["cpu_argmax_match"])
        assembly = (path/"selected-assembly.txt").read_text()
        for clip in report["profiles"].values():
            self.assertEqual(clip["transcriptions"],202)
            self.assertEqual(clip["environment"]["sampling_after_excluded_warmups"],2)
            self.assertEqual(clip["environment"]["driver_returncode"],0)
            for role in clip["roles"].values():
                self.assertGreater(role["inclusive_stack_count"],0)
                self.assertEqual(sum(r["self_stack_count"] for r in role["leaves"]),role["inclusive_stack_count"])
            for name in ("libggml-cpu.0.25.1.dylib","libwhisper.1.9.4.dylib","libsystem_m.dylib"):
                self.assertIn(clip["images"][name]["uuid"],assembly)
        self.assertIn("fcvtl",assembly)
        self.assertIn("fmla.4s",assembly)
        self.assertEqual(len(report["selected_vocabulary_kernels"]),2)


if __name__ == "__main__":
    unittest.main()
