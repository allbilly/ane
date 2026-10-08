"""Check that native precision evidence preserves the full gates and isolation."""
import json
from pathlib import Path
import unittest

from whisper.scripts.benchmark_macos import check_runtime


class NativePrecisionTests(unittest.TestCase):
    def test_every_captured_cpu_and_ane_vector_passes_the_unchanged_hf_gate(self):
        path = Path(__file__).resolve().parents[1] / "whisper/results/macos-native-precision-20261007.json"
        report = json.loads(path.read_text())
        counts = {"jfk":25,"jfk-first-5s":8,"jfk-repeat":47}
        self.assertEqual(report["full_logit_gate_nrmse"], .005)
        self.assertEqual(report["encoder_cpu_gate_nrmse"], .005)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["timings_accepted"])
        for clip in report["correctness"]:
            self.assertEqual(len(clip["hf_logit_checks"]), counts[clip["audio"]])
            self.assertEqual(len(clip["logit_checks"]), counts[clip["audio"]])
            for vector in clip["hf_logit_checks"]:
                for backend in ("cpu","ane"):
                    self.assertLess(vector[backend]["nrmse"], .005)
                    self.assertTrue(vector[backend+"_argmax_match"])
            self.assertTrue(clip["all_histories_match"])

    def test_partial_projection_execution_and_wrong_baseline_are_rejected(self):
        prefix = "whisper_init_with_params_no_state: use gpu = 0\nMACOS_PRECISION ready: paired projections\n"
        log = prefix+"MACOS_PRECISION encoder: projections=24 submissions=24\n"
        check_runtime(log,"precision_cpu",1,require_dispatch=True)
        for bad in (log.replace("submissions=24","submissions=23"),
                    log.replace("projections=24","projections=23"),
                    log+"aneforge: encoder ready\n",log+"MACOS_ANE encoder: submissions=1\n"):
            with self.assertRaises(ValueError):
                check_runtime(bad,"precision_cpu",1,require_dispatch=True)
        with self.assertRaises(ValueError):
            check_runtime(log,"cpu_cpu",1,require_dispatch=True)
        with self.assertRaises(ValueError):
            check_runtime(log,"ane_cpu",1,require_dispatch=True)


if __name__ == "__main__":
    unittest.main()
