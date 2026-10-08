"""Keep experimental fusion from silently relaxing the native benchmark gates."""
import copy
import json
from pathlib import Path
import unittest

from whisper.validation import matrix_profiles


class CrossKVProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "whisper/results/macos-cross-kv-fusion-20261007.json"
        cls.receipt = json.loads(path.read_text())

    def profiles(self, layout):
        row = self.receipt["configurations"][layout]["ane_complete_cpu"]["jfk"]["runs"][0]
        return [dict(self.receipt["matrix_metadata"][m["metadata"]],
                     **dict(zip(self.receipt["matrix_timing_columns"], m["timing_us"])))
                for m in row["cross_kv_matrices"]]

    def log(self, profiles):
        return "\n".join("MATRIX_PROFILE\t" + json.dumps(m) for m in profiles)

    def test_real_eight_product_profile_still_requires_all_products(self):
        profiles = self.profiles("separate")
        self.assertEqual(matrix_profiles(self.log(profiles), required=True), profiles)
        with self.assertRaises(ValueError):
            matrix_profiles(self.log(profiles[:-1]), required=True)
        with self.assertRaises(ValueError):
            matrix_profiles(self.log(profiles[:-1] + [profiles[0]]), required=True)

    def test_real_fused_profile_requires_explicit_layout_and_cached_weights(self):
        profiles = self.profiles("fused")
        with self.assertRaises(ValueError):
            matrix_profiles(self.log(profiles), required=True)
        self.assertEqual(matrix_profiles(self.log(profiles), required=True, layout="fused"), profiles)
        for field, value in (("n", 384), ("weight_type", "f16"), ("weight", "uncached")):
            altered = copy.deepcopy(profiles)
            altered[0][field] = value
            with self.assertRaises(ValueError):
                matrix_profiles(self.log(altered), required=True, layout="fused")

    def test_unchanged_cpu_gate_checked_every_full_vector(self):
        expected = {"jfk": 25, "jfk-first-5s": 8, "jfk-repeat": 47}
        self.assertEqual(self.receipt["full_logit_gate_nrmse"], .005)
        for clip in self.receipt["correctness"]:
            self.assertEqual(clip["decoder_calls"], expected[clip["audio"]])
            self.assertEqual(len(clip["hf_logit_checks"]), expected[clip["audio"]])
            for vector in clip["hf_logit_checks"]:
                self.assertLess(vector["cpu"]["nrmse"], .005)
                self.assertTrue(vector["cpu_argmax_match"])
        self.assertFalse(self.receipt["timings_accepted"])
        self.assertTrue(self.receipt["numerical_failures"])


if __name__ == "__main__":
    unittest.main()
