"""Guard complete accuracy evidence and actual batched-attention execution."""
import copy
import json
from pathlib import Path
import unittest

from whisper.attention import attention_profiles

RECEIPT = Path(__file__).resolve().parents[1]/"whisper/results/macos-encoder-attention-20261008.json"


class EncoderAttentionTests(unittest.TestCase):
    def test_both_algorithms_keep_every_full_vector_and_argmax_gate(self):
        report = json.loads(RECEIPT.read_text())
        counts = {"jfk":25,"jfk-first-5s":8,"jfk-repeat":47}
        self.assertEqual(report["status"],"PASS")
        self.assertTrue(report["timings_accepted"])
        self.assertEqual(report["full_logit_gate_nrmse"],.005)
        self.assertEqual(set(report["correctness"]),{"flash","blas"})
        for clips in report["correctness"].values():
            self.assertEqual({c["audio"]:c["decoder_calls"] for c in clips},counts)
            for clip in clips:
                self.assertTrue(clip["all_histories_match"])
                self.assertEqual(len(clip["hf_logit_checks"]),counts[clip["audio"]])
                for vector in clip["hf_logit_checks"]:
                    for backend in ("cpu","ane"):
                        self.assertLess(vector[backend]["nrmse"],.005)
                        self.assertTrue(vector[backend+"_argmax_match"])
        for kinds in report["configurations"].values():
            for clips in kinds.values():
                for clip in clips.values():
                    self.assertEqual(len(clip["runs"]),20)
                    self.assertEqual(len(clip["warmups"]),4)

    def test_partial_wrong_precision_and_wrong_backend_attention_are_rejected(self):
        report = json.loads(RECEIPT.read_text())
        row = report["configurations"]["blas"]["ane_precision_cpu"]["jfk"]["runs"][0]
        records = []
        for encoded in row["attention_matrices"]:
            matrix = dict(report["matrix_metadata"][encoded["metadata"]])
            matrix.update(zip(report["matrix_timing_columns"],encoded["timing_us"]))
            records.append(matrix)
        def log(rows):
            return "ENCODER_ATTENTION encoder: layers=4 heads=6 positions=1500 implementation=blas\n"+"".join(
                "ATTENTION_MATRIX_PROFILE\t"+json.dumps(r)+"\n" for r in rows)
        attention_profiles(log(records),True)
        # The empty conversion interval can still span a timer tick.
        tick = copy.deepcopy(records)
        tick[0]["total_us"] += 1-tick[0]["convert_us"]
        tick[0]["convert_us"] = 1
        attention_profiles(log(tick),True)
        for field,value in (("batch",5),("m",1499),("backend","CPU"),("weight_type","f16"),("requested_threads",1)):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                attention_profiles(log(bad),True)
        for bad in (records[:-1],[records[0]]+records[:-1]):
            with self.assertRaises(ValueError):
                attention_profiles(log(bad),True)
        with self.assertRaises(ValueError):
            attention_profiles(log(records),False)


if __name__ == "__main__":
    unittest.main()
