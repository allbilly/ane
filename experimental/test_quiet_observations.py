"""Check that paging at child exit affects the recorded benchmark classification."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen35.tools import quiet_benchmark, summarize_benchmark


class QuietObservationTests(unittest.TestCase):
    def test_busy_lock_retains_failure_receipt(self):
        def flock(fd, operation):
            if operation != quiet_benchmark.fcntl.LOCK_UN:
                raise BlockingIOError()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            argv = ["quiet_benchmark", "--model", "unused", "--traces", "unused", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(Path, "home", return_value=root), \
                 patch.object(quiet_benchmark.fcntl, "flock", side_effect=flock), \
                 patch.object(quiet_benchmark.time, "monotonic", side_effect=[0, 601]):
                with self.assertRaisesRegex(RuntimeError, "remained busy"):
                    quiet_benchmark.main()
            report = json.loads((output / "session.json").read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("remained busy", report["error"])
            self.assertEqual(report["tasks"], [])

    def test_exit_swapouts_are_observed(self):
        state = dict(swapouts=0)

        def observe(excluded=()):
            return dict(utc="test", loadavg=[0, 0, 0], external_cpu_percent=0,
                        vm={"Swapouts": state["swapouts"]}, observation_errors={})

        class Child:
            pid = 999999

            def __init__(self, command, **kwargs):
                Path(command[command.index("--output") + 1]).write_text(json.dumps(dict(results=[])))

            def wait(self):
                state["swapouts"] += 8
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            argv = ["quiet_benchmark", "--model", "unused", "--traces", "unused",
                    "--output", str(output), "--rounds", "1"]
            with patch.object(sys, "argv", argv), patch.object(Path, "home", return_value=root), \
                 patch.object(quiet_benchmark, "observe", side_effect=observe), \
                 patch.object(quiet_benchmark.time, "sleep"), \
                 patch.object(quiet_benchmark.subprocess, "Popen", Child), \
                 patch.object(quiet_benchmark.subprocess, "run", return_value=SimpleNamespace(stdout="", stderr="", returncode=0)), \
                 patch.object(summarize_benchmark, "summarize", return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                quiet_benchmark.main()
            report = json.loads((output / "session.json").read_text())
            self.assertEqual(len(report["tasks"]), 6)
            self.assertTrue(all(t["affected"] for t in report["tasks"]))
            self.assertEqual([t["observed_swapout_pages"] for t in report["tasks"]], [8] * 6)


if __name__ == "__main__":
    unittest.main()
