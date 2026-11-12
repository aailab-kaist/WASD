"""Exercise the real summary CLI using only synthetic logs and NumPy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(os.environ.get("WASD_SCRIPT", Path(__file__).resolve().parents[1] / "parse_result.py"))
METRICS = ["dolly-512", "self_inst-512", "vicuna-512", "sinst_11_-512", "uinst_11_-512"]
SEEDS = [10, 20, 30, 40, 50]


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "results/model/eval_main"
        self.summary = self.base / "_summary/train_run#step.txt"
        for metric in METRICS:
            for seed in SEEDS:
                path = self.log_path(metric, seed)
                path.parent.mkdir(parents=True)
                path.write_text("{'rougeL': 0.9}\n", encoding="utf-8")

    def log_path(self, metric, seed):
        return self.base / metric / "train_run/step" / str(seed) / "log.txt"

    def run_cli(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--model", "model", "--method", "train/run/step"],
            cwd=self.root, capture_output=True, text=True,
        )
        if os.environ.get("AUDIT_LOG_DIR"):
            dest = Path(os.environ["AUDIT_LOG_DIR"])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / (self._testMethodName + ".json")).write_text(json.dumps({
                "command": result.args, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "summary": self.summary.read_text() if self.summary.exists() else None,
            }, indent=2), encoding="utf-8")
        return result

    def expected(self, first=None):
        return "".join(
            f"{m:<15} | {np.mean(values):.2f} ({np.std(values):.2f}) | {values}\n"
            for m in METRICS
            for values in [first if m == METRICS[0] and first else [0.9] * 5]
        )

    def assert_invalid(self, path, metric_missing=False):
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        diagnostic = result.stderr.replace("\\\\", "\\").replace("/", os.sep)
        self.assertIn(str(path.relative_to(self.root)), diagnostic)
        if metric_missing:
            self.assertIn("ROUGE-L", result.stderr)
        self.assertFalse(self.summary.exists())

    def test_complete(self):
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.summary.read_text(), self.expected())

    def test_middle_metric_missing(self):
        path = self.log_path(METRICS[0], 20)
        path.write_text("evaluation interrupted\n")
        self.assert_invalid(path, metric_missing=True)

    def test_first_metric_missing(self):
        path = self.log_path(METRICS[0], 10)
        path.write_text("no metric\n")
        self.assert_invalid(path, metric_missing=True)

    def test_file_missing(self):
        path = self.log_path(METRICS[2], 30)
        path.unlink()
        self.assert_invalid(path)

    def test_final_file_missing(self):
        path = self.log_path(METRICS[-1], 50)
        path.unlink()
        self.assert_invalid(path)

    def test_existing_summary_preserved(self):
        self.summary.parent.mkdir(parents=True)
        self.summary.write_text("previous summary\n")
        self.log_path(METRICS[2], 30).write_text("missing score\n")
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.summary.read_text(), "previous summary\n")

    def test_later_distinct_score(self):
        self.log_path(METRICS[0], 20).write_text("{'rougeL': 0.3}\n")
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.summary.read_text(), self.expected([0.9, 0.3, 0.9, 0.9, 0.9]))

    def test_multiple_entries_keep_last(self):
        self.log_path(METRICS[0], 20).write_text("{'rougeL': 0.1}\n{'rougeL': 3e-1}\n")
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.summary.read_text(), self.expected([0.9, 0.3, 0.9, 0.9, 0.9]))


if __name__ == "__main__":
    unittest.main()
