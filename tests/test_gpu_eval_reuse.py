import json
from pathlib import Path
import tempfile
import unittest

from miniport.gpu_eval import reused_baseline_reports


class BaselineReuseTests(unittest.TestCase):
    def test_partial_baseline_and_seed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline" / "mlp-heldout" / "evaluation.json"
            path.parent.mkdir(parents=True)
            report = dict(condition="baseline", checkpoint=None, task_id="mlp-heldout",
                          policy_seed=42, version="test", integrity_audit_passes=False)
            path.write_text(json.dumps(report))
            rows = reused_baseline_reports(tmp, ["mlp-heldout", "attention-heldout"], 42, "test")
            self.assertEqual(rows, [report])
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                reused_baseline_reports(tmp, ["mlp-heldout"], 43, "test")
