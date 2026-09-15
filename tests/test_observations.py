import json
from pathlib import Path
import tempfile
import unittest

from miniport.controller import Controller
from miniport.observations import visible_feedback
from miniport.tasks import training_tasks


class StaticFeedbackTests(unittest.TestCase):
    def test_findings_reach_policy_and_saved_log(self):
        for source, rule in (("import numpy as np\n", "forbidden_import"),
                             ("```py\n", "syntax")):
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                submission = root / "submission"
                submission.mkdir()
                (submission / "candidate.py").write_text("import torch\n")
                log = root / "protected.jsonl"
                controller = Controller(training_tasks()[0], submission, log)
                observation = controller.step({"type": "edit", "source": source})
                findings = observation["feedback"]["static_scan_findings"]
                self.assertEqual(findings[0]["line"], 1)
                self.assertEqual(findings[0]["rule"], rule)
                self.assertTrue(findings[0]["message"])
                saved = json.loads(log.read_text())
                self.assertEqual(saved["policy_observation"], observation)

    def test_other_feedback_does_not_expose_findings(self):
        for report in ({"passed": True},
                       {"environment_outage": True},
                       {"passed": False, "current_gate": "layer",
                        "error_type": "parity_mismatch", "max_abs_error": 0.5}):
            with self.subTest(report=report):
                self.assertNotIn("static_scan_findings", visible_feedback(report))
