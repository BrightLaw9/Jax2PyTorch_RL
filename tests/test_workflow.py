import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from miniport.controller import Controller
from miniport.review import create_notebook, save_annotation
from miniport.sealed import summary
from miniport.tasks import training_tasks


class WorkflowTests(unittest.TestCase):
    def test_actions_log_and_review_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            submission.mkdir()
            candidate = submission / "candidate.py"
            candidate.write_text("import torch\n")
            log = root / "rollout.jsonl"
            controller = Controller(training_tasks()[0], submission, log)
            with patch("miniport.trajectory.verify_submission", return_value=summary(error_type="incomplete", location="topology")):
                observation = controller.step({"type": "edit", "source": "import torch\n# edit\n"})
                self.assertNotIn("diff", observation)
                changed = controller.recorder.records[-1]
                self.assertEqual(changed["changed_files"], ["candidate.py"])
                self.assertIn("+# edit", changed["diff"])
                controller.step({"type": "test"})
                controller.step({"type": "stop"})
                stopped = controller.recorder.records[-1]
            with self.assertRaises(ValueError):
                controller.step({"type": "edit", "source": "pass"})
            review = root / "review"
            self.assertEqual(create_notebook([log], review), 3)
            notebook = json.loads((review / "review.ipynb").read_text())
            compile("".join(notebook["cells"][1]["source"]), "review", "exec")
            row = save_annotation(review / "annotations.jsonl", stopped, "reviewer", "unclear", 0, "unclear")
            self.assertEqual(row["ordinal"], 0)
            with self.assertRaises(ValueError):
                save_annotation(review / "annotations.jsonl", stopped, "reviewer", "unclear", 0)

    def test_shell_and_path_injection_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            submission.mkdir()
            candidate = submission / "candidate.py"
            candidate.write_text("import torch\n")
            controller = Controller(training_tasks()[0], submission, root / "log.jsonl")
            for action in ({"type": "shell", "command": "echo bad"},
                           {"type": "edit", "path": "../test.py", "source": "bad"},
                           {"type": "edit", "source": "bad", "proposed_next_action": []}):
                with self.assertRaises(ValueError):
                    controller.step(action)
            self.assertEqual(candidate.read_text(), "import torch\n")
