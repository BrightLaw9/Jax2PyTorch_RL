import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from miniport.action_value import dataset_sha256, fit, load
from miniport.teacher_collect import TeacherConfig, _verified_teacher_actions, mine_stuck_states
from miniport.policy import Completion
from miniport.sealed import summary
from miniport.tasks import GATES, training_tasks
from miniport.experiment import ExperimentConfig
from miniport.train import combined_value_rows


class TeacherCollectionTests(unittest.TestCase):
    def test_single_state_manual_override_is_verified_and_labeled(self):
        task = next(task for task in training_tasks() if task.id == "attention-0")
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(message["content"] for message in messages)
            def encode(self, text, **kwargs):
                return [1, 2]
        class Policy:
            revision = "teacher-revision"
            tokenizer = Tokenizer()
            config = ExperimentConfig()
            def generate(self, *args):
                raise AssertionError("manual override must precede generation")
        item = {"state_id": 1, "task_id": task.id,
            "snapshot": {"source": "import torch\n", "history": [], "remaining": 5},
            "student_action": {"prompt": "student prompt", "text": '{"type":"test"}',
                "prompt_ids": [3], "output_ids": [4]}}
        with tempfile.TemporaryDirectory() as temporary:
            manual = Path(temporary) / "manual.py"
            manual.write_text("import torch\nfixed=True\n")
            with patch("miniport.sealed.verify_submission", return_value=summary(GATES)):
                accepted = _verified_teacher_actions(Policy(), [item], {task.id: task},
                    Path(temporary) / "candidates", object(),
                    SimpleNamespace(candidate_resamples=4, seed=42,
                        manual_corrections={"1": {"path": str(manual), "model": "gpt-5.6-luna"}}), False)
            self.assertEqual(len(accepted), 1)
            saved = json.loads((Path(temporary) / "candidates/state-001.json").read_text())
            self.assertEqual(saved["candidate_origin"], "external_model_override")
            self.assertEqual(saved["source_model"], "gpt-5.6-luna")

    def test_teacher_retry_receives_previous_candidate_and_diagnostic(self):
        task = next(task for task in training_tasks() if task.id == "attention-0")
        class Policy:
            revision = "teacher-revision"
            calls = []
            def generate(self, task, source, history, remaining, seed):
                self.calls.append((source, list(history), remaining, seed))
                fixed = len(self.calls) > 1
                return Completion("teacher prompt", json.dumps({"type": "edit",
                    "source": "import torch\nfixed=True\n" if fixed else "import torch\nbroken=True\n"}), [1], [2])
        item = {"state_id": 0, "task_id": task.id,
            "snapshot": {"source": "import torch\n", "history": [], "remaining": 5},
            "student_action": {"prompt": "student prompt", "text": '{"type":"test"}',
                "prompt_ids": [3], "output_ids": [4]}}
        failure = summary(GATES[:2], error_type="candidate_exception", location="attention.layer")
        failure["diagnostic"] = {"code": "runtime_error", "message": "bad shape"}
        with tempfile.TemporaryDirectory() as temporary, patch(
                "miniport.sealed.verify_submission", side_effect=[failure, summary(GATES)]):
            policy = Policy()
            accepted = _verified_teacher_actions(policy, [item], {task.id: task},
                Path(temporary) / "candidates", object(),
                SimpleNamespace(candidate_resamples=2, seed=42, manual_corrections={}), False)
            self.assertEqual(len(accepted), 1)
            self.assertIn("broken=True", policy.calls[1][0])
            self.assertIn("bad shape", policy.calls[1][1][-1]["content"])
            attempts = [json.loads(line) for line in
                        (Path(temporary) / "candidates/state-000.attempts.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["protocol"] == "iterative-diagnostic-v1" for row in attempts))

    def test_memory_cap_and_task_validation(self):
        TeacherConfig("run", gpu_memory_gb=16).validate()
        with self.assertRaisesRegex(ValueError, "16 GiB"):
            TeacherConfig("run", gpu_memory_gb=17).validate()
        with self.assertRaisesRegex(ValueError, "paired-state limit"):
            TeacherConfig("run", paired_state_limit=0).validate()
        with self.assertRaisesRegex(ValueError, "canonical"):
            TeacherConfig("run", task_ids=("attention-heldout",)).validate()
        with self.assertRaisesRegex(ValueError, "numeric state IDs"):
            TeacherConfig("run", manual_corrections={"attention": "fix.py"}).validate()
        with self.assertRaisesRegex(ValueError, "path and model"):
            TeacherConfig("run", manual_corrections={"1": {"path": "fix.py"}}).validate()

    def test_mines_repeated_noop_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "update-0-group-0-member-0"
            episode.mkdir()
            steps = []
            for i in range(3):
                steps.append({"source": "import torch\nx=1\n", "history": [], "remaining": 18-i,
                    "prompt": "repair", "completion": '{"type":"edit","source":"import torch\\nx=1\\n"}',
                    "prompt_ids": [1], "output_ids": [2]})
            (episode / "trajectory.json").write_text(json.dumps(
                {"task_id": "attention-0", "reward": 0, "steps": steps}))
            records = []
            for i in range(3):
                records.append({"index": i, "protected_summary": {"error_type": "RuntimeError",
                    "failure_location": "attention.layer"}, "policy_observation": {
                    "feedback": {"diagnostic": {"message": "4 must match 2"}},
                    "edit_feedback": {"code": "no_executable_change"}}})
            (episode / "protected.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
            states = mine_stuck_states(root, ("attention-0",), 2)
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0]["origin_action"], 2)
            self.assertEqual(states[0]["task_id"], "attention-0")
            # A family with no stuck state is coverage metadata, not a fatal
            # collection error when another requested family has evidence.
            states = mine_stuck_states(root, ("mlp-0", "attention-0"), 2)
            self.assertEqual([state["task_id"] for state in states], ["attention-0"])

    def test_mines_schema_two_step_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "update-0-group-0-member-0"
            (episode / "steps").mkdir(parents=True)
            concise = []
            for i in range(3):
                detail = {"source": "import torch\nx=1\n", "history": [], "remaining": 18-i,
                    "prompt": "repair", "completion": '{"type":"edit","source":"import torch\\nx=1\\n"}',
                    "prompt_ids": [1], "output_ids": [2]}
                detail_path = f"steps/{i:03d}-trajectory.json"
                (episode / detail_path).write_text(json.dumps(detail))
                concise.append({"index": i, "details_file": detail_path})
            (episode / "trajectory.json").write_text(json.dumps(
                {"schema_version": 2, "task_id": "attention-0", "reward": 0, "steps": concise}))
            record = {"protected_summary": {"error_type": "RuntimeError", "failure_location": "attention.layer"},
                "policy_observation": {"feedback": {"diagnostic": {"message": "same shape error"}},
                "edit_feedback": {"code": "no_executable_change"}}}
            (episode / "protected.jsonl").write_text("".join(json.dumps(dict(record, index=i))+"\n" for i in range(3)))
            states = mine_stuck_states(root, ("attention-0",), 1)
            self.assertEqual(states[0]["student_action"]["prompt"], "repair")

    def test_combined_rows_bind_critic_to_dataset(self):
        rows = [{"task_id": task, "prompt": "p", "action": action,
                 "successes": successes, "count": 4}
                for task in ("attention-0", "mlp-0")
                for action, successes in (("good", 4), ("bad", 0))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "value-data.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows[:2]))
            (root / "teacher-value-data.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows[2:]))
            combined = combined_value_rows(root)
            path = root / "critic.json"
            _, report = fit(combined, path, epochs=2)
            self.assertEqual(report["dataset_sha256"], dataset_sha256(combined))
            load(path, combined)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load(path, combined[:-1])


if __name__ == "__main__":
    unittest.main()
