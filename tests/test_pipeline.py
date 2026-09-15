from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from miniport.controller import Controller
from miniport.experiment import ExperimentConfig
from miniport.policy import Completion
from miniport.rollouts import Snapshot, rollout
from miniport.sealed import SubprocessBackend, summary, verify_submission
from miniport.tasks import GATES, training_tasks, heldout_tasks

ORACLE = Path(__file__).parent / "fixtures" / "correct_port.py"


class SubprocessTests(unittest.TestCase):
    def test_all_templates_without_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "candidate.py").write_text(ORACLE.read_text())
            with patch("miniport.sealed.DockerBackend.run", side_effect=AssertionError("Docker must never be called")):
                for task in training_tasks()[::3]:
                    with self.subTest(task=task.id):
                        result = verify_submission(task, directory, seeds=(11,), backend=SubprocessBackend())
                        self.assertTrue(result["passed"], result)
                        self.assertEqual(result["isolation"], "subprocess")

    def test_candidate_cuda_hidden_and_no_host_env_mutation(self):
        import os
        source = '''import os
import torch
def build(config):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_available()
    raise LookupError("CPU confirmed")
'''
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        result = SubprocessBackend().run(source, {"cases": [{"config": {}}]})
        self.assertEqual(result["cases"][0]["error_type"], "LookupError")
        self.assertEqual(previous, os.environ.get("CUDA_VISIBLE_DEVICES"))


class RolloutTests(unittest.TestCase):
    def test_invalid_action_costs_budget_and_evaluation_has_no_training_reward(self):
        class Policy:
            config = ExperimentConfig()
            def generate(self, *args):
                return Completion("prompt", "not json", [1], [2])
        with tempfile.TemporaryDirectory() as directory:
            with patch("miniport.trajectory.verify_submission", return_value=summary()) as verifier:
                result = rollout(Policy(), heldout_tasks()[0], Snapshot("import torch\n", remaining=2),
                                 Path(directory) / "eval", object(), 42, evaluation=True)
            self.assertEqual(len(result["steps"]), 2)
            self.assertIsNone(result["reward"])
            self.assertEqual(verifier.call_count, 2)
            self.assertIn("JSONDecodeError", result["records"][0]["policy_observation"]["action_error"])

    def test_raw_completions_survive_interrupted_episode(self):
        class Policy:
            config = ExperimentConfig()
            calls = 0
            def generate(self, *args):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("interrupted")
                return Completion("prompt", '{"type":"edit","source":"unfinished', [1],
                                  [2] * self.config.max_new_tokens)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch("miniport.trajectory.verify_submission", return_value=summary()):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    rollout(Policy(), training_tasks()[0], Snapshot("import torch\n", remaining=2),
                            root, object(), 42)
            from miniport.resume import read_jsonl
            raw = read_jsonl(root / "completions.jsonl")[0]
            self.assertEqual(raw["completion"], '{"type":"edit","source":"unfinished')
            self.assertTrue(raw["token_limit_reached"])
            record = json.loads((root / "protected.jsonl").read_text())
            self.assertIn("output token limit reached", record["policy_observation"]["action_error"])

    def test_branches_restore_source_and_do_not_mutate_snapshot(self):
        class Policy:
            config = ExperimentConfig()
            def generate(self, *args):
                return Completion("prompt", '{"type":"stop"}', [1], [2])
        initial = Snapshot("import torch\n", remaining=2)
        with tempfile.TemporaryDirectory() as directory:
            forced = Completion("prompt", json.dumps({"type": "edit", "source": "import torch\n# branch A\n"}), [1], [2])
            with patch("miniport.trajectory.verify_submission", return_value=summary(GATES)):
                first = rollout(Policy(), training_tasks()[0], initial, Path(directory) / "a", object(), 42, forced=forced)
                second = rollout(Policy(), training_tasks()[0], initial, Path(directory) / "b", object(), 42)
            self.assertIn("branch A", (Path(first["submission"]) / "candidate.py").read_text())
            self.assertNotIn("branch A", (Path(second["submission"]) / "candidate.py").read_text())
            self.assertEqual(initial.source, "import torch\n")
            self.assertEqual(initial.history, [])


class TrainingMathTests(unittest.TestCase):
    def test_action_value_condition_includes_negative_diagnostic_shaping(self):
        import torch
        from types import SimpleNamespace
        from miniport.rl import accumulate
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.zeros(5, 5))
                self.enabled = True
            @contextmanager
            def disable_adapter(self):
                yield
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.adapter[input_ids])
        trajectories = []
        for reward, token, events in ((0, 2, {"repetition": -0.25}), (0, 3, {})):
            trajectories.append({"trainable": True, "reward": reward,
                "records": [{"diagnostic_reward": {"total": sum(events.values()), "events": events}}],
                "steps": [{"completion": Completion("", "", [0, 1], [token]), "value_advantage": 0}]})
        model = Model()
        report = accumulate(model, trajectories, condition="action_value", kl_coefficient=0,
            immediate_coefficient=0.1)
        self.assertEqual(report["rewards"], [-0.025, 0.0])

    def test_leave_one_out_and_action_mask(self):
        import torch
        from types import SimpleNamespace
        from miniport.rl import leave_one_out, action_log_probs
        self.assertEqual(leave_one_out([1, 0, 0]), [1, -0.5, -0.5])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.arange(25, dtype=torch.float32).reshape(5, 5))
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.table[input_ids])
        model = Model()
        completion = Completion("", "", [0, 1], [2, 3])
        actual = action_log_probs(model, completion)
        expected = torch.stack([model.table[1].log_softmax(-1)[2], model.table[2].log_softmax(-1)[3]])
        torch.testing.assert_close(actual, expected)
        tempered = action_log_probs(model, completion, temperature=0.6)
        expected_tempered = torch.stack([(model.table[1] / 0.6).log_softmax(-1)[2],
                                         (model.table[2] / 0.6).log_softmax(-1)[3]])
        torch.testing.assert_close(tempered, expected_tempered)

    def test_rl_updates_only_adapter_and_has_finite_gradients(self):
        import torch
        from types import SimpleNamespace
        from miniport.rl import accumulate
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.zeros(5, 5), requires_grad=False)
                self.adapter = torch.nn.Parameter(torch.zeros(5, 5))
                self.enabled = True
            @contextmanager
            def disable_adapter(self):
                self.enabled = False
                try:
                    yield
                finally:
                    self.enabled = True
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=(self.base + self.adapter if self.enabled else self.base)[input_ids])
        model = Model()
        trajectories = [{"trainable": True, "reward": reward, "records": [],
                         "steps": [{"completion": Completion("", "", [0, 1], [token]), "value_advantage": 0}]}
                        for reward, token in ((1, 2), (0, 3))]
        optimizer = torch.optim.SGD([model.adapter], lr=0.1)
        report = accumulate(model, trajectories, condition="terminal", kl_coefficient=0.01)
        self.assertEqual(report["used"], 2)
        self.assertTrue(torch.isfinite(model.adapter.grad).all())
        self.assertGreater(model.adapter.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertIsNone(model.base.grad)
        self.assertEqual(model.base.abs().sum().item(), 0)
        self.assertGreater(model.adapter.abs().sum().item(), 0)

    def test_action_value_requires_varied_outcomes_and_persists(self):
        from miniport.action_value import fit
        rows = [{"task_id": task, "prompt": "repair", "action": action, "successes": successes, "count": 4}
                for task in ("mlp-0", "sampling-0") for action, successes in (("fix weight", 4), ("stop", 0))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            model, report = fit(rows, path, epochs=5)
            self.assertTrue(path.exists())
            self.assertTrue(0 <= model.predict("repair", "fix weight") <= 1)
            self.assertEqual(report["validation_examples"], 2)
            with self.assertRaises(ValueError):
                fit([dict(r, successes=0) for r in rows], path, epochs=1)
