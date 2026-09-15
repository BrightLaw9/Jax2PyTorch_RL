import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from miniport.experiment import ExperimentConfig, assert_paired
from miniport.integrity import scan, submission_source, audit_passes, AUDIT_CHECKS
from miniport.sealed import (BackendUnavailable, CandidateFailure, DockerBackend, bounded_run,
                             grade, make_cases, summary, verify_submission)
from miniport.sealed import fresh_seeds
from miniport.tasks import GATES, training_tasks, heldout_tasks
from miniport.trajectory import Recorder, select_checkpoints

ORACLE = Path(__file__).parent / "fixtures" / "correct_port.py"


class AdmissionTests(unittest.TestCase):
    def test_oracle_admitted(self):
        self.assertTrue(scan(ORACLE.read_text())["passed"])

    def test_shortcuts_rejected(self):
        for source in ("import jax", "from flax import linen", "import subprocess",
                       "import importlib", "eval('1')", "__import__('os')",
                       "torch.load('answer')", "torch.zeros = custom",
                       "from torch import load as replay", "from torch.utils import cpp_extension"):
            with self.subTest(source=source):
                self.assertFalse(scan(source)["passed"])

    def test_underscore_names_admitted(self):
        for source in ("super().__init__()", "self._parameters", "x.__class__",
                       "__custom = 1", "import torch._custom", "from torch import _custom"):
            with self.subTest(source=source):
                self.assertTrue(scan(source)["passed"])

    def test_extra_files_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate.py").write_text("import torch")
            self.assertEqual(submission_source(root), "import torch")
            (root / "test.py").write_text("pass")
            with self.assertRaises(ValueError):
                submission_source(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate.py").symlink_to(ORACLE.resolve())
            with self.assertRaises(ValueError):
                submission_source(root)

    def test_audit_bound_to_source_and_all_checks(self):
        audit = {"reviewer": "test-reviewer", "decision": "valid", "submission_sha256": "a",
                 "checks": {k: "valid" for k in AUDIT_CHECKS}}
        self.assertTrue(audit_passes(audit, "a"))
        self.assertFalse(audit_passes(audit, "b"))
        audit["checks"]["generic_implementation"] = "unclear"
        self.assertFalse(audit_passes(audit, "a"))

    def test_resource_identity(self):
        config = ExperimentConfig().validate()
        self.assertEqual(config.context_cap, 4096)
        a = config.identity("prompt", ["edit", "test", "stop"])
        assert_paired(a, a)
        with self.assertRaises(ValueError):
            assert_paired(a, config.identity("changed", ["edit", "test", "stop"]))
        with self.assertRaises(ValueError):
            ExperimentConfig(concurrent_rollouts=2).validate()


class ProtocolTests(unittest.TestCase):
    def test_timeout_and_output_limit(self):
        with self.assertRaisesRegex(CandidateFailure, "timeout"):
            bounded_run([sys.executable, "-c", "import time; time.sleep(5)"], {}, timeout=0.1)
        with patch("miniport.sealed.MAX_OUTPUT_BYTES", 1000):
            with self.assertRaisesRegex(CandidateFailure, "output_limit"):
                bounded_run([sys.executable, "-c", "print('x' * 10000)"], {})

    def test_response_claims_do_not_advance_gates(self):
        task = training_tasks()[0]
        request, expected = make_cases(task, (11,))
        self.assertNotIn("expected", request)
        self.assertNotIn("seed", request["cases"][0])
        result = grade(task, {"cases": [{"topology": True, "passed": True, "parameters": {}}]}, expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["current_gate"], "parameters")

    def test_randomization_changes_shapes(self):
        task = heldout_tasks()[0]
        request, _ = make_cases(task, (123, 456), randomized=True)
        self.assertNotEqual(request["cases"][0]["config"]["length"], task.length)
        self.assertNotEqual(request["cases"][0]["config"]["batch"], task.batch)

    def test_fresh_challenge_reused_across_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenge.json"
            task = heldout_tasks()[0]
            first = fresh_seeds(task, path)
            self.assertEqual(first, fresh_seeds(task, path))
            with self.assertRaises(ValueError):
                fresh_seeds(heldout_tasks()[1], path)

    def test_malformed_response_fails_closed(self):
        task = training_tasks()[0]
        _, expected = make_cases(task, (11,))
        for response in (None, [], {"cases": [None]}, {"cases": []}):
            self.assertFalse(grade(task, response, expected)["passed"])

    def test_unavailable_backend_is_outage(self):
        class Missing:
            def run(self, source, request):
                raise BackendUnavailable("unavailable")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "candidate.py").write_text(ORACLE.read_text())
            result = verify_submission(training_tasks()[0], directory, backend=Missing())
            self.assertFalse(result["passed"])
            self.assertTrue(result["environment_outage"])

    def test_recorder_uses_external_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            submission.mkdir()
            (submission / "candidate.py").write_text("import torch\n")
            recorder = Recorder(training_tasks()[0], submission, root / "rollout.jsonl")
            failure = summary(GATES[:2], 0.1, "parity_mismatch", "layer")
            with patch("miniport.trajectory.verify_submission", return_value=failure):
                recorder.record("test", "verify")
                repeated = recorder.record("test", "verify")
                self.assertIn("repetition", repeated["diagnostic_reward"]["events"])
                stopped = recorder.record("stop", "stop")
                self.assertIn("premature_termination", stopped["diagnostic_reward"]["events"])
                with self.assertRaises(ValueError):
                    recorder.record("test", "verify")
            self.assertEqual(len((root / "rollout.jsonl").read_text().splitlines()), 3)
            self.assertEqual(len(select_checkpoints(recorder.records)), 3)

    def test_repeated_attention_failure_gets_one_recovery_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            submission.mkdir()
            source = "import torch\nx=1\n"
            (submission / "candidate.py").write_text(source)
            task = next(task for task in training_tasks() if task.template == "attention")
            recorder = Recorder(task, submission, root / "rollout.jsonl")
            failure = summary(GATES[:2], error_type="runtime_error", location="attention.layer")
            failure["diagnostic"] = {"code": "runtime_error", "message": "size 4 must match size 2"}
            with patch("miniport.trajectory.verify_submission", return_value=failure):
                recorder.record("test", "test")
                reflected = recorder.record("test", "test")
                guidance = reflected["policy_observation"]["recovery_guidance"]
                self.assertTrue(any("[batch, heads, sequence, head_dim]" in line
                                    for line in guidance["expected_shape_ledger"]))
                terminal = recorder.record("edit", "edit candidate.py",
                    policy_action={"type": "edit", "source": source})
            self.assertTrue(terminal["done"])
            self.assertEqual(terminal["termination_reason"], "recovery_edit_made_no_executable_change")

    def test_environment_outage_is_not_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            submission.mkdir()
            (submission / "candidate.py").write_text("import torch\n")
            recorder = Recorder(training_tasks()[0], submission, root / "rollout.jsonl")
            with patch("miniport.trajectory.verify_submission", return_value=summary(GATES)):
                recorder.record("test", "verify")
            with patch("miniport.trajectory.verify_submission", return_value=summary(outage=True)):
                first = recorder.record("test", "verify")
                self.assertEqual(first["diagnostic_reward"]["total"], 0)
                self.assertIsNone(first["training_reward"])
                self.assertFalse(first["trainable"])
                with self.assertRaises(ValueError):
                    recorder.record("stop", "stop")


@unittest.skipUnless(os.environ.get("MINIPORT_DOCKER_TESTS") == "1", "Set MINIPORT_DOCKER_TESTS=1 for container integration tests")
class DockerTests(unittest.TestCase):
    def test_all_variants_in_real_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "candidate.py").write_text(ORACLE.read_text())
            for task in training_tasks() + heldout_tasks():
                with self.subTest(task=task.id):
                    result = verify_submission(task, directory, seeds=(11, 23))
                    self.assertTrue(result["passed"], result)

    def test_final_pass_requires_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "candidate.py").write_text(ORACLE.read_text())
            result = verify_submission(heldout_tasks()[0], directory, seeds=(191, 227), final=True)
            self.assertTrue(result["hidden_parity_passes"], result)
            self.assertTrue(result["fresh_randomization_passes"], result)
            self.assertFalse(result["valid"])
            self.assertEqual(result["evaluation_terminal_reward"], 0)

    def test_corrupt_output_fails_external_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ORACLE.read_text().replace('"output": output}', '"output": output + 1}')
            (Path(directory) / "candidate.py").write_text(source)
            result = verify_submission(training_tasks()[0], directory, seeds=(11,))
            self.assertFalse(result["passed"])
            self.assertEqual(result["current_gate"], "end_to_end")

    def test_container_boundary(self):
        # Explicitly bypass admission to probe the OS boundary with trusted test code.
        source = '''
import os, socket, importlib.util
def build(config):
    assert importlib.util.find_spec("jax") is None
    assert importlib.util.find_spec("miniport") is None
    assert not os.path.exists("/var/run/docker.sock")
    try:
        open("/workspace/submission/changed.py", "w").write("bad")
    except OSError:
        pass
    else:
        raise RuntimeError("Writable submission")
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1)
    except OSError:
        pass
    else:
        raise RuntimeError("Network allowed")
    raise LookupError("All boundary checks passed")
'''
        request, _ = make_cases(training_tasks()[0], (11,))
        result = DockerBackend().run(source, request)
        self.assertEqual(result["cases"][0]["error_type"], "LookupError", result)

    def test_timeout_cleans_up_container(self):
        request, _ = make_cases(training_tasks()[0], (11,))
        with self.assertRaisesRegex(CandidateFailure, "timeout"):
            DockerBackend(timeout=2).run("while True: pass\n", request)
