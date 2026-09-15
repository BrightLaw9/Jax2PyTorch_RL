import importlib.util
from pathlib import Path
import tempfile
import unittest

from miniport.tasks import training_tasks, heldout_tasks, GATES

AVAILABLE = all(importlib.util.find_spec(name) for name in ("jax", "torch", "numpy"))
ORACLE = Path(__file__).parent / "fixtures" / "correct_port.py"


@unittest.skipUnless(AVAILABLE, "Install project numerical dependencies")
class NumericalTests(unittest.TestCase):
    def test_correct_ports_all_variants(self):
        from miniport.verify import verify
        for task in training_tasks() + heldout_tasks():
            with self.subTest(task=task.id):
                report = verify(task, ORACLE, (11, 23))
                self.assertTrue(report["passed"], report)
                self.assertEqual([g["name"] for g in report["gates"]], list(GATES))

    def test_stub_fails_topology(self):
        from miniport.verify import verify
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text("def build(config): raise NotImplementedError('stub')\n")
            report = verify(training_tasks()[0], path, (11,))
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["gates"]), 1)

    def test_corrupt_output_fails_end_to_end(self):
        from miniport.verify import verify
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(ORACLE.read_text().replace('"output": output}', '"output": output + 1}'))
            report = verify(training_tasks()[0], path, (11,))
            self.assertFalse(report["passed"])
            self.assertEqual(report["gates"][-1]["name"], "end_to_end")

    def test_bad_mapping_fails_parameters(self):
        from miniport.verify import verify
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(ORACLE.read_text().replace("return self.p\n", "return {k: v + 1 for k, v in self.p.items()}\n"))
            report = verify(training_tasks()[0], path, (11,))
            self.assertEqual(report["gates"][-1]["name"], "parameters")
            self.assertFalse(report["passed"])
