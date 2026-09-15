import unittest

import numpy as np

from miniport.integrity import scan
from miniport.reference import fixture, reference, mapped_parameters
from miniport.starter import jax_source, starter_source
from miniport.tasks import training_tasks, heldout_tasks
from miniport.train import starter


class StarterTests(unittest.TestCase):
    def test_public_source_matches_verifier_for_every_template(self):
        for task in training_tasks() + heldout_tasks():
            with self.subTest(template=task.template):
                namespace = {}
                exec(jax_source(task), namespace)
                parameters, inputs, draws = fixture(task, 19)
                expected = reference(task, parameters, inputs, draws)
                model = namespace["build"](task.to_dict())
                self.assertEqual(model.topology(), task.template)
                model.load_parameters(parameters)
                for key, value in mapped_parameters(task, parameters).items():
                    np.testing.assert_array_equal(model.export_parameters()[key], value)
                actual = model.run(inputs, draws)
                for key in ("layer", "output"):
                    np.testing.assert_array_equal(actual[key], expected[key])
                if task.template == "rope_cache":
                    for _ in range(2):
                        np.testing.assert_allclose(model.run_cached(inputs, draws), expected["output"],
                                                   atol=2e-5, rtol=2e-4)

    def test_starters_are_admitted_and_exclude_fixtures(self):
        for task in training_tasks() + heldout_tasks():
            with self.subTest(task=task.id):
                source = starter_source(task)
                self.assertTrue(scan(source)["passed"])
                self.assertIn("# def build(config):", source)
                self.assertNotIn("def fixture", source)
                self.assertNotIn("PRNGKey", source)
                self.assertNotIn("random.split", source)
                self.assertEqual(starter(task, {}), source)

    def test_custom_source_is_preserved(self):
        source = "import torch\nCUSTOM = 1\n"
        self.assertTrue(starter_source(training_tasks()[0], source).endswith(source))

    def test_stub_contains_full_oop_reference(self):
        from pathlib import Path
        source = jax_source(None)
        stub = Path("miniport/stub.py.txt").read_text()
        commented = "\n".join("# " + line if line else "#" for line in source.splitlines())
        self.assertIn(commented, stub)
        self.assertNotIn('name ==', source)
        self.assertEqual(starter_source(training_tasks()[0]), stub)
