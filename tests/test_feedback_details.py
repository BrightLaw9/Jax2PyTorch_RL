import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from miniport.controller import Controller
from miniport.observations import visible_feedback
from miniport.requirements import task_text
from miniport.sealed import grade, summary
from miniport.tasks import training_tasks


class FeedbackDetailTests(unittest.TestCase):
    def test_layer_and_output_numerical_summaries(self):
        target = np.array([-2., 1., 3., 0.], dtype=np.float32)
        expected = [{'parameters': {'w': target}, 'layer': target, 'output': target}]
        payload = {'shape': [4], 'dtype': 'torch.float32', 'values': target.tolist()}
        base = {'topology': True, 'parameters': {'w': copy.deepcopy(payload)}, 'layer': copy.deepcopy(payload),
                'output': copy.deepcopy(payload), 'repeat': copy.deepcopy(payload)}
        for field in ('layer', 'output'):
            reply = copy.deepcopy(base)
            reply[field]['values'] = [0., 1., 3., 0.]
            feedback = visible_feedback(grade(training_tasks()[0], {'cases': [reply]}, expected))
            numerical = feedback['diagnostic']['numerical']
            self.assertEqual(numerical['tensor'], 'mlp.' + field)
            self.assertEqual(numerical['max_abs_error'], 2.)
            self.assertEqual(numerical['mean_abs_error'], .5)
            self.assertEqual(numerical['fraction_outside_tolerance'], .25)
            self.assertEqual(numerical['actual_range'], {'min': 0., 'max': 3.})
            self.assertEqual(numerical['expected_range'], {'min': -2., 'max': 3.})
            if field == 'layer':
                self.assertEqual(numerical['inference']['code'], 'possible_early_sign_clipping')
                self.assertIn('nonnegative', numerical['inference']['message'])
            else:
                self.assertNotIn('inference', numerical)
            self.assertNotIn('values', numerical)

    def test_comment_only_edit_gets_explicit_feedback(self):
        with tempfile.TemporaryDirectory() as d, patch('miniport.trajectory.verify_submission', return_value=summary(error_type='parity_mismatch', location='cnn.layer')):
            root = Path(d)
            submission = root / 'submission'; submission.mkdir()
            (submission / 'candidate.py').write_text('x = 1\n')
            controller = Controller(training_tasks()[0], submission, root/'protected.jsonl')
            feedback = controller.step({'type': 'edit', 'source': '# changed comment\nx=1\n'})
            self.assertEqual(feedback['edit_feedback']['code'], 'no_executable_change')
            self.assertIn('cnn.layer', feedback['edit_feedback']['message'])
            changed = controller.step({'type': 'edit', 'source': 'x=2\n'})
            self.assertNotIn('edit_feedback', changed)

    def test_cnn_contract_names_activation_and_layout(self):
        task = next(t for t in training_tasks() if t.id == 'cnn-0')
        text = task_text(task)
        self.assertIn('BEFORE ReLU', text)
        self.assertIn('(NHWC)', text)
        self.assertIn('(OIHW)', text)
        self.assertIn('mean(dim=(2,3))', text)
        self.assertIn('dimension 1 is the output-channel axis', text)
        self.assertIn('Returning ReLU(layer) as layer is incorrect', text)
