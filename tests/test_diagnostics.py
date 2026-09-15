import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from miniport.diagnostics import runtime_diagnostic
from miniport.observations import visible_feedback
from miniport.sealed import SubprocessBackend, grade, verify_submission
from miniport.tasks import training_tasks


class DiagnosticTests(unittest.TestCase):
    def test_runtime_attribute_and_source_line_reach_feedback(self):
        source = '''import torch
class Model:
    def topology(self): return "mlp"
    def load_parameters(self, parameters):
        self.p = torch.tensor([1.]).astype("float32")
    def export_parameters(self): return {}
    def run(self, x, draws): return {}
def build(config): return Model()
'''
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'candidate.py').write_text(source)
            report = verify_submission(training_tasks()[0], d, seeds=(11,), backend=SubprocessBackend())
        feedback=visible_feedback(report)
        self.assertEqual(report['passed_gates'],['topology'])
        self.assertEqual(feedback['diagnostic']['attribute'],'astype')
        self.assertEqual(feedback['diagnostic']['stage'],'load_parameters')
        self.assertEqual(feedback['diagnostic']['line'],5)
        self.assertNotIn('values',json.dumps(feedback))

    def test_runtime_exception_message_is_preserved(self):
        message = "original candidate error"
        source = f'def build(config):\n    raise RuntimeError({message!r})\n'
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'candidate.py').write_text(source)
            report = verify_submission(training_tasks()[0], d, seeds=(11,), backend=SubprocessBackend())
        self.assertFalse(report['passed'])
        self.assertEqual(report['diagnostic']['message'], message)
        self.assertEqual(report['diagnostic']['exception_type'], 'RuntimeError')

    def test_transpose_error_is_distinct_from_output_type_validation(self):
        prefix = '''import torch
class Model:
    def topology(self): return "mlp"
    def load_parameters(self, parameters): pass
    def export_parameters(self):
        return {"w1": EXPRESSION}
    def run(self, x, draws): return {}
def build(config): return Model()
'''
        cases = [('torch.zeros(2, 2, 2, 2).transpose(3, 2, 0, 1)', 'type_error'),
                 ('torch.zeros(2).numpy()', 'output_type_mismatch')]
        for expression, code in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as d:
                (Path(d)/'candidate.py').write_text(prefix.replace('EXPRESSION', expression))
                report = verify_submission(training_tasks()[0], d, seeds=(11,), backend=SubprocessBackend())
                detail = visible_feedback(report)['diagnostic']
                self.assertEqual(detail['code'], code)
                self.assertEqual(detail['stage'], 'export_parameters')
                if code == 'type_error':
                    self.assertIn('transpose()', detail['message'])
                    self.assertNotIn('return torch.Tensor', detail['message'])
                    self.assertEqual(detail['origin'], 'candidate_runtime')
                else:
                    self.assertEqual(detail['origin'], 'output_validation')
                    self.assertEqual(detail['actual_type'], 'ndarray')
                    self.assertEqual(detail['field'], 'export_parameters.w1')

    def test_structural_and_numerical_diagnostics_do_not_reveal_values(self):
        target=np.array([0.123456789,0.987654321],dtype=np.float32)
        expected=[{'parameters':{'w':target},'layer':target,'output':target}]
        payload={'shape':[2],'dtype':'torch.float32','values':target.tolist()}
        reply={'topology':True,'parameters':{'w':copy.deepcopy(payload)},
               'layer':copy.deepcopy(payload),'output':copy.deepcopy(payload),'repeat':copy.deepcopy(payload)}
        cases=[('shape_mismatch',lambda r:r['layer'].update(shape=[1])),
               ('dtype_mismatch',lambda r:r['layer'].update(dtype='torch.float64')),
               ('value_mismatch',lambda r:r['layer'].update(values=[0.,0.])),
               ('nondeterministic_output',lambda r:r['repeat'].update(values=[0.,0.]))]
        for code, mutate in cases:
            with self.subTest(code=code):
                r=copy.deepcopy(reply);mutate(r)
                report=grade(training_tasks()[0],{'cases':[r]},expected)
                self.assertFalse(report['passed'])
                self.assertEqual(report['diagnostic']['code'],code)
                feedback=visible_feedback(report)
                self.assertNotIn('values',feedback['diagnostic'])
                self.assertNotIn(target.tolist(),list(feedback['diagnostic'].values()))
