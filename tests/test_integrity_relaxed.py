import unittest
from miniport.integrity import scan


class RelaxedIntegrityTests(unittest.TestCase):
    def test_local_pytorch_state_and_python_utilities_allowed(self):
        source = '''import torch
from collections import OrderedDict
from functools import partial
conv = torch.nn.Conv2d(3, 8, 3)
conv.weight.data = torch.zeros_like(conv.weight)
conv.bias.data = torch.zeros_like(conv.bias)
x = getattr(conv, "weight")
y = type(x)
z = vars(conv)
torch.manual_seed(42)
class Model:
    def save(self): pass
m = Model()
m.save()
'''
        self.assertTrue(scan(source)['passed'])
        self.assertTrue(scan('x = ' + repr(list(range(200))))['passed'])

    def test_remaining_rejections_explain_exact_operation(self):
        for source, text in [('import torch\ntorch.save(x, "file")', 'torch.save'),
                             ('from torch import save', 'torch.save'),
                             ('import torch as t\nt.load("file")', 'torch.load'),
                             ('import torch\ntorch.nn.Conv2d = object', 'torch.nn.Conv2d'),
                             ('import os', 'os'), ('open("file")', 'open')]:
            with self.subTest(source=source):
                report = scan(source)
                self.assertFalse(report['passed'])
                self.assertTrue(any(text in f['message'] for f in report['findings']))
