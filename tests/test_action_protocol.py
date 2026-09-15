import json
import unittest
import torch
from miniport.action_protocol import complete_object, parse_action, StopAfterAction


class ActionProtocolTests(unittest.TestCase):
    def test_boundary_handles_python_braces_and_escapes(self):
        text = json.dumps({'type': 'edit', 'source': 'x = {"a": "}\\n"}\n'})
        for end in range(len(text)):
            self.assertFalse(complete_object(text[:end]))
        self.assertTrue(complete_object(text))
        self.assertEqual(parse_action(text)['type'], 'edit')
        self.assertTrue(complete_object(text + '\n{"type":"test"}'))
        with self.assertRaisesRegex(ValueError, 'multiple actions.*Nothing was applied'):
            parse_action(text + '\n{"type":"test"}')
        with self.assertRaisesRegex(ValueError, 'Extra content'):
            parse_action(text + '\nHere is my explanation')

    def test_generation_stops_before_second_action(self):
        class Tokenizer:
            def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)
        stop = StopAfterAction(Tokenizer(), 3)
        output = '{"type":"stop"}\n{"type":"test"}'
        emitted = ''
        for char in output:
            emitted += char
            if stop(torch.tensor([[1, 2, 3] + list(map(ord, emitted))]), None):
                break
        self.assertEqual(emitted, '{"type":"stop"}')
