import unittest

from miniport.experiment import ExperimentConfig
from miniport.policy import render_prompt
from miniport.tasks import training_tasks
from miniport.requirements import task_text


class PromptTests(unittest.TestCase):
    def test_attention_contract_contains_complete_shape_ledger(self):
        task = next(task for task in training_tasks() if task.template == "attention")
        text = task_text(task)
        self.assertIn("[batch,sequence,heads,head_dim]", text)
        self.assertIn("[batch,heads,sequence,sequence]", text)
        self.assertIn("Do not multiply attention while Q/K/V remain", text)

    def test_pass_requests_only_stop(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "\n".join(m["content"] for m in messages)
            def encode(self, text, **kwargs):
                return [0]
        tokenizer = Tokenizer()
        render_prompt(tokenizer, training_tasks()[0], "import torch\n",
                      [{"role": "user", "content": '{"feedback":{"status":"pass"}}'}],
                      12, ExperimentConfig())
        current = tokenizer.messages[-1]["content"]
        self.assertTrue(current.endswith('Return exactly {"type":"stop"}. Do not edit or test again.'))
        self.assertNotIn("For an edit", current)

    def test_latest_feedback_survives_history_trimming(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "\n".join(m["content"] for m in messages)

            def encode(self, text, **kwargs):
                return [0] * (len(text) // 4)

        tokenizer = Tokenizer()
        feedback = '{"action_error":"Unterminated string; output token limit reached"}'
        history = [{"role": "assistant", "content": "x" * 20000},
                   {"role": "user", "content": feedback}]
        prompt, _ = render_prompt(tokenizer, training_tasks()[0], "import torch\n", history,
                                  17, ExperimentConfig())
        self.assertEqual(len(tokenizer.messages), 2)
        self.assertIn(feedback, tokenizer.messages[-1]["content"])
        self.assertGreater(prompt.index("Your next action is for mlp-0 ONLY"),
                           prompt.index("END OF CURRENT FILE"))
