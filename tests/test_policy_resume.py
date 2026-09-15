import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from miniport.checkpoints import save_checkpoint, restore_training_state
from miniport.experiment import ExperimentConfig
from miniport.policy import Completion
from miniport.rollouts import Snapshot, rollout
from miniport.sealed import summary
from miniport.tasks import GATES, training_tasks


class SavedModel(torch.nn.Linear):
    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), Path(path) / 'weights.pt')


class Tokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir()


class PolicyResumeTests(unittest.TestCase):
    def test_checkpoint_restores_optimizer_and_commits_metrics_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'requirements-resolved.txt').write_text('')
            model = SavedModel(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step(); optimizer.zero_grad()
            policy = SimpleNamespace(model=model, tokenizer=Tokenizer())
            metric = {'update': 0, 'completed_nonzero_updates': 1}
            save_checkpoint(policy, optimizer, root, {}, 1, next_update=1, scaler=scaler, metrics=metric)
            saved = root / 'checkpoint'
            model2 = SavedModel(2, 1)
            model2.load_state_dict(torch.load(saved / 'weights.pt', weights_only=True))
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=.5)
            with patch('torch.cuda.set_rng_state_all'):
                self.assertEqual(restore_training_state(saved, optimizer2, scaler, root), (1, 1))
                restore_training_state(saved, optimizer2, scaler, root)
            self.assertEqual(len((root / 'updates.jsonl').read_text().splitlines()), 1)
            for m, o in [(model, optimizer), (model2, optimizer2)]:
                m(torch.ones(1, 2)).sum().backward(); o.step()
            for a, b in zip(model.parameters(), model2.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            old = saved.resolve()
            save_checkpoint(policy, optimizer, root, {}, 2, next_update=2, scaler=scaler)
            self.assertTrue(old.exists())
            self.assertNotEqual(old, saved.resolve())
            # A failed save must leave the last committed pointer intact.
            latest = saved.resolve()
            with patch.object(model, 'save_pretrained', side_effect=RuntimeError('disk failure')):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(policy, optimizer, root, {}, 3, next_update=3, scaler=scaler)
            self.assertEqual(saved.resolve(), latest)

    def test_partial_policy_rollout_preserves_action_credit(self):
        class Policy:
            config = ExperimentConfig()
            interrupted = True
            calls = []
            def generate(self, task, source, history, remaining, seed):
                self.calls.append(seed)
                if seed == 43 and self.interrupted:
                    raise RuntimeError('interrupt')
                action = {'type': 'edit', 'source': 'import torch\n# edited'} if seed == 42 else {'type': 'stop'}
                return Completion('prompt', json.dumps(action), [1], [2])
        value = SimpleNamespace(predict=lambda prompt, action: .8 if 'edit' in action else .2)
        with tempfile.TemporaryDirectory() as d, patch('miniport.trajectory.verify_submission', return_value=summary(GATES)), patch('miniport.sealed.verify_submission', return_value=summary(GATES)):
            directory = Path(d) / 'run'
            policy = Policy()
            initial = Snapshot('import torch\n', remaining=3)
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                rollout(policy, training_tasks()[0], initial, directory, object(), 42, value_model=value)
            policy.interrupted = False
            before = len(policy.calls)
            result = rollout(policy, training_tasks()[0], initial, directory, object(), 42, value_model=value, resume=True)
            self.assertEqual(policy.calls[before:], [43, 100043])
            self.assertAlmostEqual(result['steps'][0]['value_advantage'], .3)
            self.assertEqual(result['proposal_tokens'], 2)
            before = len(policy.calls)
            again = rollout(policy, training_tasks()[0], initial, directory, object(), 42, value_model=value, resume=True)
            self.assertEqual(before, len(policy.calls))
            self.assertEqual(again['steps'][0]['value_advantage'], result['steps'][0]['value_advantage'])
            # Legacy partial logs lack sidecar credits: reconstruct from the same state/seed.
            (directory / 'trajectory.json').unlink()
            (directory / 'steps/000-value.json').unlink()
            recovered = rollout(policy, training_tasks()[0], initial, directory, object(), 42, value_model=value, resume=True)
            self.assertAlmostEqual(recovered['steps'][0]['value_advantage'], .3)
