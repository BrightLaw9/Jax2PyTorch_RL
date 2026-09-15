import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from miniport.collection import action_key, choose_actions
from miniport.experiment import ExperimentConfig
from miniport.policy import Completion
from miniport.resume import read_jsonl
from miniport.rollouts import Snapshot, rollout
from miniport.sealed import summary
from miniport.tasks import GATES, training_tasks
from miniport.train import collect_value_data, validate_resume_config, value_checkpoint_seed


def completion(action):
    return Completion('prompt', json.dumps(action), [1], [2])


class CandidateTests(unittest.TestCase):
    def test_formatting_and_comments_are_duplicates(self):
        a = completion({'type': 'edit', 'source': 'x=1\n'})
        b = completion({'type': 'edit', 'source': '# same\nx = 1\n'})
        self.assertEqual(action_key(a), action_key(b))

    def test_duplicate_resampling_and_durable_attempts(self):
        first = completion({'type': 'edit', 'source': 'x=1'})
        second = completion({'type': 'edit', 'source': 'x=2'})
        class Policy:
            calls = 0
            def generate(self, *args):
                self.calls += 1
                return first if self.calls == 1 else second
        with tempfile.TemporaryDirectory() as d:
            p = Policy()
            actions, attempts = choose_actions(p, training_tasks()[0], Snapshot(''), first, 42, 3, Path(d)/'choice.json')
            self.assertEqual(len(actions), 2)
            self.assertEqual(len(attempts), 2)
            self.assertTrue(attempts[0]['duplicate'])
            again, _ = choose_actions(p, training_tasks()[0], Snapshot(''), first, 42, 3, Path(d)/'choice.json')
            self.assertEqual(p.calls, 2)
            self.assertEqual(again, actions)

    def test_all_duplicates_have_bounded_cost(self):
        first = completion({'type': 'stop'})
        class Policy:
            calls = 0
            def generate(self, *args):
                self.calls += 1
                return first
        with tempfile.TemporaryDirectory() as d:
            p = Policy()
            actions, _ = choose_actions(p, training_tasks()[0], Snapshot(''), first, 42, 2, Path(d)/'choice.json')
            self.assertEqual(len(actions), 1)
            self.assertEqual(p.calls, 3)


class ResumeTests(unittest.TestCase):
    def test_resume_can_only_increase_update_target(self):
        old = {'updates': 4, 'task_ids': ['mlp-0', 'cnn-0', 'layernorm-0', 'attention-0'],
               'learning_rate': 1e-5, 'experiment': {'task_limit': 3}}
        new = {'updates': 8, 'task_ids': ['mlp-0', 'layernorm-0', 'attention-0'],
               'learning_rate': 1e-5, 'experiment': {'task_limit': 3}}
        validate_resume_config(old, new, refresh_pending=False)
        with self.assertRaisesRegex(ValueError, 'cannot reduce'):
            validate_resume_config(new, old, refresh_pending=False)
        changed = json.loads(json.dumps(new)); changed['learning_rate'] = 2e-5
        with self.assertRaisesRegex(ValueError, 'original run configuration'):
            validate_resume_config(old, changed, refresh_pending=False)
        reordered = json.loads(json.dumps(new)); reordered['task_ids'] = ['attention-0', 'mlp-0']
        with self.assertRaisesRegex(ValueError, 'order-preserving subset'):
            validate_resume_config(old, reordered, refresh_pending=False)

    def test_refresh_retry_uses_disjoint_reproducible_seed_schedule(self):
        refresh = {'indices': [1], 'resample_seed_offset': 20_000_000}
        self.assertEqual(value_checkpoint_seed(42, 0, refresh), 1_000_042)
        self.assertEqual(value_checkpoint_seed(42, 1, refresh), 21_010_042)
        self.assertEqual(value_checkpoint_seed(42, 1, refresh), 21_010_042)

    def test_partial_rollout_restores_committed_source_and_budget(self):
        class Policy:
            config = ExperimentConfig()
            calls = 0
            resumed = False
            def generate(self, task, source, history, remaining, seed):
                self.calls += 1
                if self.resumed:
                    self.assertions = (source, remaining, seed, history)
                    return completion({'type': 'stop'})
                if self.calls == 1:
                    return completion({'type': 'edit', 'source': 'import torch\n# committed'})
                raise RuntimeError('interrupted')
        with tempfile.TemporaryDirectory() as d, patch('miniport.trajectory.verify_submission', return_value=summary(GATES)), patch('miniport.sealed.verify_submission', return_value=summary(GATES)):
            p = Policy(); root = Path(d)/'run'; initial = Snapshot('import torch\n', remaining=3)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                rollout(p, training_tasks()[0], initial, root, object(), 42)
            (root/'submission/candidate.py').write_text('uncommitted write')
            p.resumed = True
            result = rollout(p, training_tasks()[0], initial, root, object(), 42, resume=True)
            self.assertEqual(p.assertions[:3], ('import torch\n# committed', 2, 43))
            self.assertEqual(len(result['records']), 2)
            self.assertEqual(result['reward'], 1)
            count = p.calls
            again = rollout(p, training_tasks()[0], initial, root, object(), 42, resume=True)
            self.assertEqual(p.calls, count)
            self.assertEqual(again['reward'], 1)
            # Recover a terminal action committed before trajectory finalization.
            (root/'trajectory.json').unlink()
            terminal = rollout(p, training_tasks()[0], initial, root, object(), 42, resume=True)
            self.assertEqual(p.calls, count)
            self.assertEqual(terminal['reward'], 1)

    def test_collection_skips_completed_checkpoint_action_and_continuation(self):
        class Policy:
            config = ExperimentConfig()
            fail = True
            calls = []
            def generate(self, task, source, history, remaining, seed):
                self.calls.append(seed)
                if seed == 1011043 and self.fail:
                    raise RuntimeError('interrupted continuation')
                if seed in (1000042, 1010042):
                    return completion({'type': 'edit', 'source': 'import torch\n# edited'})
                if seed in (1009042, 1019042):
                    return completion({'type': 'test'})
                return completion({'type': 'stop'})
        raw = {'value_checkpoints': 2, 'continuations_per_action': 2}
        with tempfile.TemporaryDirectory() as d, patch('miniport.trajectory.verify_submission', return_value=summary(GATES)), patch('miniport.sealed.verify_submission', return_value=summary(GATES)):
            root=Path(d); p=Policy()
            with self.assertRaisesRegex(RuntimeError, 'interrupted continuation'):
                collect_value_data(p, training_tasks()[:2], raw, root, object())
            first=root/'branch-0-0-0/trajectory.json'
            digest=hashlib.sha256(first.read_bytes()).hexdigest()
            before=len(p.calls); p.fail=False
            rows=collect_value_data(p, training_tasks()[:2], raw, root, object(), resume=True)
            self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(),digest)
            self.assertEqual(len(rows),4)
            self.assertNotIn(1000042,p.calls[before:])
            self.assertNotIn(1010042,p.calls[before:])
            before=len(p.calls)
            again=collect_value_data(p,training_tasks()[:2],raw,root,object(),resume=True)
            self.assertEqual(rows,again)
            self.assertEqual(before,len(p.calls))
            self.assertEqual(len(read_jsonl(root/'value-data.jsonl')),4)

    def test_task_limit_does_not_cap_value_checkpoint_count(self):
        class Policy:
            config = ExperimentConfig(task_limit=1)
            def generate(self, task, source, history, remaining, seed):
                return completion({'type': 'stop'})
        raw = {'value_checkpoints': 3, 'continuations_per_action': 2}
        with tempfile.TemporaryDirectory() as d, patch('miniport.trajectory.verify_submission', return_value=summary(GATES)), patch('miniport.sealed.verify_submission', return_value=summary(GATES)):
            # Stop-only baselines are valid but supply no branch state; the three
            # baseline directories still prove that collection cycled three times.
            collect_value_data(Policy(), training_tasks()[:1], raw, Path(d), object())
            self.assertEqual(len(list(Path(d).glob('baseline-*/trajectory.json'))), 3)

    def test_truncated_tail_is_archived_and_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'records.jsonl'; p.write_text('{"index":0}\n{"index":')
            self.assertEqual(read_jsonl(p,repair=True),[{'index':0}])
            self.assertEqual(len(list(Path(d).glob('*.interrupted-*'))),1)
