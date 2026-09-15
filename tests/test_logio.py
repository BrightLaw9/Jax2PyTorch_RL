import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from miniport.experiment import ExperimentConfig
from miniport.logio import format_existing
from miniport.policy import Completion
from miniport.resume import read_jsonl, completed_rollout
from miniport.rollouts import Snapshot, rollout
from miniport.sealed import summary
from miniport.tasks import GATES, training_tasks, heldout_tasks, limit_tasks
from miniport.train import read_config


class ReadableLogTests(unittest.TestCase):
    def test_lossless_indexes_and_code_files(self):
        class Policy:
            config = ExperimentConfig()
            def generate(self, *args):
                return Completion('exact prompt', json.dumps({'type':'stop'}), [4,5], [6,7])
        with tempfile.TemporaryDirectory() as d, patch('miniport.trajectory.verify_submission', return_value=summary(GATES)):
            root=Path(d)/'run'
            result=rollout(Policy(),training_tasks()[0],Snapshot('import torch\n',remaining=2),root,object(),42)
            index=json.loads((root/'completions.jsonl').read_text())
            self.assertNotIn('prompt_ids',index)
            self.assertNotIn('completion',index)
            self.assertEqual(read_jsonl(root/'completions.jsonl')[0]['prompt_ids'],[4,5])
            self.assertEqual(read_jsonl(root/'protected.jsonl'),json.loads(json.dumps(result['records'])))
            trajectory=json.loads((root/'trajectory.json').read_text())
            self.assertNotIn('prompt',trajectory['steps'][0])
            loaded=completed_rollout(root)
            self.assertEqual(loaded['steps'][0]['completion'].text,result['steps'][0]['completion'].text)
            self.assertTrue((root/'progress.md').exists())
            # Migrate legacy indexes without dropping fields.
            raw=read_jsonl(root/'completions.jsonl')[0]
            (root/'completions.jsonl').write_text(json.dumps(raw)+'\n')
            record=json.loads(json.dumps(result['records'][0]))
            (root/'protected.jsonl').write_text(json.dumps(record)+'\n')
            format_existing(root)
            self.assertEqual(read_jsonl(root/'completions.jsonl')[0],raw)
            self.assertEqual(read_jsonl(root/'protected.jsonl')[0],record)


class TaskLimitTests(unittest.TestCase):
    def test_training_focus_and_evaluation_scope(self):
        raw,config,tasks=read_config('configs/train.json')
        self.assertEqual(config.task_limit,4)
        self.assertEqual([t.template for t in tasks],['mlp','layernorm','attention'])
        self.assertEqual([t.template for t in limit_tasks(heldout_tasks(),config.task_limit)],['mlp','cnn','layernorm','attention'])
        self.assertEqual(len(limit_tasks(heldout_tasks(),None)),6)
        with self.assertRaises(ValueError):
            ExperimentConfig(task_limit=0).validate()
