import json
from pathlib import Path
import tempfile
import unittest

from miniport.refresh import prepare
from miniport.resume import atomic_json, read_jsonl
from miniport.tasks import training_tasks, limit_tasks
from miniport.train import read_config


class RefreshTests(unittest.TestCase):
    def test_refresh_archives_selected_collection_and_uncommitted_training(self):
        raw, _, _ = read_config('configs/train.json')
        raw['experiment']['task_limit'] = 4
        tasks = limit_tasks(training_tasks(), 4)
        # Use one canonical variant per family, as the training configuration does.
        tasks = [next(t for t in tasks if t.id == name) for name in raw['task_ids'][:4]]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = json.loads(json.dumps(raw)); old['experiment']['task_limit'] = 3
            atomic_json(root / 'config.json', old)
            atomic_json(root / 'checkpoint/manifest.json', {'completed_updates': 2, 'next_update': 2})
            (root / 'updates.jsonl').write_text('{"update":1,"completed_nonzero_updates":2}\n')
            collection = root / 'value-collection'; collection.mkdir()
            for i in range(3):
                (collection / f'baseline-{i}').mkdir()
                (collection / f'branch-{i}-0-0').mkdir()
                atomic_json(collection / f'checkpoint-{i}.json', {'checkpoint': i})
            (collection / 'value-data.jsonl').write_text(''.join(json.dumps({'checkpoint':i,'action_index':0})+'\n' for i in range(3)))
            (root / 'update-1-group-0-member-0').mkdir()
            (root / 'update-2-group-0-member-0').mkdir()
            (root / 'action-value.json').write_text('{}')
            plan = prepare(root, raw, tasks, ['layernorm'])
            self.assertEqual(plan['indices'], [1])
            self.assertEqual(plan['next_update'], 2)
            self.assertEqual(plan['resample_attempt'], 1)
            self.assertEqual(plan['resample_seed_offset'], 10_000_000)
            self.assertTrue((collection / 'baseline-0').exists())
            self.assertTrue((collection / 'baseline-2').exists())
            self.assertFalse((collection / 'baseline-1').exists())
            self.assertFalse((root / 'update-2-group-0-member-0').exists())
            self.assertTrue((root / 'update-1-group-0-member-0').exists())
            self.assertTrue((root / plan['archive'] / 'value-collection/baseline-1').exists())
            self.assertEqual([r['checkpoint'] for r in read_jsonl(collection/'value-data.jsonl')], [0,2])
            self.assertEqual(prepare(root, raw, tasks, ['layernorm']), plan)
            (collection / 'baseline-1').mkdir()
            restarted = prepare(root, raw, tasks, ['layernorm'], restart=True)
            self.assertNotEqual(restarted['archive'], plan['archive'])
            self.assertEqual(restarted['resample_attempt'], 2)
            self.assertEqual(restarted['resample_seed_offset'], 20_000_000)
            self.assertTrue((root / plan['archive'] / 'collection-refresh-plan.json').exists())
            self.assertTrue((root / restarted['archive'] / 'value-collection/baseline-1').exists())
