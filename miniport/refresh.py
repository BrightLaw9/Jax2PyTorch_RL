"""Prepare an auditable collection refresh while preserving committed training."""
import argparse
import json
from pathlib import Path
import uuid

from .checkpoints import checkpoint_path
from .resume import atomic_json, read_jsonl
from .train import read_config


def prepare(root, raw, tasks, templates, *, restart=False):
    root = Path(root)
    pending = root / 'collection-refresh.json'
    previous_attempt = 0
    if restart and pending.exists():
        previous = json.loads(pending.read_text())
        previous_attempt = previous.get('resample_attempt', 1)
        if type(previous_attempt) is not int or previous_attempt < 1:
            raise ValueError('Invalid collection refresh resample attempt')
        atomic_json(root / previous['archive'] / 'collection-refresh-plan.json', previous)
        pending.unlink()
    if pending.exists() and json.loads(pending.read_text()).get('status') != 'complete':
        plan = json.loads(pending.read_text())
        if plan['config'] != raw or plan['templates'] != templates:
            raise ValueError('Finish the pending refresh before requesting a different one')
    else:
        unknown = set(templates) - {t.template for t in tasks}
        if unknown:
            raise ValueError(f'Templates excluded from configuration: {unknown}')
        old = json.loads((root / 'config.json').read_text())
        a, b = json.loads(json.dumps(old)), json.loads(json.dumps(raw))
        a['experiment'].pop('task_limit', None); b['experiment'].pop('task_limit', None)
        old_updates, new_updates = a.pop('updates'), b.pop('updates')
        if new_updates < old_updates:
            raise ValueError('Collection refresh cannot reduce the update target')
        if a != b:
            raise ValueError('Collection refresh only permits changing task_limit and increasing updates')
        checkpoint = checkpoint_path(root)
        journal = read_jsonl(root / 'updates.jsonl')
        if checkpoint:
            manifest = json.loads((checkpoint / 'manifest.json').read_text())
            next_update = manifest.get('next_update')
            if next_update is None:
                if not journal or journal[-1]['completed_nonzero_updates'] != manifest['completed_updates']:
                    raise ValueError('Legacy checkpoint and journal disagree')
                next_update = journal[-1]['update'] + 1
        else:
            next_update = 0
        count = min(raw['value_checkpoints'], len(tasks)) if raw['experiment'].get('task_limit') else raw['value_checkpoints']
        indices = [i for i in range(count) if tasks[i % len(tasks)].template in templates]
        paths = []
        collection = root / 'value-collection'
        for i in indices:
            for pattern in (f'baseline-{i}', f'branch-{i}-*', f'checkpoint-{i}.*', f'checkpoint-{i}-*'):
                paths.extend(p.relative_to(root).as_posix() for p in collection.glob(pattern))
        paths.extend(p.relative_to(root).as_posix() for p in root.glob('update-*-group-*-member-*')
                     if int(p.name.split('-')[1]) >= next_update)
        if (root / 'action-value.json').exists():
            paths.append('action-value.json')
        resample_attempt = previous_attempt + 1
        plan = {'status': 'preparing', 'config': raw, 'templates': templates, 'indices': indices,
                'next_update': next_update, 'archive': 'refresh-archives/' + uuid.uuid4().hex,
                'paths': sorted(set(paths)), 'collection_policy': 'original frozen policy',
                'resample_attempt': resample_attempt,
                'resample_seed_offset': 10_000_000 * resample_attempt}
        atomic_json(pending, plan)
    archive = root / plan['archive']
    archive.mkdir(parents=True, exist_ok=True)
    rows_path = root / 'value-collection/value-data.jsonl'
    original = archive / 'value-data.jsonl'
    if not original.exists():
        original.write_text(rows_path.read_text() if rows_path.exists() else '')
    for relative in plan['paths']:
        source, destination = root / relative, archive / relative
        if source.exists():
            if destination.exists():
                raise ValueError(f'Both archive and active artifact exist: {relative}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    rows = [r for r in read_jsonl(original) if r['checkpoint'] not in plan['indices']]
    from .collection import write_rows
    write_rows(rows_path, {(r['checkpoint'], r['action_index']): r for r in rows})
    plan['status'] = 'pending'
    atomic_json(pending, plan)
    return plan


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--templates', nargs='+', required=True)
    parser.add_argument('--restart', action='store_true', help='Archive and replace a pending refresh')
    args = parser.parse_args()
    raw, _, tasks = read_config(args.config)
    root = Path(raw['output_dir'])
    with (root / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(prepare(root, raw, tasks, args.templates, restart=args.restart), indent=2))


if __name__ == '__main__':
    main()
