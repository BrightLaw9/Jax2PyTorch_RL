"""Readable log indexes with lossless, referenced per-action payloads."""
import json
import os
from pathlib import Path

from .resume import atomic_json, read_jsonl


def append_row(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def completion_entry(directory, raw):
    directory = Path(directory)
    steps = directory / 'steps'
    steps.mkdir(exist_ok=True)
    prefix = f"{raw['index']:03d}"
    attempt = len(list(steps.glob(prefix + '-completion-*.json')))
    base = f'steps/{prefix}-completion-{attempt}'
    payload = base + '.json'
    atomic_json(directory / payload, raw)
    (directory / (base + '-response.txt')).write_text(raw['completion'])
    (directory / (base + '-prompt.txt')).write_text(raw['prompt'])
    try:
        action = json.loads(raw['completion'])
        kind = action.get('type') if isinstance(action, dict) else 'invalid'
    except ValueError:
        kind = 'invalid'
    return {'schema_version': 2, 'index': raw['index'], 'action': kind,
            'output_tokens': raw.get('output_tokens', len(raw['output_ids'])),
            'token_limit_reached': raw.get('token_limit_reached', False),
            'response_file': base + '-response.txt', 'prompt_file': base + '-prompt.txt',
            'details_file': payload}


def write_completion(directory, raw):
    append_row(Path(directory) / 'completions.jsonl', completion_entry(directory, raw))


def record_entry(path, record):
    path = Path(path)
    steps = path.parent / 'steps'
    steps.mkdir(exist_ok=True)
    base = f"steps/{path.stem}-{record['index']:03d}"
    details = base + '-record.json'
    atomic_json(path.parent / details, record)
    action = record.get('policy_action') or {}
    links = {}
    if 'source' in action:
        links['source_file'] = base + '-candidate.py'
        (path.parent / links['source_file']).write_text(action['source'])
    if record.get('diff'):
        links['diff_file'] = base + '.diff'
        (path.parent / links['diff_file']).write_text(record['diff'])
    feedback = record['policy_observation']['feedback']
    return {'schema_version': 2, 'index': record['index'], 'task_id': record['task_id'],
            'action': record['checkpoint']['action'], 'status': feedback['status'],
            'gate': record['protected_summary']['current_gate'],
            'passed_gates': record['checkpoint']['passed_gates'],
            'remaining_actions': record['checkpoint']['remaining_actions'],
            'done': record['done'], 'training_reward': record['training_reward'],
            'trainable': record['trainable'], 'changed': bool(record.get('diff')),
            'policy_observation': record['policy_observation'],
            'details_file': details, **links}


def progress(path, records):
    path = Path(path)
    lines = ['# Rollout progress', '', 'Action indices start at 0. Reward is terminal; zero before stopping is normal.', '',
             '| Index | Action | Status / gate | Gates | Remaining | Changed | Reward | Files |',
             '|---:|---|---|---:|---:|---|---:|---|']
    notes = []
    for r in records:
        c = r['checkpoint']; feedback = r['policy_observation']['feedback']
        base = f"steps/{path.stem}-{r['index']:03d}"
        links = [f'[details]({base}-record.json)']
        if (r.get('policy_action') or {}).get('type') == 'edit':
            links.append(f'[code]({base}-candidate.py)')
        if r.get('diff'):
            links.append(f'[diff]({base}.diff)')
        reward = str(r['training_reward']) if r['done'] else 'pending'
        lines.append(f"| {r['index']} | {c['action']} | {feedback['status']} / {r['protected_summary']['current_gate']} | "
                     f"{len(c['passed_gates'])}/5 | {c['remaining_actions']} | {'yes' if r.get('diff') else 'no'} | {reward} | {' · '.join(links)} |")
        diagnostic = feedback.get('diagnostic') or feedback.get('static_scan_findings') or r['policy_observation'].get('action_error')
        if diagnostic:
            notes.extend([f"### Action {r['index']}: diagnostic", '', '```json', json.dumps(diagnostic, indent=2), '```', ''])
    name = 'progress.md' if path.name == 'protected.jsonl' else path.stem + '-progress.md'
    (path.parent / name).write_text('\n'.join(lines + ['', *notes]))


def write_record(path, record, records):
    append_row(path, record_entry(path, record))
    progress(path, [*records, record])


def save_trajectory(directory, result, records):
    directory = Path(directory)
    steps = directory / 'steps'
    steps.mkdir(exist_ok=True)
    concise = []
    for i, (step, record) in enumerate(zip(result['steps'], records)):
        file = f'steps/{i:03d}-trajectory.json'
        atomic_json(directory / file, step)
        concise.append({'index': i, 'action': record['checkpoint']['action'],
                        'passed_gates': record['checkpoint']['passed_gates'],
                        'done': record['done'], 'training_reward': record['training_reward'],
                        'output_tokens': len(step['output_ids']), 'details_file': file})
    atomic_json(directory / 'trajectory.json', {k:v for k,v in result.items() if k != 'steps'} |
                {'schema_version': 2, 'steps': concise})


def format_existing(directory):
    directory = Path(directory)
    records = read_jsonl(directory / 'protected.jsonl', repair=True)
    for name, render in [('completions.jsonl', lambda r: completion_entry(directory, r)),
                         ('protected.jsonl', lambda r: record_entry(directory / 'protected.jsonl', r))]:
        path = directory / name
        if not path.exists():
            continue
        rows = read_jsonl(path, repair=True)
        indexes = [json.loads(line) for line in path.read_text().splitlines()]
        if all(row.get('schema_version') == 2 for row in indexes):
            continue
        replacement = path.with_name(path.name + '.formatting')
        replacement.write_text(''.join(json.dumps(render(row), allow_nan=False) + '\n' for row in rows))
        replacement.replace(path)
    trajectory = directory / 'trajectory.json'
    if trajectory.exists():
        result = json.loads(trajectory.read_text())
        if result.get('schema_version') != 2:
            save_trajectory(directory, result, records)
    if records:
        progress(directory / 'protected.jsonl', records)
