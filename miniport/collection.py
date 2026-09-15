"""Distinct candidate sampling with durable attempts and bounded retries."""
from dataclasses import asdict
import ast
import json
import os
from pathlib import Path
import uuid

from .resume import atomic_json, read_jsonl


def action_key(completion):
    try:
        action = json.loads(completion.text)
        if not isinstance(action, dict):
            raise ValueError('Expected action object')
        kind = action.get('type')
        if kind == 'edit' and isinstance(action.get('source'), str):
            try:
                source = ast.dump(ast.parse(action['source']), include_attributes=False)
            except SyntaxError:
                source = action['source'].strip()
            return ('edit', source)
        if kind in ('test', 'stop'):
            return (kind,)
        return ('invalid', completion.text.strip())
    except (ValueError, TypeError):
        return ('invalid', completion.text.strip())


def choose_actions(policy, task, state, first, seed, retries, selection, recovered=None):
    from .policy import Completion
    if not isinstance(retries, int) or not 0 <= retries <= 20:
        raise ValueError('candidate_resamples must be an integer from 0 to 20')
    if recovered is not None:
        if action_key(first) == action_key(recovered):
            print('Legacy second action duplicates first; keeping existing evidence without new duplicate continuations', flush=True)
            return [first], [{'legacy_duplicate': True}]
        return [first, recovered], [{'recovered_existing_action': True}]
    path = Path(selection).with_suffix('.attempts.jsonl')
    attempts = read_jsonl(path, repair=True)
    for index in range(retries + 1):
        if index < len(attempts):
            candidate = Completion(**attempts[index]['completion'])
        else:
            candidate = policy.generate(task, state.source, state.history, state.remaining, seed + index)
            row = {'attempt': index, 'seed': seed + index, 'completion': asdict(candidate),
                   'duplicate': action_key(candidate) == action_key(first)}
            with path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            attempts.append(row)
        if action_key(candidate) != action_key(first):
            return [first, candidate], attempts
    print(f'No distinct second action after {retries + 1} attempts; skipping duplicate continuations', flush=True)
    return [first], attempts


def write_rows(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        for key in sorted(rows):
            stream.write(json.dumps(rows[key]) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def replace_legacy_duplicate(root, checkpoint, task, state, rows, backend):
    """Archive old action-1 evidence before sampling a different action into its slot."""
    import shutil
    import tempfile
    from .rollouts import Snapshot
    from .sealed import verify_submission
    from .observations import visible_feedback
    root = Path(root)
    migration = root / f'checkpoint-{checkpoint}-resample-state.json'
    archive = root / 'superseded' / f'checkpoint-{checkpoint}-action-1'
    archive.mkdir(parents=True, exist_ok=True)
    if not migration.exists():
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'candidate.py').write_text(state.source)
            feedback = visible_feedback(verify_submission(task, directory, backend=backend))
        state = Snapshot(state.source, state.history + [{'role': 'user', 'content': json.dumps({
            'feedback': feedback, 'remaining_actions': state.remaining, 'done': False}, sort_keys=True)}], state.remaining)
        atomic_json(migration, {'snapshot': asdict(state), 'reason': 'Legacy action 1 duplicates action 0',
                               'superseded_value_row': rows.get((checkpoint, 1))})
    else:
        state = Snapshot(**json.loads(migration.read_text())['snapshot'])
    for directory in sorted(root.glob(f'branch-{checkpoint}-1-*')):
        destination = archive / directory.name
        if destination.exists():
            raise ValueError(f'Both original and archived continuation exist: {directory}')
        shutil.move(str(directory), str(destination))
    rows.pop((checkpoint, 1), None)
    write_rows(root / 'value-data.jsonl', rows)
    print(f'Archived duplicate action 1 at checkpoint {checkpoint}; resampling with refreshed feedback', flush=True)
    return state
