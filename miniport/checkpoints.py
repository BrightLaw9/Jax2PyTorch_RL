"""Committed policy snapshots and optimizer recovery."""
import json
import os
from pathlib import Path
import random
import shutil
import uuid


def checkpoint_path(root):
    root = Path(root)
    path = root / 'checkpoint'
    if path.exists():
        return path
    legacy = root / 'checkpoints' / 'legacy'
    return legacy if legacy.exists() else None


def save_checkpoint(policy, optimizer, root, metadata, completed_updates, *, next_update, scaler, metrics=None):
    import torch
    root = Path(root)
    generations = root / 'checkpoints'
    generations.mkdir(exist_ok=True)
    path = generations / f'update-{next_update:04d}-{uuid.uuid4().hex}'
    path.mkdir()
    policy.model.save_pretrained(path, safe_serialization=True)
    policy.tokenizer.save_pretrained(path / 'tokenizer')
    torch.save({'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'completed_updates': completed_updates,
                'next_update': next_update, 'metrics': metrics}, path / 'trainer_state.pt')
    (path / 'manifest.json').write_text(json.dumps(dict(metadata, completed_updates=completed_updates,
                                                      next_update=next_update), indent=2))
    shutil.copyfile(root / 'requirements-resolved.txt', path / 'requirements-resolved.txt')
    for file in path.rglob('*'):
        if file.is_file():
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
    pointer = root / 'checkpoint'
    if pointer.exists() and not pointer.is_symlink():
        pointer.rename(generations / 'legacy')
    temporary = root / ('.checkpoint-' + uuid.uuid4().hex)
    temporary.symlink_to(path.relative_to(root), target_is_directory=True)
    os.replace(temporary, pointer)


def restore_training_state(path, optimizer, scaler, root):
    import torch
    from .resume import read_jsonl
    state = torch.load(Path(path) / 'trainer_state.pt', map_location='cpu', weights_only=False)
    rows = read_jsonl(Path(root) / 'updates.jsonl', repair=True)
    next_update = state.get('next_update')
    if next_update is None:
        # Legacy saves lack an iteration cursor. Require matching evidence rather
        # than guessing from the number of nonzero optimizer steps.
        if not rows or rows[-1]['completed_nonzero_updates'] != state['completed_updates']:
            raise ValueError('Legacy checkpoint does not match the update journal')
        next_update = rows[-1]['update'] + 1
    if any(r['update'] >= next_update for r in rows):
        raise ValueError('Update journal is ahead of the committed checkpoint')
    metric = state.get('metrics')
    if metric and not any(r['update'] == metric['update'] for r in rows):
        from .logio import append_row
        append_row(Path(root) / 'updates.jsonl', metric)
    optimizer.load_state_dict(state['optimizer'])
    if 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    elif scaler.is_enabled():
        raise ValueError('Legacy FP16 checkpoint lacks gradient scaler state')
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    return next_update, state['completed_updates']
