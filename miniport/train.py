"""Slurm training entry point: CUDA policy + CPU subprocess verifier, never Docker."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import shutil

from .experiment import ExperimentConfig
from .tasks import training_tasks, limit_tasks, VERIFIER_VERSION


def read_config(path):
    raw = json.loads(Path(path).read_text())
    experiment = ExperimentConfig(**raw["experiment"]).validate()
    if raw["condition"] not in {"terminal", "immediate", "action_value"}:
        raise ValueError("Unknown training condition")
    if raw["group_size"] < 2 or experiment.gradient_accumulation % raw["group_size"]:
        raise ValueError("Gradient accumulation must be a multiple of RLOO group size >= 2")
    if raw["updates"] < 1 or raw["candidate_actions"] != 2 or raw["continuations_per_action"] < 2:
        raise ValueError("Use positive updates, two candidate actions and at least two continuations")
    tasks = {t.id: t for t in training_tasks()}
    if not raw["task_ids"] or any(t not in tasks for t in raw["task_ids"]):
        raise ValueError("Training task IDs must be canonical training variants")
    selected = limit_tasks([tasks[t] for t in raw["task_ids"]], experiment.task_limit)
    if not selected:
        raise ValueError("task_limit excludes all configured training tasks")
    return raw, experiment, selected


def validate_resume_config(old, new, *, refresh_pending):
    """Allow an update extension and order-preserving task narrowing."""
    old_updates, new_updates = old["updates"], new["updates"]
    if new_updates < old_updates:
        raise ValueError("Resume cannot reduce the configured update target")
    old_tasks, new_tasks = old["task_ids"], new["task_ids"]
    if not new_tasks or new_tasks != [task for task in old_tasks if task in set(new_tasks)]:
        raise ValueError("Resume task IDs must be a nonempty order-preserving subset of the prior schedule")
    comparable_old, comparable_new = json.loads(json.dumps(old)), json.loads(json.dumps(new))
    comparable_old["experiment"].pop("task_limit", None)
    comparable_new["experiment"].pop("task_limit", None)
    comparable_old.pop("updates")
    comparable_new.pop("updates")
    comparable_old.pop("task_ids")
    comparable_new.pop("task_ids")
    if comparable_old != comparable_new:
        raise ValueError("Resume must use the original run configuration")
    if not refresh_pending:
        exact_old, exact_new = json.loads(json.dumps(old)), json.loads(json.dumps(new))
        exact_old.pop("updates")
        exact_new.pop("updates")
        exact_old.pop("task_ids")
        exact_new.pop("task_ids")
        if exact_old != exact_new:
            raise ValueError("Policy resume only permits increasing updates and narrowing task IDs")


def starter(task, raw):
    from .starter import starter_source
    if raw.get("starter_dir"):
        source = (Path(raw["starter_dir"]) / task.id / "submission" / "candidate.py").read_text()
        return starter_source(task, source)
    return starter_source(task)


def value_checkpoint_seed(policy_seed, checkpoint, refresh=None):
    seed = policy_seed + 1_000_000 + checkpoint * 10_000
    if refresh is not None and checkpoint in refresh.get("indices", []):
        offset = refresh.get("resample_seed_offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("Invalid collection refresh seed offset")
        seed += offset
    return seed


def collect_value_data(policy, tasks, raw, root, backend, *, resume=False, refresh=None):
    from .rollouts import Snapshot, rollout
    from .policy import Completion
    from .resume import atomic_json, read_jsonl
    from .collection import action_key, choose_actions, write_rows, replace_legacy_duplicate
    root = Path(root)
    rows = {(r["checkpoint"], r["action_index"]): r
            for r in read_jsonl(root / "value-data.jsonl", repair=resume)}
    # More checkpoints than task families intentionally cycle through new seeded
    # trajectories; task_limit restricts families, not value-data volume.
    checkpoint_count = raw["value_checkpoints"]
    rows = {key: row for key, row in rows.items() if key[0] < checkpoint_count}
    for i in range(checkpoint_count):
        task = tasks[i % len(tasks)]
        base_seed = value_checkpoint_seed(policy.config.seed, i, refresh)
        baseline = rollout(policy, task, Snapshot(starter(task, raw), remaining=policy.config.max_actions),
                           root / f"baseline-{i}", backend, base_seed, resume=resume)
        choices = [s for s in baseline["steps"] if s["snapshot"].remaining >= 3]
        if not baseline["trainable"] or not choices:
            continue
        chosen = choices[len(choices) // 2]
        state = chosen["snapshot"]
        selection = root / f"checkpoint-{i}.json"
        if selection.exists():
            saved = json.loads(selection.read_text())
            state = Snapshot(**saved["snapshot"])
            actions = [Completion(**a) for a in saved["actions"]]
        else:
            # Legacy runs persisted the forced second action in each continuation.
            recovered = None
            for directory in sorted(root.glob(f"branch-{i}-1-*")):
                samples = read_jsonl(directory / "completions.jsonl", repair=resume)
                if samples:
                    r = samples[0]
                    recovered = Completion(r["prompt"], r["completion"], r["prompt_ids"], r["output_ids"])
                    break
            migration = root / f"checkpoint-{i}-resample-state.json"
            if migration.exists() or (recovered is not None and action_key(recovered) == action_key(chosen["completion"])):
                state = replace_legacy_duplicate(root, i, task, state, rows, backend)
                recovered = None
            actions, attempts = choose_actions(policy, task, state, chosen["completion"],
                base_seed + 9000, raw.get("candidate_resamples", 3), selection,
                recovered=recovered)
            atomic_json(selection, {"checkpoint": i, "task_id": task.id,
                "snapshot": asdict(state), "actions": [asdict(a) for a in actions],
                "sampling_attempts": attempts, "distinct_candidates": len(actions)})
        for j, action in enumerate(actions):
            outcomes = []
            for k in range(raw["continuations_per_action"]):
                continuation = rollout(policy, task, state, root / f"branch-{i}-{j}-{k}", backend,
                                       base_seed + 1000 + k * 100, forced=action, resume=resume)
                if continuation["trainable"]:
                    outcomes.append(continuation["reward"])
            row = {"task_id": task.id, "checkpoint": i, "action_index": j,
                   "prompt": action.prompt, "action": action.text,
                   "source_sha256": hashlib.sha256(state.source.encode()).hexdigest(),
                   "remaining": state.remaining, "successes": int(sum(outcomes)), "count": len(outcomes),
                   "continuation_seed_schedule": [base_seed + 1000 + k * 100 for k in range(raw["continuations_per_action"])]}
            rows[(i, j)] = row
            if len(actions) == 1 and (i, 1) in rows:
                rows[(i, 1)]["excluded_duplicate"] = True
                rows[(i, 1)]["duplicate_of"] = 0
            write_rows(root / "value-data.jsonl", rows)
        print(f"Collected checkpoint {i + 1}/{checkpoint_count} ({len(actions)} distinct actions)", flush=True)
    return [rows[key] for key in sorted(rows)]


def combined_value_rows(root, primary=None):
    """Load ordinary and independently verified teacher-paired value rows."""
    from .resume import read_jsonl
    root = Path(root)
    rows = list(primary) if primary is not None else read_jsonl(root / "value-data.jsonl", repair=False)
    rows.extend(read_jsonl(root / "teacher-value-data.jsonl", repair=False))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume collection or policy training from saved progress")
    parser.add_argument("--collect-only", action="store_true",
                        help="Complete collection and action-value fitting, then exit before policy updates")
    args = parser.parse_args(argv)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    raw, config, tasks = read_config(args.config)
    root = Path(raw["output_dir"])
    existing = root.exists()
    if existing and not args.resume:
        raise ValueError("Output already exists; use --resume to continue training")
    from .checkpoints import checkpoint_path, save_checkpoint, restore_training_state
    saved_checkpoint = checkpoint_path(root) if args.resume else None
    refresh_path = root / "collection-refresh.json"
    refresh = json.loads(refresh_path.read_text()) if refresh_path.exists() else None
    refresh_pending = refresh is not None and refresh["status"] != "complete"
    if refresh_pending and (refresh["status"] != "pending" or refresh["config"] != raw):
        raise ValueError("Collection refresh must be prepared for this configuration")
    previous = None
    if args.resume:
        if not existing:
            raise ValueError("Resume requires an existing run directory")
        previous = json.loads((root / "run-manifest.json").read_text())
        old_config = json.loads((root / "config.json").read_text())
        validate_resume_config(old_config, raw, refresh_pending=refresh_pending)
        if not saved_checkpoint and (root / "updates.jsonl").exists():
            raise ValueError("Update journal exists without a recoverable checkpoint")
    import torch
    from huggingface_hub import model_info
    from .policy import HFPolicy, SYSTEM_PROMPT
    from .sealed import SubprocessBackend
    from .rollouts import Snapshot, rollout
    from .action_value import fit, load
    from .rl import accumulate
    from .provenance import source_hash
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this job")
    # Resolve once, persist the immutable revision, and reuse it for all downloads.
    revision = previous["base_revision"] if previous else model_info(config.model, revision=raw.get("base_revision", "main")).sha
    root.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (root / ".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume:
        (root / "config.json").write_text(json.dumps(raw, indent=2))
        (root / "requirements-resolved.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    metadata = {"base_model": config.model, "base_revision": revision,
                "experiment": asdict(config), "training_config": raw, "isolation": "subprocess",
                "verifier_version": VERIFIER_VERSION, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes")},
                "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "source_sha256": source_hash()}
    from .resume import atomic_json
    if previous:
        from datetime import datetime, timezone
        if not (root / "run-manifest.initial.json").exists():
            atomic_json(root / "run-manifest.initial.json", previous)
        if not (root / "config.initial.json").exists():
            atomic_json(root / "config.initial.json", old_config)
        atomic_json(root / "config.json", raw)
        metadata["mixed_collection_protocols"] = (previous.get("mixed_collection_protocols", False)
            or previous["source_sha256"] != metadata["source_sha256"])
        with (root / "resume-history.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                "previous_manifest": previous, "new_source_sha256": metadata["source_sha256"],
                "new_system_prompt_sha256": metadata["system_prompt_sha256"],
                "note": "Saved results retained; unfinished rollouts use current feedback and prompts"}) + "\n")
    atomic_json(root / "run-manifest.json", metadata)
    backend = SubprocessBackend(timeout=raw["verifier_timeout"])
    policy = HFPolicy(config, revision, checkpoint=None if refresh_pending else saved_checkpoint)
    value = None
    if raw["condition"] == "action_value" and saved_checkpoint and not refresh_pending:
        value = load(root / "action-value.json", combined_value_rows(root / "value-collection"))
    elif raw["condition"] == "action_value":
        data_dir = root / "value-collection"
        data_dir.mkdir(exist_ok=args.resume)
        rows = collect_value_data(policy, tasks, raw, data_dir, backend, resume=args.resume, refresh=refresh)
        rows = combined_value_rows(data_dir, rows)
        value, report = fit(rows, root / "action-value.json", raw["value_epochs"], config.seed)
        print("Action-value validation:", report, flush=True)
        if refresh_pending:
            refresh['status'] = 'complete'
            atomic_json(refresh_path, refresh)
            if args.collect_only:
                print("Collection refresh complete; policy training was not started", flush=True)
                return
            if saved_checkpoint:
                import gc
                del policy
                gc.collect()
                torch.cuda.empty_cache()
                policy = HFPolicy(config, revision, checkpoint=saved_checkpoint)
    if args.collect_only:
        print("Collection already complete; policy training was not started", flush=True)
        return
    params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=raw["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=policy.dtype == torch.float16)
    microgroups = config.gradient_accumulation // raw["group_size"]
    start_update, completed = (restore_training_state(saved_checkpoint, optimizer, scaler, root)
                               if saved_checkpoint else (0, 0))
    print(f"Policy training starts at update {start_update}; {completed} optimizer steps restored", flush=True)
    try:
        for update in range(start_update, raw["updates"]):
            optimizer.zero_grad(set_to_none=True)
            metrics = []
            for micro in range(microgroups):
                task = tasks[(update * microgroups + micro) % len(tasks)]
                group = []
                for member in range(raw["group_size"]):
                    result = rollout(policy, task, Snapshot(starter(task, raw), remaining=config.max_actions),
                        root / f"update-{update}-group-{micro}-member-{member}", backend,
                        config.seed + update * 100000 + micro * 10000 + member * 100,
                        value_model=value, resume=args.resume)
                    group.append(result)
                metrics.append(accumulate(policy.model, group, condition=raw["condition"],
                    kl_coefficient=raw["kl_coefficient"], value_coefficient=raw["value_coefficient"],
                    immediate_coefficient=raw.get("immediate_coefficient", 0.1),
                    accumulation_groups=microgroups, scaler=scaler, temperature=config.temperature))
            if any(p.grad is not None for p in params):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                if norm > 0:
                    scaler.step(optimizer)
                    completed += 1
                scaler.update()
            metric = {"update": update, "groups": metrics, "completed_nonzero_updates": completed,
                      "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30}
            save_checkpoint(policy, optimizer, root, metadata, completed,
                            next_update=update + 1, scaler=scaler, metrics=metric)
            from .logio import append_row
            append_row(root / "updates.jsonl", metric)
            print(f"Update {update + 1}/{raw['updates']}: {metrics}", flush=True)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("CUDA OOM: last committed checkpoint retained; resume after resolving memory pressure")
    if not completed:
        raise RuntimeError("No nonzero policy update occurred; checkpoint is not a trained result. Calibrate tasks/rewards.")


if __name__ == "__main__":
    main()
