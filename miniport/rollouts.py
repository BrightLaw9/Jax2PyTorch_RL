"""Exact source/history snapshots and sequential continuations under a frozen policy."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

from .controller import Controller


@dataclass
class Snapshot:
    source: str
    history: list = field(default_factory=list)
    remaining: int = 18


def rollout(policy, task, initial, directory, backend, seed, *, forced=None, evaluation=False, value_model=None,
            resume=False):
    from .resume import atomic_json, completed_rollout, recover_partial
    from .logio import write_completion, save_trajectory
    directory = Path(directory)
    existing = directory.exists()
    if resume and existing:
        saved = completed_rollout(directory)
        if saved is not None:
            print(f"Reused completed rollout {directory.name}", flush=True)
            return saved
    directory.mkdir(parents=True, exist_ok=resume)
    sub = directory / "submission"
    sub.mkdir(exist_ok=resume)
    history, steps, prior = list(initial.history), [], None
    if resume and existing:
        prior, steps, source, history = recover_partial(directory, initial)
        print(f"Resuming {directory.name} at action {len(steps)} with {initial.remaining - len(steps)} actions left", flush=True)
    else:
        (sub / "candidate.py").write_text(initial.source)
    controller = Controller(task, sub, directory / "protected.jsonl", policy.config, backend,
                            evaluation=evaluation, remaining_actions=initial.remaining,
                            resume_records=prior, resume_sources=[s["snapshot"].source for s in steps])
    if resume and existing and not controller.recorder.stopped:
        from .sealed import verify_submission
        from .observations import visible_feedback
        refresh = {"feedback": visible_feedback(verify_submission(task, sub, backend=backend)),
                   "remaining_actions": initial.remaining - len(steps), "done": False}
        history.append({"role": "user", "content": json.dumps(refresh, sort_keys=True)})
        with (directory / "resume-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"at_action": len(steps), "observation": refresh}) + "\n")
    proposal_tokens = 0

    def action_credit(snapshot, completion, index, *, committed=False):
        nonlocal proposal_tokens
        credit_path = directory / 'steps' / f'{index:03d}-value.json'
        if credit_path.exists():
            saved = json.loads(credit_path.read_text())
            if saved['prompt'] != completion.prompt or saved['completion'] != completion.text:
                if committed:
                    raise ValueError('Saved action credit does not match completion')
                import uuid
                credit_path.rename(credit_path.with_suffix('.superseded-' + uuid.uuid4().hex + '.json'))
        if not credit_path.exists():
            alternative = policy.generate(task, snapshot.source, snapshot.history, snapshot.remaining,
                                          seed + 100000 + index)
            if alternative.prompt != completion.prompt:
                raise ValueError('Cannot reconstruct action credit with a different prompt')
            qa = value_model.predict(completion.prompt, completion.text)
            qb = value_model.predict(alternative.prompt, alternative.text)
            saved = {'prompt': completion.prompt, 'completion': completion.text,
                     'alternative': alternative.text, 'alternative_output_ids': alternative.output_ids,
                     'value_advantage': (qa - qb) / 2}
            atomic_json(credit_path, saved)
        proposal_tokens += len(saved['alternative_output_ids'])
        return saved['value_advantage']

    if value_model is not None:
        for index, step in enumerate(steps):
            step['value_advantage'] = action_credit(step['snapshot'], step['completion'], index, committed=True)
    while not controller.recorder.stopped:
        remaining = initial.remaining - len(steps)
        snapshot = Snapshot((sub / "candidate.py").read_text(), list(history), remaining)
        start = time.monotonic()
        completion = forced if len(steps) == 0 and forced is not None else policy.generate(
            task, snapshot.source, history, remaining, seed + len(steps))
        # Persist before parsing/verifying so interrupted episodes retain evidence.
        token_limit_reached = len(completion.output_ids) >= policy.config.max_new_tokens
        write_completion(directory, {"index": len(steps), "prompt": completion.prompt,
                "snapshot": asdict(snapshot),
                "completion": completion.text, "prompt_ids": completion.prompt_ids,
                "output_ids": completion.output_ids, "output_tokens": len(completion.output_ids),
                "token_limit_reached": token_limit_reached,
                "wall_seconds": time.monotonic() - start})
        value_advantage = 0.0
        if value_model is not None:
            value_advantage = action_credit(snapshot, completion, len(steps))
        try:
            from .action_protocol import parse_action
            action = parse_action(completion.text)
            observation = controller.step(action)
        except (ValueError, TypeError, AttributeError) as exc:
            reason = f"invalid_action: {type(exc).__name__}: {exc}"
            if token_limit_reached:
                reason += "; output token limit reached: shorten the edit and omit reference comments and unused classes"
            observation = controller.reject_action(reason)
        steps.append({"snapshot": snapshot, "completion": completion, "value_advantage": value_advantage,
                      "wall_seconds": time.monotonic() - start})
        history.extend([{"role": "assistant", "content": completion.text},
                        {"role": "user", "content": json.dumps(observation, sort_keys=True)}])
    records = controller.recorder.records
    result = {"task_id": task.id, "steps": steps, "records": records,
              "reward": records[-1]["training_reward"], "trainable": records[-1]["trainable"],
              "submission": str(sub), "proposal_tokens": proposal_tokens}
    serializable = {k: v for k, v in result.items() if k not in {"steps", "records"}}
    serializable["steps"] = [{"source": s["snapshot"].source, "history": s["snapshot"].history,
        "remaining": s["snapshot"].remaining, "prompt": s["completion"].prompt,
        "completion": s["completion"].text, "prompt_ids": s["completion"].prompt_ids,
        "output_ids": s["completion"].output_ids, "value_advantage": s["value_advantage"],
        "wall_seconds": s["wall_seconds"]} for s in steps]
    save_trajectory(directory, serializable, records)
    return result
