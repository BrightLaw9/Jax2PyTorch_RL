"""Offline teacher-guided value collection under a hard CUDA-memory cap."""
import argparse
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .action_protocol import parse_action
from .resume import atomic_json, load_details, read_jsonl
from .rollouts import Snapshot, rollout
from .tasks import training_tasks, VERIFIER_VERSION


@dataclass(frozen=True)
class TeacherConfig:
    run_dir: str
    model: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    revision: str = "main"
    gpu_memory_gb: int = 16
    context_cap: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 0.6
    states_per_task: int = 8
    candidate_resamples: int = 4
    continuations_per_action: int = 4
    paired_state_limit: int | None = None
    verifier_timeout: int = 60
    seed: int = 42
    task_ids: tuple = ("mlp-0", "layernorm-0", "attention-0")
    manual_corrections: dict = field(default_factory=dict)

    def validate(self):
        if not self.run_dir or not self.model:
            raise ValueError("Teacher model and run directory are required")
        if not 1 <= self.gpu_memory_gb <= 16:
            raise ValueError("Teacher CUDA memory must be capped at 16 GiB")
        if self.context_cap not in (2048, 4096) or not 128 <= self.max_new_tokens <= 1024:
            raise ValueError("Invalid teacher context/output budget")
        if not 1 <= self.states_per_task <= 40 or not 0 <= self.candidate_resamples <= 20:
            raise ValueError("Invalid teacher state or resampling budget")
        if not 2 <= self.continuations_per_action <= 16:
            raise ValueError("Use at least two teacher-pair continuations")
        if self.paired_state_limit is not None and not 1 <= self.paired_state_limit <= 40:
            raise ValueError("Teacher paired-state limit must be between one and 40")
        known = {task.id for task in training_tasks()}
        if not self.task_ids or any(task not in known for task in self.task_ids):
            raise ValueError("Teacher task IDs must be canonical training variants")
        if not isinstance(self.manual_corrections, dict):
            raise ValueError("Manual corrections must be a mapping")
        for key, correction in self.manual_corrections.items():
            if not str(key).isdigit():
                raise ValueError("Manual corrections must use numeric state IDs")
            if isinstance(correction, str):
                correction = {"path": correction, "model": "manual"}
            if (not isinstance(correction, dict) or set(correction) != {"path", "model"}
                    or not all(isinstance(correction[name], str) and correction[name]
                               for name in ("path", "model"))):
                raise ValueError("Each manual correction requires nonempty path and model")
        return self


def read_config(path):
    raw = json.loads(Path(path).read_text())
    if isinstance(raw.get("task_ids"), list):
        raw["task_ids"] = tuple(raw["task_ids"])
    return TeacherConfig(**raw).validate()


def _error_signature(record):
    summary = record.get("protected_summary", {})
    diagnostic = record.get("policy_observation", {}).get("feedback", {}).get("diagnostic", {})
    return (summary.get("error_type"), summary.get("failure_location"), diagnostic.get("message"))


def mine_stuck_states(run_dir, task_ids, states_per_task):
    """Select states after repeated no-op/error observations, without private arrays."""
    run_dir = Path(run_dir)
    wanted = set(task_ids)
    selected, seen = [], set()
    paths = sorted(run_dir.glob("update-*/trajectory.json"))
    paths += sorted((run_dir / "value-collection").glob("baseline-*/trajectory.json"))
    counts = {task: 0 for task in task_ids}
    for trajectory_path in paths:
        trajectory = json.loads(trajectory_path.read_text())
        task_id = trajectory.get("task_id")
        if task_id not in wanted or counts[task_id] >= states_per_task or trajectory.get("reward") == 1:
            continue
        records = read_jsonl(trajectory_path.parent / "protected.jsonl", repair=False)
        steps = [load_details(trajectory_path.parent, step) for step in trajectory.get("steps", [])]
        for index in range(1, min(len(records), len(steps))):
            current, previous = records[index], records[index - 1]
            unchanged = current.get("policy_observation", {}).get("edit_feedback", {}).get("code") == "no_executable_change"
            repeated = _error_signature(current) == _error_signature(previous) and any(_error_signature(current))
            if not (unchanged and repeated):
                continue
            step_index = index + 1 if index + 1 < len(steps) else index
            step = steps[step_index]
            identity = (task_id, hashlib.sha256(step["source"].encode()).hexdigest(), _error_signature(current))
            if identity in seen:
                continue
            seen.add(identity)
            selected.append({"state_id": len(selected), "task_id": task_id,
                "origin": str(trajectory_path.relative_to(run_dir)), "origin_action": step_index,
                "snapshot": {"source": step["source"], "history": step["history"], "remaining": step["remaining"]},
                "student_action": {"prompt": step["prompt"], "text": step["completion"],
                    "prompt_ids": step["prompt_ids"], "output_ids": step["output_ids"]},
                "error_signature": list(_error_signature(current))})
            counts[task_id] += 1
            break
    if not selected:
        raise ValueError("No repeated no-op/error states were found in the configured task families")
    return selected


class TeacherPolicy:
    def __init__(self, config, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for teacher collection")
        self.config, self.revision = config, revision
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(config.gpu_memory_gb * 2**30 / total, 1.0), 0)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, revision=revision,
            quantization_config=quant, torch_dtype=self.dtype, device_map={"": 0},
            max_memory={0: f"{config.gpu_memory_gb}GiB"}, attn_implementation="sdpa",
            low_cpu_mem_usage=True, trust_remote_code=False)
        self.model.eval()

    def generate(self, task, source, history, remaining, seed):
        import torch
        from transformers import StoppingCriteriaList
        from .action_protocol import StopAfterAction
        from .policy import Completion, render_prompt
        prompt, ids = render_prompt(self.tokenizer, task, source, history, remaining, self.config)
        torch.manual_seed(seed)
        with torch.no_grad():
            tokens = self.model.generate(input_ids=torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
                max_new_tokens=self.config.max_new_tokens, do_sample=True,
                temperature=self.config.temperature, top_p=1.0, top_k=0,
                pad_token_id=self.tokenizer.eos_token_id, use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopAfterAction(self.tokenizer, len(ids))]))
        output = tokens[0, len(ids):].tolist()
        if torch.cuda.max_memory_reserved() > self.config.gpu_memory_gb * 2**30:
            raise RuntimeError("Teacher exceeded configured CUDA-memory cap")
        return Completion(prompt, self.tokenizer.decode(output, skip_special_tokens=True), ids, output)


def _manual_spec(config, state_id):
    correction = config.manual_corrections.get(str(state_id))
    return {"path": correction, "model": "manual"} if isinstance(correction, str) else correction


def _verified_teacher_actions(policy, states, tasks, directory, backend, config, resume):
    from .policy import Completion
    from .observations import visible_feedback
    directory.mkdir(parents=True, exist_ok=resume)
    accepted = []
    for item in states:
        target = directory / f"state-{item['state_id']:03d}.json"
        manual = _manual_spec(config, item["state_id"])
        if target.exists():
            saved = json.loads(target.read_text())
            manual_hash = hashlib.sha256(Path(manual["path"]).read_bytes()).hexdigest() if manual else None
            if (not manual or (saved.get("candidate_origin") == "external_model_override"
                    and saved.get("source_model") == manual["model"]
                    and saved.get("manual_source_sha256") == manual_hash)):
                accepted.append((item, Completion(**saved["value_action"])))
                continue
            target.rename(target.with_name(target.name + ".superseded-" + uuid.uuid4().hex))
        attempts_path = directory / f"state-{item['state_id']:03d}.attempts.jsonl"
        attempts = read_jsonl(attempts_path, repair=resume)
        if attempts and (manual or any(row.get("protocol") != "iterative-diagnostic-v1" for row in attempts)):
            archive = attempts_path.with_name(attempts_path.name + ".independent-" + uuid.uuid4().hex)
            attempts_path.rename(archive)
            attempts = []
            print(f"Archived superseded teacher retries for state {item['state_id']}", flush=True)
        snapshot = Snapshot(**item["snapshot"])
        chosen = None
        for attempt in range(config.candidate_resamples + 1):
            candidate_origin = "teacher"
            if attempt < len(attempts):
                completion = Completion(**attempts[attempt]["teacher_completion"])
                candidate_origin = attempts[attempt].get("candidate_origin", "teacher")
            elif attempt == 0 and manual:
                manual_path = Path(manual["path"])
                if not manual_path.is_file():
                    raise ValueError(f"Manual correction source does not exist: {manual_path}")
                source = manual_path.read_text()
                student = item["student_action"]
                text = json.dumps({"type": "edit", "source": source}, separators=(",", ":"))
                completion = Completion(student["prompt"], text, student["prompt_ids"],
                    policy.tokenizer.encode(text, add_special_tokens=False))
                candidate_origin = "external_model_override"
            else:
                completion = policy.generate(tasks[item["task_id"]], snapshot.source, snapshot.history,
                    snapshot.remaining, config.seed + 50_000_000 + item["state_id"] * 1000 + attempt)
            report = None
            action = None
            try:
                action = parse_action(completion.text)
                if action.get("type") != "edit":
                    raise ValueError("Teacher must propose an edit")
                with tempfile.TemporaryDirectory() as temporary:
                    (Path(temporary) / "candidate.py").write_text(action["source"])
                    from .sealed import verify_submission
                    from .trajectory import TRAIN_TERMINAL_SEEDS
                    report = verify_submission(tasks[item["task_id"]], temporary,
                        seeds=TRAIN_TERMINAL_SEEDS, backend=backend, randomized=True)
            except (ValueError, TypeError, AttributeError) as error:
                report = {"passed": False, "error_type": type(error).__name__, "message": str(error)}
            if attempt >= len(attempts):
                with attempts_path.open("a") as stream:
                    stream.write(json.dumps({"protocol": "iterative-diagnostic-v1", "attempt": attempt,
                        "state_source_sha256": hashlib.sha256(snapshot.source.encode()).hexdigest(),
                        "candidate_origin": candidate_origin,
                        "teacher_completion": asdict(completion),
                        "verification": report}) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
            if report.get("environment_outage"):
                raise RuntimeError("Verifier outage during teacher filtering")
            if report.get("passed"):
                student = item["student_action"]
                value_action = Completion(student["prompt"], completion.text,
                    student["prompt_ids"], completion.output_ids)
                atomic_json(target, {"state": item, "teacher_revision": policy.revision,
                    "candidate_origin": candidate_origin,
                    **({"source_model": manual["model"],
                        "manual_source_sha256": hashlib.sha256(Path(manual["path"]).read_bytes()).hexdigest()}
                       if manual and candidate_origin == "external_model_override" else {}),
                    "teacher_completion": asdict(completion), "value_action": asdict(value_action),
                    "verification": report})
                chosen = value_action
                break
            # Unlike the original independent retries, the next teacher call sees
            # the failed candidate and its policy-visible verifier diagnostic.
            feedback = (visible_feedback(report) if "current_gate" in report else
                {"status": "fail", "error_type": report.get("error_type"),
                 "diagnostic": {"message": report.get("message", "Invalid teacher action")}})
            if isinstance(action, dict) and action.get("type") == "edit" and isinstance(action.get("source"), str):
                snapshot.source = action["source"]
            snapshot.history.extend([
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": json.dumps({"feedback": feedback,
                    "teacher_retry": "Correct the proposed implementation using this verifier diagnostic.",
                    "remaining_teacher_attempts": config.candidate_resamples - attempt}, sort_keys=True)},
            ])
            snapshot.remaining = max(1, snapshot.remaining - 1)
        if chosen is not None:
            accepted.append((item, chosen))
    if not accepted:
        raise RuntimeError("Teacher produced no verifier-passing corrections")
    return accepted


def _collect_pairs(student, accepted, tasks, directory, backend, config, resume):
    from .policy import Completion
    rows = {}
    for item, teacher_action in accepted:
        state = Snapshot(**item["snapshot"])
        actions = (("student_negative", Completion(**item["student_action"])),
                   ("teacher_positive", teacher_action))
        for role, action in actions:
            outcomes = []
            for continuation in range(config.continuations_per_action):
                result = rollout(student, tasks[item["task_id"]], state,
                    directory / f"pair-{item['state_id']:03d}-{role}-{continuation}", backend,
                    config.seed + 10_000_000 + item["state_id"] * 10_000 + continuation * 100,
                    forced=action, resume=resume)
                if result["trainable"]:
                    outcomes.append(result["reward"])
            key = (item["state_id"], role)
            rows[key] = {"task_id": item["task_id"], "checkpoint": f"teacher-{item['state_id']:03d}",
                "action_index": 0 if role == "student_negative" else 1,
                "pair_id": item["state_id"], "pair_role": role, "collection": "verified_teacher_pair",
                "prompt": action.prompt, "action": action.text,
                "source_sha256": hashlib.sha256(state.source.encode()).hexdigest(),
                "remaining": state.remaining, "successes": int(sum(outcomes)), "count": len(outcomes),
                "continuation_seed_schedule": [config.seed + 10_000_000 + item["state_id"] * 10_000 + k * 100
                    for k in range(config.continuations_per_action)]}
    from .collection import write_rows
    write_rows(directory.parent / "teacher-value-data.jsonl", rows)
    return [rows[key] for key in sorted(rows)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    config = read_config(args.config)
    run_dir = Path(config.run_dir)
    manifest_path = run_dir / "run-manifest.json"
    data_dir = run_dir / "value-collection"
    if not manifest_path.exists() or not (data_dir / "value-data.jsonl").exists():
        raise ValueError("Run ordinary value collection before the teacher job")
    output = data_dir / "teacher-collection"
    output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    run_lock = (run_dir / ".training.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = (output / ".teacher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    states_path = output / "states.json"
    states = json.loads(states_path.read_text()) if states_path.exists() else mine_stuck_states(
        run_dir, config.task_ids, config.states_per_task)
    if not states_path.exists():
        atomic_json(states_path, states)
    tasks = {task.id: task for task in training_tasks()}
    from .sealed import SubprocessBackend
    backend = SubprocessBackend(timeout=config.verifier_timeout)
    manifest = json.loads(manifest_path.read_text())
    from .experiment import ExperimentConfig
    student_config = ExperimentConfig(**manifest["experiment"]).validate()
    complete_external_coverage = all(_manual_spec(config, item["state_id"]) for item in states)
    if complete_external_coverage:
        from transformers import AutoTokenizer
        from .checkpoints import checkpoint_path
        checkpoint = checkpoint_path(run_dir)
        tokenizer_path = checkpoint / "tokenizer" if checkpoint else student_config.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
            **({} if checkpoint else {"revision": manifest["base_revision"]}), trust_remote_code=False)
        from types import SimpleNamespace
        correction_policy = SimpleNamespace(tokenizer=tokenizer, revision="external-model-corrections")
        teacher_revision = None
        print("All stuck states use external model corrections; local teacher model will not be loaded", flush=True)
    else:
        from huggingface_hub import model_info
        teacher_revision = model_info(config.model, revision=config.revision).sha
        correction_policy = TeacherPolicy(config, teacher_revision)
    accepted = _verified_teacher_actions(correction_policy, states, tasks,
        output / "candidates", backend, config, args.resume)
    paired = accepted[:config.paired_state_limit] if config.paired_state_limit is not None else accepted
    import torch
    del correction_policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    pair_outputs_complete = args.resume and all(
        (output / f"pair-{item['state_id']:03d}-{role}-{continuation}" / "trajectory.json").exists()
        for item, _ in paired
        for role in ("student_negative", "teacher_positive")
        for continuation in range(config.continuations_per_action))
    if pair_outputs_complete:
        student = None
        print("All selected paired rollouts are complete; skipping student model load", flush=True)
    else:
        from .policy import HFPolicy
        from .checkpoints import checkpoint_path
        student = HFPolicy(student_config, manifest["base_revision"], checkpoint=checkpoint_path(run_dir))
    teacher_rows = _collect_pairs(student, paired, tasks, output, backend, config, args.resume)
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from .train import combined_value_rows
    from .action_value import fit, dataset_sha256
    rows = combined_value_rows(data_dir)
    value, report = fit(rows, run_dir / "action-value.json",
        manifest["training_config"]["value_epochs"], student_config.seed)
    del value
    provenance = {"teacher_model": None if complete_external_coverage else config.model,
        "teacher_revision": teacher_revision, "local_teacher_loaded": not complete_external_coverage,
        "gpu_memory_gb": config.gpu_memory_gb, "verifier_version": VERIFIER_VERSION,
        "states_mined": len(states), "verified_teacher_actions": len(accepted),
        "paired_states": len(paired),
        "paired_state_ids": [item["state_id"] for item, _ in paired],
        "requested_task_ids": list(config.task_ids),
        "external_corrections": {state_id: {**_manual_spec(config, state_id),
            "source_sha256": hashlib.sha256(
                Path(_manual_spec(config, state_id)["path"]).read_bytes()).hexdigest()}
            for state_id in config.manual_corrections},
        "mined_task_counts": {task: sum(item["task_id"] == task for item in states)
                              for task in config.task_ids},
        "teacher_rows": len(teacher_rows), "combined_rows": len(rows),
        "combined_dataset_sha256": dataset_sha256(rows), "critic_report": report}
    atomic_json(output / "complete.json", provenance)
    print("Teacher-guided value collection complete:", provenance, flush=True)


if __name__ == "__main__":
    main()
