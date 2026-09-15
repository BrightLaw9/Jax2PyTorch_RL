"""Paired baseline/adapter held-out evaluation on CUDA with CPU verification."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path


class HFEvalPolicy:
    def __init__(self, config, model, revision, checkpoint=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU evaluation")
        self.config = config
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(config.gpu_memory_gb * 2**30 / total, 1.0), 0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        tokenizer_path = Path(checkpoint) / "tokenizer" if checkpoint else model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
            **({} if checkpoint else {"revision": revision}), trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(model, revision=revision,
            quantization_config=quant, torch_dtype=dtype, device_map={"": 0},
            max_memory={0: f"{config.gpu_memory_gb}GiB"}, attn_implementation="sdpa",
            low_cpu_mem_usage=True, trust_remote_code=False)
        if checkpoint:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        self.model = base.eval()

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
            raise RuntimeError("GPU evaluation exceeded configured CUDA-memory cap")
        return Completion(prompt, self.tokenizer.decode(output, skip_special_tokens=True), ids, output)


def reused_baseline_reports(directory, task_ids, seed, version):
    reports = []
    for task_id in task_ids:
        path = Path(directory) / "baseline" / task_id / "evaluation.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if (report.get("condition") != "baseline" or report.get("checkpoint") is not None
                or report.get("task_id") != task_id or report.get("policy_seed") != seed
                or report.get("version") != version):
            raise ValueError(f"Incompatible saved baseline report: {path}")
        reports.append(report)
    return reports


def evaluate_pair(checkpoint, manifest_path, output, challenges, task_ids, seed=42, baseline_results=None):
    import torch
    from .checkpoints import checkpoint_path
    from .experiment import ExperimentConfig
    from .rollouts import Snapshot, rollout
    from .sealed import SubprocessBackend, verify_submission
    from .starter import starter_source
    from .tasks import Task, VERIFIER_VERSION, heldout_tasks
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir() and not (checkpoint / "manifest.json").exists():
        checkpoint = checkpoint_path(checkpoint)
    if checkpoint is None or not (checkpoint / "manifest.json").exists():
        raise ValueError("A committed adapter checkpoint is required")
    metadata = json.loads((checkpoint / "manifest.json").read_text())
    private = json.loads(Path(manifest_path).read_text())
    if private.get("version") != VERIFIER_VERSION:
        raise ValueError("Held-out manifest version mismatch")
    available = {task.id: task for task in heldout_tasks()}
    provisioned = {raw["id"]: Task(**raw) for raw in private["tasks"]}
    if not task_ids or any(task not in available or provisioned.get(task) != available[task] for task in task_ids):
        raise ValueError("Use provisioned canonical held-out tasks")
    saved_baselines = (reused_baseline_reports(baseline_results, task_ids, seed, VERIFIER_VERSION)
                       if baseline_results is not None else None)
    output, challenges = Path(output), Path(challenges)
    if output.exists():
        raise ValueError("Use a fresh paired evaluation output directory")
    output.mkdir(parents=True)
    challenges.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(**metadata["experiment"]).validate()
    backend = SubprocessBackend(timeout=metadata["training_config"]["verifier_timeout"])
    reports = {}
    for condition, adapter in (("baseline", None), ("trained", checkpoint)):
        if condition == "baseline" and saved_baselines is not None:
            reports[condition] = saved_baselines
            for report in saved_baselines:
                episode = output / condition / report["task_id"]
                episode.mkdir(parents=True)
                (episode / "evaluation.json").write_text(json.dumps(report, indent=2))
            print(f"Reused {len(saved_baselines)} baseline reports; no baseline policy will run", flush=True)
            continue
        policy = HFEvalPolicy(config, metadata["base_model"], metadata["base_revision"], adapter)
        condition_reports = []
        for task_id in task_ids:
            task = available[task_id]
            episode = output / condition / task_id
            result = rollout(policy, task, Snapshot(starter_source(task), remaining=config.max_actions),
                episode, backend, seed, evaluation=True)
            report = verify_submission(task, result["submission"], private["seeds"], backend=backend,
                final=True, challenge_path=challenges / f"{task.id}.json")
            report.update(task_id=task.id, condition=condition, actions=len(result["steps"]),
                generated_tokens=sum(len(step["completion"].output_ids) for step in result["steps"]),
                policy_seed=seed, checkpoint=str(adapter) if adapter else None)
            (episode / "evaluation.json").write_text(json.dumps(report, indent=2))
            condition_reports.append(report)
            print(condition, task.id, "pass:", bool(report.get("hidden_parity_passes")
                and report.get("fresh_randomization_passes")), flush=True)
        reports[condition] = condition_reports
        del policy
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    summary = {condition: {"tasks": len(rows),
        "automated_passes": sum(bool(row.get("hidden_parity_passes") and row.get("fresh_randomization_passes")) for row in rows),
        "mean_actions": sum(row["actions"] for row in rows) / len(rows) if rows else None} for condition, rows in reports.items()}
    summary["paired"] = {"task_ids": task_ids, "policy_seed": seed,
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint), "completed_updates": metadata.get("completed_updates")}
    if baseline_results is not None:
        present = {row["task_id"] for row in saved_baselines}
        summary["paired"].update(baseline_results=str(baseline_results),
            paired_task_ids=[task for task in task_ids if task in present],
            missing_baseline_task_ids=[task for task in task_ids if task not in present],
            baseline_provenance_note="Saved reports validate seed/task/verifier only; original model, prompt, and manifest hashes were not recorded.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-results", type=Path,
                        help="Reuse available baseline reports; never run the baseline, even for missing tasks")
    args = parser.parse_args(argv)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    evaluate_pair(args.checkpoint, args.manifest, args.output, args.challenges, args.task, args.seed,
                  args.baseline_results)


if __name__ == "__main__":
    main()
