"""Convert transferred checkpoints and evaluate MLX policies using Docker only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def require_mac():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Native Apple Silicon macOS is required")


def convert_pair(source, output, bits=4):
    require_mac()
    import mlx.core as mx
    from mlx_lm.convert import convert
    if not mx.metal.is_available():
        raise RuntimeError("MLX Apple GPU unavailable")
    source, output = Path(source).resolve(), Path(output).resolve()
    if not (source / "COMPLETE").is_file():
        raise ValueError("Incomplete export bundle")
    if output.exists():
        raise ValueError("Use a fresh MLX output directory")
    manifests = [json.loads((source / f"{label}-hf" / "export-manifest.json").read_text()) for label in ("baseline", "trained")]
    for key in ("base_model", "base_revision", "experiment", "verifier_version", "system_prompt_sha256"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Mismatched export provenance: {key}")
    output.mkdir(parents=True)
    for label, manifest in zip(("baseline", "trained"), manifests):
        target = output / label
        convert(hf_path=str(source / f"{label}-hf"), mlx_path=str(target), quantize=True,
                q_bits=bits, q_group_size=64, dtype="float16")
        (target / "export-manifest.json").write_text(json.dumps(dict(manifest,
            mlx_bits=bits, mlx_group_size=64, mlx_lm_version=importlib.metadata.version("mlx-lm")), indent=2))
        mx.clear_cache()
    (output / "requirements-resolved.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (output / "COMPLETE").write_text("Both checkpoints converted with identical settings.\n")


def evaluate(model_path, manifest_path, output, challenge_dir, seed=42, image="miniport-verifier:local", task_ids=None):
    require_mac()
    os.environ["JAX_PLATFORMS"] = "cpu"
    from .policy import MLXPolicy, SYSTEM_PROMPT
    from .experiment import ExperimentConfig
    from .rollouts import Snapshot, rollout
    from .sealed import DockerBackend, verify_submission
    from .tasks import Task, VERIFIER_VERSION, heldout_tasks, limit_tasks
    from .provenance import source_hash
    import hashlib
    model_path, output = Path(model_path), Path(output)
    metadata = json.loads((model_path / "export-manifest.json").read_text())
    if metadata["verifier_version"] != VERIFIER_VERSION or metadata["system_prompt_sha256"] != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
        raise ValueError("Model export and evaluator prompt/verifier versions differ")
    if metadata.get("source_sha256") != source_hash():
        raise ValueError("Evaluator source differs from training export; transfer the matching project source")
    config = ExperimentConfig(**metadata["experiment"]).validate()
    private = json.loads(Path(manifest_path).read_text())
    if private["version"] != VERIFIER_VERSION:
        raise ValueError("Held-out manifest version mismatch")
    tasks = [Task(**t) for t in private["tasks"]]
    if any(t not in heldout_tasks() for t in tasks):
        raise ValueError("Noncanonical held-out task")
    tasks = limit_tasks(tasks, config.task_limit)
    if task_ids:
        if set(task_ids) - {t.id for t in tasks}:
            raise ValueError("Unknown held-out task ID")
        tasks = [t for t in tasks if t.id in task_ids]
    if output.exists():
        raise ValueError("Use a fresh evaluation output directory")
    output.mkdir(parents=True)
    Path(challenge_dir).mkdir(parents=True, exist_ok=True)
    backend = DockerBackend(image=image)
    policy = MLXPolicy(model_path, config)
    from .starter import starter_source
    reports = []
    for task in tasks:
        result = rollout(policy, task, Snapshot(starter_source(task), remaining=config.max_actions),
                         output / task.id, backend, seed, evaluation=True)
        report = verify_submission(task, result["submission"], private["seeds"], backend=backend,
                                   final=True, challenge_path=Path(challenge_dir) / f"{task.id}.json")
        report.update(task_id=task.id, actions=len(result["steps"]),
                      episode_valid=bool(result["trainable"] and not report.get("environment_outage")
                                         and not report.get("fresh_randomization", {}).get("environment_outage")),
                      generated_tokens=sum(len(s["completion"].output_ids) for s in result["steps"]),
                      prompt_tokens=sum(len(s["completion"].prompt_ids) for s in result["steps"]),
                      policy_seed=seed, model_export=metadata)
        (output / task.id / "evaluation.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(task.id, "numerical pass:", bool(report.get("hidden_parity_passes") and report.get("fresh_randomization_passes")), flush=True)
    summary = {"tasks": len(reports), "valid_episodes": sum(r["episode_valid"] for r in reports),
               "automated_passes": sum(bool(r["episode_valid"] and r.get("hidden_parity_passes") and r.get("fresh_randomization_passes")) for r in reports),
               "audited_valid": sum(r.get("valid", False) for r in reports),
               "note": "Numerical passers require artifact audit before final acceptance."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert")
    convert.add_argument("--source", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--bits", type=int, choices=(4, 8), default=4)
    run = commands.add_parser("evaluate")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--challenges", required=True, type=Path)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--image", default="miniport-verifier:local")
    run.add_argument("--task", action="append")
    args = parser.parse_args(argv)
    if args.command == "convert":
        convert_pair(args.source, args.output, args.bits)
    else:
        evaluate(args.model, args.manifest, args.output, args.challenges, args.seed, args.image, args.task)


if __name__ == "__main__":
    main()
