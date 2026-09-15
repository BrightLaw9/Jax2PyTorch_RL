import argparse
import json
from pathlib import Path
import secrets
import shutil
import os

from .tasks import Task, training_tasks, heldout_tasks, VERIFIER_VERSION
from .observations import visible_feedback


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="MiniPort benchmark pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Export the 18 visible training tasks")
    generate.add_argument("directory", type=Path)
    verify = commands.add_parser("verify", help="Run visible gates on one training workspace")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--image", default="miniport-verifier:local")
    verify.add_argument("--backend", choices=("docker", "subprocess"), default="docker")
    verify.add_argument("--protected-report", action="store_true", help="Evaluator-only full report; never expose to policy")
    provision = commands.add_parser("provision-eval", help="Create private evaluator-only manifest")
    provision.add_argument("manifest", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen port against private held-out configuration")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("task_id")
    evaluate.add_argument("candidate", type=Path)
    evaluate.add_argument("--audit", type=Path)
    evaluate.add_argument("--image", default="miniport-verifier:local")
    evaluate.add_argument("--challenge", type=Path, required=True,
                          help="Private per-task fresh challenge file; reused for both evaluation conditions")
    review = commands.add_parser("review-notebook", help="Select 30–40 observable checkpoints for human review")
    review.add_argument("directory", type=Path)
    review.add_argument("logs", type=Path, nargs="+")
    review.add_argument("--count", type=int, default=36)
    script = commands.add_parser("run-script", help="Exercise the restricted controller with JSON actions")
    script.add_argument("task_id")
    script.add_argument("submission", type=Path)
    script.add_argument("actions", type=Path)
    script.add_argument("log", type=Path)
    script.add_argument("--image", default="miniport-verifier:local")
    script.add_argument("--backend", choices=("docker", "subprocess"), default="docker")
    args = parser.parse_args()
    if args.command == "generate":
        if args.directory.exists():
            parser.error("Output directory already exists; choose a fresh directory")
        args.directory.mkdir(parents=True)
        for task in training_tasks():
            target = args.directory / task.id
            target.mkdir()
            write_json(target / "task.json", task.to_dict())
            from .requirements import task_text
            (target / "submission").mkdir()
            from .starter import starter_source
            (target / "submission" / "candidate.py").write_text(starter_source(task))
            (target / "task.md").write_text(task_text(task))
        print(f"Created 18 training tasks in {args.directory}")
    elif args.command == "run-script":
        from .controller import run_script
        from .sealed import DockerBackend, SubprocessBackend
        matches = [task for task in training_tasks() if task.id == args.task_id]
        if not matches:
            parser.error("Unknown training task")
        if args.backend == "subprocess":
            os.environ["JAX_PLATFORMS"] = "cpu"
        backend = SubprocessBackend() if args.backend == "subprocess" else DockerBackend(args.image)
        records = run_script(matches[0], args.submission, args.log, json.loads(args.actions.read_text()), backend=backend)
        print(json.dumps({"checkpoints": len(records), "log": str(args.log)}, indent=2))
    elif args.command == "review-notebook":
        from .review import create_notebook
        count = create_notebook(args.logs, args.directory, args.count)
        print(f"Selected {count} checkpoints; open {args.directory / 'review.ipynb'}")
    elif args.command == "provision-eval":
        # Exclusive creation avoids silently replacing seeds between conditions.
        manifest = {"version": VERIFIER_VERSION, "tasks": [t.to_dict() for t in heldout_tasks()],
                    "seeds": [secrets.randbits(32) for _ in range(8)]}
        with args.manifest.open("x") as stream:
            json.dump(manifest, stream, indent=2)
        args.manifest.chmod(0o600)
        print("Created private evaluator manifest. Keep it outside agent workspaces.")
    elif args.command == "verify":
        from .sealed import verify_submission, DockerBackend, SubprocessBackend
        task = Task(**json.loads((args.directory / "task.json").read_text()))
        if task not in training_tasks():
            parser.error("Visible verification requires an unchanged training task configuration")
        if args.backend == "subprocess":
            os.environ["JAX_PLATFORMS"] = "cpu"
        backend = SubprocessBackend() if args.backend == "subprocess" else DockerBackend(args.image)
        result = verify_submission(task, args.directory / "submission", backend=backend)
        print(json.dumps(result if args.protected_report else visible_feedback(result), indent=2))
        return 0 if result["passed"] else 1
    else:
        from .sealed import verify_submission, DockerBackend
        manifest = json.loads(args.manifest.read_text())
        if manifest["version"] != VERIFIER_VERSION:
            parser.error("Evaluator version mismatch")
        matches = [Task(**t) for t in manifest["tasks"] if t["id"] == args.task_id]
        if len(matches) != 1:
            parser.error("Unknown or duplicate held-out task")
        audit = json.loads(args.audit.read_text()) if args.audit else None
        result = verify_submission(matches[0], args.candidate, manifest["seeds"],
                                   backend=DockerBackend(args.image), final=True, audit=audit, challenge_path=args.challenge)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
