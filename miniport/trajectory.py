"""Controller-owned checkpoint records. Never accept verifier results from policy text."""
from dataclasses import asdict
import difflib
import ast
import hashlib
import json
from pathlib import Path
import time

from .integrity import submission_source
from .rewards import Checkpoint, RewardTracker, normalize_command
from .sealed import verify_submission
from .tasks import GATES
from .observations import visible_feedback

# Training-only terminal cases, distinct from visible seeds (11, 23, 47).
# Not exposed in the policy workspace. Held-out evaluation uses separate manifests.
TRAIN_TERMINAL_SEEDS = (101, 211, 307, 401)


class Recorder:
    """One trusted recorder per rollout; its log must be outside the submission."""
    def __init__(self, task, submission, log_path, max_actions=18, backend=None, *, evaluation=False,
                 resume_records=None, resume_sources=None):
        self.task, self.submission = task, Path(submission).resolve()
        self.log_path = Path(log_path).resolve()
        if self.log_path.is_relative_to(self.submission):
            raise ValueError("Protected log cannot live inside submission")
        if self.log_path.exists() and resume_records is None:
            raise ValueError("Use a fresh rollout log")
        if task.split != "train" and not evaluation:
            raise ValueError("Training recorder cannot consume held-out evaluation")
        self.max_actions, self.backend = max_actions, backend
        self.evaluation = evaluation
        self.tracker = RewardTracker()
        self.records, self.prior_sources = [], []
        self.source = submission_source(self.submission)
        self.latest = None
        self.stopped = False
        self.trainable = True
        if resume_records is not None:
            self.records = list(resume_records)
            self.prior_sources = list(resume_sources or [])
            for record in self.records:
                values = dict(record["checkpoint"])
                values["passed_gates"] = tuple(values["passed_gates"])
                if not values["environment_outage"]:
                    self.tracker.score(Checkpoint(**values))
                    self.latest = record["protected_summary"] | {"passed_gates": list(values["passed_gates"])}
            if self.records:
                self.stopped = self.records[-1]["done"]
                self.trainable = self.records[-1]["trainable"]

    def record(self, action, command, *, proposed_next_action="", tokens=0, wall_seconds=0.0, policy_action=None,
               action_error=None):
        if self.stopped or len(self.records) >= self.max_actions:
            raise ValueError("Rollout has ended")
        if action not in {"edit", "test", "profile", "stop", "invalid"}:
            raise ValueError("Unknown action")
        if tokens < 0 or wall_seconds < 0:
            raise ValueError("Usage must be nonnegative")
        # Parse before verifier work so malformed commands cannot partially advance state.
        command = normalize_command(command)
        source = submission_source(self.submission)
        before_hash, after_hash = [hashlib.sha256(s.encode()).hexdigest() for s in (self.source, source)]
        verification_start = time.monotonic()
        result = verify_submission(self.task, self.submission, backend=self.backend)
        wall_seconds += time.monotonic() - verification_start
        # Infrastructure failure provides no evidence of regression or task progress.
        outage = result["environment_outage"]
        passed = tuple(self.latest["passed_gates"]) if outage and self.latest else tuple(result["passed_gates"])
        protected = {k: result.get(k) for k in ("current_gate", "max_abs_error", "error_type", "failure_location", "environment_outage")}
        if result.get("diagnostic"):
            protected["diagnostic"] = result["diagnostic"]
        signature = json.dumps({**protected, "pass_count": len(passed)}, sort_keys=True)
        diagnostic = ":".join(str(result.get(k) or "") for k in ("failure_location", "error_type"))
        if diagnostic == ":" or outage:
            diagnostic = ""
        previous_error = self.latest.get("max_abs_error") if self.latest else None
        metric = result["current_gate"] if not outage else ""
        remaining = self.max_actions - len(self.records) - 1
        c = Checkpoint(action, command, before_hash, after_hash, signature, passed,
                       diagnostic=diagnostic, error=result.get("max_abs_error") if not outage else None,
                       metric_name=metric, remaining_actions=remaining,
                       actionable_failure=not result["passed"] and not outage,
                       environment_outage=outage,
                       immediate_revert=bool(self.prior_sources and source == self.prior_sources[-1] and source != self.source))
        reward = ({"execution": 0.0, "prm": 0.0, "total": 0.0, "events": {}}
                  if outage else self.tracker.score(c))
        self.trainable = self.trainable and not outage
        try:
            unchanged = action == "edit" and ast.dump(ast.parse(source)) == ast.dump(ast.parse(self.source))
        except SyntaxError:
            unchanged = action == "edit" and source == self.source
        previous_record = self.records[-1] if self.records else None
        identical_failure = bool(previous_record and not outage and not result["passed"]
            and previous_record["checkpoint"]["after_hash"] == after_hash
            and previous_record["checkpoint"]["result_signature"] == signature)
        prior_reflection = bool(previous_record and
            previous_record.get("policy_observation", {}).get("recovery_guidance"))
        recovery_exhausted = bool(action == "edit" and unchanged and identical_failure and prior_reflection)
        done = action == "stop" or remaining == 0 or outage or recovery_exhausted
        terminal_report = None
        training_reward = 0.0
        if done and self.trainable and not self.evaluation:
            terminal_start = time.monotonic()
            terminal_report = verify_submission(self.task, self.submission, seeds=TRAIN_TERMINAL_SEEDS,
                                                backend=self.backend, randomized=True)
            wall_seconds += time.monotonic() - terminal_start
            self.trainable = not terminal_report["environment_outage"]
            training_reward = float(terminal_report["passed"]) if self.trainable else None
        if not self.trainable:
            training_reward = None  # Discard the entire episode from policy updates.
        if self.evaluation:
            training_reward = None
        observation = {"feedback": visible_feedback(result),
                       "remaining_actions": remaining, "done": done}
        if action == "edit":
            if unchanged:
                observation["edit_feedback"] = {
                    "code": "no_executable_change",
                    "message": "This edit made no executable change (comments and formatting do not count). "
                               + (f"The {result.get('failure_location') or result['current_gate']} failure remains. "
                                  "Change the implementation to address the latest diagnostic."
                                  if not result['passed'] and not outage else "Review the latest verification status.")}
        if identical_failure and not prior_reflection and not result["passed"]:
            from .requirements import recovery_shape_ledger
            observation["recovery_guidance"] = {
                "code": "repeated_failure_reflection",
                "message": "The source hash and verifier error are unchanged for two consecutive actions. "
                           "Before editing, compare every current tensor axis with the required semantic layout.",
                "observed_source_sha256": after_hash,
                "observed_error_signature": signature,
                "current_shape_evidence": result.get("diagnostic"),
                "expected_shape_ledger": recovery_shape_ledger(self.task),
                "one_recovery_edit_remaining": True,
            }
        if recovery_exhausted:
            observation["termination_reason"] = "recovery_edit_made_no_executable_change"
        if action == "invalid":
            observation["action_error"] = action_error or "invalid_action_use_the_documented_json_schema"
        labels = list(reward["events"])
        if reward["events"].get("numerical_change", 0) < 0 and "regression" not in labels:
            labels.append("regression")
        record = {"index": len(self.records), "task_id": self.task.id, "task_template": self.task.template,
                  "checkpoint": asdict(c), "protected_summary": protected,
                  "previous_error": previous_error, "before_summary": self.latest,
                  "gate_advanced": bool(reward["events"].get("new_gate")),
                  "furthest_gate": len(self.tracker.reached), "diagnostic_reward": reward, "automatic_labels": labels,
                  "policy_observation": observation,
                  "policy_action": policy_action,
                  "training_reward": training_reward, "trainable": self.trainable,
                  "terminal_report": terminal_report, "done": done,
                  "termination_reason": observation.get("termination_reason"),
                  "isolation": getattr(self.backend, "isolation", "docker"), "evaluation": self.evaluation,
                  "changed_files": ["candidate.py"] if source != self.source else [],
                  "diff": "".join(difflib.unified_diff(self.source.splitlines(True), source.splitlines(True),
                                                     fromfile="before/candidate.py", tofile="after/candidate.py"))[:24000],
                  "recent_actions": [{"action": r["checkpoint"]["action"], "command": r["checkpoint"]["command"],
                                      "summary": r["protected_summary"]} for r in self.records[-4:]],
                  "proposed_next_action": proposed_next_action, "tokens": tokens, "wall_seconds": wall_seconds}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        from .logio import write_record
        write_record(self.log_path, record, self.records)
        self.records.append(record)
        self.prior_sources.append(self.source)
        self.source = source
        if not outage:
            self.latest = protected | {"passed_gates": list(passed)}
        self.stopped = done
        return record


def select_checkpoints(records, count=36):
    """Prefer ambiguous failures and workflow events; deterministic and no CoT."""
    if not 30 <= count <= 40:
        raise ValueError("Select 30–40 checkpoints")
    def priority(record):
        events = record.get("diagnostic_reward", record.get("reward", {}))["events"]
        return (bool(set(events) & {"repetition", "regression", "premature_termination", "premature_optimization"}),
                bool(record["protected_summary"]["error_type"]), bool(record["changed_files"]))
    return sorted(records, key=priority, reverse=True)[:count]
