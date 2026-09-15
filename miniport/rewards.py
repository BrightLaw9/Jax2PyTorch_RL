"""Stateful observable-evidence rewards, with no hidden training feedback."""
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import shlex

from .tasks import GATES


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".toml", ".yaml", ".yml"}:
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def normalize_command(command):
    # Preserve argument order and case: changing either can change semantics.
    return shlex.join(shlex.split(command))


@dataclass(frozen=True)
class Checkpoint:
    action: str
    command: str
    before_hash: str
    after_hash: str
    result_signature: str
    passed_gates: tuple[str, ...]
    diagnostic: str = ""
    error: float | None = None
    metric_name: str = ""
    remaining_actions: int = 0
    actionable_failure: bool = True
    environment_outage: bool = False
    focused_diagnostic: bool = False
    immediate_revert: bool = False


@dataclass
class RewardTracker:
    reached: set = field(default_factory=set)
    previous: Checkpoint | None = None
    seen: set = field(default_factory=set)
    diagnostics: set = field(default_factory=set)
    error_relative_threshold: float = 1e-3
    error_absolute_threshold: float = 1e-7

    def score(self, checkpoint, prm=0.0):
        c = checkpoint
        if not math.isfinite(prm) or not -1 <= prm <= 1:
            raise ValueError("PRM prediction must be finite and in [-1, 1]")
        if c.passed_gates != GATES[:len(c.passed_gates)]:
            raise ValueError("Passed gates must be an ordered prefix")
        if c.error is not None and (not math.isfinite(c.error) or c.error < 0):
            raise ValueError("Parity error must be finite and nonnegative")
        events = {}
        gates = set(c.passed_gates)
        new = gates - self.reached
        if new:
            events["new_gate"] = float(len(new))
        if self.previous:
            regressed = set(self.previous.passed_gates) - gates
            if regressed:
                events["regression"] = -0.75 * len(regressed)
            p = self.previous
            if (p.error is not None and c.error is not None and c.metric_name
                    and p.metric_name == c.metric_name
                    and abs(p.error - c.error) > max(self.error_absolute_threshold,
                                                     self.error_relative_threshold * p.error)):
                events["numerical_change"] = 0.25 * max(-1.0, min(1.0,
                    math.log(max(p.error, 1e-12) / max(c.error, 1e-12))))
        signature = (c.after_hash, normalize_command(c.command), c.result_signature)
        new_diagnostic = bool(c.diagnostic and c.diagnostic not in self.diagnostics)
        if c.before_hash == c.after_hash and signature in self.seen and not new_diagnostic:
            events["repetition"] = -0.25
        if c.immediate_revert:
            events["immediate_revert"] = -0.15
        if c.action == "profile" and len(gates) < len(GATES):
            events["premature_optimization"] = -0.5
        if c.focused_diagnostic and new_diagnostic and self.previous and len(self.previous.passed_gates) < len(GATES):
            events["focused_diagnostic"] = 0.15
        premature = (c.action == "stop" and len(gates) < len(GATES)
                     and c.remaining_actions > 0 and c.actionable_failure and not c.environment_outage)
        if premature:
            events["premature_termination"] = -2.0
        self.reached.update(gates)
        self.seen.add(signature)
        if c.diagnostic:
            self.diagnostics.add(c.diagnostic)
        self.previous = c
        return {"execution": sum(events.values()), "prm": 0.25 * prm,
                "total": sum(events.values()) + 0.25 * prm, "events": events}
