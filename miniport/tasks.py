"""Public training catalog. Evaluation configurations are provisioned separately."""
from dataclasses import asdict, dataclass

TEMPLATES = ("mlp", "cnn", "layernorm", "attention", "rope_cache", "sampling")
GATES = ("topology", "parameters", "layer", "end_to_end", "determinism")
VERIFIER_VERSION = "miniport-v3-outcomes"


def limit_tasks(tasks, limit=None):
    allowed = set(TEMPLATES[:limit]) if limit is not None else set(TEMPLATES)
    return [task for task in tasks if task.template in allowed]


@dataclass(frozen=True)
class Task:
    id: str
    template: str
    split: str
    batch: int
    length: int
    width: int
    hidden: int
    heads: int
    epsilon: float = 1e-5
    offset: int = 2

    def to_dict(self):
        return asdict(self)


def training_tasks():
    return [Task(f"{name}-{i}", name, "train", b, n, w, h, heads)
            for name in TEMPLATES
            for i, (b, n, w, h, heads) in enumerate(
                ((1, 4, 8, 12, 2), (2, 6, 12, 8, 3), (2, 5, 16, 20, 4)))]


def heldout_tasks():
    # Call only in the evaluator process; these are never exported to training tasks.
    return [Task(f"{name}-heldout", name, "heldout", 3, 7, 24, 28, 3,
                 epsilon=1e-3, offset=5) for name in TEMPLATES]
