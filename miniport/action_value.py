"""Small CPU action-value predictor; outcome counts, not human workflow labels."""
import hashlib
import json
from pathlib import Path
import re

FEATURES = 512


def dataset_sha256(rows):
    """Stable identity for the exact rows eligible to fit the critic."""
    eligible = [row for row in rows if row.get("count", 0) > 0 and not row.get("excluded_duplicate", False)]
    encoded = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in eligible]
    return hashlib.sha256(("\n".join(sorted(encoded)) + "\n").encode()).hexdigest()


def features(prompt, action):
    import torch
    vector = torch.zeros(FEATURES)
    for prefix, text in (("state", prompt), ("action", action)):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        for a, b in zip(tokens, tokens[1:]):
            digest = hashlib.blake2b(f"{prefix}:{a}:{b}".encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "little") % FEATURES] += 1
    return vector / vector.norm().clamp_min(1)


class ActionValue:
    def __init__(self):
        import torch
        self.model = torch.nn.Sequential(torch.nn.Linear(FEATURES, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def predict(self, prompt, action):
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.model(features(prompt, action)).sigmoid().item()

    def save(self, path, metadata):
        Path(path).write_text(json.dumps({"features": FEATURES, "metadata": metadata,
            "state": {k: v.detach().tolist() for k, v in self.model.state_dict().items()}}))


def fit(rows, output, epochs=100, seed=42):
    import torch
    torch.manual_seed(seed)
    rows = [r for r in rows if r["count"] > 0 and not r.get("excluded_duplicate", False)]
    family = lambda r: r["task_id"].rsplit("-", 1)[0]
    groups = sorted({family(r) for r in rows})
    if len(groups) < 2:
        raise ValueError("Action-value fitting requires at least two task groups for validation")
    validation_group = groups[-1]
    train = [r for r in rows if family(r) != validation_group]
    val = [r for r in rows if family(r) == validation_group]
    if sum(r["successes"] for r in train) in (0, sum(r["count"] for r in train)):
        raise ValueError("Training continuations have no outcome variation; calibrate tasks before fitting action value")
    model = ActionValue()
    x = torch.stack([features(r["prompt"], r["action"]) for r in train])
    count = torch.tensor([r["count"] for r in train], dtype=torch.float32)
    target = torch.tensor([r["successes"] / r["count"] for r in train])
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model.model(x).squeeze(-1)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * count).sum() / count.sum()
        loss.backward()
        optimizer.step()
    brier = sum((model.predict(r["prompt"], r["action"]) - r["successes"] / r["count"])**2 for r in val) / len(val)
    family_counts = {}
    for row in rows:
        name = family(row)
        bucket = family_counts.setdefault(name, {"actions": 0, "successes": 0, "trials": 0})
        bucket["actions"] += 1
        bucket["successes"] += row["successes"]
        bucket["trials"] += row["count"]
    metadata = {"validation_task": validation_group, "validation_brier_vs_empirical_rate": brier,
                "training_examples": len(train), "validation_examples": len(val),
                "dataset_sha256": dataset_sha256(rows), "family_counts": family_counts,
                "note": "Small hashed-feature baseline; empirical rates are noisy, not independent ground-truth probabilities"}
    model.save(output, metadata)
    return model, metadata


def load(path, rows=None):
    import torch
    saved = json.loads(Path(path).read_text())
    if saved['features'] != FEATURES:
        raise ValueError('Action-value feature schema mismatch')
    recorded = saved.get('metadata', {}).get('dataset_sha256')
    if rows is not None and recorded != dataset_sha256(rows):
        raise ValueError('Action-value model does not match the current value dataset; refit it before training')
    value = ActionValue()
    value.model.load_state_dict({k: torch.tensor(v) for k, v in saved['state'].items()})
    value.model.eval()
    return value
