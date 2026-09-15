"""Portable source identity, independent of checkout directory and absent Git metadata."""
import hashlib
from pathlib import Path


def source_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.glob("*.py"), *root.glob("*.txt")]):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
