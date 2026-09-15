"""Complete public JAX source embedded as comments in the candidate stub."""
from pathlib import Path


def jax_source(task):
    return Path(__file__).with_name("jax_interface.py.txt").read_text()


def starter_source(task, source=None):
    if source is None:
        source = Path(__file__).with_name("stub.py.txt").read_text()
    if source.startswith("# Complete JAX source to port"):
        return source
    header = (
        "Complete JAX source to port: all templates and parameter mapping.\n"
        "Documentation only; implement the candidate with PyTorch.\n"
        "Entry point: build(config). Its model exposes the exact verifier method signatures.\n"
        "Port JAX array results to torch.Tensor results; run inputs are torch tensors in the verifier.\n"
    )
    reference = "\n".join("# " + line if line else "#" for line in (header + jax_source(task)).splitlines())
    return reference + "\n\n" + source
