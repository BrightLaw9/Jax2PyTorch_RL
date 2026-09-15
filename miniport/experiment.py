"""Validated resource limits and paired-condition identity; no GPU allocation."""
from dataclasses import asdict, dataclass
import hashlib
import json

from .tasks import VERIFIER_VERSION


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization: str = "nf4"
    lora_rank: int = 8
    context_cap: int = 4096
    batch_size: int = 1
    gradient_accumulation: int = 8
    concurrent_rollouts: int = 1
    max_actions: int = 18
    max_new_tokens: int = 1024
    temperature: float = 0.6
    gpu_memory_gb: int = 10
    seed: int = 42
    verifier_version: str = VERIFIER_VERSION
    task_limit: int | None = None

    def validate(self):
        if self.task_limit is not None and (type(self.task_limit) is not int or not 1 <= self.task_limit <= 6):
            raise ValueError("task_limit must be null or an integer from 1 to 6")
        if self.model != "Qwen/Qwen3-4B-Instruct-2507" or self.quantization != "nf4" or self.lora_rank != 8:
            raise ValueError("Pilot requires Qwen3-4B, NF4, rank-8 QLoRA")
        if self.context_cap not in (2048, 4096, 8192) or not 128 <= self.max_new_tokens <= 1024:
            raise ValueError("Invalid context/output budget")
        if self.batch_size != 1 or self.concurrent_rollouts != 1 or not 8 <= self.gradient_accumulation <= 16:
            raise ValueError("Use batch 1, sequential rollouts and accumulation 8–16")
        if not 16 <= self.max_actions <= 20 or not 10 <= self.gpu_memory_gb <= 16:
            raise ValueError("Action or GPU memory budget exceeds pilot limits")
        if self.temperature != 0.6 or self.verifier_version != VERIFIER_VERSION:
            raise ValueError("Temperature/verifier mismatch")
        return self

    def identity(self, prompt, tools):
        self.validate()
        payload = {"config": asdict(self), "prompt": prompt, "tools": tools}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()



def assert_paired(baseline, trained):
    if baseline != trained:
        raise ValueError("Experimental conditions differ; restart both conditions with the same configuration")
