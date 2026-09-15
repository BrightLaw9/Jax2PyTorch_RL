"""Opt-in tiny real Qwen/PEFT export and native MLX smoke test; no model download."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("MINIPORT_MLX_TESTS") == "1", "Enable native MLX/PEFT export test")
class ExportMLXTests(unittest.TestCase):
    def test_tiny_adapter_merge_conversion_and_apple_gpu_generation(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from peft import LoraConfig, get_peft_model
        from miniport.export_model import export_pair
        from miniport.mac import convert_pair
        from miniport.policy import MLXPolicy, SYSTEM_PROMPT
        from miniport.experiment import ExperimentConfig
        from miniport.tasks import VERIFIER_VERSION, training_tasks
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(42)
            config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, max_position_embeddings=4096, eos_token_id=1, pad_token_id=0)
            model = Qwen3ForCausalLM(config)
            base = root / "tiny-base"
            model.save_pretrained(base, safe_serialization=True)
            raw = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "hello": 3}, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
            tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
            tokenizer.save_pretrained(base)
            adapted = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            # A real tiny adapter update tests the same token-masked loss and PEFT
            # context manager as training, without claiming a 4B/CUDA experiment.
            from miniport.rl import accumulate
            from miniport.policy import Completion
            parameters = [p for p in adapted.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=0.001)
            trajectories = [{"trainable": True, "reward": reward, "records": [],
                "steps": [{"completion": Completion("", "", [3, 2], [token]), "value_advantage": 0}]}
                for reward, token in ((1, 4), (0, 5))]
            accumulate(adapted, trajectories, condition="terminal", kl_coefficient=0.01)
            self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters if p.grad is not None), 0)
            optimizer.step()
            checkpoint = root / "checkpoint"
            adapted.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint / "tokenizer")
            metadata = {"base_model": str(base), "base_revision": "local-fixture",
                "completed_updates": 1, "experiment": asdict(ExperimentConfig()),
                "verifier_version": VERIFIER_VERSION,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
            (checkpoint / "manifest.json").write_text(json.dumps(metadata))
            export_pair(checkpoint, root / "export")
            self.assertTrue((root / "export" / "COMPLETE").exists())
            from transformers import AutoModelForCausalLM
            merged = AutoModelForCausalLM.from_pretrained(root / "export" / "trained-hf", torch_dtype=torch.float32)
            adapted.eval()
            with torch.no_grad():
                x = torch.tensor([[3, 2, 3]])
                expected = adapted(x).logits
                actual = merged(x).logits
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
            convert_pair(root / "export", root / "mlx")
            policy = MLXPolicy(root / "mlx" / "trained", ExperimentConfig(max_new_tokens=128))
            completion = policy.generate(training_tasks()[0], "import torch\n", [], 18, 42)
            self.assertGreater(len(completion.output_ids), 0)
            self.assertLessEqual(len(completion.output_ids), 128)
            self.assertLessEqual(len(completion.prompt_ids) + len(completion.output_ids), 4096)

    @unittest.skipUnless(os.environ.get("MINIPORT_DOCKER_TESTS") == "1", "Enable Docker for Mac evaluation smoke test")
    def test_mac_evaluation_runner_uses_docker_and_holds_audit_gate(self):
        from unittest.mock import patch
        from miniport.mac import evaluate
        from miniport.experiment import ExperimentConfig
        from miniport.policy import Completion, SYSTEM_PROMPT
        from miniport.provenance import source_hash
        from miniport.tasks import heldout_tasks, VERIFIER_VERSION
        oracle = (Path(__file__).parent / "fixtures" / "correct_port.py").read_text()
        class ScriptedPolicy:
            def __init__(self, path, config):
                self.config = config
            def generate(self, task, source, history, remaining, seed):
                action = {"type": "stop"} if source == oracle else {"type": "edit", "source": oracle}
                return Completion("test-prompt", json.dumps(action), [1], [2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "export-manifest.json").write_text(json.dumps({"experiment": asdict(ExperimentConfig()),
                "verifier_version": VERIFIER_VERSION, "source_sha256": source_hash(),
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}))
            manifest = root / "private.json"
            manifest.write_text(json.dumps({"version": VERIFIER_VERSION,
                "tasks": [heldout_tasks()[0].to_dict()], "seeds": [101, 211]}))
            with patch("miniport.policy.MLXPolicy", ScriptedPolicy):
                evaluate(model, manifest, root / "evaluation", root / "challenges")
            summary = json.loads((root / "evaluation" / "summary.json").read_text())
            self.assertEqual(summary["automated_passes"], 1)
            self.assertEqual(summary["audited_valid"], 0)
            report = json.loads((root / "evaluation" / "mlp-heldout" / "evaluation.json").read_text())
            self.assertEqual(report["isolation"], "docker")
