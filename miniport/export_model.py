"""Export exact-base baseline and merged LoRA checkpoints for transfer to macOS."""
import argparse
import gc
import json
from pathlib import Path
import shutil


def export_pair(checkpoint, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    checkpoint, output = Path(checkpoint), Path(output)
    metadata = json.loads((checkpoint / "manifest.json").read_text())
    if metadata.get("completed_updates", 0) < 1 or metadata.get("interrupted"):
        raise ValueError("Checkpoint has no completed training update or was interrupted")
    if output.exists():
        raise ValueError("Use a fresh export directory")
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", trust_remote_code=False)
    for label in ("baseline", "trained"):
        # Reload original floating-point weights: never merge into a packed NF4 base.
        model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], revision=metadata["base_revision"],
            torch_dtype=torch.float16, device_map={"": "cpu"}, low_cpu_mem_usage=True, trust_remote_code=False)
        if label == "trained":
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False).merge_and_unload(safe_merge=True)
        target = output / f"{label}-hf"
        model.save_pretrained(target, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(target)
        (target / "export-manifest.json").write_text(json.dumps(dict(metadata, condition=label,
            export_dtype="float16", export_quantization=None), indent=2))
        del model
        gc.collect()
    shutil.copyfile(checkpoint / "manifest.json", output / "training-manifest.json")
    if (checkpoint / "requirements-resolved.txt").exists():
        shutil.copyfile(checkpoint / "requirements-resolved.txt", output / "training-requirements.txt")
    (output / "COMPLETE").write_text("Both baseline and trained checkpoints exported.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    export_pair(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
