"""Shared action protocol; CUDA training and native MLX inference implementations."""
from dataclasses import dataclass
import json

SYSTEM_PROMPT = '''You are implementing a numerical PyTorch port. Return exactly one JSON object, no markdown or explanation.
Every edit is tested automatically and receives verifier feedback. Emit one action and end your response. Never append a test or stop action after an edit.
Actions: {"type":"edit","source":"complete replacement Python source"}, {"type":"test"}, or {"type":"stop"}.
For edit actions, source must contain only valid, executable Python code that can be saved directly as candidate.py and imported by the verifier. Never include Markdown code fences (``` or ~~~), language tags, or explanatory prose inside source. Do not wrap the JSON object in code fences either.
The source must implement build(config), model.topology(), load_parameters(numpy_parameters), export_parameters(),
and run(torch_input, torch_uniform_draws) returning layer/output tensors. RoPE/cache additionally needs run_cached.
Only candidate.py can be changed. Use PyTorch and ordinary Python utilities; allowed imports are torch, math, typing, dataclasses, collections, functools, itertools, operator, enum, abc, numbers and copy. No files, network, dynamic source execution or JAX.
Use the task equations and visible feedback. If the latest feedback.status is "pass", return exactly {"type":"stop"}; do not edit or test again. Otherwise repair the failure. Invalid JSON consumes an action.
The JAX code in comments is reference documentation, not content to preserve. Implement only the requested task template. In edit.source omit the reference comments, unused task classes, and interface docstring. Replace the placeholder build(config) with a working PyTorch implementation. Keep the complete JSON response concise enough to finish within the output token limit.'''


@dataclass
class Completion:
    prompt: str
    text: str
    prompt_ids: list
    output_ids: list


def render_prompt(tokenizer, task, source, history, remaining, config):
    from .requirements import task_text
    latest_feedback = next((message["content"] for message in reversed(history)
                            if message["role"] == "user"), None)
    try:
        passed = json.loads(latest_feedback or "{}").get("feedback", {}).get("status") == "pass"
    except (ValueError, AttributeError):
        passed = False
    instruction = ('All visible checks passed. Return exactly {"type":"stop"}. Do not edit or test again.'
                   if passed else
                   f"Your next action is for {task.id} ONLY.\n"
                   f"Implement only {task.template}; the other reference classes are unrelated to this task. "
                   "Omit all reference comments and the interface docstring from your replacement source. "
                   "Return one complete JSON action object, starting with { and ending with }. "
                   "For an edit, put concise, executable PyTorch source in the source string, including build(config). "
                   "Edits are tested automatically. Emit one action and end the response; do not append test or stop. "
                   "Do not output a Python file directly or use Markdown fences.")
    current = (task_text(task) + "\nConfiguration: " + json.dumps(task.to_dict(), sort_keys=True)
               + f"\nRemaining actions: {remaining}\nCurrent candidate.py:\n{source}"
               + "\nEND OF CURRENT FILE. "
               + (f"\nLatest action feedback:\n{latest_feedback}\n" if latest_feedback else "")
               + instruction)
    recent = list(history[-6:])
    while True:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(recent)
        messages.append({"role": "user", "content": current})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + config.max_new_tokens <= config.context_cap:
            return prompt, ids
        if recent:
            recent = recent[2:]
        else:
            raise ValueError("Task and current source exceed context budget; restart all conditions with revised limits")


class HFPolicy:
    def __init__(self, config, revision, checkpoint=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU training; use MLXPolicy for Mac evaluation")
        self.config, self.revision = config, revision
        torch.manual_seed(config.seed)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(config.gpu_memory_gb * 2**30 / total, 1.0), 0)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint / "tokenizer") if checkpoint else config.model,
            **({} if checkpoint else {"revision": revision}), trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(config.model, revision=revision,
                quantization_config=quant, torch_dtype=self.dtype, device_map={"": 0},
                attn_implementation="sdpa", trust_remote_code=False)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True) if checkpoint else get_peft_model(base, LoraConfig(r=config.lora_rank, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        self.model.config.use_cache = False
        self.model.eval()

    def generate(self, task, source, history, remaining, seed):
        import torch
        from transformers import StoppingCriteriaList
        from .action_protocol import StopAfterAction
        prompt, ids = render_prompt(self.tokenizer, task, source, history, remaining, self.config)
        torch.manual_seed(seed)
        self.model.eval()
        with torch.no_grad():
            tokens = self.model.generate(input_ids=torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
                max_new_tokens=self.config.max_new_tokens, do_sample=True,
                temperature=self.config.temperature, top_p=1.0, top_k=0,
                pad_token_id=self.tokenizer.eos_token_id, use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopAfterAction(self.tokenizer, len(ids))]))
        output = tokens[0, len(ids):].tolist()
        return Completion(prompt, self.tokenizer.decode(output, skip_special_tokens=True), ids, output)


class MLXPolicy:
    def __init__(self, model_path, config):
        import mlx.core as mx
        from mlx_lm import load
        if not mx.metal.is_available():
            raise RuntimeError("MLX evaluation requires the Apple GPU")
        mx.set_default_device(mx.gpu)
        self.model, self.tokenizer = load(str(model_path))
        self.config = config

    def generate(self, task, source, history, remaining, seed):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt, ids = render_prompt(self.tokenizer, task, source, history, remaining, self.config)
        mx.random.seed(seed)
        pieces, output = [], []
        for response in stream_generate(self.model, self.tokenizer, prompt=ids,
                max_tokens=self.config.max_new_tokens, sampler=make_sampler(temp=self.config.temperature, top_p=1.0)):
            pieces.append(response.text)
            output.append(int(response.token))
            from .action_protocol import complete_object
            if complete_object("".join(pieces)):
                break
        return Completion(prompt, "".join(pieces), ids, output)
