"""Single-policy, sequential RLOO gradients. Only generated action tokens are scored."""
import torch


def leave_one_out(rewards):
    if len(rewards) < 2:
        raise ValueError("RLOO requires at least two valid trajectories")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def action_log_probs(model, completion, temperature=1.0):
    if not completion.prompt_ids or not completion.output_ids:
        raise ValueError("Prompt and action tokens must be nonempty")
    device = next(model.parameters()).device
    tokens = torch.tensor([completion.prompt_ids + completion.output_ids], device=device)
    n = len(completion.output_ids)
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                   use_cache=False, logits_to_keep=n + 1).logits
    # Qwen logits_to_keep returns last n+1 positions. Full-logit test models are
    # also accepted: the same suffix scores only continuation tokens.
    logits = logits[:, -(n + 1):-1, :] / temperature
    targets = tokens[:, -n:]
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    # Small chunks avoid materializing a full float32 [context,vocabulary] copy.
    denominators = torch.cat([torch.logsumexp(part.float(), -1) for part in logits.split(128, dim=1)], dim=1)
    return (chosen - denominators).squeeze(0)


def accumulate(model, trajectories, *, condition, kl_coefficient, value_coefficient=0.25,
               immediate_coefficient=0.1, accumulation_groups=1, scaler=None, temperature=1.0):
    trajectories = [t for t in trajectories if t["trainable"] and t["reward"] is not None]
    if len(trajectories) < 2:
        return {"used": 0, "loss": 0.0}
    def diagnostic_credit(trajectory):
        if condition == "immediate":
            return sum(record["diagnostic_reward"]["total"] for record in trajectory["records"])
        if condition == "action_value":
            # Preserve terminal correctness as the primary signal while making
            # observable regressions and repeated no-ops costly. Positive gate
            # shaping remains exclusive to the immediate-reward ablation.
            return sum(min(0.0, value) for record in trajectory["records"]
                       for value in record["diagnostic_reward"]["events"].values())
        return 0.0
    rewards = [t["reward"] + immediate_coefficient * diagnostic_credit(t) for t in trajectories]
    advantages = leave_one_out(rewards)
    total_loss = 0.0
    for trajectory, advantage in zip(trajectories, advantages):
        total_tokens = sum(len(s["completion"].output_ids) for s in trajectory["steps"])
        if not total_tokens:
            continue
        for step in trajectory["steps"]:
            completion = step["completion"]
            if not completion.output_ids:
                continue
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                reference = action_log_probs(model, completion, temperature)
            model.train()  # Enables gradient checkpointing; all policy dropout is zero.
            logp = action_log_probs(model, completion, temperature)
            delta = reference - logp
            kl = delta.exp() - delta - 1
            credit = advantage + (value_coefficient * step["value_advantage"] if condition == "action_value" else 0)
            loss = (-credit * logp + kl_coefficient * kl).sum() / (total_tokens * len(trajectories) * accumulation_groups)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite policy loss; aborting update")
            (scaler.scale(loss) if scaler else loss).backward()
            total_loss += loss.detach().item()
    return {"used": len(trajectories), "loss": total_loss, "rewards": rewards}
