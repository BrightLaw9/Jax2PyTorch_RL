# MiniPort Process-Reward RL Pilot for Long-Horizon Coding Agents

## Goal

Build a 1–2 day research pilot answering:

> Can execution-grounded process rewards reduce premature termination and unproductive exploration in framework-porting coding agents?

The project should produce a clear before/after comparison on held-out JAX→PyTorch porting tasks. It is inspired by Abundant AI’s SWE-Marathon `jax-pytorch-rewrite` task, but does not attempt to train on or fully solve that official task.

## Why this task

SWE-Marathon’s `jax-pytorch-rewrite` task asks an agent to port a renamed JAX vision-language-action policy to PyTorch, establish topology/layer/E2E/deterministic-sampling parity, and then optimize latency. Correctness is required before latency contributes to reward.

It has clear failure modes: wrong parameter layouts, repeated debugging loops, regression after correct changes, premature optimization, and premature termination.

Do not use the Rust C compiler task as the primary project. It is better as a future reward-hacking/integrity study, but is too broad for a short RL pilot.

Relevant sources:

- SWE-Marathon repo: https://github.com/abundant-ai/swe-marathon
- Task metadata: https://raw.githubusercontent.com/abundant-ai/swe-marathon/main/tasks/jax-pytorch-rewrite/task.toml
- Paper: https://www.swe-marathon.org/swe-marathon-paper.pdf

## Scope: MiniPort

Build a small local benchmark named `MiniPort`. Do not include the A100 latency phase in the first version. Focus on numerical correctness and deterministic behavior on CPU or ordinary CUDA.

### Six task templates

1. MLP port: linear-weight transpose and bias layout
2. CNN port: NHWC/NCHW and convolution-kernel layout
3. LayerNorm/residual block: epsilon, broadcasting, and dtype behavior
4. Attention block: packed QKV, head reshape, scale, and masking
5. RoPE/cache task: position offsets and KV-cache ordering
6. Deterministic sampling: JAX PRNG splitting versus PyTorch RNG behavior

For each template, generate four configuration variants.

- Training: 18 variants total
- Held-out: 6 variants total with unseen dimensions, layer ordering, or composition
- Every variant contains a JAX reference implementation, incomplete PyTorch stub, visible staged verifier, and hidden-seed final verifier.

### Required gate order

1. Imports and topology
2. Parameter loading/mapping
3. Layer parity
4. End-to-end parity
5. Deterministic sampling
6. Optional latency, excluded from the initial pilot

The final held-out evaluation must use hidden seeds/configurations never shown during RL.

## Model, hardware, and resource limits

Use `Qwen/Qwen3-4B-Instruct-2507` as the policy. It is a dense 4B open-weight model that can be fine-tuned locally while still making enough structured mistakes to evaluate process rewards.

Available GPU memory is limited to approximately **10–16 GB** on an RTX 6000. Treat this as the hard operating constraint.

- Load model in NF4 4-bit quantization with QLoRA.
- LoRA rank: `8`; use a small adapter rather than full fine-tuning.
- Context cap: `4K` initially; increase to `8K` only if memory permits.
- Batch size: `1`; gradient accumulation: `8–16`.
- Rollouts: sequential only; never hold several model rollouts concurrently in GPU memory.
- Maximum actions: `16–20` per task.
- Maximum model output per action: `1,024` tokens.
- Temperature: `0.6`.
- Keep prompt, tools, action budget, context cap, seeds, and verifier version identical for baseline and PRM-RL evaluation.

Do not train Muse Glimmer-30B, Kimi, or Z.ai/GLM models locally under this memory budget. They may be used only as optional external reference agents, never as one side of the trained before/after comparison.

Stop and reduce context or output length if CUDA out-of-memory occurs; do not solve this by silently changing one experimental condition.

## Sealed evaluation and ground truth

Input/output parity alone is insufficient because an agent could hard-code outputs, call JAX from the purported PyTorch port, modify tests, or cache answers. A valid final solution must satisfy:

`hidden_parity_passes AND fresh_randomization_passes AND PyTorch_only_passes AND integrity_audit_passes`

The agent may write only to `/workspace/submission/`. Reference JAX code and hidden tests live outside the agent container. At each check, the evaluator copies only the submission into a separate verifier process, mounts it read-only, and returns a compact diagnostic summary. At final grading, use a fresh container and never copy agent-modified tests, caches, shell history, or the rest of the workspace.

### Automated annotation protocol

After every meaningful checkpoint (edit, test run, profiler run, or stop action), an external evaluator records:

- Workspace fingerprint: hashes of allow-listed source/configuration files.
- Normalized action/command signature.
- Normalized result signature: gate, error type, pass count, and numerical-parity metric.
- Current and furthest-reached gate.
- Diff summary and changed-file list.

The evaluator returns only a protected summary such as:

```json
{
  "current_gate": "layer_parity",
  "max_abs_error": 0.038,
  "previous_error": 0.091,
  "new_failure_location": "attention.q_proj",
  "gate_advanced": false
}
```

Automatic labels:

- **Gate progress:** a mandatory verifier gate passes.
- **Numerical progress:** parity error decreases by a thresholded amount.
- **Regression:** a previously passed gate now fails or error materially worsens.
- **No-progress repetition:** unchanged workspace, repeated normalized command, same normalized result, and no new diagnostic.
- **Premature optimization:** profiler/optimization work before all correctness gates pass.
- **Premature termination:** the agent stops with mandatory visible gates unresolved, budget remaining, and an actionable visible failure.

Fresh randomized weights, inputs, seeds, batch sizes, and eligible sequence lengths are generated only after submission. Grade candidate and JAX reference in separate processes. Reject forbidden imports (`jax`, `flax`, reference packages), dynamic imports, subprocess delegation, test modification, altered test runners, network access, fixed-output lookup tables, and cache-based output replay.

## Human review process and interface

Humans do not decide technical correctness. The protected verifier is the source of capability ground truth. Human review has two narrow purposes: label ambiguous workflow quality for the PRM and audit final passing artifacts for shortcuts.

### Process annotation

Label only 30–40 ambiguous, high-information checkpoints after baseline rollouts. Do not label chain-of-thought.

The local annotation page contains:

- **Left panel:** task requirement and current required verifier gate.
- **Center panel:** changed files and compact code diff.
- **Right panel:** protected before/after parity metrics, test result, and new diagnostic if any.
- **Bottom panel:** last 3–5 observable tool actions, proposed next action or stop action, and remaining action budget.

The reviewer selects one workflow label: `useful diagnosis`, `genuine implementation progress`, `repetition/waste`, `regression`, `premature optimization`, `premature termination`, or `unclear`. They may select `unclear` whenever framework details are required to judge the action.

### Final artifact audit

Review only submissions that already pass sealed hidden parity tests. This should be a small number of final candidates.

Show the reviewer the submitted source files, diff against the stub, external verifier summary, static-scan report, and observable action log. Do not show reference code or hidden-test details.

Mark a candidate **valid**, **invalid**, or **unclear** against this binary rubric:

| Check | Valid when |
| --- | --- |
| PyTorch-only | No JAX, Flax, reference-model, subprocess, remote-delegation, or dynamic-import use. |
| Generic implementation | No hard-coded output tensors, seed-specific lookup tables, or input fingerprints. |
| Generic parameter mapping | Mapping follows parameter structure/shapes rather than fixed parameter values. |
| Test integrity | No test edits, monkey-patching, altered environment, or verifier tampering. |
| Sensible artifact | Submitted code instantiates and executes a PyTorch model rather than forwarding or replaying outputs. |

Do not reject a novel or ugly implementation merely because it differs from an expected implementation. Reject only concrete shortcut evidence. An `unclear` audit is reviewed a second time or excluded from reported aggregate results.

## Process reward

Score each action checkpoint: code edit, test run, profiler run, or stop/final action.

`r_t = r_exec + 0.25 × r_PRM`

### Terminal reward

- Hidden final verifier pass: `+10`
- Hidden final verifier fail: `0`
- Stop/final action while mandatory visible gates remain unresolved: `-2`

### Automatic execution rewards

| Event | Reward |
| --- | ---: |
| New required gate reached | +1.0 |
| Current numerical parity error improves | +0.1 to +0.5 |
| Previously passed gate regresses | -0.75 |
| Uninformative repeated action | -0.25 |
| Edit immediately reverted / no useful diagnostic | -0.15 |
| Optimization/profiling before correctness gates pass | -0.5 |
| Focused diagnostic after a failure | +0.15 |

Use a bounded log ratio for numerical improvement:

`0.25 × clamp(log(error_before / error_after), -1, 1)`

The hidden final verifier is evaluation-only and remains the dominant reward signal.

### Detecting no-progress actions

Store after every action:

- Workspace fingerprint: hashes of relevant code/config files
- Normalized command/action signature
- Normalized result signature: test name, error type, pass/fail count, parity metric
- Current gate index

Flag an action as no-progress when all are true:

1. It does not change relevant files.
2. It repeats a normalized command in the same workspace state.
3. It produces the same normalized output/result signature.
4. It does not expose a new failure location or diagnostic.

Re-running a test after source changes is not repetition. A test that reveals a new mismatch is useful even if the scalar score does not improve. Automatic flags are candidate labels; human annotations resolve ambiguous cases.

### Detecting premature termination

Flag a stop/final action when:

- At least one mandatory visible gate is unresolved.
- Action/time budget remains.
- The latest visible test points to an actionable failure, not an environment outage.

## Process reward model

PRM input must use observable evidence only, not chain-of-thought:

- Task and current required gate
- Last 3–5 tool actions and outputs
- Diff summary
- Before/after visible test and parity metrics
- Proposed next action or stop/final action

Human targets are ordinal:

- `-2`: strongly harmful
- `-1`: harmful
- `0`: neutral/unclear
- `+1`: useful
- `+2`: strongly useful

Train an ordinal classifier or pairwise preference model, normalize its prediction to `[-1, 1]`, and use it only as the small PRM reward component.

## Annotation workflow

Annotate 30–40 selected checkpoints, not every action. Sample after test failures, major edits, strategy pivots, repeated actions, optimization attempts, and stop/final attempts.

Build a minimal local review page or notebook UI:

- Left: task requirements and current gate
- Center: changed files and diff
- Right: before/after tests and parity metrics
- Bottom: recent actions, proposed next action, and remaining budget

Reviewer selects:

1. Progress label: `-2, -1, 0, +1, +2`
2. Primary reason: useful diagnosis, genuine implementation progress, regression, repetition, premature optimization, premature termination, or unclear
3. For stop actions: appropriate stop or should continue

Use only observable tool actions, code diffs, and execution output.

## Experiment

1. Run the baseline agent on MiniPort training variants.
2. Select and annotate checkpoints.
3. Train the PRM.
4. Collect grouped rollouts on training variants.
5. Run one small LoRA GRPO/RLOO-style update using the combined reward.
6. Evaluate baseline and PRM-RL versions on held-out variants.

Keep model, prompt, action budget, seeds, verifier version, and environment identical between baseline and RL evaluation.

## Required metrics

- Held-out hidden-test pass rate
- Furthest gate reached
- Premature-termination rate
- No-progress actions per rollout
- Regression rate after passing a gate
- Tokens/actions/wall time per successful task
- Qualitative trajectory examples showing baseline vs. PRM-RL behavior

This is a pilot: do not claim it solves SWE-Marathon or proves broad transfer. A valid result is improved gate completion and reduced wasted exploration even if final held-out pass rate remains low.

## Optional final transfer

After MiniPort is complete, run frozen baseline and PRM-RL policies on SWE-Marathon `jax-pytorch-rewrite` under the same budget. Report it only as exploratory transfer. Do not train on official-task outputs, hidden evaluator feedback, or manually solve the official task before the comparison.
