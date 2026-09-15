# MiniPort: Outcome-Grounded Action Value for Coding Agents

## Research question

Does learning longer-term action value add anything beyond immediate verifier
rewards when training a coding agent?

Restore the original MiniPort benchmark/RL direction. Replace the proposed small
human-labeled PRM with an outcome-grounded action-value model trained from verified
outcomes of multiple continuations after an action. The feedback-ablation study
is discontinued. The earlier full PRM pilot is preserved in
`docs/plan-prm-pilot.md` for historical context; this document is the active plan.

## What stays fixed

- MiniPort's six JAX-to-PyTorch templates: MLP, CNN, LayerNorm/residual, attention,
  RoPE/cache, and deterministic sampling.
- Ordered correctness gates and PyTorch-only admission. Training checks use CPU
  subprocesses in the GPU server venv; Mac evaluation uses sealed Docker execution,
  fresh final randomization, and human artifact audits of final numerical passers.
- Qwen/Qwen3-4B-Instruct-2507 with NF4 4-bit QLoRA and rank-8 adapters.
- 10–16 GB GPU budget; 4K context, at most 1,024 output tokens per action,
  18 actions, temperature 0.6, batch 1, gradient accumulation 8–16.
- Sequential rollouts. Freeze model/prompt/tools/budgets/cases/image versions
  within matched comparisons. OOM changes must be applied to all affected arms.
- Identical structured visible feedback in every experimental condition.

## Conditions to compare

| Condition | Training signal | Purpose |
| --- | --- | --- |
| Frozen baseline | No policy update | Establish starting capability |
| Terminal-only RL | Verified terminal training-task success | Outcome-supervision baseline |
| Immediate-verifier RL | Terminal outcome plus current gate/error/repetition signals | Existing process-reward baseline |
| Outcome-grounded action-value RL | Terminal outcome plus estimated longer-term action advantage | Test whether learned longer-term credit adds value |

Start with the first three and validate action-value predictions offline before
adding the fourth. Add a fourth-arm ablation without immediate shaping if the
final combined method includes it. Use the same policy-visible feedback; reward
and credit assignment are the treatment, not diagnostic information disclosure.

## Action-value supervision

1. Collect baseline trajectories on TRAINING tasks.
2. Select checkpoints with failures, substantial edits, temporary regressions,
   repeated actions or stopping decisions. Save full source, observable history,
   remaining budget, task identity and RNG/continuation-policy metadata.
3. From the same checkpoint, sample two candidate actions initially.
4. Fork the exact saved state, execute one candidate action per branch, then run
   four independent continuations per action initially. Keep the continuation
   policy frozen and versioned; use matched continuation seed schedules and equal
   post-action budgets. Reset all mutable state between continuations.
5. Grade terminal submissions against the same CPU-subprocess TRAINING-task outcome
   protocol. Failed continuations are zero outcomes; outages are invalid samples.
6. For each state/action pair, retain success count, valid continuation count,
   uncertainty, action cost and remaining budget. Never store only a hard label.

Q_pi(s,a,b) estimates the probability of verified success after taking action a
in state s with budget b and continuing under the frozen policy pi. It is not
universal action correctness. The history/observations and budget are part of the
state; an action can be useful despite temporarily worsening a visible metric.

A 3/4 versus 1/4 success difference is noisy evidence, not a certain preference.
Allocate additional continuations to informative uncertain comparisons within a
predeclared budget. Keep independent validation continuations for checking the
learned model, rather than evaluating predictions against the same noisy labels.

Human review audits a small set for shortcuts, inconsistent branch states and
misleading labels. It is not the main training-label source. Do not collect
chain-of-thought annotations.

## Learned model and integration

Begin with a small supervised action-value predictor using observable task,
source/diff, recent tool evidence, candidate action and remaining budget. Choose
its capacity based on collected data and compare against simple gate/error feature
baselines. The initial implementation uses a small CPU MLP over hashed state/action token
bigrams. It is a lightweight baseline, not a pretrained code-understanding critic;
stronger architecture selection remains a research question.

Train with the verified success/failure counts (e.g. count-weighted binomial loss),
or uncertainty-aware action preferences. Split by task/source/bug family before
checkpoint extraction; never scatter adjacent checkpoints across train/test.

First measure out-of-family action ranking, calibration, and performance on cases
where immediate verifier progress disagrees with continuation outcomes. If the
model cannot beat simple immediate signals, do not proceed to policy optimization
against it.

For RL, convert predicted Q values to a centered action-advantage signal relative
to actions sampled from the SAME checkpoint, e.g. Q_hat(s,a) minus the mean over
sampled alternatives. Do not add raw predicted success probability at every step,
which can reward lingering in already-promising states. Freeze the value model
within an experiment round and refresh data explicitly if the policy changes.

Use one shared sequential RLOO-style QLoRA implementation. Keep the terminal
objective, KL regularization, optimizer, group budgets and loss conventions fixed;
change only the specified credit signal. Fix the action-value coefficient using
development tasks, not final held-out results. The precise estimator and loss
must be validated before any substantive training claim.

## Outcomes and evaluation

Terminal TRAINING reward: +1 for automated sealed randomized success, otherwise 0.
The existing recorder supplies this independently of visible-test pass/fail.
Process scores remain available separately for the immediate-verifier condition.
Training outcome is an automated proxy; held-out final validity also requires the
artifact audit. Final held-out cases, audits and diagnostics never supervise
training or action-value labels.

Expand beyond the existing 18 training and 6 held-out calibration variants with
interacting bugs and compositions before broad claims. Establish that the baseline
has nontrivial successes and failures and that tasks require meaningful multi-step
repairs. Split by task/bug/source-construction family, not just random inputs.

Primary result: audited held-out success under a fixed inference budget.
Secondary: gate completion, regression/repetition and premature-stop rates,
actions/tokens/verifier calls per success, and disagreement cases where immediate
progress is misleading. Report uncertainty by task families and training seeds.

Count ALL continuation-generation, value-model training and inference compute.
Compare both at matched policy-update budgets and with a baseline granted the same
total additional compute. Sequential sampling reduces memory, not total runtime.
A starting collection of 50 checkpoints x 2 actions x 4 continuations already
requires 400 continuation rollouts; measure throughput before scaling up.

## Implementation milestones

1. Implemented: Qwen integration and persisted model-driven trajectories/tokens.
2. Implemented: exact source/history snapshot restoration and sequential branches.
3. Locally tested: branch independence, budget handling and training-only labels.
4. Implemented: continuation dataset collection and a small predictive baseline.
5. Implemented: value fitting/validation and its advantage in policy updates;
   predictive quality on real 4B rollouts remains unmeasured.
6. Remaining: full CUDA validation and paired terminal-only, immediate-verifier
   and learned-value experiments with uncertainty and compute accounting.

The repository now includes sequential branch collection, a small hashed-feature
action-value baseline, a CUDA policy wrapper and RLOO-style LoRA updates. These have
local unit/smoke coverage; full 4B CUDA training still needs cluster validation. No feedback-removal
experiment or human-trained 30–40-label PRM is on the active critical path.

## Execution architecture — user-approved GPU venv / Mac MLX deployment

This section supersedes earlier Docker-on-GPU assumptions. GPU training runs on a
bare-metal Slurm node entirely in a Python venv, with NO Docker installation or
Docker calls. Candidate verification runs in CPU subprocesses from that venv.
This deliberately provides weaker isolation; record `isolation=subprocess` in
training provenance and do not describe training checks as sealed. Static checks,
separate source snapshots, output limits and timeouts remain enabled. GPU visibility
is removed from verifier children, and JAX reference execution is forced to CPU.

Slurm entry points create a venv, install platform-specific pinned dependencies,
check CUDA, resolve an immutable base-model revision, collect baseline/continuation
outcomes, fit the action-value model, and run sequential adapter policy updates.
The host RAM request is capped at 16 GB per the user's limit (separate from VRAM); adjust partition,
module loads and resource names for the cluster. The logs directory must exist
before sbatch. Network/model-cache access is needed for first-time setup.
Full-size CPU export must be measured under this same 16 GB limit. Sequential
low-memory loading reduces peak use but does not guarantee the merge fits; if it
fails, preserve the adapter and perform the merge on a machine with sufficient RAM.

Checkpoint handoff:
1. Save adapter safetensors, exact base-model revision, tokenizer, full run config,
   optimizer/RNG checkpoint, dependency versions and training provenance.
2. Reload that exact base model without NF4 quantization in float16, merge the
   adapter, and export a standard Hugging Face checkpoint. Export a baseline
   checkpoint through the same floating-point path.
3. Transfer these exported checkpoints and project source (not a venv) to the Mac.
4. Create a fresh Mac venv with MLX-LM and CPU verifier dependencies. Convert BOTH
   baseline and trained exports with the same MLX quantization settings.
5. Run model-driven held-out agent inference on the Apple GPU through MLX; use the
   existing Docker CPU verifier on the Mac for visible and final checks. Held-out
   seed/challenge manifests stay outside policy observations and remain shared
   across baseline/trained evaluation. Human audits still gate final validity.

MLX quantization is not CUDA NF4, and merging into the original floating-point
base changes the numerical execution path. Record both formats, use identical Mac
conversion/inference settings for both arms, and treat results as evaluation of
exported policies. Do not use Mac-generated log probabilities for CUDA on-policy
updates. The learned action-value model is needed during training, not standard
final policy inference.

No GPU cluster endpoint/checkpoint is currently attached to this workspace. Code
and local smoke tests can be completed here; actual Slurm execution, CUDA VRAM
behavior and trained-model export must be validated on the user's cluster. Do not
claim a real training experiment completed from mocked or tiny-model tests.


## Current implementation and operational limitations

`slurm_train.sh` installs `requirements-gpu.txt` and calls `miniport.train`. The
config selects terminal, immediate, or action_value training. Action-value runs
first collect frozen-policy baseline checkpoints and paired continuations, then
fit the count-weighted CPU predictor with a held-out template-family validation
split. GPU generation stays sequential, and only action-token log probabilities
receive policy gradients. Invalid actions consume budget; outages invalidate
episodes. Full tokenized trajectories and branch outcomes are persisted.

`slurm_export.sh` exports exact-base floating-point baseline/trained model pairs.
`setup_mac.sh` installs native MLX, and `miniport.mac` converts and evaluates them
using Docker exclusively. Tiny real Qwen/PEFT merge + MLX generation is tested;
full-size exports and cluster execution are not.

Checkpoints contain optimizer/RNG state, but automatic job resume is not supported
yet: use fresh output directories. All-identical continuation outcomes abort
action-value fitting rather than fabricate supervision. Runs with no nonzero
policy update are marked unsuccessful and cannot be exported as trained results.
The default stubs may be too difficult; calibrate or supply starter files first.
The code is an initial experimental loop, not a claim of validated research gains.
