# MiniPort

Project repository: [Jax2PyTorch_RL](https://github.com/BrightLaw9/Jax2PyTorch_RL).
Research report: [brightlaw9.github.io/Jax2PyTorch_RL](https://brightlaw9.github.io/Jax2PyTorch_RL/).

Pilot status (September 2026): 16 policy updates completed. Paired CUDA evaluation
improved numerical success from 1/3 to 2/3 held-out tasks; human integrity audits
remain outstanding. This three-task, single-seed pilot does not isolate the
critic's contribution or establish a latency improvement. See the report and
[collected results](docs/results.json). CUDA evaluation is available through
`slurm_gpu_eval.sh`, alongside the Mac evaluation path below. Training
artifacts and private evaluation manifests are deliberately excluded from Git.

**Research question: Does learning longer-term action value add anything beyond
immediate verifier rewards?** See [plan.md](plan.md) for the research protocol.

Training runs on a Linux CUDA server in a Python venv, including CPU subprocess
verification. **There is no Docker dependency in training or export.** Evaluation
can run on CUDA with CPU subprocess verification, or on an Apple Silicon Mac,
where MLX generates edits and Docker checks submitted code on CPU.

## What is implemented

- Six JAX-to-PyTorch task templates (18 training, six held-out configurations), an
  edit/test/stop controller, fixed visible feedback, and protected trajectory logs.
- Qwen3-4B-Instruct-2507 generation and sequential NF4 QLoRA updates. The config
  selects `terminal`, `immediate`, or `action_value` rewards. A frozen baseline is
  exported alongside each trained policy for matched Mac evaluation.
- Action-value data collection: snapshot the source and conversation before an
  action, sample two actions, then run multiple continuations after each action
  under the frozen policy. The target is the fraction of valid continuations that
  eventually pass the separate training verifier cases.
- A small CPU neural predictor over hashed state/action text features, fitted with
  count-weighted binary cross entropy and validated on a held-out training template
  family. This is an initial predictive baseline, not a pretrained language critic.
- RLOO-style policy gradients on generated action tokens, a frozen-base KL penalty,
  and centered predicted action value for the action-value condition. Sampling and
  training log probabilities use the same temperature. Only LoRA weights update.
- Adapter/tokenizer/config/revision checkpointing, floating-point model export,
  paired MLX conversion, and model-driven held-out evaluation through Docker.

The value predictor supplies training credit; ordinary Mac policy inference does
not load it. Held-out cases never supply training rewards or value labels. Human
reviews audit final artifacts; they are not the action-value training dataset.

Teacher-guided value collection is a separate, optional GPU job. It mines frozen
stuck states, generates corrections with a larger inference-only model, admits only
corrections that pass the sealed verifier, and measures matched teacher/student
actions with frozen-policy continuations. The teacher and trainable policy are
loaded sequentially, never concurrently.

## 1. Train on the GPU server

Copy this project to the server. Use Python 3.11 or 3.12. Edit
[configs/train.json](configs/train.json) and cluster module/resource directives in
[slurm_train.sh](slurm_train.sh), then run from the project directory:

```sh
mkdir -p logs
sbatch slurm_train.sh
```

The script creates `.venv-gpu`, installs [requirements-gpu.txt](requirements-gpu.txt),
checks CUDA, and runs `python -m miniport.train --config configs/train.json`.
Candidates run in CPU subprocesses using that venv; CUDA is hidden from those
children and JAX is forced to CPU. This is deliberately not a security sandbox.
Static checks, fresh source snapshots, process cleanup, and time/output limits
remain enabled. Training logs explicitly record `isolation=subprocess`.

The supplied Slurm allocation uses `gpu-gen`, one `rtx6000`, eight CPUs, ten hours,
and **16 GB host RAM**. Host RAM is separate from the config's 16 GB CUDA allocation
cap. Adapt resource names/modules to the cluster. The default PyTorch wheel is
CUDA 12.6; it requires a compatible NVIDIA driver. Select a compatible wheel index
with `TORCH_INDEX_URL` if necessary, for example:

```sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 sbatch slurm_train.sh
```

Initial dependency and model downloads require network access or a prepopulated
cache. The job resolves `base_revision` to an immutable Hugging Face commit once
and saves it in `artifacts/gpu-run/run-manifest.json`. For subsequent experimental
conditions, set that exact revision in each config, keep task/seed/budget settings
matched, and use a new output directory. Select another config with:

```sh
TRAIN_CONFIG=configs/your-condition.json sbatch slurm_train.sh
```

The default config is a small operational pilot: four active checkpoints (with
a configured upper bound of six), two actions and four continuations each,
followed by four policy update iterations. It is not a
research-sized experiment. Incomplete stubs may produce only failures. You can
supply calibrated training starters via `starter_dir`, organized as
`TASK_ID/submission/candidate.py`. Establish mixed successes and failures before
scaling. Value fitting rejects all-identical training outcomes; a run with no
nonzero policy updates cannot be exported as a trained result.

Checkpoint contents include adapter safetensors, tokenizer, exact base revision,
full configuration, dependency versions, source identity, and optimizer/RNG state.
Branch trajectories and the fitted value model remain in the run directory.
Collection and policy training can resume in the same output directory:

```sh
RESUME=1 sbatch slurm_train.sh
# Equivalent entry point:
python -m miniport.train --config configs/train.json --resume
```

### Optional teacher-guided value collection

First finish ordinary value collection without starting policy updates, then run
the separate teacher job and resume training:

```sh
COLLECT_ONLY=1 sbatch slurm_train.sh
sbatch slurm_teacher_collect.sh
RESUME=1 sbatch slurm_train.sh
```

For an existing run containing failed trajectories, the teacher job can mine those
states directly. Configure it in `configs/teacher-value.json`. The default
Qwen2.5-Coder-14B teacher uses NF4, batch size one, and a hard 16 GiB CUDA allocator
cap. It is unloaded before the frozen 4B student is loaded for continuation
measurement. The separate job also requests only 16 GB of host RAM; low-memory
model loading avoids materializing a full-precision teacher copy on the CPU.

The job writes `value-collection/teacher-value-data.jsonl` and refits
`action-value.json` from ordinary plus teacher-paired rows. Each pair shares one
state: the observed stuck student action is the hard negative and the sealed-pass
teacher edit is the positive candidate. Continuation outcomes remain the labels;
the role names do not override empirical rewards. `teacher-collection/complete.json`
records the immutable teacher revision, verified count, and combined dataset hash.
Training refuses to load a critic whose recorded dataset hash differs from the
current combined value data.

Teacher correction attempts are iterative rather than independent resamples. A
failed candidate becomes the next attempt's current source and its policy-visible
sealed-verifier diagnostic is appended to the teacher dialogue. On resume, older
unaccepted independent-attempt logs are archived; already verified corrections
remain reusable.

`manual_corrections` may bind individual persisted state IDs to corrections from
an explicitly named external model. Every correction is still sealed-verifier
checked and continuation-scored. If all mined states have external corrections,
the local 14B teacher is not loaded; the completion manifest records each source
model, file hash, and `local_teacher_loaded: false`.

Stop the previous job before resuming. Completed baselines and continuations are
reused; interrupted continuations recover their last committed action, candidate,
history, and remaining budget. Uncommitted actions are regenerated. Candidate
choices and sampling attempts are persisted, and value rows are written atomically
without duplicates. The original base revision and configuration are retained.
Policy checkpoints restore the adapter and optimizer; completed rollouts in the
unfinished update are reused. Prompt/feedback changes are recorded in `resume-history.jsonl`
and mark the run as containing mixed collection protocols.

Second candidate actions are compared structurally, ignoring whitespace and
comments. Duplicates are resampled up to three times by default
(`candidate_resamples`); if none is distinct, duplicate continuations are skipped.
Existing duplicate labels are excluded from value fitting rather than counted twice.

Each rollout writes `completions.jsonl` immediately after generation, including
the raw response, token counts, and whether it reached the output token limit.
In `protected.jsonl`, `policy_observation.action_error` explains rejected JSON or
actions, and `policy_observation.feedback.static_scan_findings` lists rejected
source lines and rules. `feedback.diagnostic` preserves original candidate runtime
exception messages, with exception types, candidate line numbers and execution stages.
Returned tensor types are checked separately: `output_type_mismatch` identifies the
field, required type and actual type. Shape, dtype and numerical checks remain external
to the candidate; reference outputs are not added to feedback. Runtime messages are
untrusted candidate diagnostics, not grading evidence. Passing feedback explicitly
instructs the model to emit `{"type":"stop"}`. An invalid action leaves the candidate unchanged; a
subsequent interface failure can therefore still refer to the original stub.
The complete JAX reference in the starting stub is documentation: edit actions
should return only the requested task's PyTorch implementation, without copying
the reference comments or implementing unrelated templates.

The current `experiment.task_limit: 4` restricts collection, policy-training tasks,
and exported-model evaluation to MLP, CNN, LayerNorm, and attention. Set it to `null` for all
six families. `value_checkpoints` may exceed the selected task count: collection
cycles through the selected families with distinct deterministic state seeds.
Rollout batch sizes and policy update counts remain independent. Task-limit expansion
is the one configuration change allowed when resuming collection.

Log schema version 2 keeps each JSONL entry short. `completions.jsonl` links to
the exact response and prompt text; `protected.jsonl` shows action/status/gate,
remaining budget, reward, feedback, and code/diff links. Large original records,
snapshots, and token IDs live under `steps/` and are loaded losslessly by
`miniport.resume.read_jsonl`. `trajectory.json` is an indented step index with
links to full training payloads. Open `progress.md` for a readable action timeline.
Old inline logs remain readable, and `miniport.logio.format_existing` converts
them without changing their recorded actions or outcomes.

## 2. Export and transfer

After successful training, submit [slurm_export.sh](slurm_export.sh). It is a CPU
job using the same venv; customize the CPU partition if your cluster requires it.
You can queue it behind the training job (replace `12345` with its actual ID):

```sh
sbatch --dependency=afterok:12345 slurm_export.sh
```

It reloads the exact original base in FP16, merges the trained adapter, and saves
`artifacts/export/trained-hf`. It separately saves the same original base as
`artifacts/export/baseline-hf`. It never merges into the packed NF4 training model.
For nondefault paths:

```sh
CHECKPOINT=artifacts/your-run/checkpoint EXPORT_DIR=artifacts/your-export sbatch slurm_export.sh
```

Allow roughly 16 GB for the two FP16 4B exports, plus model caches, checkpoints,
MLX outputs and trajectory logs. Merge peak RAM needs validation at full size;
the export job is limited to 16 GB, so full-size merging may exceed the allocation
despite loading and exporting the two models sequentially. If it does, retain the
adapter checkpoint and move the merge step to a machine with sufficient RAM.
Preserve the whole run directory for research
reproducibility, even though Mac inference only needs exported models and source.

From the Mac, transfer the matching source and exported checkpoint pair. Substitute
your SSH host and server path:

```sh
rsync -av --exclude='.venv*' --exclude='venv' --exclude='artifacts' --exclude='logs' \
  USER@GPU:/path/to/miniport-process-reward-rl/ ./
mkdir -p artifacts/export
rsync -av USER@GPU:/path/to/miniport-process-reward-rl/artifacts/export/ artifacts/export/
```

Do not transfer the Linux venv to macOS. Evaluation checks the source identity
against the export, so keep the matching project source rather than silently
changing the evaluator after training.

## 3. Set up the Mac and convert to MLX

Use native Apple Silicon Python 3.11/3.12, with Docker Desktop running:

```sh
PYTHON_BIN=python3.12 bash setup_mac.sh
export PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH
docker build -f containers/verifier.Dockerfile -t miniport-verifier:local .
.venv-mac/bin/python -m miniport.mac convert \
  --source artifacts/export --output artifacts/mlx
```

[setup_mac.sh](setup_mac.sh) creates `.venv-mac`, installs
[requirements-mac.txt](requirements-mac.txt), and checks Apple GPU availability.
The pinned MLX 0.32.2 wheel used here requires macOS 26; older macOS versions need
a compatible, separately validated dependency set. Rosetta/Intel Python is not
supported. This machine's native Mac environment has been installed and tested.

Conversion happens on the Mac because MLX is the target backend. Both baseline
and trained checkpoints receive identical 4-bit MLX quantization; use `--bits 8`
for both if desired. CUDA NF4 and MLX quantization differ, so CPU/Apple inference
is not numerically identical to CUDA training. Report results as performance of
the exported policies and always evaluate the matched exported baseline too.

## 4. Run held-out agents locally

Provision one private manifest and reuse it and the fresh challenge directory
across conditions. Use fresh output directories for each evaluation:

```sh
.venv-mac/bin/miniport provision-eval /private/tmp/miniport-evaluation.json
.venv-mac/bin/python -m miniport.mac evaluate \
  --model artifacts/mlx/baseline \
  --manifest /private/tmp/miniport-evaluation.json \
  --challenges /private/tmp/miniport-challenges \
  --output artifacts/eval-baseline --seed 42
.venv-mac/bin/python -m miniport.mac evaluate \
  --model artifacts/mlx/trained \
  --manifest /private/tmp/miniport-evaluation.json \
  --challenges /private/tmp/miniport-challenges \
  --output artifacts/eval-trained --seed 42
```

Add `--task mlp-heldout` to both commands for a small smoke run. These commands run
the actual agent, not just a frozen submission. MLX generates edits on Apple GPU;
visible checks and final randomized checks use the existing Docker verifier.
Prompts, tokens, actions, provenance and final reports are saved under each output.

`summary.json` distinguishes automated numerical passes from audited validity.
Numerical passers still require a source-bound artifact audit. Use
`miniport.integrity.audit_bundle` to prepare the audit form; the existing
`miniport evaluate ... --audit ...` command can regrade the frozen submission with
its completed audit. Infrastructure outages invalidate episodes. Keep private
manifests and protected logs out of policy observations.

## Validation and remaining work

Forty tests passed locally, including all six subprocess task templates with
Docker disabled, actual Docker tests, branch independence, reward masking and
adapter gradients. A tiny real Qwen received an adapter update, was merged,
converted through MLX and generated tokens on the Apple GPU. The Mac evaluation
runner was also tested against the real Docker verifier with a scripted policy.

```sh
.venv-mac/bin/python -m unittest discover -s tests -v
# Optional export smoke test dependencies (not needed for normal Mac evaluation):
.venv-mac/bin/python -m pip install peft==0.17.1 accelerate==1.10.1
MINIPORT_MLX_TESTS=1 MINIPORT_DOCKER_TESTS=1 \
  .venv-mac/bin/python -m unittest discover -s tests -v
```

The 4B CUDA/Slurm pilot completed 16 policy updates, with 12.55 GiB peak CUDA
allocation during the additional eight updates. Paired held-out GPU evaluation
completed; broader learned-policy performance and controlled reward comparisons
remain to be measured. OOM does not trigger a silent smaller-model or backend
fallback.

Before research claims: calibrate meaningful multi-step repairs, expand beyond
the small template set, increase continuation coverage, evaluate critic quality,
and run terminal/immediate/action-value conditions across multiple seeds. Count
all branch collection and alternative-action generation compute; compare at both
matched policy-update budgets and matched total compute. The current code does
not automate that multi-run statistical study.

Policy-training resume uses the same `RESUME=1 sbatch slurm_train.sh` command.
It restores the saved LoRA adapter, tokenizer, optimizer, random states, gradient
scaler (new checkpoints), and next update index. The existing action-value model
is loaded without recollecting or refitting. Completed rollouts in an unfinished
update are reused; partial rollouts resume at the last committed action. Gradients
for the unfinished update are recomputed from those saved trajectories before one
optimizer step. Legacy partial action-value credits are reconstructed using their
original state and deterministic alternative seed. Updated feedback applies to
subsequent actions, and protocol changes are recorded in resume-history.jsonl.

New checkpoints live in `checkpoints/update-<next-index>-<unique-id>/`. The
`checkpoint` symlink switches only after all files have been written; previous
generations remain available. The legacy checkpoint is retained under
`checkpoints/legacy/`. A checkpoint also stores the completed update's metrics so
a restart can repair a missing metrics journal entry without repeating training.
Policy resume requires the same training configuration. Legacy FP16 checkpoints
without scaler state are rejected; the current BF16 run does not need a scaler.

To refresh selected collection families and expand the task limit, stop the running
job, edit the config, then run:

```sh
python -m miniport.refresh --config configs/train.json --templates cnn
RESUME=1 sbatch slurm_train.sh
```

Preparation archives the selected families' collection artifacts, their value-data
rows, the old critic, and rollouts from uncommitted policy updates. Committed model
checkpoints and completed policy updates are retained. A persistent refresh plan
makes preparation recoverable. Collection reuses other completed families and adds
newly enabled families with the original frozen policy, then refits the value model
and restores the trained adapter and optimizer. The new task rotation applies to
the remaining updates. All archived records live under `refresh-archives/`.

Layer and output parity failures include `diagnostic.numerical`: tensor name,
zero-based case index, shape, dtypes, maximum/mean absolute error, fraction outside
tolerance, and actual/expected minimum and maximum. These statistics describe the
first failing tensor and case; top-level `max_abs_error` summarizes the gate.
Full reference arrays are not returned. Task prompts specify the meaning and layout
of intermediate and output tensors. Comment-only or identical edits receive
`edit_feedback.code = "no_executable_change"` with the current failure location.
Use `miniport.refresh --restart` to archive and replace an already pending refresh
when intentionally recollecting again under changed feedback.

The admission scanner allows local model/parameter assignments, ordinary Python
introspection helpers, common standard-library utilities, and arbitrary literal
sizes within the overall source-size limit. Remaining rejections include a
plain-language message alongside the source line and rule. It retains explicit
checks for non-PyTorch numerical imports, external I/O/execution facilities,
imported-object assignment, and submission layout/size. This static scan is an
admission aid, not an isolation boundary or proof that arbitrary Python is safe.

Action generation stops at the first complete JSON object (with proper handling
of braces and escaped quotes inside source strings), for both HF and MLX policies.
Generated tokens are retained exactly; no response text is silently stripped or
repaired. Strict parsing still rejects malformed JSON or additional content in the
last emitted token. Multiple-action errors explain that nothing was applied and
that edits are tested automatically. The prompt states this protocol explicitly.
