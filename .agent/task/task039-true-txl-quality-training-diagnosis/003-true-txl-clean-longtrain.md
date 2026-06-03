# 003 True-TXL Clean Longtrain

## Route

Run Task038 true-TXL on the clean train variant long enough to detect an actual
learning trend, not just one-iteration smoke plumbing.

This slice is still clean-gait diagnosis only. It does not claim held-out
adaptation or LocoFormer reproduction.

Local plumbing for the clean train eval wrapper is ready. Bounded H200 evidence
now exits to `004` because memory debug is missing and clean-gait quality fails.

## Minimal Closed Loop

Close this slice with:

- one H200 true-TXL clean training run;
- checkpoint and training log path;
- Task039 clean feedback-loop eval JSON;
- memory debug fields showing inference cache is active;
- comparison to the one-iteration Task038 checkpoint and the MLP baseline when
  available.

## Evidence Gate

Evidence must record:

- train task id;
- runner and actor model class;
- action dimension;
- env count and rollout steps;
- iterations or wall-time budget;
- reward trend summary;
- checkpoint path;
- eval JSON path;
- memory debug fields;
- `quality_gate_pass` or explicit failure reasons.

## Subagent Ownership

Worker owns true-TXL clean-training launch/eval scripts and docs for this slice
only. Worker must not change PPO memory semantics in this slice; that belongs
to `004`.

Reviewer checks whether the evidence shows a quality trend and whether failure
reasons are specific enough for the router to choose the next diagnosis.

## Failure Exit

If true-TXL does not improve on clean train variant, do not continue to held-out
morphology. Route to `004` memory-update diagnostics and/or clean curriculum
diagnosis.

## Log

- 2026-05-30 Opened as the first true-TXL quality-training slice.
- 2026-05-30 Added local Task039/003 true-TXL clean eval plumbing without
  running H200:
  - `src/h200_locomotion_lab/tools/task039_true_txl_clean_eval.py` wraps
    `task037_multitrial_eval_checkpoint.run_eval`, allow-lists only
    `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`, preflights
    checkpoint/task/numeric args, writes structured JSON failures, sets
    `policy_label=True-TXL`, sets `task039_true_txl_clean_only=true`, and keeps
    all no-overclaim flags false.
  - The wrapper separates `pipeline_pass` from `quality_gate_pass` by delegating
    quality metrics to `task039_quality_feedback.evaluate_quality_feedback`.
    Pipeline health rejects runner/model/action-dim mismatch, missing
    `trial_0`/`final_trial`/`aggregate`, eval error/traceback, overclaim flags,
    and absent or inactive true-TXL memory debug.
  - Memory debug contract accepts the existing `txl_debug` shape and the
    explicit aliases `memory_debug` and `policy_debug`; it requires present
    memory lengths, positive incremental steps, and previous-memory exposure.
  - `tests/test_task039_true_txl_clean_eval.py` covers parse defaults,
    preflight missing checkpoint and unrelated/heldout task rejection, delegated
    wrapping, pipeline rejections for runner/model/dim/memory-debug failures,
    quality failure separation, structured failure JSON, and `--help`.
- 2026-05-30 Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp
  .test_tmp_task039_003_worker tests\test_task039_true_txl_clean_eval.py
  tests\test_task039_quality_feedback.py
  tests\test_task038_true_txl_multitrial_eval_smoke.py
  tests\test_agent_inventory.py` returned `48 passed in 0.24s`.

## H200 Command

After a router-approved true-TXL clean checkpoint exists, evaluate it through
the local diagnostic wrapper:

```bash
cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
export PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_DISABLED=true
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task039_true_txl_clean_eval \
  --task Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke \
  --checkpoint /path/to/model.pt \
  --output-json /path/to/task039_true_txl_clean_eval.json \
  --num-envs 64 \
  --steps 360 \
  --trial-length-s 2.0 \
  --lin-vel-x 0.4
```

Expected JSON evidence must keep:

```text
quality_claim:false
training_claim:false
eval_claim:false
reproduction_claim:false
superiority_claim:false
task039_true_txl_clean_only:true
policy_label:True-TXL
```

No H200 quality, training, eval, reproduction, or superiority evidence has been
recorded by this local plumbing update.
- 2026-05-30 Router ran the bounded H200 true-TXL clean diagnostic:
  - train task/config id:
    `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`;
  - runner class: `Task038TrueTxlMemoryK160Runner`;
  - actor model class: `Task038TrueTxlMemoryModel`;
  - action dimension: `31`;
  - train envs: `4096`;
  - eval envs: `64`;
  - command speed: `lin_vel_x=0.4`;
  - train budget: `30` iterations, seed `3900301`, GPU `cuda:0`;
  - train stdout:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/true_txl_clean_longtrain/039_true_txl_clean_env4096_iter30_gpu0_seed3900301.stdout.log`;
  - train metadata JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/true_txl_clean_longtrain/true_txl_clean_env4096_iter30_model29_train_metadata.json`;
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task039_true_txl_clean_train/2026-05-30_16-58-19_039_true_txl_clean_env4096_iter30_gpu0_seed3900301/model_29.pt`;
  - eval JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/true_txl_clean_longtrain/true_txl_clean_env4096_iter30_model29_vx0p4_eval.json`.
- 2026-05-30 H200 eval result:
  - `pipeline_pass=false`;
  - `quality_gate_pass=false`;
  - `pass=false`;
  - pipeline failure reason: `memory_debug_missing`;
  - `memory_debug_present=false`;
  - `memory_debug_active=false`;
  - `final_trial.fall_ratio=1.0`;
  - `final_trial.gravity_xy.max=0.9480438828468323`;
  - `final_trial.root_z.min=0.1402791440486908`;
  - `final_trial.lin_vel_error.mean=0.63901287317276`;
  - `final_trial.yaw_vel_error.mean=0.6005142331123352`;
  - quality failure reasons:
    `final_fall_ratio_too_high`, `final_gravity_xy_too_high`,
    `final_root_z_too_low`, `final_lin_vel_error_too_high`,
    `final_yaw_vel_error_too_high`, `completion_ratio_regressed_from_trial0`,
    `gravity_xy_max_regressed_from_trial0`, and
    `root_z_min_regressed_from_trial0`.
  - All diagnostic no-overclaim flags remained false:
    `quality_claim`, `training_claim`, `eval_claim`, `reproduction_claim`,
    and `superiority_claim`.

## Review

Status: H200 evidence recorded as a failure-exit diagnostic. This slice does
not produce a true-TXL quality result because the eval path does not expose
active memory debug and gait quality fails. Route to `004` before any held-out
morphology or TXL superiority claim.
