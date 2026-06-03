# 002 MLP Clean Baseline

## Route

Produce the clean G1-like baseline that true-TXL must be compared against.

The point is not to optimize MLP. The point is to avoid diagnosing TXL without a
same-task quality reference.

## Minimal Closed Loop

Close this slice with one of:

- an H200 MLP clean-training JSON plus eval JSON under the Task039 feedback
  loop; or
- an explicit blocked reason showing why the comparable MLP run cannot be
  launched from current infra.

The run should start with the smallest useful clean setup: one train variant,
fixed command speed, fixed seed or small seed set, and no held-out morphology.

## Evidence Gate

Evidence must record:

- train task/config id;
- eval task/config id;
- policy class;
- action dimension;
- command speed;
- number of envs;
- iteration count or wall-time budget;
- checkpoint path on H200;
- eval JSON path;
- `pipeline_pass` and `quality_gate_pass` separately.

## Subagent Ownership

Worker owns launch scripts, CLI wrappers, and docs for the MLP clean baseline
slice only. Worker must not modify true-TXL code or morphology generation.

Reviewer checks that the MLP evidence is comparable to the planned true-TXL run
and that failed quality is not written as a pipeline failure.

## Failure Exit

If MLP cannot learn any clean gait under the same env/reward/action contract,
stop before blaming TXL. The next router decision should diagnose clean-gait
reward/curriculum/env stability first.

## Log

- 2026-05-30 Opened as the same-task quality reference slice.
- 2026-05-30 Added local plumbing for the MLP clean baseline without running
  H200:
  - `task039_register_mlp_clean_baseline.py` registers
    `Unitree-G1-Gripper-Flat-Task039-MlpClean-Train` against the Task038 train
    XML env cfg helper and `Task037BufferOnlyK4DeterministicInnerResetRunner`.
  - `src/h200_locomotion_lab/tools/task039_mlp_clean_eval.py` wraps
    `task037_multitrial_eval_checkpoint.run_eval`, allow-lists only the
    Task039 MLP clean task id, preflights checkpoint/task/numeric args, writes
    structured JSON failures, sets all no-overclaim flags false, and separates
    `pipeline_pass` from `quality_gate_pass` through
    `task039_quality_feedback.evaluate_quality_feedback`.
  - Tightened `pipeline_pass` to fail on missing/mismatched runner class, actor
    model class, action dimensions, missing trial/aggregate output, or delegated
    eval error/traceback. Quality metric failures remain separate from pipeline
    failures.
  - Added non-breaking metadata fields to the shared Task037 eval result:
    `runner_cls`, `actor_model_class`, `action_dim`, and `total_action_dim`.
- 2026-05-30 Router verified the local 002 plumbing:
  - `PYTHONPATH=src python -m pytest -q -p no:cacheprovider --basetemp
    .test_tmp_task039_002_003_fix tests/test_task039_mlp_clean_eval.py
    tests/test_task039_true_txl_clean_eval.py tests/test_task039_train_metadata.py
    tests/test_task039_quality_feedback.py
    tests/test_task037_mjlab_smoke_scripts.py tests/test_agent_inventory.py`
    returned `75 passed in 0.39s`.
  - `python -m h200_locomotion_lab.tools.inspect_agent` completed
    successfully.
- 2026-05-30 Ran the smallest H200 MLP clean baseline. This is a pipeline and
  diagnostic-quality sample only, not a training success claim:
  - train task/config id:
    `Unitree-G1-Gripper-Flat-Task039-MlpClean-Train`;
  - runner class:
    `Task037BufferOnlyK4DeterministicInnerResetRunner`;
  - policy class: `MLPModel`;
  - action dimension: `31`;
  - command speed: `lin_vel_x=0.4`;
  - train envs: `4096`;
  - eval envs: `64`;
  - train budget: `30` iterations, seed `3900201`, GPU `cuda:0`;
  - train stdout:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/mlp_clean_baseline/039_mlp_clean_env4096_iter30_gpu0_seed3900201.stdout.log`;
  - train metadata JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/mlp_clean_baseline/mlp_clean_env4096_iter30_model29_train_metadata.json`;
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task039_mlp_clean_train/2026-05-30_16-52-26_039_mlp_clean_env4096_iter30_gpu0_seed3900201/model_29.pt`;
  - eval JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/mlp_clean_baseline/mlp_clean_env4096_iter30_model29_vx0p4_eval_v2.json`.
- 2026-05-30 H200 eval result:
  - `pipeline_pass=true`;
  - `quality_gate_pass=false`;
  - `pass=false`;
  - `final_trial.fall_ratio=0.953125`;
  - `final_trial.gravity_xy.max=0.9452863335609436`;
  - `final_trial.root_z.min=0.2165585458278656`;
  - `final_trial.lin_vel_error.mean=0.519557774066925`;
  - failure reasons:
    `final_fall_ratio_too_high`, `final_gravity_xy_too_high`,
    `final_root_z_too_low`, `final_lin_vel_error_too_high`,
    and `completion_ratio_regressed_from_trial0`.
  - All diagnostic no-overclaim flags remained false:
    `quality_claim`, `training_claim`, `eval_claim`, `reproduction_claim`,
    and `superiority_claim`.
- 2026-05-30 Fixed the Task039 pass boundary after read-only review: Task039
  eval wrappers now set top-level `pass = pipeline_pass and quality_gate_pass`.
  Pipeline-only health remains available through `pipeline_pass`; a failed
  gait quality gate can no longer produce `pass=true`.

## H200 Commands

Register the external MJLab task after Task038 XML helpers are already patched:

```bash
python .agent/task/task039-true-txl-quality-training-diagnosis/task039_register_mlp_clean_baseline.py \
  --root /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
```

Train the clean MLP baseline on the Task038 train XML variant:

```bash
cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
export PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_DISABLED=true
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  scripts/train.py Unitree-G1-Gripper-Flat-Task039-MlpClean-Train \
  --gpu-ids=[0] \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=1500
```

Evaluate a produced checkpoint with Task039 diagnostic gates:

```bash
cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
export PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_DISABLED=true
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task039_mlp_clean_eval \
  --task Unitree-G1-Gripper-Flat-Task039-MlpClean-Train \
  --checkpoint /path/to/model.pt \
  --output-json /path/to/task039_mlp_clean_eval.json \
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
task039_mlp_clean_baseline_only:true
policy_label:MLP
```

No H200 pass, training pass, eval pass, quality pass, reproduction claim, or
superiority claim is made by this slice yet.

## Review

Status: evidence ready for independent read-only review. The minimal closed loop
for this slice is satisfied by an H200 MLP clean train/eval JSON with
`pipeline_pass=true`, `quality_gate_pass=false`, and `pass=false`. This is not
a clean-gait, training, reproduction, eval, or superiority pass.
