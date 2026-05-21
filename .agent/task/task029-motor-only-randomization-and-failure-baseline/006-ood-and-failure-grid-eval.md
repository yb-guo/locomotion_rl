# 006: OOD And Failure Grid Eval

## Route

Evaluate the accepted MLP checkpoint beyond the training distribution without
changing the first-pass training objective.

Eval groups:

1. Clean eval.
   - No motor randomization or failure.
   - Confirms baseline walking was not destroyed.

2. Motor-only randomized eval.
   - Uses the task029 training distribution.
   - Confirms in-distribution robustness.

3. Doubled motor randomization holdout.
   - Widens motor ranges relative to training.
   - Mimics LocoFormer-style OOD evaluation without changing topology.

4. Dead-motor grid.
   - Force one leg motor dead at a time.
   - Report per-joint survival, tracking, yaw, gravity, root height, and fall
     metrics.

5. Optional diagnostic holdouts.
   - Locked joint and stuck command are eval-only diagnostics in task029.
   - They are not training randomization for first acceptance.

## Minimal Closed Loop

Feedback loop:

1. Load one saved checkpoint.
2. Run all eval groups with deterministic seeds and fixed commands.
3. Save one JSON per group plus an aggregate summary.
4. Include actual motor scale/failure mask settings in JSON diagnostics.
5. Compare clean, in-distribution, doubled holdout, and grid metrics.

Pass:

- Clean eval remains stable.
- In-distribution motor-randomized eval meets predeclared thresholds.
- Doubled holdout and dead-motor grid produce complete reports even if some
  joints fail.
- Eval summaries include exact fault settings used per run.

Fail:

- Eval requires privileged actor fault inputs.
- Only training reward is reported.
- Failed grid cases are omitted instead of reported.
- The same checkpoint cannot be evaluated across all groups.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/`.

## Log

- 2026-05-19 Opened to separate convergence claims from robustness claims.
  The first task029 pass can succeed even if some grid joints remain hard
  failures, as long as the grid is complete and diagnostic.
- 2026-05-19 Prepared the same-checkpoint eval gate script:
  `.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_eval_failure_checkpoint.py`.
  The script runs clean, motor-primitives, in-distribution motor-failure,
  doubled motor holdout, and dead-motor grid cases from one checkpoint. Each
  case writes JSON independently, so failed grid joints are still reported.
- 2026-05-19 Eval script behavior:
  - Clean case uses `Unitree-G1-Gripper-Flat-MotorOnly-Control` and strips
    non-control events. The clean JSON records `clean_strip_report` and the
    aggregate requires `clean_motor_failure_stripped=true`.
  - In-distribution failure uses
    `Unitree-G1-Gripper-Flat-MotorOnly-Failure`.
  - Doubled holdout widens motor primitive ranges at runtime and widens failure
    ranges to weak `(0.15, 0.85)` and dead `(0.0, 0.20)`. If a helper is not
    present, the script records that in JSON instead of claiming a pass.
  - Dead-motor grid discovers all task029 leg failure targets, then forces one
    `dead` motor per run and writes every joint case, including errors.
    Final eval must not use `--skip-grid`; the aggregate requires
    `failure_targets_nonempty=true` and `grid_case_count_equals_target_count=true`.
  - JSON records checkpoint path, command, seeds, fixed command, event/failure
    settings, thresholds, metrics, and pass/fail.
- 2026-05-19 Local checks passed:
  `python .../task029_eval_failure_checkpoint.py --help` and AST parse. A
  `py_compile` attempt was not used as evidence because the existing ignored
  `__pycache__` directory denied pyc replacement on Windows.
- 2026-05-19 Copied the script to H200:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_eval_failure_checkpoint.py`.
  H200 `--help` passed from the MJLab checkout with:
  `PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true`.
- 2026-05-19 Checkpoint blocker: no task029 `model_*.pt` was found under
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl`
  and no task029 eval output exists yet under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/`.
  No eval was run because 005 has not provided a final checkpoint.
- Planned H200 command once 005 provides a checkpoint:
  ```bash
  cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
  PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_eval_failure_checkpoint.py \
    --checkpoint <005-final-model.pt> \
    --output-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/ \
    --device cuda:0
  ```
  Do not add `--skip-grid` for the final gate.
- 2026-05-19 `model_600.pt` became available before `model_1200.pt`; training
  was still running on GPU1, so the formal gate ran on GPU0 with this
  same-checkpoint path:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_600.pt`.
  Command:
  ```bash
  cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
  PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_eval_failure_checkpoint.py \
    --checkpoint /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_600.pt \
    --output-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/ \
    --device cuda:0
  ```
- 2026-05-19 Eval aggregate JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/task029_eval_failure_aggregate.json`.
  Gate completeness checks passed: `failure_targets_nonempty=true`,
  `complete_grid=true`, `grid_case_count=12`,
  `grid_case_count_equals_target_count=true`, and
  `clean_motor_failure_stripped=true`.
- 2026-05-19 Eval result on `model_600.pt`: aggregate `pass=false`.
  The full grid is complete, but the checkpoint does not meet every threshold.
  Clean fixed-command eval had zero falls but missed the strict tracking
  threshold: linear velocity error mean `0.3542388` versus max `0.35`.
  The dead-motor grid also found a hard failure on
  `right_knee_joint`: zero-fall ratio `0.359375` versus min `0.50`,
  max done count `5`.
- 2026-05-19 Per-case JSON evidence:
  - Clean:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/clean_forward_0p5.json`.
  - Motor primitives:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/motor_primitives_forward_0p5.json`.
  - In-distribution failure:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/failure_indistribution_forward_0p5.json`.
  - Doubled holdout:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/doubled_motor_holdout_forward_0p5.json`.
  - Dead grid:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_00_left_hip_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_01_left_hip_yaw_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_02_right_hip_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_03_right_hip_yaw_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_04_left_hip_roll_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_05_left_knee_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_06_right_hip_roll_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_07_right_knee_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_08_left_ankle_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_09_left_ankle_roll_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_10_right_ankle_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/dead_motor_grid_11_right_ankle_roll_joint.json`.
- 2026-05-19 Final training process exited with no `model_1200.pt`; the final
  saved checkpoint is:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_1199.pt`.
  Reran the full eval gate on GPU0 into a distinct directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/`.
  Command:
  ```bash
  cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
  PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_eval_failure_checkpoint.py \
    --checkpoint /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_1199.pt \
    --output-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/ \
    --device cuda:0
  ```
- 2026-05-19 Final `model_1199.pt` eval aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/task029_eval_failure_aggregate.json`.
  Gate completeness checks passed again: `failure_targets_nonempty=true`,
  `complete_grid=true`, `grid_case_count=12`,
  `grid_case_count_equals_target_count=true`, and
  `clean_motor_failure_stripped=true`.
- 2026-05-19 Final `model_1199.pt` eval result: aggregate `pass=false`.
  Improvements over `model_600.pt`: clean, motor-primitives,
  in-distribution failure, and doubled holdout all passed their numeric
  thresholds. The only failed case in the final aggregate is group
  `dead_motor_grid`, exact case `dead_motor_grid_05_left_knee_joint`:
  `left_knee_joint` forced-dead had `zero_fall_ratio=0.0` versus min `0.50`,
  `max_done_count=5`, `lin_vel_error_mean=0.5513105392456055`,
  `yaw_vel_error_mean=0.3230378329753876`, and
  `gravity_xy_mean=0.13182896375656128`.
- 2026-05-19 Final `model_1199.pt` per-case JSON evidence:
  - Clean:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/clean_forward_0p5.json`.
  - Motor primitives:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/motor_primitives_forward_0p5.json`.
  - In-distribution failure:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/failure_indistribution_forward_0p5.json`.
  - Doubled holdout:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/doubled_motor_holdout_forward_0p5.json`.
  - Dead grid:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_00_left_hip_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_01_left_hip_yaw_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_02_right_hip_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_03_right_hip_yaw_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_04_left_hip_roll_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_05_left_knee_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_06_right_hip_roll_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_07_right_knee_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_08_left_ankle_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_09_left_ankle_roll_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_10_right_ankle_pitch_joint.json`;
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/dead_motor_grid_11_right_ankle_roll_joint.json`.

## Review

Status: complete for the final `model_1199.pt` gate, not passed.

This subtask should produce the table that tells us whether task029 is enough
for the current MLP or whether the next task needs long-context adaptation.
Do not hide failure cases; the grid is useful precisely because it names which
motors break the gait.

The earlier `model_600.pt` gate failed clean tracking and the right-knee
dead-motor grid case. The final `model_1199.pt` checkpoint improved the clean,
motor-primitives, in-distribution failure, and doubled holdout metrics enough
to pass those groups, but the complete dead-motor grid still failed on the
single `dead_motor_grid_05_left_knee_joint` case because the left-knee
dead-motor zero-fall threshold was not met. Task029 remains not passed; no
robustness pass is claimed.
