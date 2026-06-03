# 002 Right-Knee Focused Stage

## Route

Add the smallest H200 training stage that changes only the failure sampling
curriculum.

Requirements:

- Use Task033 frozen-base StackMLP K4 runner.
- Use K4 shared history input.
- Oversample `right_knee_joint` forced weak/dead failures.
- Keep some hip/knee rehearsal so the fix does not overfit one joint.
- Keep speed fixed or tightly centered at `2.0 m/s` for the first pass.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Added
  `.agent/task/task034-right-knee-dead-history-curriculum/task034_create_right_knee_stage.py`.
  It registers weak/mixed/hard right-knee-focused frozen-base stages on H200.
- 2026-05-28 H200 registry contains:
  `Unitree-G1-Gripper-Flat-Task034-RightKneeWeak-FrozenBase-Fast2p0`,
  `Unitree-G1-Gripper-Flat-Task034-RightKneeMixed-FrozenBase-Fast2p0`,
  and `Unitree-G1-Gripper-Flat-Task034-RightKneeHard-FrozenBase-Fast2p0`.
- 2026-05-28 Env64 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_smoke/2026-05-28_14-12-22_034_mixed_rightknee_env64_iter1_gpu1_seed3403400`.

## Review

Status: passed for stage registration and smoke. Later subtasks show the stage
is trainable but not beneficial for the target metric.
