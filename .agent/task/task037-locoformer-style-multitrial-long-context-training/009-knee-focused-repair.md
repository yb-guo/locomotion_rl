# 009 Knee-Focused Repair

## Route

Subtask 008 narrowed the AdaptK160 blocker:

- clean gait passes at `0.4`, `1.2`, and `2.0 m/s`;
- dynamic switch passes at `0.4`, `1.2`, and `2.0 m/s`;
- `2.0 m/s` forced deadgrid is `10/12`;
- only `left_knee_joint` and `right_knee_joint` fail.

This subtask tries a knee-focused repair from `model_5467.pt`.

Fixed boundaries:

- Start from 007 clean-prior checkpoint `model_5467.pt`, not the generic mixed
  `model_5496.pt`.
- Keep AdaptK160 and the Task037 multi-trial contract unchanged.
- Do not expose explicit fault labels, active joint ids, motor scales, trial
  index, or final-trial flag to the actor.
- Reuse existing MJLab failure stages first; do not invent a new scheduler
  until existing knee-focused stages are tested.
- A candidate must preserve clean and dynamic gates before any promotion.

Initial stage:

- `Unitree-G1-Gripper-Flat-Task037-AdaptK160-KneeHipRollVx2p0`
  - uses existing `unitree_g1_gripper_flat_task030_knee_hiproll_vx2p0_env_cfg`;
  - fixed `2.0 m/s`;
  - focuses right knee, left knee, hip roll, and right hip yaw guard cases.

Fallback stage if right knee remains the only blocker:

- `Unitree-G1-Gripper-Flat-Task037-AdaptK160-RightKneeMixedVx2p0`
  - uses existing `unitree_g1_gripper_flat_task034_rightknee_mixed_env_cfg`.

Acceptance:

- H200 registration and launch dry-run pass.
- One bounded knee-focused continuation writes log/checkpoints.
- Checkpoint sweep includes at least final and saved intermediate checkpoints
  for `left_knee_joint` and `right_knee_joint`.
- Full eval for the best knee candidate covers:
  - clean `0.4`, `1.2`, `2.0`;
  - dynamic `0.4`, `1.2`, `2.0`;
  - `2.0 m/s` forced deadgrid.
- Pass requires deadgrid `12/12` without regressing clean/dynamic.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_knee_repair/`

## Log

- 2026-05-29 Opened after user asked to try knee-focused repair.
- 2026-05-29 Added AdaptK160 task ids for the existing Task030 knee+hiproll
  `2.0 m/s` stage and Task034 right-knee mixed `2.0 m/s` stage. Extended the
  stage-selectable H200 launch script with `STAGE=knee2p0` and
  `STAGE=rightknee2p0`.
- 2026-05-29 H200 registration/tests passed for the new Task037 AdaptK160
  knee task ids. Launch dry-run passed with `STAGE=knee2p0`.
- 2026-05-29 Ran bounded `knee2p0` continuation from clean prior
  `model_5467.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_adaptk160_failure_curriculum/2026-05-29_17-32-34_037_adapt_k160_knee2p0_from_clean5467_env8192_iter30_gpu0_seed3700820/model_5496.pt`.
  Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_knee_repair/knee2p0_sweep_seed3700820/task037_adaptk160_knee2p0_sweep_summary.json`.
  Result: no checkpoint passed both knee blockers. Right knee passed at some
  checkpoints; left knee remained just outside the velocity gate, with best
  left-knee final-trial velocity error around `1.27 > 1.20`.
- 2026-05-29 Ran full eval for the final `knee2p0` candidate `model_5496.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_knee_repair/eval_knee2p0_model5496/task037_adaptk160_knee2p0_model5496_eval_summary.json`.
  Result: clean `3/3` pass, dynamic switch `3/3` pass, forced deadgrid `10/12`
  pass. `left_knee_joint` and `right_knee_joint` still failed, so the checkpoint
  is not promoted.
- 2026-05-29 Ran a conservative second continuation from `model_5496.pt` with
  lower learning rate:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_adaptk160_failure_curriculum/2026-05-29_17-47-13_037_adapt_k160_knee2p0_from5496_lr1e6_env8192_iter20_gpu0_seed3700830/model_5515.pt`.
  Sweep evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_knee_repair/knee2p0_lr1e6_sweep_seed3700830/task037_adaptk160_knee2p0_lr1e6_sweep_summary.json`.
  Result: no checkpoint passed both knees. Best right knee passed at
  `model_5514.pt` with fall ratio `0.0` and velocity error `0.895`; best left
  knee stayed upright but failed velocity with `model_5502.pt`, velocity error
  `1.276 > 1.20`.

## Review

Status: closed as a rejected repair route. The current knee2p0 stage improves
right-knee robustness intermittently and keeps left-knee trials upright, but it
does not close the left-knee velocity gate. No knee-repaired checkpoint is
promoted. The next useful route should change the left-knee objective/stage or
failure curriculum, not keep extending this same knee2p0 continuation.
