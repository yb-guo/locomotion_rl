# 001: Current Best and Gate Audit

## Route

Freeze the current best known checkpoint and the exact eval blocker before
changing training again. This prevents Task045 from redefining success around
new experiments that do not improve the actual gate.

## Acceptance

- Record the current best checkpoint path.
- Record repeated-seed normal continuous eval metrics.
- State the single active blocker and the unchanged pass gate.

## Log

- 2026-06-02 Current best checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_scale0p5_immediate_leftknee_pose_forward_all_env2048_iter40_lr1e5_seed4419502/model_39.pt`.
- 2026-06-02 Repeated normal continuous evals:
  - seed `4419601`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/scale0p5_immediate_leftknee_pose_forward_all_model39_continuous_normal_left_knee_seed4419601.json`,
    `fall_ratio=0.10546875`,
    `lin_vel_error.mean=0.40938737988471985`;
  - seed `4419602`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/scale0p5_immediate_leftknee_pose_forward_all_model39_continuous_normal_left_knee_seed4419602.json`,
    `fall_ratio=0.125`,
    `lin_vel_error.mean=0.4212338924407959`;
  - seed `4419603`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/scale0p5_immediate_leftknee_pose_forward_all_model39_continuous_normal_left_knee_seed4419603.json`,
    `fall_ratio=0.10546875`,
    `lin_vel_error.mean=0.41330718994140625`.
- 2026-06-02 Intermediate checkpoints from the same 40-iteration run were not
  better. Seed `4419901` evals had `model_0 fall_ratio=0.17578125` and
  `model_20 fall_ratio=0.12109375`.

## Review

Status: passed for baseline audit, not a task pass.

The active blocker is now narrow: speed, root height, and gravity are within
the normal continuous post-fault gate, but post-fault fall ratio is consistently
above the required `<=0.05`. This is the only metric Task045 should optimize
before running expensive memory-ablation triplets.
