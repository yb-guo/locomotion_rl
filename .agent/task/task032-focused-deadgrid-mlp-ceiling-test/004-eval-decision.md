# 004 Eval Decision

## Route

Evaluate whether focused tuning materially improves Task031 Level A forced
persistent dead-grid without breaking Level B dynamic switch.

Required eval:

- Speeds: `0.4`, `0.8`, `1.2`, `1.6`, `2.0 m/s`.
- Cases:
  - clean;
  - random persistent motor failure;
  - forced persistent 12-joint dead-grid;
  - canonical dynamic switch.

Decision criteria:

- Promising:
  - forced dead-grid failed cases shrink materially versus Task031 baseline;
  - clean and random persistent still pass;
  - dynamic switch still passes at all speed bins.
- Ceiling reached:
  - same joints remain failed after focused tuning; or
  - dynamic switch regresses; or
  - training only trades one failure joint for another.

## Log

- 2026-05-28 Planned as the final close step.
- 2026-05-28 Evaluated weak-focused final `model_5388.pt` on blocker subset:
  speeds `0.4`, `1.6`, and `2.0 m/s`; cases `dead_grid` and
  `dynamic_switch`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/eval_weak_model5388_subset/task031_eval_speed_matrix_summary.json`.
  Dynamic switch passed at all three speeds, but dead-grid failed at all three:
  - `0.4`: `left_hip_roll_joint`, `right_knee_joint`;
  - `1.6`: `left_hip_yaw_joint`, `right_knee_joint`;
  - `2.0`: `left_hip_yaw_joint`, `left_hip_roll_joint`,
    `left_knee_joint`, `right_knee_joint`.
- 2026-05-28 Evaluated early `model_5350.pt` to check whether the final
  checkpoint over-trained. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/eval_weak_model5350_subset/task031_eval_speed_matrix_summary.json`.
  Dynamic switch passed at all three speeds, but dead-grid still failed:
  - `0.4`: `left_hip_yaw_joint`, `right_hip_pitch_joint`,
    `left_hip_roll_joint`, `right_knee_joint`;
  - `1.6`: `right_knee_joint`;
  - `2.0`: `right_knee_joint`.

## Review

Status: MLP weak-curriculum ceiling test is negative. The weak-focused stage did
not materially improve the Task031 forced dead-grid blocker, and the final
checkpoint traded failures rather than shrinking them. Dynamic switch stayed
healthy, so the failure is specific to forced persistent dead-grid robustness.
Do not continue to mixed/hard under this task without a new hypothesis; the next
substantive route should add memory/history or change the failure-conditioning
mechanism.
