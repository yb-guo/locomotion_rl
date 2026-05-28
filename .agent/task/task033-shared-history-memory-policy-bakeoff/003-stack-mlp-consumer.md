# 003 Stack MLP Consumer

## Route

Implement the smallest memory consumer first: flatten shared history frames and
feed an MLP.

Scope:

- Reuse MJLab/RSL-RL training path if possible.
- Keep action `31D`.
- Keep actor no-fault-label contract.
- Test at least `K=4`; try `K=8` only after overhead is acceptable.

Eval:

- Blocker subset first:
  - speeds `0.4`, `1.6`, `2.0 m/s`;
  - forced persistent dead-grid;
  - canonical dynamic switch.

## Log

- 2026-05-28 Planned as first policy consumer.
- 2026-05-28 Added `Task033StackMlpK4Runner` and
  `Task033StackMlpK8Runner` in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`. Both reuse the
  shared history wrapper and switch only actor obs groups to `actor_history`;
  critic remains on `critic`.
- 2026-05-28 Registered H200 tasks:
  `Unitree-G1-Gripper-Flat-Task033-StackMlpK4-Fast2p0` and
  `Unitree-G1-Gripper-Flat-Task033-StackMlpK8-Fast2p0`.
- 2026-05-28 H200 env64 train smoke passed for StackMLP K4:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_smoke/2026-05-28_11-48-31_033_stackmlp_k4_env64_iter2_gpu0_seed3303302`.
- 2026-05-28 H200 env8192 one-iteration overhead smokes passed initial gates:
  K4 actor input dim `540`, `10805` steps/s, `9.3%` overhead; K8 actor input
  dim `1080`, `11596` steps/s, `2.6%` overhead. See
  `task033_h200_history_policy_smoke_summary.json`.
- 2026-05-28 Added `model_5349.pt` to StackMLP K4 migration. The source actor
  first layer `104D` obs weights are copied into the newest-frame obs slice
  `[405, 509)` of the K4 `540D` history input; older frames and previous-action
  columns start at zero. The migrated checkpoint uses fresh optimizer state.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/migration_smoke/model_5349_stackmlp_k4_migration.json`.
- 2026-05-28 H200 migration load/train smoke passed with env64 and one PPO
  iteration:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_migration_warmstart/2026-05-28_12-20-22_033_stackmlp_k4_migration_load_smoke_env64_iter1_gpu0_seed3303341`.
- 2026-05-28 Longer naive StackMLP K4 warm-start training from the migrated
  checkpoint collapsed gait quality. Final checkpoint `model_5378.pt` failed
  the apples-to-apples dynamic-switch eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/model5378_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`
  (`pass=false`, `zero_fall_ratio=0.0`). The migrated checkpoint before PPO
  still passed the same eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/migrated_model5349_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`
  (`pass=true`).
- 2026-05-28 Added `Task033StackMlpK4FrozenBaseRunner`, which freezes the
  migrated base actor path and trains only the new history/action columns in
  the first layer. H200 env64 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_smoke/2026-05-28_12-40-34_033_frozenbase_focused_env64_iter1_gpu1_seed3303361`.
- 2026-05-28 H200 frozen-base focused training completed 30 iterations:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5378.pt`.
  Dynamic-switch eval at `2.0 m/s` passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`
  (`pass=true`, `zero_fall_ratio=1.0`, `recovery_success_ratio=1.0`).
  Forced persistent dead-grid at `2.0 m/s` improved to `11/12` pass, with
  `right_knee_joint` still failing:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_deadgrid_vx2p0_seed3303500/task033_failure_grid_eval_aggregate.json`
  (`pass=false`, `pass_count=11`, failed `right_knee_joint`,
  `zero_fall_ratio=0.2109375`).

## Review

Status: StackMLP is the only consumer with policy-quality eval evidence in
Task033. Naive PPO on the expanded actor input regressed the base gait.
Frozen-base StackMLP preserves dynamic-switch performance and materially
improves the `2.0 m/s` forced dead-grid blocker to `11/12`, but it is not a
full dead-grid pass because `right_knee_joint` remains below threshold.
