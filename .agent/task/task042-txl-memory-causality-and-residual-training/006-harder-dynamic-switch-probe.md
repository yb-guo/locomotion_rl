# 006: Harder Dynamic Switch Probe

## Route

Subtask 004 proved the compatible Task042 True-TXL dynamic task can run, but the
easy right-knee 0.4 m/s case did not require memory: normal, zero-residual, and
stateless-memory evals all preserved quality.

This slice probes a harder memory-causality target without changing the task
contract:

- same `txl_residual_only` checkpoint;
- same Task042 True-TXL dynamic MJLab task id;
- `dynamic_case=switch`;
- `lin_vel_x=1.6`;
- same seed and rollout shape across ablations.

The goal is diagnostic, not a pass claim. A useful positive signal would be
normal mode clearly outperforming zero-residual/stateless ablations under the
same harder switch setting.

## Acceptance

- Record normal, zero-residual, and stateless-memory JSON summaries.
- Confirm the dynamic eval still uses `Task038TrueTxlMemoryK160Runner`.
- Compare final-trial fall, root height, base tilt, linear velocity error, yaw
  velocity error, and residual/memory debug fields.
- Do not claim memory causality unless normal mode passes quality and ablated
  modes degrade materially.

## Log

- 2026-05-31 Normal 1.6 m/s switch eval completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/harder_dynamic_probe/txl_residual_only_switch_task042_none_vx1p6_seed4104401.json`.
  Result: `pipeline_pass=true`, `final_trial_pass=true`, `pass=false`,
  `quality_gate_pass=false`, `memory_debug_active=true`,
  `stateful_memory_enabled=true`, `memory_residual_enabled=true`,
  `txl_residual_output_norm=0.8399460911750793`, `fall_ratio=0.0`,
  `gravity_xy.max=0.09833084046840668`,
  `root_z.min=0.7620277404785156`,
  `lin_vel_error.mean=0.4827834665775299`, and
  `yaw_vel_error.mean=0.09547217190265656`. Failure reasons:
  `final_lin_vel_error_too_high` and `yaw_vel_error_mean_regressed_from_trial0`.
- 2026-05-31 Zero-residual 1.6 m/s switch eval completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/harder_dynamic_probe/txl_residual_only_switch_task042_zero_residual_vx1p6_seed4104401.json`.
  Result: `pipeline_pass=true`, `final_trial_pass=true`, `pass=false`,
  `quality_gate_pass=false`, `memory_debug_active=true`,
  `memory_residual_enabled=false`, `txl_residual_output_norm=0.0`,
  `fall_ratio=0.0`, `gravity_xy.max=0.09857313334941864`,
  `root_z.min=0.7611386179924011`,
  `lin_vel_error.mean=0.48645174503326416`, and
  `yaw_vel_error.mean=0.09539386630058289`. Failure reasons:
  `final_lin_vel_error_too_high` and `yaw_vel_error_mean_regressed_from_trial0`.
- 2026-05-31 Stateless-memory 1.6 m/s switch eval completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/harder_dynamic_probe/txl_residual_only_switch_task042_stateless_vx1p6_seed4104401.json`.
  Result: `final_trial_pass=true`, `pass=false`, `quality_gate_pass=false`,
  `stateful_memory_enabled=false`, `memory_debug_active=false`,
  `txl_residual_output_norm=0.9408254623413086`, `fall_ratio=0.0`,
  `gravity_xy.max=0.09973301738500595`,
  `root_z.min=0.760913074016571`,
  `lin_vel_error.mean=0.4801263213157654`, and
  `yaw_vel_error.mean=0.09307045489549637`. Top-level `pipeline_pass=false`
  is expected for this ablation because active memory is intentionally disabled.

## Review

Status: passed for diagnostic coverage, failed for positive memory-causality
evidence.

The harder 1.6 m/s switch case is stable enough to be useful: none of the three
modes falls, and all complete the final trial. It is not a Task042 success
case, because normal mode misses the strict quality gate on forward speed
tracking.

More importantly, ablations still do not hurt behavior. Final linear velocity
error is effectively tied across modes: normal `0.4828`, zero residual
`0.4865`, and stateless memory `0.4801`. This rules out a positive memory
causality claim for the current residual-only checkpoint.

Next boundary: Task042 has answered its question. The current True-TXL memory
path is plumbed, trainable, and measurable, but the training target has not made
stateful memory behaviorally necessary. The next task should train directly on a
memory-required dynamic-switch/multi-trial target instead of continuing to probe
this checkpoint.
