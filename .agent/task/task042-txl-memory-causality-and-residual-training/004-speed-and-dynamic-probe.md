# 004: Speed and Dynamic Probe

## Route

After clean memory-ablation eval is stable, probe whether the same checkpoint
survives a slightly broader matrix:

- clean 1.2 m/s;
- clean 2.0 m/s;
- one specified dynamic-switch case already used in prior tasks.

This is not the full robustness benchmark. It only decides whether the next
task should scale training/eval or change the policy contract.

The True-TXL dynamic probe must use a Task042-specific MJLab registration. The
older `Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DynamicMotorFailure-Fast1p6`
uses `Task037TxlMemoryK160DeterministicRunner`, so it is not compatible with a
Task041/Task042 `Task038TrueTxlMemoryModel` checkpoint.

## Acceptance

- Each probe writes a JSON summary path.
- Failures are not hidden behind a broad pass claim.
- Review recommends the next task boundary based on evidence.
- Dynamic eval uses `Task038TrueTxlMemoryK160Runner`, not the older Task037
  TXL-style runner.

## Log

- 2026-05-30 Opened. Runs use the `txl_residual_only` checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/logs_txl_residual_only_env1024_iter5/model_4.pt`.
- 2026-05-30 Clean 1.2 m/s probe completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_none_vx1p2_seed4104301.json`.
  Result: `pass=true`, `quality_gate_pass=true`, `final_trial_pass=true`,
  `pipeline_pass=true`, `fall_ratio=0.0`, `gravity_xy.max=0.1334141343832016`,
  `root_z.min=0.7423623204231262`,
  `lin_vel_error.mean=0.3558266758918762`,
  `yaw_vel_error.mean=0.11504746228456497`, and
  `txl_residual_output_norm=0.9224275350570679`.
- 2026-05-30 Clean 2.0 m/s probe completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_none_vx2p0_seed4104302.json`.
  Result: `pass=false`, `quality_gate_pass=false`,
  `final_trial_pass=true`, `pipeline_pass=true`, `fall_ratio=0.0`,
  `gravity_xy.max=0.34090450406074524`,
  `root_z.min=0.6932728886604309`,
  `lin_vel_error.mean=0.7326226830482483`,
  `yaw_vel_error.mean=0.1743871420621872`, and
  `txl_residual_output_norm=0.7724313139915466`. Failure reasons:
  `final_lin_vel_error_too_high`, `gravity_xy_max_regressed_from_trial0`,
  and `root_z_min_regressed_from_trial0`.
- 2026-05-30 Dynamic right-knee probe attempted:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_right_knee_dynamic_none_vx0p4_seed4104201.json`.
  Result: `pipeline_pass=false`, `failure_reasons=["eval_wrapper_exception"]`,
  with `RuntimeError('dynamic_motor_failure event is absent')`. This is an env
  registration/setup failure, not a policy-quality result.
- 2026-05-31 Added a Task042-specific True-TXL dynamic task registration script:
  `task042_register_true_txl_dynamic_stage.py`. It registers
  `Unitree-G1-Gripper-Flat-Task042-TrainTrueTxlDynamicMotorFailure-Fast1p6`
  with `unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg()` and
  `runner_cls=Task038TrueTxlMemoryK160Runner`. H200 registry confirmation:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/tasks/velocity/config/g1_gripper/__init__.py`
  contains the new task id and runner.
- 2026-05-31 Dynamic right-knee normal eval completed on the new Task042 task:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_right_knee_dynamic_task042_none_vx0p4_seed4104204.json`.
  Result: `pass=true`, `task042_pass=true`, `pipeline_pass=true`,
  `quality_gate_pass=true`, `memory_debug_active=true`,
  `stateful_memory_enabled=true`, `memory_residual_enabled=true`,
  `txl_residual_output_norm=1.095429539680481`, `fall_ratio=0.0`,
  `gravity_xy.max=0.06805049628019333`,
  `root_z.min=0.7738457918167114`,
  `lin_vel_error.mean=0.17293773591518402`, and
  `yaw_vel_error.mean=0.14015839993953705`.
- 2026-05-31 Dynamic right-knee zero-residual eval completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_right_knee_dynamic_task042_zero_residual_vx0p4_seed4104204.json`.
  Result: `pass=true`, `task042_pass=true`, `pipeline_pass=true`,
  `quality_gate_pass=true`, `memory_debug_active=true`,
  `memory_residual_enabled=false`, `txl_residual_output_norm=0.0`,
  `fall_ratio=0.0`, `gravity_xy.max=0.0679953470826149`,
  `root_z.min=0.7735569477081299`,
  `lin_vel_error.mean=0.1747198849916458`, and
  `yaw_vel_error.mean=0.13876710832118988`.
- 2026-05-31 Dynamic right-knee stateless-memory eval completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_right_knee_dynamic_task042_stateless_vx0p4_seed4104204.json`.
  Result: `quality_gate_pass=true`, `task042_pass=true`,
  `stateful_memory_enabled=false`, `memory_debug_active=false`,
  `txl_residual_output_norm=1.4166319370269775`, `fall_ratio=0.0`,
  `gravity_xy.max=0.0695144385099411`,
  `root_z.min=0.7738620042800903`,
  `lin_vel_error.mean=0.1728627234697342`, and
  `yaw_vel_error.mean=0.135950967669487`. Top-level `pass=false` and
  `pipeline_pass=false` are expected for this ablation because the active-memory
  debug contract is intentionally disabled.

## Review

Status: passed for the Task042 probe boundary.

The checkpoint can complete 1.2 m/s clean walking and can finish 2.0 m/s
without falling, but it does not satisfy the strict 2.0 m/s quality gate.

The right-knee dynamic single-onset eval is now executable through a compatible
True-TXL Task042 task id and passes in normal mode. However, both
zero-residual and stateless-memory ablations still preserve locomotion quality.
That is negative memory-causality evidence: the current checkpoint is robust to
this right-knee dynamic case, but the measured behavior does not depend on the
TXL residual or stateful memory path.
