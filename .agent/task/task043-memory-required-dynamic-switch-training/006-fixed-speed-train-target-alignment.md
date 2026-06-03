# 006: Fixed Speed Train Target Alignment

## Route

The first Task043 scale probes trained successfully but did not improve the
fixed 1.6 m/s dynamic-switch eval. Before spending more H200 time, align the
train command target with the eval gate.

Keep the change minimal:

- preserve the existing dynamic failure schedule;
- preserve the same runner, actor, PPO, reward, action, and observation
  contracts;
- change only the Task043 train registration command range from sampled
  `lin_vel_x=(1.0, 1.6)` to fixed `lin_vel_x=(1.6, 1.6)`;
- rerun train smoke and only then scale training again.

## Acceptance

- Local tests prove the Task043 H200 registry patcher inserts a fixed-1.6
  helper and replaces any older range-based Task043 registration.
- H200 patcher rerun updates the Task043 task id in MJLab.
- H200 smoke train summary still passes:
  - `task043_train_pipeline_pass=true`;
  - `runner_cls=Task038TrueTxlMemoryK160Runner`;
  - `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
  - sequence update batches active;
  - no stateless fallback.
- A fixed-1.6 candidate checkpoint is evaluated by the normal/zero/stateless
  dynamic eval triplet before any success claim.

## Log

- 2026-05-31 H200 diagnosis found
  `unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg()` calls
  `_configure_task029_forward_speed_command(cfg, min_speed=1.0, max_speed=1.6)`.
  This is not the same target as the fixed 1.6 m/s eval gate.
- 2026-05-31 Updated
  `task043_register_dynamic_switch_train_stage.py` to insert
  `_task043_dynamic_failure_fixed1p6_env_cfg()`, which wraps the existing
  dynamic-failure env config and sets `twist_cmd.ranges.lin_vel_x=(1.6, 1.6)`.
- 2026-05-31 Local validation passed:
  - `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task043_dynamic_switch_training_contract.py tests\test_agent_inventory.py --tb=short --basetemp pytest_tmp_task043_fixed1p6`
    with sandbox escalation for pytest temp creation: 9 passed.
  - `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent`
    passed.
- 2026-05-31 Synced the fixed-1.6 patcher to H200 and reran it. The H200
  MJLab registry now contains:
  `env_cfg=_task043_dynamic_failure_fixed1p6_env_cfg()` and
  `play_env_cfg=_task043_dynamic_failure_fixed1p6_env_cfg(play=True)`.
- 2026-05-31 H200 fixed-1.6 train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_fixed1p6_smoke_env64_iter1_seed4300501.json`.
  It produced
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_fixed1p6_smoke_env64_iter1_seed4300501/model_0.pt`
  with `task043_train_pipeline_pass=true`,
  `sequence_update_forward_batches=20`, and
  `stateless_fallback_forward_batches=0`.
- 2026-05-31 H200 fixed-1.6 residual-only 1024-env, 25-iteration training
  passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_fixed1p6_residual_env1024_iter25_seed4300601.json`.
  Normal dynamic-switch eval did not pass quality:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/fixed1p6_residual_env1024_iter25_model24_switch_none_vx1p6_seed4300701.json`,
  final `lin_vel_error.mean=0.48685064911842346`.
- 2026-05-31 Continued residual-only training for another 75 iterations from
  the fixed-1.6 model_24:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_fixed1p6_residual_continue_env1024_iter75_seed4300801.json`.
  Normal dynamic-switch eval improved but still did not pass:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/fixed1p6_residual_continue_env1024_iter75_model74_switch_none_vx1p6_seed4300901.json`,
  final `lin_vel_error.mean=0.4760773777961731`,
  `quality_gate_pass=false`.
- 2026-05-31 Ran a fixed-1.6 all-scope 25-iteration contrast:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_fixed1p6_all_env1024_iter25_seed4301001.json`.
  It trained successfully but eval regressed badly:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/fixed1p6_all_env1024_iter25_model24_switch_none_vx1p6_seed4301101.json`,
  final `lin_vel_error.mean=0.876373827457428` and
  `yaw_vel_error.mean=0.33968353271484375`.

## Review

Status: fixed-speed target alignment tested, not passed.

The fixed-speed registration works and the train pipeline remains healthy.
However, target alignment alone does not close the quality gate. Residual-only
longer training gives a small improvement and is still the least bad branch;
all-scope training destabilizes tracking. The next diagnosis should inspect
reward/curriculum pressure and whether the frozen base MLP speed prior limits
residual-only adaptation, rather than adding more memory machinery.
