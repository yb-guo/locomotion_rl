# 002: H200 Train Smoke

## Route

After local contract tests pass, sync the Task043 files to H200 and run a short
train smoke. Start from the Task042 residual-only checkpoint so this tests the
dynamic training path rather than scratch gait learning.

## Acceptance

- Registry patcher runs on H200 and the Task043 task id is present.
- Train smoke returns and writes JSON.
- JSON records active sequence update counters and no stateless fallback.
- Produced checkpoint path exists.
- No quality or memory-causality claim is made from train smoke alone.

## Log

- 2026-05-31 Opened.
- 2026-05-31 H200 registry patcher ran and inserted
  `Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6`
  in
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/tasks/velocity/config/g1_gripper/__init__.py`.
- 2026-05-31 H200 `--help` for
  `python -m h200_locomotion_lab.tools.task043_dynamic_switch_train` passed.
- 2026-05-31 H200 train smoke completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_smoke_env64_iter1.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_smoke_env64_iter1/model_0.pt`.
  Result: `task043_train_pipeline_pass=true`, `train_pipeline_pass=true`,
  `failure_reasons=[]`, `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`,
  `algorithm_class=Task040SequenceAwareTrueTxlPPO`,
  `sequence_update_forward_batches=20`,
  `algorithm_debug.sequence_update_batches=20`,
  `stateless_fallback_forward_batches=0`,
  `checkpoint_exists=true`, `trainable_parameter_delta_norm=1.0637896060943604`,
  `frozen_parameter_delta_norm=0.0`, and
  `frozen_obs_normalizer_delta_norm=0.0`.

## Review

Status: passed for train smoke. This is not a quality or memory-causality pass;
it only proves the Task043 dynamic training path is valid.
