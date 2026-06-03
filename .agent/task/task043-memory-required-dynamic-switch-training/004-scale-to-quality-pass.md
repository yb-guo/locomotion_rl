# 004: Scale To Quality Pass

## Route

If the train smoke is valid but the first candidate checkpoint fails the normal
dynamic-switch quality gate, run additional training chunks. Keep every chunk
bounded by JSON train evidence and an eval triplet.

## Acceptance

- Each training chunk records a train summary and candidate checkpoint.
- Each candidate that is promoted has a normal/zero/stateless eval triplet.
- Stop conditions are explicit:
  - normal mode passes and ablations degrade;
  - normal mode passes but ablations remain tied;
  - training destabilizes clean/dynamic gait;
  - H200/runtime blocker prevents progress.

## Log

- 2026-05-31 Opened.
- 2026-05-31 Ran two H200 scale probes from the residual-only prior:
  - residual-only scope:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_residual_env1024_iter25_seed4300301.json`;
    checkpoint
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_residual_env1024_iter25_seed4300301/model_24.pt`.
  - all-scope:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_all_env1024_iter25_seed4300302.json`;
    checkpoint
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_all_env1024_iter25_seed4300302/model_24.pt`.
  Both train summaries passed the training pipeline and used
  `Task040SequenceAwareTrueTxlPPO` with active sequence update batches.
- 2026-05-31 Normal-mode evals at fixed 1.6 m/s did not pass quality:
  - residual-only model_24:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/residual_env1024_iter25_model24_none_vx1p6_seed4300401.json`,
    `quality_gate_pass=false`, final `lin_vel_error.mean=0.4807707965373993`,
    `yaw_error.mean=0.09130024909973145`.
  - all-scope model_24:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/all_env1024_iter25_model24_none_vx1p6_seed4300402.json`,
    `quality_gate_pass=false`, final `lin_vel_error.mean=0.7321958541870117`,
    `yaw_error.mean=0.2126370221376419`.
- 2026-05-31 Diagnosed a target mismatch before launching longer runs: the
  current Task043 H200 registration uses
  `unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg()`, which sets the
  train command distribution to `lin_vel_x=(1.0, 1.6)`, while the eval gate is
  fixed at 1.6 m/s. Opened 006 to test fixed-1.6 training target alignment.

## Review

Status: blocked by training-target diagnosis.

The scale probes prove the sequence-aware training path runs, but they do not
justify more of the same run. The next minimal experiment is to align the
training command target with the fixed 1.6 m/s eval gate before increasing
iteration count.
