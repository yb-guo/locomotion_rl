# 004: Hidden-Fault Train Target

## Route

Only after the baseline-negative gate proves the eval target is meaningful,
train on hidden-fault schedules that require the policy to infer actuator state
from delayed observation/action response history.

The training target should preserve the current True-TXL/sequence-aware PPO
stack and avoid changing reward/action/visible-observation contracts unless a
review explicitly documents why the contract must change.

## Acceptance

- Train summary records active sequence-aware PPO and no stateless fallback.
- Hidden fault labels are not placed in actor observations.
- Checkpoint path exists.
- Training summary does not claim success without triplet eval.

## Log

- 2026-05-31 Planned after 003.
- 2026-05-31 H200 train smoke passed from the Task043 bridge:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_smoke_env64_iter1_seed4400101.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_smoke_env64_iter1_seed4400101/model_0.pt`.
  The runner/actor/algorithm matched the expected True-TXL stack, no fault
  labels were exposed through actor observation terms, and
  `txl_residual_output_norm_last` was nonzero. This was only a smoke.
- 2026-05-31 H200 1024-env 25-iteration train completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_model5349_hidden_fault_env1024_iter25_seed4400301.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt`.
  Sequence-aware update counters were active, stateless fallback forward
  batches stayed zero, and residual/memory module delta norms were nonzero.
  This does not close Task044 because acceptance requires triplet eval.

## Review

Status: closed for train-pipeline evidence.

The training wrapper and H200 registration work, but the target is not accepted
until subtask 005/006 produces normal quality plus degraded ablations.
