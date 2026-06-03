# 003: Sequence-Aware PPO Update Smoke

## Route

Prove the fixed update boundary can execute one PPO update and avoid the
Task039 stateless fallback.

## Contract

The smoke uses the existing Task038 train true-TXL runner task id:

```text
Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke
```

Only the algorithm class is changed by the smoke config:

```text
h200_locomotion_lab.training.rsl_history_wrapper:Task040SequenceAwareTrueTxlPPO
```

Pass requires:

- `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
- `learn_returned=true`;
- `stateless_fallback_forward_batches == 0`;
- `stateless_fallback_forward_samples == 0`;
- `sequence_update_forward_batches > 0`;
- `algorithm_debug.sequence_update_batches > 0`;
- `algorithm_debug.last_loss_dict` is present;
- log directory exists.

No quality claim is allowed:

- `quality_claim:false`
- `training_claim:false`
- `eval_claim:false`
- `reproduction_claim:false`
- `superiority_claim:false`

## Log

- 2026-05-30 Added
  `h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke`, with
  local tests for parse/preflight/config mutation/pass gates/no-overclaim
  behavior.
- 2026-05-30 H200 smoke passed at
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task040/sequence_txl_ppo_update_smoke/sequence_txl_ppo_update_env8_steps2_iter1.json`.
  The run used env8, rollout steps 2, one mini-batch, one iteration on
  `cuda:0`. It wrote `model_0.pt`, returned `pass=true`, and recorded
  `algorithm_debug.last_loss_dict`.

## Review

Status: implemented and verified. This is smoke/plumbing evidence only.
