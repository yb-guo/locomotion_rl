# 005 LocoFormer-Style Consumer

## Route

Implement a minimal tokenized history consumer on top of the shared history
buffer.

Scope:

- This is not full LocoFormer morphology generalization.
- Fixed G1-like topology only.
- Use the existing 31D action and existing eval tasks.
- Tokenization may include joint/body/time tokens derived from actor-visible
  history only.

Non-goals:

- no random morphology;
- no link mass/COM/inertia variation;
- no new simulator;
- no separate eval harness.

## Log

- 2026-05-28 Planned as third policy consumer.
- 2026-05-28 Added a minimal tokenized history actor,
  `Task033HistoryTokenMlpModel`, and `Task033TokenK4Runner` in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`. This is a
  fixed-topology token smoke only: it reshapes K4 actor history into time
  tokens, applies a token projection plus time embedding, and keeps the same
  action/env/eval contract.
- 2026-05-28 Registered H200 task
  `Unitree-G1-Gripper-Flat-Task033-TokenK4-Fast2p0`.
- 2026-05-28 H200 runner construction smoke showed token projection
  `Linear(135, 128)` and actor history input dim `540`.
- 2026-05-28 H200 env64 train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_smoke/2026-05-28_12-00-50_033_token_k4_env64_iter1_gpu0_seed3303305`.
- 2026-05-28 H200 env8192 one-iteration overhead smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_12-48-44_033_token_k4_env8192_iter1_gpu1_seed3303315`,
  actor history input dim `540`, token projection `Linear(135, 128)`,
  `56745` steps/s.

## Review

Status: minimal tokenized consumer train smoke passed. This is not full
LocoFormer morphology generalization. Env8192 cost smoke passed, but trained
checkpoint and blocker-subset eval remain pending.
