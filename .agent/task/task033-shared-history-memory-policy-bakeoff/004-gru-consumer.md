# 004 GRU Consumer

## Route

Implement a recurrent consumer using the same history stream and reset/done
signals.

Scope:

- Do not introduce a second history buffer.
- Hidden state belongs to the policy backend, but reset ids come from the shared
  env/done stream.
- Prefer any existing recurrent support if MJLab/RSL-RL exposes it; otherwise
  document the minimal runner/storage changes required before implementation.

Eval:

- Same blocker subset as StackMLP.
- Same JSON schema plus recurrent metadata: hidden size, sequence length,
  reset handling.

## Log

- 2026-05-28 Planned as second policy consumer.
- 2026-05-28 Added `Task033GruK4Runner` in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`. It reuses the
  same `actor_history` stream and configures RSL-RL `RNNModel` with
  `rnn_type=gru`, hidden dim `256`, and one recurrent layer for actor and
  critic.
- 2026-05-28 Registered H200 task
  `Unitree-G1-Gripper-Flat-Task033-GruK4-Fast2p0`.
- 2026-05-28 H200 runner construction smoke showed recurrent actor/critic
  setup with actor history input dim `540` and critic input dim `119`.
- 2026-05-28 H200 env64 train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_smoke/2026-05-28_12-00-18_033_gru_k4_env64_iter1_gpu0_seed3303304`.
- 2026-05-28 H200 env8192 one-iteration overhead smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_12-47-40_033_gru_k4_env8192_iter1_gpu1_seed3303314`,
  actor history input dim `540`, recurrent hidden dim `256`, `52312`
  steps/s.

## Review

Status: runner, env64 train smoke, and env8192 cost smoke passed. Checkpoint
migration/training and blocker-subset eval remain pending, so Task033 does not
claim GRU policy quality.
