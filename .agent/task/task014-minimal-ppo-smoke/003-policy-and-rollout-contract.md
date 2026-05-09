# 003: Policy And Rollout Contract

## Goal

Bind task13 env to a minimal actor-critic contract without coupling env to
policy/trainer internals.

## Route

1. Add separate actor/value MLPs:
   - input `90`;
   - output action `27`;
   - hidden `2x128`;
   - activation `tanh` or `elu`, fixed by implementation tests.
2. Add tanh Gaussian actor:
   - sample raw normal;
   - tanh squash to `[-1, 1]`;
   - corrected log-prob;
   - learned log std.
3. Add rollout collection:
   - obs `[steps + 1, n_envs, 90]`;
   - actions `[steps, n_envs, 27]`;
   - rewards/dones/values/logprobs `[steps, n_envs]`.
4. Keep tensors on `cuda:0` in H200 runs.
5. Record collection throughput separate from PPO update throughput.

## Stop Rules

- If env action, obs, reward, done shapes do not match task13 contract, stop.
- If rollout storage forces CPU conversion before final metrics, stop.
- If action distribution emits NaN/Inf or out-of-range action after tanh, stop.

## Verification

- Unit tests for actor/value shape.
- Unit test for rollout storage shape.
- H200 smoke device report includes obs/action/reward/done/logprob/value.

## Log

- 2026-05-09 Added separate MLP actor and value networks:
  - obs dim `90`;
  - action dim `27`;
  - hidden layers `2x128`;
  - `Tanh` activations;
  - learned `log_std`.
- Added rollout collection for:
  - observations `[steps, n_envs, 90]`;
  - actions `[steps, n_envs, 27]`;
  - rewards/dones/values/log_probs `[steps, n_envs]`;
  - next observation and next value for bootstrap.
- H200 smoke confirmed tensor device residency with
  `tensor_device_ok=true` in every metrics row.
- Collection throughput was recorded separately from update throughput in
  `metrics.jsonl`.
- Rollout diagnostics aggregate finite and terminal flags after the rollout
  instead of synchronizing once per simulation step.

## Review

Status: passed.

- Actor/value and rollout tensors match the task013 env contract.
- Actions are tanh-squashed to `[-1, 1]` before env step.
- No rollout storage path forced per-step CPU conversion before PPO math;
  scalar diagnostics are converted at rollout/update reporting boundaries.
