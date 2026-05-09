# 005: H200 Three-Seed Verification

## Goal

Run the real task014 PPO smoke on H200 and record pass/fail evidence.

## Route

1. Copy task014 code to:
   `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`.
2. Run remote focused pytest through guarded command.
3. Run PPO smoke through guarded command:
   - seeds `0,1,2`;
   - `n_envs=1024`;
   - `rollout_steps=32`;
   - `ppo_updates=5`;
   - physical GPU `1`;
   - logical device `cuda:0`.
4. Record:
   - command;
   - output path;
   - summary metrics;
   - any warnings;
   - blocker if failed.

## Stop Rules

- If seed 0 fails, do not run seeds 1/2 until diagnosis or fix.
- If any seed has NaN/Inf, stop and record ranked hypotheses.
- If throughput `<10000 env_policy_steps_per_sec`, stop and diagnose CPU sync.
- If params do not change, stop and diagnose optimizer/graph/logprob.
- If output lands outside project workspace, stop and delete only if explicitly
  safe inside project workspace.

## Verification

- `metrics.jsonl` has 15 rows: 3 seeds x 5 updates.
- `summary.json` marks all seeds passed.
- `final_checkpoint.pt` exists.
- Task log includes key metrics and output paths.

## Log

Pending implementation.

## Review

Status: pending.
