# Task 053: Procedural Specialist Normal Walking

## Route

Train one masked MLP/PPO specialist per procedural family on a fixed primitive
topology first.  Use the MuJoCo shard for the reference loop; MJLab on the RTX
5060 Ti can replace the shard internals without changing the task, policy, or rollout
contracts.

## Log

- 2026-08-19: Added `WholeBodyMuJoCoShard`, masked MLP actor-critic, PPO
  trainer, deterministic quality-gate evaluator, and a checkpoint-writing
  smoke CLI.
- 2026-08-19: One-update biped CPU smoke completed with finite PPO metrics and
  a schema-hashed checkpoint.

## Review

The biped and quadruped one-update PPO smoke paths both complete with finite
losses and schema-hashed checkpoints.  The predeclared 100-trial × 10-second
quality gates are intentionally not marked passed: this RTX workstation has
not completed the long-run gate, and the reference shard is CPU-bound.
