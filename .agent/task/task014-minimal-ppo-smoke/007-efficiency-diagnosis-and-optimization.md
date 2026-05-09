# 007: Efficiency Diagnosis And Optimization

## Goal

Extend task014 with a small performance diagnosis loop and optimize the hot
path without changing the task013 env contract or task014 smoke scope.

This subtask targets training-loop efficiency, not walking quality.

## Route

1. Reproduce the task014 H200 smoke efficiency baseline.
2. Record ranked hypotheses before changing code:
   - collect timing is asynchronous and too optimistic without CUDA sync;
   - rollout finite checks create repeated reductions in the hot loop;
   - PPO metrics use repeated `.item()` syncs per minibatch;
   - fall/reset frequency dominates env-side cost.
3. Instrument metrics enough to separate collect and update wall time:
   - record `collect_time_s`;
   - record `update_time_s`;
   - keep `collect_env_policy_steps_per_sec`;
   - keep `update_samples_per_sec`.
4. Optimize only trainer-local synchronization:
   - synchronize timing boundaries for real CUDA timing;
   - batch rollout finite checks after stack;
   - batch PPO metric scalar conversion after minibatches.
5. Re-run local tests.
6. Re-run H200 focused tests.
7. Re-run H200 PPO smoke on physical GPU 1.
8. Update task014 log and review with before/after evidence.

## Stop Rules

- If synchronized collect throughput drops below the task014 acceptance threshold
  of `10000 env_policy_steps_per_sec`, stop and record blocker.
- If any seed fails, do not call the optimization passed.
- If the optimization touches `GenesisG1SceneBackend`, stop.
- If the route needs reward tuning, curriculum, render, GIF, SONIC, ONNX,
  planner, LocoFormer, downloads, or `/mnt/workspace*` writes, stop.

## Verification

- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass.
- H200 PPO smoke passes seeds `0,1,2`.
- `metrics.jsonl` includes `collect_time_s` and `update_time_s`.
- Review records whether the bottleneck is trainer sync or env/reset behavior.

## Log

- 2026-05-09 Baseline symptom from task014 v2:
  - synchronized timing was not present;
  - minimum reported collect throughput was
    `12201.757460784085 env_policy_steps_per_sec`;
  - total smoke command wall time was about `73.3s`;
  - PPO update throughput was high enough that collect/env path was the likely
    bottleneck.
- Ranked hypotheses:
  1. If collect timing was asynchronous, adding CUDA synchronization would make
     the metric more honest and expose true collect wall time.
  2. If rollout finite checks were expensive, moving them from per-step
     reductions to post-stack batch checks would improve smoke wall time while
     preserving NaN/Inf detection.
  3. If PPO metric `.item()` calls were expensive, accumulating metrics on GPU
     and converting once per update would improve update throughput.
  4. If fall/reset frequency dominates, trainer-local changes would help but
     final metrics would still show high `fallen_count`.
- Implemented:
  - `synchronize_device()` for CUDA timing boundaries;
  - batched rollout finite checks after tensor stack;
  - PPO metric accumulation on device with one scalar conversion per metric;
  - per-update `collect_time_s` and `update_time_s` in `metrics.jsonl`.
- Local verification:
  - focused tests: `6 passed, 4 skipped`;
  - full pytest: `178 passed, 4 skipped`.
- H200 focused verification:
  - `10 passed in 4.73s`.
- H200 optimized smoke:
  - command used `CUDA_VISIBLE_DEVICES=1`;
  - physical GPU `1`, logical device `cuda:0`;
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-three-seed-v3`;
  - `metrics.jsonl`: 15 rows;
  - `summary.json`: `all_seeds_passed=true`;
  - `final_checkpoint.pt`: exists;
  - synchronized min collect throughput:
    `19381.815355781637 env_policy_steps_per_sec`;
  - mean final reward mean: `1.4781097173690796`;
  - observed total smoke command wall time: about `53.6s`.
- Per-seed final update metrics:
  - seed 0: collect `29075.040446951498/s`, collect time
    `1.1270147692412138s`, update time `0.043640587478876114s`,
    `fallen_count=2048`;
  - seed 1: collect `19381.815355781637/s`, collect time
    `1.6906569069251418s`, update time `0.044599851593375206s`,
    `fallen_count=2048`;
  - seed 2: collect `20574.187580320217/s`, collect time
    `1.592675281688571s`, update time `0.04406619444489479s`,
    `fallen_count=2048`.

## Review

Status: passed.

- The likely trainer-side bottleneck was repeated synchronization for
  validation and metric extraction.
- The optimization preserved task014 acceptance and improved observed smoke
  wall time.
- Synchronized collect throughput still exceeds `10000 env_policy_steps_per_sec`.
- PPO update is not the current bottleneck: final update times were about
  `0.044s`, while collect time was about `1.1s-1.7s`.
- Remaining bottleneck is env/reset behavior: every final update still reports
  `fallen_count=2048`, so task015 should diagnose stability/reset frequency
  before larger learning runs.
