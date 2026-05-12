# 004: H200 Three-Seed Final

## Goal

Run the longer-horizon curriculum smoke on H200 for three seeds if the dev probe
passes.

## Route

1. Use the same guarded H200 project path.
2. Run curriculum with:
   - `CUDA_VISIBLE_DEVICES=1`;
   - physical GPU `1`;
   - logical device `cuda:0`;
   - seeds `0,1,2`;
   - `updates_per_stage=50`.
3. Record stage-level reset/tilt/height trends and throughput.

## Log

- 2026-05-09 H200 final curriculum smoke:
  - command used `CUDA_VISIBLE_DEVICES=1`;
  - recorded `physical_gpu=1` and `logical_cuda_device=cuda:0`;
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo/outputs/task015/g1_curriculum_ppo/h200-gpu1-final-seeds012-updates50-v1`;
  - seeds `0,1,2`;
  - `updates_per_stage=50`;
  - stages: `standing`, `small_vx`, `small_yaw`, `small_vxyaw`;
  - global updates per seed: `200`;
  - total metrics rows: `600`;
  - artifacts present: `config.json`, `metrics.jsonl`, `summary.json`,
    `final_checkpoint.pt`;
  - `summary.json`: `all_seeds_passed=true`;
  - min collect throughput:
    `10977.639073614948 env_policy_steps_per_sec`;
  - mean final reward over completed stages:
    `1.5364224016666412`.
- Per-seed final summary:
  - seed 0 passed, global updates `200`, final `small_vxyaw` reward
    `1.4989290237426758`, min collect throughput
    `10977.639073614948`;
  - seed 1 passed, global updates `200`, final `small_vxyaw` reward
    `1.4697320461273193`, min collect throughput
    `17115.420033578364`;
  - seed 2 passed, global updates `200`, final `small_vxyaw` reward
    `1.4734476804733276`, min collect throughput
    `17274.843023475016`.
- Residual observed in the final update:
  - completed stages still show `reset_count=1024` and
    `tilt_bad_count=1024` in final rows, with
    `termination_height_bad_count=0`;
  - this does not violate task015 runner acceptance, but it means task015 is a
    longer-horizon curriculum smoke, not a sustained no-fall locomotion claim.

## Review

Status: passed as longer-horizon curriculum smoke.

- H200 final run completed all stages for all requested seeds.
- Throughput, finite/device, artifact, and stage-order acceptance are met.
- Next task should target reducing recurring long-horizon tilt resets rather
  than increasing curriculum length.
