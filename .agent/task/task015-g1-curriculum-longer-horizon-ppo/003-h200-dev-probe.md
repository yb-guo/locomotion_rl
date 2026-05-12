# 003: H200 Dev Probe

## Goal

Run the new curriculum runner on H200 with one seed before spending time on the
three-seed final run.

## Route

1. Sync task015 code to `/root/agent_workspace/project`.
2. Run guarded focused tests.
3. Run curriculum with:
   - `CUDA_VISIBLE_DEVICES=1`;
   - physical GPU `1`;
   - logical device `cuda:0`;
   - `seed=0`;
   - `updates_per_stage=20`.
4. Record blocker if any stage stops.

## Log

- 2026-05-09 H200 focused verification:
  - guarded command under
    `/root/agent_workspace/project/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo`;
  - command:
    `PYTHONPATH=src python -m pytest tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`;
  - result: `20 passed in 22.51s`.
- 2026-05-09 H200 dev probe:
  - command used `CUDA_VISIBLE_DEVICES=1`;
  - recorded `physical_gpu=1` and `logical_cuda_device=cuda:0`;
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo/outputs/task015/g1_curriculum_ppo/h200-gpu1-dev-seed0-updates20-v1`;
  - seed `0`, `updates_per_stage=20`, 4 stages, 80 global updates;
  - `summary.json`: `all_seeds_passed=true`;
  - min collect throughput:
    `12567.562857868475 env_policy_steps_per_sec`;
  - final `small_vxyaw` update:
    `reset_count=0`, `termination_height_bad_count=0`,
    `tilt_bad_count=0`, `root_height_min=0.5744914412498474`.

## Review

Status: passed.

- Dev probe passed the upstream stop rules, so three-seed final verification
  may proceed.
