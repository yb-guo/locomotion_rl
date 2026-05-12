# 003: H200 Baseline Reproduction

## Goal

Reproduce the task015 long-horizon tilt reset wave with task016 diagnostics.

## Route

1. Sync task016 to H200 project path.
2. Run H200 focused tests.
3. Run baseline seed-0 50-update curriculum with diagnostics.
4. Stop if baseline does not reproduce tilt resets.

## Log

- 2026-05-09 Remote project:
  `/root/agent_workspace/project/h200-locomotion-lab-task016-g1-long-horizon-tilt-reset-ablation`.
- 2026-05-09 H200 focused tests:
  `PYTHONPATH=src python -m pytest tests/test_g1_tilt_reset_ablation.py tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  passed with 27 passed in 23.01s.
- 2026-05-09 H200 run:
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, run id
  `h200-gpu1-seed0-updates50-v2`.
- 2026-05-09 Baseline reproduced the task015 reset wave. Aggregate summary:
  `/root/agent_workspace/project/h200-locomotion-lab-task016-g1-long-horizon-tilt-reset-ablation/outputs/task016/tilt_reset_ablation/h200-gpu1-seed0-updates50-v2/summary.json`.

Baseline seed-0 evidence:

| Stage | first tilt | final reset | final tilt | max reset | mean reset | final root mean | final root min | final upright |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standing | 2 | 1024 | 1024 | 1024 | 348.16 | 0.804 | 0.316 | 0.833 |
| small_vx | 2 | 1024 | 1024 | 1024 | 348.16 | 0.807 | 0.316 | 0.838 |
| small_yaw | 2 | 1024 | 1024 | 1024 | 348.16 | 0.807 | 0.316 | 0.839 |
| small_vxyaw | 2 | 1024 | 1024 | 1024 | 348.16 | 0.805 | 0.315 | 0.835 |

`termination_height_bad_count=0` for all baseline final rows.

## Review

Status: passed.

- Stop rule satisfied: baseline reproduced before interpreting tuning variants.
- Hardware isolation recorded: physical GPU 1, logical `cuda:0`,
  `CUDA_VISIBLE_DEVICES=1`.
