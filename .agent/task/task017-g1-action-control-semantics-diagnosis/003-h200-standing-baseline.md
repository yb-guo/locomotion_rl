# 003: H200 Standing Baseline

## Goal

Reproduce the task016 standing reset wave with the task017 standing-only runner.

## Route

1. Sync task017 to H200 project path.
2. Run H200 focused tests.
3. Run standing-only baseline seed 0.
4. Stop if baseline does not reproduce tilt resets.

## Log

- 2026-05-11 Remote project:
  `/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis`.
- 2026-05-11 H200 focused tests:
  `PYTHONPATH=src python -m pytest tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_action_control_semantics.py tests/test_g1_tilt_reset_ablation.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  passed with 40 passed in 2.59s.
- 2026-05-11 H200 standing-only u10 run:
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, run id
  `h200-gpu1-seed0-standing-u10-v1`.
- 2026-05-11 Baseline reproduced the early reset wave:

| updates | first tilt | max reset | mean reset | final reset | final tilt | final root mean | final root min | final upright |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 1024 | 307.20 | 0 | 0 | 0.807 | 0.784 | 1.000 |

Interpretation:

- The fast loop reproduced the update-2 reset wave.
- The 10-update baseline recovered by the final row, so it is a fast
  reproduction of early reset instability, not the full task016 final-collapse
  symptom.
- For that reason, Router added a targeted 50-update standing-only check in
  subtask 004.

## Review

Status: passed.

- Stop rule satisfied: baseline reproduced before interpreting variants.
- Hardware isolation recorded: physical GPU 1, logical `cuda:0`,
  `CUDA_VISIBLE_DEVICES=1`.
