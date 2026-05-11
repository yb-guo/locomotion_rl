# 003: H200 No-Update Baseline

## Goal

Run the H200 no-update matrix and decide whether PPO-update isolation is
allowed by the stop rules.

## Route

1. Sync task018 to H200 project path.
2. Run H200 focused tests.
3. Run seed-0 no-update probes:
   - `zero_action`;
   - `untrained_mean_action`;
   - `untrained_sampled_action`.
4. Compare reset waves against task017 PPO u50.
5. Apply stop rules before subtask 004.

## Log

- 2026-05-11 Synced task018 to H200 project:
  `/root/agent_workspace/project/h200-locomotion-lab-task018-g1-no-update-ppo-causality-diagnosis`.
- 2026-05-11 H200 focused tests passed:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 900 python -m pytest tests/test_g1_no_update_ppo_causality.py tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_action_control_semantics.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  returned `43 passed in 5.06s`.
- 2026-05-11 H200 seed-0 no-update matrix completed under:
  `/root/agent_workspace/project/h200-locomotion-lab-task018-g1-no-update-ppo-causality-diagnosis/outputs/task018/no_update_ppo_causality/h200-gpu1-seed0-no-update-v1`.

Run settings:

- `CUDA_VISIBLE_DEVICES=1`;
- `physical_gpu=1`;
- `logical_cuda_device=cuda:0`;
- `n_envs=1024`;
- `chunks=50`;
- `chunk_steps=32`;
- `seed=0`;
- `default_pose=tall_crouch`;
- `root_z=1.20`;
- `action_scale_mult=0.10`;
- `action_joint_group=all`;
- `termination_height_min=0.20`.

Seed-0 no-update matrix:

| Mode | first tilt chunk | max reset | mean reset | final reset | final tilt | termination height final | action abs mean | action abs max | root mean | root min | upright | min env-policy steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_action` | 2 | 1024 | 348.16 | 1024 | 1024 | 0 | 0.0000 | 0.0000 | 0.794 | 0.330 | 0.818 | 41051 |
| `untrained_mean_action` | 2 | 1024 | 348.16 | 1024 | 1024 | 0 | 0.0024 | 0.0129 | 0.795 | 0.330 | 0.818 | 47787 |
| `untrained_sampled_action` | 2 | 1024 | 348.16 | 1024 | 1024 | 0 | 0.0652 | 0.3994 | 0.800 | 0.315 | 0.827 | 12104 |

Interpretation:

- `zero_action` alone reproduces the same chunk-2 tilt/reset wave as task017
  PPO u50.
- The no-update `zero_action` metrics match task017 u50 shape:
  `first_tilt=2`, `mean_reset=348.16`, `final_reset=1024`, and
  `final_tilt=1024`.
- `termination_height_bad_count=0` in all modes, so the immediate failure path
  is tilt/fall reset rather than hard height reset.
- The task018 stop rule is triggered by `zero_action`: PPO-update isolation is
  not allowed in subtask 004.

## Review

Status: passed.

- Final read-only reviewer found no blocking findings.
- H200 evidence is sufficient for the no-update baseline boundary.
