# 003: H200 Six-Case Control/Pose Gate

## Goal

Run the highest-value six zero-action experiments and decide which diagnostic
branch is justified next.

## Route

1. Sync task019 to the H200 project path.
2. Run H200 focused tests.
3. Run six gate cases:

| Case | Pose profile | Root z | Control mode |
| --- | --- | ---: | --- |
| 1 | `current` | 1.20 | `genesis_position` |
| 2 | `current` | 1.20 | `genesis_position_resend_physics` |
| 3 | `current` | 1.20 | `custom_pd_torque` |
| 4 | `unitree_gym` | 0.80 | `genesis_position` |
| 5 | `unitree_gym` | 0.80 | `genesis_position_resend_physics` |
| 6 | `unitree_gym` | 0.80 | `custom_pd_torque` |

`current` means the task018 failing operational baseline: task018
`tall_crouch` leg values with `root_z=1.20`, not the raw YAML default pose.

4. Apply stop rules before any gain/force or reset/contact follow-up.

## Log

- 2026-05-11 Synced task019 to H200 project:
  `/root/agent_workspace/project/h200-locomotion-lab-task019-g1-zero-action-standing-causality-diagnosis`.
- 2026-05-11 H200 focused tests initially exposed two fake-runtime test bugs
  that local Windows skipped because torch was unavailable. Coding subagent
  fixed the fake backend to use `self.torch` and to mirror real backend motor
  multiplier defaults.
- 2026-05-11 H200 focused tests passed:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 900 python -m pytest tests/test_g1_zero_action_standing_causality.py tests/test_vectorized_genesis_backend.py tests/test_g1_no_update_ppo_causality.py tests/test_g1_velocity_tracking_env.py -q -p no:cacheprovider`
  returned `38 passed in 1.67s`.
- 2026-05-11 H200 six-case gate completed under:
  `/root/agent_workspace/project/h200-locomotion-lab-task019-g1-zero-action-standing-causality-diagnosis/outputs/task019/zero_action_standing_causality/h200-gpu1-seed0-six-case-gate-v1`.

Run settings:

- `CUDA_VISIBLE_DEVICES=1`;
- `physical_gpu=1`;
- `logical_cuda_device=cuda:0`;
- `n_envs=1024`;
- `chunks=50`;
- `chunk_steps=32`;
- `seed=0`;
- zero normalized action;
- no policy model;
- no PPO update.

Six-case result:

| Pose | Root z | Control mode | first tilt | final reset | final tilt | termination height final | root mean | root min | upright mean | joint error rms | joint vel rms | force sat |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current` | 1.20 | `genesis_position` | 2 | 1024 | 1024 | 0 | 0.794 | 0.330 | 0.818 | 0.0979 | 0.4605 | 0.000 |
| `current` | 1.20 | `genesis_position_resend_physics` | 2 | 1024 | 1024 | 0 | 0.794 | 0.330 | 0.818 | 0.0979 | 0.4605 | 0.000 |
| `current` | 1.20 | `custom_pd_torque` | 2 | 1024 | 1024 | 0 | 0.793 | 0.324 | 0.816 | 0.0973 | 0.4756 | 0.000 |
| `unitree_gym` | 0.80 | `genesis_position` | 1 | 1024 | 1024 | 0 | 0.634 | 0.303 | 0.792 | 0.1236 | 0.4208 | 0.000 |
| `unitree_gym` | 0.80 | `genesis_position_resend_physics` | 1 | 1024 | 1024 | 0 | 0.634 | 0.303 | 0.792 | 0.1236 | 0.4209 | 0.000 |
| `unitree_gym` | 0.80 | `custom_pd_torque` | 1 | 1024 | 1024 | 0 | 0.634 | 0.303 | 0.792 | 0.1233 | 0.4271 | 0.000 |

Key observations:

- No six-case candidate passed.
- Resending Genesis position targets every physics substep did not change the
  failure profile.
- Custom torque PD did not rescue either pose profile.
- `unitree_gym` pose was worse than the task018 current baseline, failing in
  chunk 1 instead of chunk 2.
- `force_saturation_ratio=0.0` in final rows, so this first gate does not
  support force-limit saturation as the immediate explanation.
- All final `termination_height_bad_count` values are zero; the reset path
  remains tilt/fall, not hard height termination.

Stop-rule outcome:

- Broad control/pose gate is complete.
- Because all six cases failed, task019 may continue only to targeted
  gain/force or reset/contact diagnostics. PPO remains disallowed.
- The best follow-up anchor is still `current` pose, because it fails later
  and has lower final joint error than `unitree_gym`.

## Review

Status: passed for six-case gate.

- Final read-only reviewer found no blocking findings.
- Reviewer agreed all six failed and that follow-up must stay limited to
  targeted gain/force or reset/contact diagnostics, with PPO disallowed.
