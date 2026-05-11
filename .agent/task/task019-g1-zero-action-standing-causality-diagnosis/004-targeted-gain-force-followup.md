# 004: Targeted Gain/Force Follow-Up

## Goal

Only after the six-case gate, run targeted gain/force diagnostics around the
best pose/control candidate.

## Route

1. Confirmed subtask 003 permits this route.
2. Choose one best pose/control candidate, not a cross-product:
   `current + root_z=1.20`; control mode should be selected deliberately per
   probe, not swept as a broad cross-product.
3. Implement named `--gain-profile` profiles for targeted follow-up:
   `current`, `global_kv_2x`, `global_kv_4x`,
   `global_kp_0_5x_kv_2x`, `ankle_kp_2x_kv_2x`,
   `knee_ankle_kp_2x_kv_2x`, `unitree_leg_gains`, and
   `force_limit_2x`.
4. Apply the selected profile to Genesis position modes through
   `set_dofs_kp`, `set_dofs_kv`, and `set_dofs_force_range`; use the same
   diagnostic gains for `custom_pd_torque`.
5. Test named gain/force profiles one variable at a time:
   - Unitree-style leg gains;
   - higher damping;
   - stiffer knee/ankle;
   - higher force limit.
6. Stop when one profile passes or the gain/force hypothesis is falsified.

## Log

- 2026-05-11 Six-case gate permits this route because all six control/pose
  candidates failed.
- The next gain/force follow-up should anchor on `current + root_z=1.20`
  because it failed later than `unitree_gym`.
- First gate final rows showed `force_saturation_ratio=0.0`, so a pure
  force-limit increase is lower priority than damping/stiffness profile
  changes.
- 2026-05-11 Implemented named `--gain-profile` support in the standalone
  probe without editing `VectorizedGenesisBackend`.
- 2026-05-11 Gain profile is included in config, per-chunk metrics rows, and
  summary output.
- 2026-05-11 Custom PD torque now reads the same diagnostic gain profile that
  is applied to Genesis robot DoFs.
- 2026-05-11 Added `force_limit_2x` after read-only reviewer found the
  subtask route required a higher force-limit diagnostic; it doubles all
  `force_limits` while leaving `kp` and `kv` unchanged.
- 2026-05-11 Local focused tests passed:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_zero_action_standing_causality.py -q -p no:cacheprovider`
  -> `12 passed, 1 skipped in 0.27s`.
- 2026-05-11 Local task-scoped tests passed:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_zero_action_standing_causality.py tests\test_vectorized_genesis_backend.py -q -p no:cacheprovider`
  -> `24 passed, 1 skipped`.
- 2026-05-11 H200 focused tests passed on physical GPU 1 / logical
  `cuda:0`:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 900 python -m pytest tests/test_g1_zero_action_standing_causality.py tests/test_vectorized_genesis_backend.py tests/test_g1_no_update_ppo_causality.py tests/test_g1_velocity_tracking_env.py -q -p no:cacheprovider`
  -> `41 passed in 1.65s`.
- 2026-05-11 H200 gain-force matrix ran under
  `/root/agent_workspace/project/h200-locomotion-lab-task019-g1-zero-action-standing-causality-diagnosis/outputs/task019/zero_action_standing_causality/h200-gpu1-seed0-gain-force-v1`.
  All rows used `pose_profile=current`, `root_z=1.20`,
  `control_mode=genesis_position`, `n_envs=1024`, `chunks=50`,
  `chunk_steps=32`, `seed=0`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`.

| Gain profile | First tilt chunk | Max reset | Final reset | Final tilt | Final root mean | Final root min | Final upright mean | Joint err RMS | Joint vel RMS | Force sat | Strict pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `current` | 2 | 1024 | 1024 | 1024 | 0.794 | 0.330 | 0.818 | 0.0979 | 0.4604 | 0.000 | no |
| `global_kv_2x` | 2 | 1024 | 1024 | 1024 | 0.677 | 0.341 | 0.774 | 0.1201 | 0.3221 | 0.000 | no |
| `global_kv_4x` | 2 | 1024 | 0 | 0 | 0.786 | 0.775 | 0.994 | 0.0437 | 0.0809 | 0.000 | no |
| `global_kp_0_5x_kv_2x` | 2 | 1024 | 0 | 0 | 0.788 | 0.784 | 0.999 | 0.0653 | 0.0712 | 0.000 | no |
| `ankle_kp_2x_kv_2x` | 3 | 1024 | 1024 | 1024 | 0.753 | 0.298 | 0.743 | 0.0710 | 0.2789 | 0.000 | no |
| `knee_ankle_kp_2x_kv_2x` | 3 | 1024 | 0 | 0 | 0.759 | 0.698 | 0.941 | 0.0734 | 0.1399 | 0.000 | no |
| `unitree_leg_gains` | 3 | 1024 | 0 | 0 | 0.790 | 0.788 | 0.999 | 0.0423 | 0.2399 | 0.000 | no |
| `force_limit_2x` | 2 | 1024 | 1024 | 1024 | 0.794 | 0.330 | 0.818 | 0.0979 | 0.4604 | 0.000 | no |

- 2026-05-11 No profile strictly passed because every profile had an early
  full-env reset wave (`max_reset_count=1024` and `max_tilt_bad_count=1024`).
- 2026-05-11 `global_kv_4x`, `global_kp_0_5x_kv_2x`,
  `knee_ankle_kp_2x_kv_2x`, and `unitree_leg_gains` recovered to final
  chunks with `final_reset_count=0` and `final_tilt_bad_count=0`.
- 2026-05-11 `force_limit_2x` matched baseline while
  `force_saturation_ratio=0.0`, so the force-limit hypothesis is not
  supported.

## Review

Status: H200 gain-force follow-up complete; task remains in progress because
strict pass evidence is still missing and reset/contact settling is now the
next diagnostic boundary.

- Boundary check: no PPO imports or actor/update path were added.
- Ownership check: edits stayed within the task004 source, tests, and subtask
  doc scope.
- Verification evidence: focused local tests cover profile values anchored on
  `G1_27DOF_NOHAND_ACTUATOR_ORDER`, fake Genesis gain/force application, and
  custom PD torque using diagnostic gains.
- Read-only review: reviewer found one blocking gap because the initial
  implementation omitted a force-limit diagnostic; the coding subagent added
  `force_limit_2x`; re-review found no blocking findings.
- Decision: gain/damping is relevant to final stability, but it does not
  explain the early all-env tilt/reset wave. The next test should distinguish
  true fall from reset/contact settling transient.
