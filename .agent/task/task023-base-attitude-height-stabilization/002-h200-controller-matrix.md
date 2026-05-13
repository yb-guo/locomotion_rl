# Subtask 002: H200 Controller Matrix

## Route

- Run only after subtask001 passes local tests and read-only review.
- Use guarded H200 commands only.
- Use physical GPU 1 with logical `cuda:0`.
- No PPO.

## H200 Command Shape

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src <command>'
```

## Required Matrix

- source asset, no stabilizer;
- source asset, attitude-only;
- source asset, height-only;
- source asset, attitude+height;
- regenerated `ankle_roll_larger_spheres`, no stabilizer;
- regenerated `ankle_roll_larger_spheres`, best bounded stabilizer candidate.

## Evidence Required

- first tilt/reset step;
- root height and upright timeline;
- ankle-roll and ankle-pitch contact force summary;
- top joint error summary;
- clipping/saturation summary;
- output paths under `/root/agent_workspace/project`.

## Stop Rules

- If baseline reproduction differs materially from task022, stop and diagnose.
- If a candidate improves by at least 20 policy steps, rerun once.
- If all candidates fail near task022 horizons, stop and record controller
  insufficient.
- If contact force exceeds task022 box-support levels without stability gain,
  stop that candidate.

## Log

- 2026-05-13 Created with task023.
- 2026-05-13 Coding subagent implemented the subtask002 local prerequisite for
  `--runner genesis` in
  `src/h200_locomotion_lab/tools/g1_base_attitude_height_stabilization.py`.
  Genesis/torch imports remain delayed to the Genesis runner path, the runner
  builds `VectorizedGenesisBackend`/`VectorizedGenesisConfig` with the G1
  profile, supports `--asset-path` profile asset replacement, generates bounded
  normalized fixed-controller actions for `none`, `attitude`, `height`, and
  `attitude_height`, and writes Genesis summaries with hardware metadata and
  contact availability/missing reasons.
- 2026-05-13 Extended
  `tests/test_g1_base_attitude_height_stabilization.py` with fake backend
  coverage for the Genesis runner path, asset override, bounded normalized
  actions, contact schema, and guarded command CUDA env construction. Local
  focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_base_attitude_height_stabilization.py -p no:cacheprovider`
  passed with 9 tests.
- 2026-05-13 Coding subagent fixed the subtask002 prerequisite issues found by
  Router before H200 matrix execution. `attitude_from_quat` now normalizes wxyz
  quaternions and reports upright with the same projected-gravity semantics as
  `g1_zero_action_standing_causality.projected_gravity_torch`/
  `standing_flags`: `projected_gravity_z = -1 + 2 * (x^2 + y^2)` and
  `upright = clamp(-projected_gravity_z, 0, 1)`. Euler roll/pitch remain
  available for the fixed controller. The contact reader now tries both
  `get_links_net_contact_force` and `get_links_net_contact_forces`, first with
  `links_idx_local=(idx,)` and then with no-arg full contact-force fallback.
  Added deterministic tests covering projected-gravity upright equivalence, the
  H200 baseline semantic discrepancy fixture (`Euler first_tilt_step=64` vs
  `projected-gravity first_tilt_step=88`), and plural contact API availability.
  Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_base_attitude_height_stabilization.py -p no:cacheprovider`
  passed with 12 tests. No H200 command was run.
- 2026-05-13 Coding subagent fixed the reset/default pose prerequisite before
  rerunning H200. `g1_base_attitude_height_stabilization` now supports
  `--pose-profile` with default `current` and `unitree_gym` alternative, using
  the same task018 tall-crouch leg values as
  `g1_zero_action_standing_causality`: hip_pitch `-0.06`, knee `0.12`, and
  ankle_pitch `-0.07` for the `current` profile. The Genesis runner now passes
  `default_positions_rad=pose` into `VectorizedGenesisConfig` for both candidate
  and baseline/no-stabilizer backend construction, so `mode none` starts from
  the same default pose as the old zero-action probe. Config and summary JSON
  now record `pose_profile` and `pose_leg_values_rad`; local_toy records this
  metadata but does not use it for physics. Added fake Genesis coverage proving
  the config receives `default_positions_rad`, plus direct pose-profile
  semantics tests for `current` and `unitree_gym`. Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_base_attitude_height_stabilization.py -p no:cacheprovider`
  passed with 14 tests. No H200 command was run.

## Review

Status: reviewed_no_blocking_for_h200_matrix.

- 2026-05-13 Read-only reviewer found no blocking findings and allowed Router
  to execute the guarded H200 matrix. Reviewer confirmed that `--runner genesis`
  is implemented, Genesis/torch imports are delayed, asset override/backend/
  n-envs/logical-device/modes are supported, actions are bounded and clipped,
  and no PPO/training path or stop-rule violation was introduced.
- Non-blocking carry-forward for H200 execution: use explicit
  `--physical-gpu 1 --logical-cuda-device cuda:0`; stop and diagnose if real
  Genesis exposes contact force only through an API not covered by the current
  link reader.
- 2026-05-13 Read-only reviewer found no blocking findings for the reset-pose
  fix and allowed Router to submit, sync remote, and rerun only the
  source/no-stabilizer baseline. Reviewer confirmed that `current` pose
  semantics match the old zero-action probe and that the H200 command path
  includes `--pose-profile current`.
