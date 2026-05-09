# 011: Reset Semantics Calibration

## Goal

Calibrate task014 reset semantics so standing-height settling is not treated as
a fall.

This subtask follows the evidence from subtask 010: long-horizon zero-action
standing crossed the old `height_min=0.45` line with no tilt failure. The goal
is to split diagnostic/base-height shaping from hard termination before further
PPO tuning.

## Route

1. Keep `height_min=0.45` / `height_max=1.20` as the diagnostic standing-height
   band.
2. Add independent hard reset bounds:
   - `termination_height_min`;
   - `termination_height_max`.
3. Make `termination_height_bad | tilt_bad` the termination condition.
4. Keep metrics for both:
   - diagnostic `height_bad_count`;
   - hard reset `termination_height_bad_count`.
5. Add a long-horizon zero-action sweep:
   - `steps=512`;
   - `pose=tall_crouch`;
   - `root_z=1.20`;
   - `termination_height_min=0.20,0.25,0.30,0.35,0.40,0.45`.
6. If the long-horizon probe finds a no-reset hard-height line, run PPO smoke
   with:
   - `action_scale_mult=0.25`;
   - `termination_height_min=0.20`;
   - `command_mode=standing` first;
   - then `command_mode=vx_yaw` if standing no longer gets reset by settling.

## Stop Rules

- Do not touch `GenesisG1SceneBackend`.
- Do not download assets, checkpoints, datasets, or upstream repos.
- Do not use render/GIF/video, SONIC, ONNX, planner, or LocoFormer.
- Do not write/delete under `/mnt/workspace` or `/mnt/workspace1`.
- Do not claim walking quality.
- Do not mark this subtask passed unless local tests, H200 focused tests,
  H200 long-horizon probe evidence, and PPO smoke evidence are recorded.

## Verification

- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass under guarded command.
- H200 long-horizon reset sweep writes `config.json`, `rows.jsonl`, and
  `summary.json`.
- H200 PPO smoke records both diagnostic and hard termination height counts.

## Log

- 2026-05-09 Added independent termination height bounds:
  - `G1VelocityTrackingConfig.termination_height_min`;
  - `G1VelocityTrackingConfig.termination_height_max`.
- Split height metrics:
  - `height_bad` remains the diagnostic standing-height band;
  - `termination_height_bad` is the hard reset height cause.
- Updated PPO rollout metrics and smoke/probe CLIs to record
  `termination_height_bad_count`.
- Updated standing reset probe to sweep `termination_height_min` values.
- Local focused verification:
  - `PYTHONPATH=src python -m pytest tests/test_g1_velocity_tracking_env.py tests/test_g1_standing_reset_pose_probe.py tests/test_g1_policy_action_safety_probe.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  - result: `21 passed, 4 skipped`.
- Local full verification:
  - `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`
  - result: `189 passed, 4 skipped`.
- H200 focused verification:
  - guarded command under
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`;
  - result: `25 passed in 5.63s`.
- H200 hard-height sweep:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-reset-semantics-512step-v1`;
  - `pose=tall_crouch`, `root_z=1.20`, `steps=512`;
  - swept `termination_height_min=0.20,0.25,0.30,0.35,0.40,0.45`;
  - `stable_found=false`;
  - at `termination_height_min=0.20`:
    `termination_height_bad_count=0`, `tilt_bad_count=10240`,
    `reset_count=10240`, `root_height_min=0.23775185644626617`,
    `upright_min=0.2702862620353699`;
  - conclusion: lowering the hard height line fixes height-caused reset, but
    512-step passive standing still eventually fails by tilt.
- H200 pose/root-z sweep:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-reset-pose-rootz-512step-v1`;
  - swept `pose=profile,half_crouch,mild_crouch,tall_crouch,straight`;
  - swept `root_z=1.00,1.10,1.20,1.30`;
  - used `termination_height_min=0.20`;
  - `stable_found=false`;
  - best row: `pose=straight`, `root_z=1.20`,
    `termination_height_bad_count=0`, `height_bad_count=0`,
    `tilt_bad_count=10240`, `root_height_min=0.6124037504196167`,
    `upright_min=0.2798134684562683`;
  - conclusion: root height and default leg pose are not enough for 512-step
    passive stability; the remaining reset cause is balance/upright control.
- H200 standing PPO smoke after reset semantics split:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-standing-termh020-actionscale025-three-seed-v1`;
  - `action_scale_mult=0.25`, `command_mode=standing`,
    `termination_height_min=0.20`, `height_min=0.45`;
  - `summary.json`: `all_seeds_passed=true`;
  - min collect throughput:
    `24342.973420741637 env_policy_steps_per_sec`;
  - final update for seeds `0,1,2`:
    `termination_height_bad_count=0`, `tilt_bad_count=1024`,
    `reset_count=1024`;
  - first update for all seeds had `reset_count=0`,
    `termination_height_bad_count=0`, and `tilt_bad_count=0`.
- Stop decision:
  - did not run `vx_yaw` PPO, because standing mode still resets from
    tilt/balance rather than height.

## Review

Status: passed as a reset-semantics diagnostic.

- Height reset semantics are calibrated: `height_bad_count` now records the
  0.45 diagnostic band, while `termination_height_bad_count` records hard
  height termination.
- H200 evidence confirms `termination_height_min=0.20` removes height-caused
  resets in long-horizon standing and standing PPO smoke.
- This does not solve standing: the next blocker is tilt/balance/upright
  control. A 512-step zero-action pass is not achievable with the current
  passive PD setpoint alone.
- Boundary review:
  - no `GenesisG1SceneBackend` change;
  - no render/GIF/video, SONIC, ONNX, planner, LocoFormer, downloads, or
    `/mnt/workspace*` path;
  - all H200 outputs stayed under `/root/agent_workspace/project`.
