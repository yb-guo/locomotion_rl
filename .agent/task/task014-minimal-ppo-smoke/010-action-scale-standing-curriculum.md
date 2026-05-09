# 010: Action Scale and Standing Curriculum Probe

## Goal

Diagnose whether task014 height resets under the untrained stochastic policy
come from action amplitude, initial policy std, or command curriculum.

This is a training-safety probe, not a walking-quality claim.

## Route

1. Add an explicit `action_scale_mult` on top of the prepared profile action
   scales.
2. Expose PPO smoke knobs for:
   - `action_scale_mult`;
   - `command_mode=vx_yaw|standing`;
   - optional base-height tracking reward.
3. Add an independent policy-action safety probe:
   - untrained policy only;
   - no render/GIF/video;
   - no SONIC/ONNX/planner/LocoFormer;
   - outputs under `/root/agent_workspace/project`.
4. Sweep:
   - action scale multipliers;
   - initial log std values;
   - vx+yaw vs standing command mode.
5. Pick a candidate only if reset/height failure counts improve without tilt
   failure and without relaxing reset thresholds.
6. Re-run focused local and H200 tests.
7. Confirm the selected candidate with H200 PPO smoke before changing defaults.

## Stop Rules

- Do not touch `GenesisG1SceneBackend`.
- Do not download assets, checkpoints, datasets, or upstream repos.
- Do not use render/GIF/video, SONIC, ONNX, planner, or LocoFormer.
- Do not write/delete under `/mnt/workspace` or `/mnt/workspace1`.
- Do not lower `height_min` to hide falls.
- Do not change PPO smoke defaults unless H200 evidence supports it.

## Verification

- Local focused tests pass.
- H200 focused tests pass.
- H200 policy-action safety probe writes `config.json`, `rows.jsonl`, and
  `summary.json`.
- If defaults change, H200 PPO smoke passes all 3 seeds and records evidence.

## Log

- 2026-05-09 Added action-safety controls:
  - `VectorizedGenesisConfig.action_scale_mult`;
  - `VectorizedGenesisConfig.action_joint_group=all|legs|legs_waist`;
  - PPO smoke CLI `--action-scale-mult`;
  - PPO smoke CLI `--action-joint-group`;
  - PPO smoke CLI `--command-mode=vx_yaw|standing`;
  - optional base-height tracking reward;
  - optional termination penalty.
- Added independent untrained-policy action safety probe:
  `src/h200_locomotion_lab/tools/g1_policy_action_safety_probe.py`.
- Local focused verification:
  - `PYTHONPATH=src python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_velocity_tracking_env.py tests/test_g1_ppo_smoke.py tests/test_g1_policy_action_safety_probe.py tests/test_g1_standing_reset_pose_probe.py -q -p no:cacheprovider`
  - final result after action group support: `30 passed in 0.63s`.
- H200 focused verification:
  - guarded command under
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`;
  - final result: `30 passed in 0.32s`.
- H200 action-safety sweep:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/policy_action_safety_probe/h200-gpu1-action-safety-sweep-v2`;
  - swept `action_scale_mult=1.0,0.5,0.25`;
  - swept `log_std_init=-0.5,-1.5,-2.5`;
  - swept `command_mode=vx_yaw,standing`;
  - `row_count=18`;
  - baseline `action_scale_mult=1.0`, `log_std_init=-0.5`,
    `command_mode=vx_yaw`: `reset_count=869`, `height_bad_count=869`,
    `tilt_bad_count=0`;
  - `action_scale_mult=0.25`, `log_std_init=-0.5`,
    `command_mode=vx_yaw`: `reset_count=0`, `height_bad_count=0`,
    `tilt_bad_count=0`;
  - standing command mode changed reward level but did not materially change
    reset safety for the untrained policy.
- H200 PPO smoke confirmations:
  - `action_scale_mult=0.25`, vx+yaw:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-actionscale025-three-seed-v1`,
    `all_seeds_passed=true`, min collect throughput
    `45181.440260100986`, but final update reset counts were all `1024`;
  - `action_scale_mult=0.5`, `log_std_init=-1.5`, vx+yaw:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-actionscale050-logstd15-three-seed-v1`,
    `all_seeds_passed=true`, min collect throughput
    `51193.840031944594`, but final update reset counts were all `1024`;
  - `action_scale_mult=0.25`, base-height reward `1.0`,
    termination penalty `-5.0`, vx+yaw:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-actionscale025-heightrew1-termpen5-three-seed-v1`,
    `all_seeds_passed=true`, but final update reset counts were all `1024`;
  - same reward settings with `command_mode=standing`:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-standing-actionscale025-heightrew1-termpen5-three-seed-v1`,
    `all_seeds_passed=true`, but final update reset counts were all `1024`;
  - `root_z=1.30`, `height_max=1.50`, `action_scale_mult=0.25`:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-rootz130-highmax-actionscale025-three-seed-v1`,
    `all_seeds_passed=true`, but final update reset counts were all `1024`;
  - `action_scale_mult=0.1` and single-seed `action_scale_mult=0.01`
    still produced final update `reset_count=1024`.
- H200 long-horizon zero-action control:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-standing-tall-rootz120-160step-v1`;
  - `pose=tall_crouch`, `root_z=1.20`, `steps=160`;
  - `reset_count=4096`, `height_bad_count=4096`, `tilt_bad_count=0`;
  - `root_height_min=0.404991090297699`;
  - `upright_mean=0.9964646697044373`.

## Review

Status: passed.

- Decision: do not change task014 PPO defaults from this subtask.
- Initial random-action safety improves strongly with smaller action scale:
  `action_scale_mult=0.25` gives 0 reset in the first 32-step rollout.
- But PPO smoke runs 5 consecutive 32-step rollouts; long-horizon zero-action
  standing already crosses `height_min=0.45` by 160 steps with no tilt failure.
- Therefore the current recurring reset problem is primarily long-horizon
  standing-height calibration/settling, not command mode, action std, PPO
  update, or upper-body action noise.
- The next subtask should calibrate the height termination against long-horizon
  zero-action standing evidence, or find a true long-horizon standing pose.
- Boundary review:
  - no `GenesisG1SceneBackend` change;
  - no render/GIF/video, SONIC, ONNX, planner, LocoFormer, downloads, or
    `/mnt/workspace*` path;
  - all H200 outputs stayed under `/root/agent_workspace/project`.
