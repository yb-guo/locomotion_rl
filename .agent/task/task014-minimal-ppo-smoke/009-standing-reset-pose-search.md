# 009: Standing Reset Pose Search

## Goal

Find a more stable zero-action standing reset pose and default joint pose for
the G1 27DoF no-hand Genesis training env.

This is a diagnosis/probe subtask, not reward tuning and not a walking claim.

## Route

1. Build an agent-runnable zero-action standing probe:
   - fixed zero command velocity;
   - zero action;
   - no PPO;
   - no render/GIF/video;
   - output under `/root/agent_workspace/project`.
2. Allow `VectorizedGenesisBackend` to override reset root pose and default
   joint pose without changing `GenesisG1SceneBackend`.
3. Sweep:
   - root heights;
   - profile default legs;
   - progressively straighter leg candidates.
4. Record:
   - reset count;
   - height/tilt failure counts;
   - root height mean/min/final;
   - upright mean/min;
   - zero-action throughput.
5. Select the best candidate only if the zero-action probe shows fewer resets
   and no tilt instability.
6. Re-run local and H200 focused tests.
7. If a candidate is found, decide whether to update PPO smoke defaults.

## Stop Rules

- Do not touch `GenesisG1SceneBackend`.
- Do not use render/GIF/video, SONIC, ONNX, planner, LocoFormer, downloads, or
  `/mnt/workspace*`.
- If no candidate reduces resets, do not change PPO defaults.
- If a candidate avoids height reset by causing tilt reset, reject it.

## Verification

- Local focused tests pass.
- H200 focused tests pass.
- H200 standing reset probe writes `config.json`, `rows.jsonl`,
  `summary.json`.
- Review records best candidate and whether it is safe for PPO smoke defaults.

## Log

- 2026-05-09 Added a zero-action G1 standing reset probe:
  `src/h200_locomotion_lab/tools/g1_standing_reset_pose_probe.py`.
- Added `VectorizedGenesisBackend.set_reset_pose(...)` so the probe can sweep
  root pose and default joint pose without rebuilding Genesis and without
  touching `GenesisG1SceneBackend`.
- Added shared G1 standing reset pose candidates:
  `src/h200_locomotion_lab/envs/g1_reset_poses.py`.
- Local focused verification:
  - `PYTHONPATH=src python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_standing_reset_pose_probe.py tests/test_g1_ppo_smoke.py tests/test_g1_velocity_tracking_env.py -q -p no:cacheprovider`
  - result: `25 passed in 0.52s`.
- H200 focused verification:
  - guarded command under
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`;
  - result: `25 passed in 0.34s`.
- H200 broad 64-step standing sweep:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-standing-sweep-v1`;
  - candidates: `profile`, `half_crouch`, `mild_crouch`, `tall_crouch`,
    `straight`;
  - root z: `0.90`, `1.00`, `1.10`, `1.20`;
  - `stable_found=false`;
  - `straight` avoided height reset but caused tilt reset, so it was rejected.
- H200 32-step rollout-length sweep:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-standing-32step-sweep-v1`;
  - `stable_found=true`;
  - stable candidates were `tall_crouch` at `root_z=1.20` and `root_z=1.30`
    when `height_max=1.50`.
- H200 default-height confirmation:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-standing-tall-rootz120-default-heightmax-v1`;
  - `pose=tall_crouch`, `root_z=1.20`, `height_min=0.45`,
    `height_max=1.20`;
  - `n_envs=1024`, `steps=32`, `env_steps=32768`;
  - `reset_count=0`, `height_bad_count=0`, `tilt_bad_count=0`;
  - `root_height_min=0.45437127351760864`;
  - `root_height_final_mean=0.45437127351760864`;
  - `upright_mean=0.9968236088752747`;
  - `upright_min=0.9743549227714539`;
  - throughput: `54420.54598520063 env_policy_steps_per_sec`.
- Selected default standing reset pose:
  - `root_z=1.20`;
  - leg joint values in radians:
    `hip_pitch=-0.06`, `knee=0.12`, `ankle_pitch=-0.07`;
  - all non-leg joints keep the profile default pose.
- Updated PPO smoke defaults to use `default_pose=tall_crouch` and
  `root_z=1.20`.
- H200 PPO smoke confirmation with new defaults:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-standing-default-three-seed-v1`;
  - `summary.json`: `all_seeds_passed=true`;
  - `config.json` records `physical_gpu=1`, `logical_cuda_device=cuda:0`,
    `CUDA_VISIBLE_DEVICES=1`, `default_pose=tall_crouch`, and `root_z=1.2`;
  - min collect throughput:
    `24606.65830209426 env_policy_steps_per_sec`;
  - final update reset counts were `960`, `962`, and `948`;
  - final update tilt counts were all `0`.

## Review

Status: passed.

- Decision: use `tall_crouch` with `root_z=1.20` as the stable standing reset
  default for the task014 G1 27DoF no-hand Genesis PPO smoke.
- This is a standing reset/default pose decision only. It is not a walking
  quality claim.
- The selected pose passes the zero-action standing probe for one 32-step
  rollout under default height bounds with no reset, no height failure, and no
  tilt failure.
- PPO smoke still records height resets under an untrained stochastic policy,
  so the remaining efficiency problem is policy-action/reset interaction, not
  the passive standing reset pose.
- Boundary review:
  - no `GenesisG1SceneBackend` change;
  - no render/GIF/video, SONIC, ONNX, planner, LocoFormer, downloads, or
    `/mnt/workspace*` path;
  - all H200 outputs stayed under `/root/agent_workspace/project`.
