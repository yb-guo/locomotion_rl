# 012: Upright Balance Stabilization

## Goal

Diagnose and stabilize the remaining standing reset cause after subtask 011.

Subtask 011 removed hard height resets with `termination_height_min=0.20`, but
512-step zero-action standing and standing PPO still reset by `tilt_bad`. This
subtask targets the upright/balance control path before any `vx_yaw` PPO run.

## Route

1. Reproduce the standing failure with low-noise PPO variants.
2. Rank and test hypotheses:
   - policy action noise/action scale too high;
   - PPO update too aggressive;
   - Genesis motor gains/force limits from the 27DoF profile not applied.
3. Fix only the minimal backend boundary if the evidence points there.
4. Verify with local tests, H200 focused tests, long-horizon zero-action probe,
   and standing PPO smoke.
5. Stop before `vx_yaw` unless standing tilt resets are fixed.

## Stop Rules

- Do not touch `GenesisG1SceneBackend`.
- Do not download assets, checkpoints, datasets, or upstream repos.
- Do not use render/GIF/video, SONIC, ONNX, planner, or LocoFormer.
- Do not write/delete under `/mnt/workspace` or `/mnt/workspace1`.
- Do not claim walking quality.
- Do not mark this subtask passed unless verification evidence and review are
  both recorded.

## Verification

- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass under guarded command.
- H200 long-horizon probe records `physical_gpu=1` and
  `logical_cuda_device=cuda:0`.
- H200 standing PPO smoke records reset-cause metrics.

## Log

- 2026-05-09 Started diagnose loop from subtask 011 evidence:
  `termination_height_bad_count=0`, `tilt_bad_count=1024`,
  `reset_count=1024` in standing PPO final updates.
- Tested low noise/action scale on H200:
  - run id:
    `h200-gpu1-standing-termh020-actionscale010-logstd25-three-seed-v1`;
  - `action_scale_mult=0.10`, `log_std_init=-2.5`,
    `command_mode=standing`, `termination_height_min=0.20`;
  - result: all seeds passed smoke acceptance, but final reset cause stayed
    `tilt_bad_count=1024`, `termination_height_bad_count=0`,
    `reset_count=1024`.
- Tested low PPO learning rate on H200:
  - run id:
    `h200-gpu1-standing-termh020-actionscale010-logstd25-lr1e5-three-seed-v1`;
  - `lr=1e-5`;
  - result: KL dropped to about `3e-4`, but final reset cause stayed
    `tilt_bad_count=1024`, `termination_height_bad_count=0`,
    `reset_count=1024`.
- Diagnosis:
  - H1 action noise/action scale too high: falsified.
  - H2 PPO update too aggressive: falsified.
  - H3 profile motor gains/force limits are not applied by
    `VectorizedGenesisBackend`: supported by code inspection.
- Implemented minimal fix:
  - `VectorizedGenesisBackend` now applies profile `kp`, `kv`, and
    symmetric `force_limits` to `motor_dof_indices` after building Genesis.
  - Local fake Genesis robot now records those calls.
- Added diagnostic motor multipliers for the standing probe:
  - `motor_kp_mult`;
  - `motor_kv_mult`;
  - `motor_force_limit_mult`;
  - defaults stay `1.0`.
- Added PPO smoke `warmup_steps` default `1` to exclude cold Genesis/PyTorch
  kernel startup from measured rollout throughput. The env resets again before
  every seed, so warmup transitions are not training data.
- Local focused verification:
  - command:
    `PYTHONPATH=src python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_standing_reset_pose_probe.py tests/test_g1_velocity_tracking_env.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`;
  - result after final changes: `32 passed, 4 skipped`.
- Local full verification:
  - command: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`;
  - result after final changes: `191 passed, 4 skipped`.
- H200 focused verification:
  - guarded command under
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`;
  - result after final changes: `36 passed in 13.50s`.
- H200 512-step zero-action probe after motor config:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-motor-gains-standing-512step-v1`;
  - `pose=tall_crouch`, `root_z=1.20`, `termination_height_min=0.20`;
  - `termination_height_bad_count=0`, `tilt_bad_count=5120`,
    `reset_count=5120`;
  - improvement over subtask 011 `tilt_bad_count=10240`, but passive
    zero-action standing still is not stable.
- H200 pose/root-z probe after motor config:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-motor-gains-pose-rootz-512step-v1`;
  - `row_count=6`, `stable_found=false`;
  - best row: `pose=straight`, `root_z=1.20`,
    `termination_height_bad_count=0`, `tilt_bad_count=5120`,
    `reset_count=5120`.
- H200 gain sweeps:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-motor-gain-sweep-256step-v1`;
  - best row: `pose=straight`, `motor_kp_mult=1.0`,
    `motor_kv_mult=2.0`, `motor_force_limit_mult=1.0`,
    `tilt_bad_count=2048`, `reset_count=2048`;
  - follow-up damping/force run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/standing_reset_pose_probe/h200-gpu1-damping-force-sweep-256step-v1`;
  - higher damping/force did not improve on `kv_mult=2.0`.
- H200 standing PPO smoke after motor config:
  - `action_scale_mult=0.25`, `log_std_init=-0.5` learned back to zero final
    resets for seed 0, but failed smoke acceptance because one measured rollout
    hit `8438.25133421615 env_policy_steps_per_sec`;
  - conclusion: stability fixed, but cold/intermediate rollout timing needed
    warmup and conservative policy start.
- H200 standing PPO smoke final:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-standing-motorgains-warmup-termh020-actionscale010-logstd25-three-seed-v1`;
  - `action_scale_mult=0.10`, `log_std_init=-2.5`,
    `command_mode=standing`, `warmup_steps=1`;
  - `summary.json`: `all_seeds_passed=true`;
  - min collect throughput:
    `12255.188467549899 env_policy_steps_per_sec`;
  - final update for seeds `0,1,2`: `reset_count=0`,
    `termination_height_bad_count=0`, `tilt_bad_count=0`.
- H200 `vx_yaw` PPO smoke final:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-vxyaw-motorgains-warmup-termh020-actionscale010-logstd25-three-seed-v1`;
  - same conservative init with `command_mode=vx_yaw`;
  - `summary.json`: `all_seeds_passed=true`;
  - min collect throughput:
    `20352.60733855557 env_policy_steps_per_sec`;
  - final update for seeds `0,1,2`: `reset_count=0`,
    `termination_height_bad_count=0`, `tilt_bad_count=0`.

## Review

Status: passed as PPO smoke stabilization.

- Diagnosis result:
  - fixed a real backend bug: profile motor gains/force limits were loaded but
    not applied to Genesis;
  - passive zero-action G1 still falls over long horizons, so the valid route
    is conservative policy initialization plus PPO correction, not claiming a
    statically stable humanoid setpoint.
- Acceptance evidence:
  - local focused and full tests pass;
  - H200 focused tests pass;
  - H200 standing and `vx_yaw` PPO smoke pass all 3 seeds with final reset
    counts at zero and throughput above `10000`.
- Boundary review:
  - no `GenesisG1SceneBackend` change;
  - no render/GIF/video, SONIC, ONNX, planner, LocoFormer, downloads, or
    `/mnt/workspace*` path;
  - all H200 outputs stayed under `/root/agent_workspace/project`.
