# 004: Randomization Curriculum

## Route

Add randomization in stages after the no-randomization MLP smoke is learnable.
One variable group changes at a time so failures are diagnosable.

Initial order:

1. No randomization control.
2. Motor strength / PD gain scale.
3. Contact friction.
4. Link mass / COM / inertia.
5. Encoder bias / observation noise.
6. Action delay / smoothing.
7. Combined randomization.

## Minimal Closed Loop

Feedback loop:

1. For each randomization stage, run a fixed short PPO budget from scratch.
2. Run deterministic eval and randomized holdout eval on the same checkpoint.
3. Compare against the previous stage using the same JSON metrics.

Pass:

- Each stage can be toggled independently from config.
- No-randomization control remains reproducible.
- The new stage does not break import, reset, or smoke training.
- If a stage fails, the failing stage is isolated to one randomization group.

Fail:

- Multiple new randomization groups are enabled at once.
- There is no deterministic control eval.
- Failure cannot be attributed to a single stage.
- Randomization changes topology, DoF, action dim, or observation dim.

Evidence:

- Per-stage train/eval summaries under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/`.

## Log

- 2026-05-19 Opened during diagnose audit to prevent all-randomization-at-once
  failures.
- 2026-05-19 Subagent API audit found the inherited
  `Unitree-G1-Gripper-Flat` config is not a deterministic control. It inherits
  actor observation corruption, `push_robot`, `foot_friction`, `encoder_bias`,
  `base_com`, and command curriculum from the base velocity task. CLI override
  alone is not enough because Tyro cannot reliably delete event/curriculum
  dict entries. Therefore 004 needs explicit stage task configs.
- 2026-05-19 Added stage-specific task IDs in the upstream gripper config,
  without modifying the original G1 baseline:
  - `Unitree-G1-Gripper-Flat-Control`
  - `Unitree-G1-Gripper-Flat-Contact`
  - `Unitree-G1-Gripper-Flat-EncoderNoise`
  - `Unitree-G1-Gripper-Flat-MassComInertia`
  - `Unitree-G1-Gripper-Flat-MotorPd`
  - `Unitree-G1-Gripper-Flat-Combined`
- 2026-05-19 Stage behavior:
  - `Control`: keeps only `reset_base` and `reset_robot_joints`, disables actor
    corruption, and clears curriculum.
  - `Contact`: control plus `foot_friction`.
  - `EncoderNoise`: control plus actor corruption and `encoder_bias`.
  - `MassComInertia`: control plus `body_com_offset` and
    `pseudo_inertia`.
  - `MotorPd`: control plus `pd_gains`, `effort_limits`, `joint_damping`, and
    `joint_friction`.
  - `Combined`: all supported stages except delay/smoothing.
- 2026-05-19 Delay/smoothing remains deferred. The audit found actuator delay
  requires wrapping actuators with `DelayedActuatorCfg` and then using
  `dr.sync_actuator_delays`; this is not included in the first curriculum
  smoke. No generic action smoothing API was found for the current position
  actuators.
- 2026-05-19 Reproducibility artifacts:
  - `.agent/task/task028-randomized-wholebody-morphology-env/artifacts/task028_create_g1_gripper_task.py`
  - `.agent/task/task028-randomized-wholebody-morphology-env/artifacts/task028_inspect_randomization_stage.py`
- 2026-05-19 Validation passed:
  `scripts/list_envs.py --keyword Gripper-Flat` showed all six stage task IDs
  plus the base `Unitree-G1-Gripper-Flat`; `py_compile` passed for the gripper
  config and generator script; inspect passed for all stages with action dim
  31, actor obs dim 104, and critic obs dim 119.
- 2026-05-19 H200 inspect JSON outputs:
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/Control.json`
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/Contact.json`
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/EncoderNoise.json`
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/MassComInertia.json`
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/MotorPd.json`
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/Combined.json`
- 2026-05-19 H200 PPO smoke passed for all six stages, each with 64 envs,
  2 learning iterations, tensorboard logger, GPU0, `save_interval=1`, and no
  residual training process:
  - Control:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_randomization_004_smoke/2026-05-19_16-12-21_004_control_env64_iter2/model_1.pt`
  - Contact:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_randomization_004_smoke/2026-05-19_16-12-51_004_contact_env64_iter2/model_1.pt`
  - EncoderNoise:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_randomization_004_smoke/2026-05-19_16-13-56_004_encoder_noise_env64_iter2/model_1.pt`
  - MassComInertia:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_randomization_004_smoke/2026-05-19_16-17-47_004_mass_com_inertia_env64_iter2_v2/model_1.pt`
  - MotorPd:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_randomization_004_smoke/2026-05-19_16-14-38_004_motor_pd_env64_iter2/model_1.pt`
  - Combined:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_randomization_004_smoke/2026-05-19_16-18-08_004_combined_env64_iter2_v2/model_1.pt`

## Review

Status: passed for staged smoke.

The first curriculum implementation now has an explicit deterministic control
and independently toggleable randomization stages. This satisfies the 004
minimum requirement: import/reset/training failures can be attributed to a
single randomization group instead of to an all-randomization bundle.

This subtask does not prove convergence under randomization. It proves that all
supported stage configs preserve the fixed topology/DoF/observation/action
contract and can run the same short PPO budget from scratch. Delay/smoothing
is deliberately deferred because it requires an actuator-wrapper change rather
than a simple event toggle.
