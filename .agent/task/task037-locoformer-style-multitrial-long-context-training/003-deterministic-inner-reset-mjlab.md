# 003 Deterministic Inner Reset MJLab

## Route

Implement the real MJLab inner-trial reset.

Acceptance criteria:

- Fall or trial timeout resets robot to deterministic standing pose.
- Phase resets to fixed start phase.
- Env/action-manager last action is cleared.
- Command is unchanged across inner trials.
- Motor failure target, failure type, severity, and actuator force range are
  unchanged across inner trials.
- Outer episode reset resamples the latent condition.
- JSON evidence records pre/post reset state and condition preservation.

## Log

- 2026-05-29 Planned.
- 2026-05-29 Implemented `Task037MjlabInnerResetController`, installed by
  `Task037BufferOnlyK4DeterministicInnerResetRunner`.
- 2026-05-29 The controller patches MJLab base env reset flow:
  - full outer reset uses the original MJLab reset path;
  - inner reset uses deterministic reset pose and restores command/failure
    condition tensors;
  - post-step restore is required because MJLab step events can resample
    dynamic failure state after `_reset_idx`.
- 2026-05-29 Added deterministic task id:
  `Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0`.
- 2026-05-29 Added H200 probe:
  `python -m h200_locomotion_lab.tools.task037_mjlab_inner_reset_probe`.
- 2026-05-29 H200 validation:
  `/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python -m pytest
  -p no:cacheprovider tests/test_task037_multitrial_contract.py
  tests/test_task037_mjlab_smoke_scripts.py -q` -> `5 passed in 2.66s`.
- 2026-05-29 First H200 probe without post-step restore failed:
  `inner_failure_max_delta=0.3787555694580078`. Cause: MJLab step-event
  scheduler resampled dynamic failure after reset.
- 2026-05-29 H200 probe after post-step restore passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/deterministic_inner_reset/task037_mjlab_inner_reset_probe_env64_poststep_restore.json`.
  Result: `pass=true`, `inner_command_max_delta=0.0`,
  `inner_failure_max_delta=0.0`, `inner_episode_length_max=0`,
  `phase_after_inner=[0.0, 1.0]`, `root_z_after_inner=0.800000011920929`,
  `outer_command_changed_any=true`.

## Review

Status: passed for short-horizon deterministic inner-reset probe.

Evidence:

- Fall/timeout path was exercised through short MJLab timeouts.
- Inner reset preserved command and captured motor-failure condition tensors.
- Inner reset used deterministic reset config and fixed phase start.
- Env/action-manager reset path cleared the reset step action history through
  the existing Task037 zero-action reset contract.
- Outer reset still occurred and command changed at least once.

Limit:

- This is an env64 reset-contract probe, not a policy-quality or long-horizon
  gait evaluation.
