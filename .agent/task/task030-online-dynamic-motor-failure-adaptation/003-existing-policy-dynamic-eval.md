# 003: Existing Policy Dynamic Eval

## Route

Evaluate current task029 accepted policies under the dynamic scheduler before
training on dynamic failures.

Purpose:

- Establish the baseline failure mode.
- Separate scheduler/eval bugs from learning problems.
- Find whether MLP already adapts through current proprioception feedback.

Checkpoints:

- Start with a known lower-speed accepted checkpoint at fixed `1.2 m/s` to
  validate the eval loop.
- Then evaluate task029 accepted `Fast1p6 model_4700.pt` at fixed `1.6 m/s`.

Required cases:

- clean fixed-command eval
- persistent weak/dead eval
- dynamic single-failure eval
- dynamic switch eval using the deterministic template

Pass:

- JSON evidence exists for every case.
- The dynamic metrics identify whether failure is onset instability, poor
  recovery, or switch-specific collapse.
- No training result is accepted before this baseline is recorded.

Fail:

- Existing-policy eval is skipped.
- Only training reward is used to infer dynamic robustness.
- The eval does not distinguish transient-window metrics from post-recovery
  metrics.

## Log

- 2026-05-21 Opened.
- 2026-05-21 Added `artifacts/task030_dynamic_eval_checkpoint.py` for
  existing-policy dynamic evaluation with transient-window metrics.
- 2026-05-21 First eval attempt caught a script dtype bug: MJLab returned
  `dones` as an integer tensor. The script now casts `dones` to bool before
  fall accounting.
- 2026-05-21 Dynamic switch baseline at fixed `1.2 m/s` passed with task029
  accepted `model_4700.pt`: `num_envs=256`, `steps=500`, `eval_time_s=10.0`,
  `zero_fall_ratio=1.0`, `recovery_success_ratio=1.0`,
  `post_recovery_lin_vel_error_mean=0.0793`,
  `post_recovery_yaw_vel_error_mean=0.1676`,
  `max_gravity_xy_after_onset.max=0.1870`, `pass=true`. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/existing_policy_dynamic_eval/vx1p2/task030_dynamic_eval_vx1p2.json`.
- 2026-05-21 Dynamic switch baseline at fixed `1.6 m/s` failed with the same
  checkpoint: `num_envs=256`, `steps=500`, `eval_time_s=10.0`,
  `zero_fall_ratio=0.2109`, `mean_done_count=1.40625`, `max_done_count=2`,
  `recovery_success_ratio=0.4883`,
  `post_recovery_lin_vel_error_mean=0.5754`,
  `post_recovery_yaw_vel_error_mean=0.6814`,
  `max_gravity_xy_after_onset.max=0.9624`, `pass=false`. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/existing_policy_dynamic_eval/vx1p6/task030_dynamic_eval_vx1p6.json`.
- 2026-05-21 Initial failure mode: the existing MLP checkpoint handles the
  lower-speed dynamic switch, but at `1.6 m/s` online adaptation is not stable.
  The largest degradation is not command tracking after recovery, which remains
  under the loose `0.8` thresholds, but fall/reset count, low base height, high
  post-onset gravity excursion, and low recovery-success coverage.

## Review

Status: partial. Dynamic switch baseline is measured at `1.2` and `1.6 m/s`.
Before marking this subtask passed, add isolated dynamic single-failure eval
and either rerun or explicitly reference task029 clean/persistent JSON evidence
under the task030 report root.
