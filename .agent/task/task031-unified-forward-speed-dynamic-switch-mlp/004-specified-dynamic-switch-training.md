# 004 Specified Dynamic Switch Training

## Route

Train Level B: unified speed plus the specified Task030 dynamic-switch route.

Implementation target:

- Reuse the Task030 canonical switch as the eval case.
- During training, add small timing jitter to onset and recovery windows so the
  policy does not memorize one exact timestamp.
- Keep the same weak/dead motor semantics used in Task030.
- Keep arbitrary per-joint onset outside the pass target for this subtask.

Evaluation:

- Run canonical dynamic-switch eval at every speed bin:
  `0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`.
- A single-seed candidate must pass all speed bins before final multiseed eval.
- Final acceptance requires `5/5` seeds passing at each of the five speed bins.
- Per-case thresholds are `zero_fall_ratio >= 0.90`,
  `recovery_success_ratio >= 0.75`,
  `post_recovery_lin_vel_error_mean <= 0.8`,
  `post_recovery_yaw_vel_error_mean <= 0.8`, and
  `max_gravity_xy_after_onset <= 0.8`.

## Log

- 2026-05-21 Planned as Level B training after Level A is stable.
- 2026-05-21 Started `model_5349.pt` canonical dynamic-switch multiseed eval
  on H200. The foreground eval was already running with 11 per-case JSON files
  written when the user requested background execution.
- 2026-05-21 Installed watchdog script:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/run_model5349_switch_s5_watchdog.sh`.
  Launched tmux session `task031_model5349_switch_s5_watchdog`; it waits for
  the existing eval PID to finish, then reruns the multiseed wrapper with
  `--reuse-existing` to fill any missing results.
- 2026-05-21 Current output directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/eval_model5349_dynamic_switch_multiseed`.
- 2026-05-28 Background eval completed. `model_5349.pt` passes Level B
  canonical dynamic-switch multiseed across all five speed bins: `5/5` seeds at
  `0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`. Worst observed metrics were
  `zero_fall_ratio_min=1.0`, `recovery_success_ratio_min=1.0`,
  `post_recovery_lin_vel_error_mean_max=0.1684289127588272`,
  `post_recovery_yaw_vel_error_mean_max=0.19292102754116058`, and
  `max_gravity_xy_after_onset_max=0.14738795161247253`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/eval_model5349_dynamic_switch_multiseed/task031_dynamic_switch_multiseed_summary.json`.

## Review

Status: passed for the written Level B gate. This does not close Task031 by
itself because Level A forced persistent dead-grid and Level C diagnostic/render
evidence still need final review.
