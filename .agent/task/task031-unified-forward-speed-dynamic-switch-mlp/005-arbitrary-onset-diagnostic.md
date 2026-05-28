# 005 Arbitrary Onset Diagnostic

## Route

Run Level C diagnostics without making arbitrary onset a Task031 pass
condition.

Diagnostic matrix:

- Speeds: `0.4`, `1.2`, and `2.0 m/s`.
- Joints: the same 12 leg joints used by the Task030 onset grid.
- Failure: mid-episode dead onset with recovery.
- Checkpoint: best Level B candidate checkpoint.

Outputs:

- Per-case JSON files.
- Aggregate JSON with pass count, failed speed/joint pairs, and worst metrics.
- Short diagnosis of whether failures are speed-specific, joint-specific, or
  both.

Classification thresholds use the same dynamic thresholds as Level B:
`zero_fall_ratio >= 0.90`, `recovery_success_ratio >= 0.75`,
`post_recovery_lin_vel_error_mean <= 0.8`,
`post_recovery_yaw_vel_error_mean <= 0.8`, and
`max_gravity_xy_after_onset <= 0.8`.

If arbitrary onset still fails, the expected conclusion is not "train longer".
The review should recommend the next policy-capacity task: history stack, GRU,
or LocoFormer-style memory.

## Log

- 2026-05-21 Planned as diagnostic-only Level C.
- 2026-05-28 Added and locally checked diagnostic wrapper:
  `.agent/task/task031-unified-forward-speed-dynamic-switch-mlp/artifacts/task031_eval_arbitrary_onset_grid.py`.
  Local checks: AST parse passed and `--help` printed successfully.
- 2026-05-28 Started H200 tmux session `task031_arbitrary_onset_grid` for
  `model_5349.pt` over speeds `0.4`, `1.2`, and `2.0 m/s` across the 12
  Task030 leg joints. Output directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/eval_model5349_arbitrary_onset_grid`.
- 2026-05-28 Level C diagnostic completed: `pass_count=28/36`,
  `aggregate_pass=false`. Failed cases:
  - `0.4 m/s`: `single-left_hip_yaw_joint`,
    `single-left_hip_roll_joint`;
  - `1.2 m/s`: `single-right_hip_pitch_joint`,
    `single-right_knee_joint`;
  - `2.0 m/s`: `single-left_hip_pitch_joint`,
    `single-left_hip_yaw_joint`, `single-right_hip_pitch_joint`,
    `single-right_knee_joint`.
  Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/eval_model5349_arbitrary_onset_grid/task031_arbitrary_onset_grid_summary.json`.

## Review

Status: completed diagnostic. Arbitrary per-joint onset remains unsolved for
the current MLP checkpoint, especially hip yaw/roll at low speed and hip
pitch/right knee at medium/high speed. This supports moving arbitrary onset to a
later history/memory-policy task rather than claiming Task031 solved it.
