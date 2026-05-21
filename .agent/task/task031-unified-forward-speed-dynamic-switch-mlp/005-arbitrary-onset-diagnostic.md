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

## Review

Status: planned. Pass means the diagnostic matrix was run and interpreted. It
does not require all arbitrary onset cases to pass.
