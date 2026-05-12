# 008: Review And Decision

## Goal

Close task020 with evidence and a narrow decision.

## Route

1. Collect all H200 run dirs.
2. Summarize success metrics.
3. Summarize reset causes and deterministic eval.
4. Read-only reviewer checks code, boundaries, and evidence.
5. If blocking findings exist, fix and re-review.

## Acceptance

- Task020 is either passed with standing PPO evidence, or explicitly blocked.
- If blocked, reason is one of:
  - PPO plumbing;
  - reset semantics;
  - reward/action energy;
  - env/contact dynamics.
- No walking/asset/importer claim is made.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Evidence collected:
  - subtask002-004: PPO plumbing, reset metrics, distribution profiling, and
    minimal reward pack all ran on H200 physical GPU 1 with finite metrics and
    no final reset regression;
  - subtask005: bounded one-seed action-energy matrix completed 12/12
    candidates and selected `action_scale_mult=0.10`, `log_std_init=-2.0`;
  - subtask006/006a: 3-seed standing gate with selected config passed device,
    finite-metric, actor/value-change, and throughput checks, but failed the
    standing gate with final mean episode length about 67 versus the 2x
    baseline threshold of about 103.84;
  - subtask006a: no single-step full-env reset wave occurred, but each seed had
    rollout-window tilt reset sweeps at chunks/updates 2, 5, 8, 11, 14, and 17;
  - subtask006c: zero-action, untrained-mean-action, and
    untrained-sampled-action no-update probes reproduced the same first tilt
    chunk and reset-sweep cadence.
- 2026-05-12 Decision: task020 is blocked, not passed. The blocker is current
  env/contact/passive-standing dynamics for the Genesis G1 27DoF no-hand setup,
  not PPO plumbing, CUDA isolation, NaN/Inf, action-energy search, or a yaw/vx
  curriculum issue.
- 2026-05-12 Stop-rule action: deterministic standing eval and yaw readiness
  were skipped. No walking, `vx_yaw`, asset/importer, MuJoCo, LocoFormer,
  SONIC, ONNX, rendering, datasets, checkpoints, or upstream downloads were
  introduced.

## Review

Status: pending final read-only review.
