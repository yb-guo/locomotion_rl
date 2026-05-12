# 000: Success Metrics Contract

## Goal

Make task020 pass/fail numeric before implementation.

## Route

1. Define standing PPO pass metrics.
2. Define baseline comparison metrics.
3. Define full-env reset wave detection.
4. Define deterministic eval pass criteria.
5. Record H200 output and review requirements.

## Acceptance

- Success metrics are numeric.
- Metrics can be computed from `metrics.jsonl` and `summary.json`.
- No walking or `vx_yaw` metric is required.
- Reviewer can reject vague claims.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Contract checked against current `g1_ppo_smoke` outputs. Existing
  smoke metrics cover reset causes, root height, upright, KL/loss/entropy,
  throughput, CUDA isolation, and parameter-change evidence. Gap found:
  `episode_len_mean`/survival metrics and deterministic eval evidence are not
  yet emitted, so task020 cannot pass on the baseline smoke alone.

## Review

Status: route accepted; read-only review pending after implementation evidence.
