# 008 — V2 Stance Matrix And R5 Gate

## Route

1. Re-run stance realization for the descriptor-driven v2 outputs only; do not
   reuse Task067 or old Task070 v1 stance evidence.
2. Run the full expected structural-center x family x region matrix with
   1000-step, 2 ms stance-hold gates.
3. Rebuild R4 matrix/gallery and R5 final verification from raw manifests, not
   summary trust.
4. Complete execution-agent local visual review for the final gallery and record
   `agent_visual_check_passed=true` with per-image observations.
5. Stop after agent visual pass and present the final gallery/matrix/R5 summary
   to the user for human visual acceptance; record `user_visual_acceptance=true`
   only after explicit user confirmation.
6. Run focused tests, Ruff, agent inspection, full pytest, and request a new
   independent read-only review.

## Log

- Not started.

## Review

- Pass only if exact motor/tree/axis/module/SHA gates are fail-closed and all
  stance/contact/visual denominators match the frozen plan, with both
  `agent_visual_check_passed=true` and `user_visual_acceptance=true`.
- Fail if any old v1 matrix/gallery/test evidence is used to satisfy v2, or if
  empty/partial records can pass R5.
- Fail if final passed evidence is claimed before the user has checked the
  agent-approved gallery.
