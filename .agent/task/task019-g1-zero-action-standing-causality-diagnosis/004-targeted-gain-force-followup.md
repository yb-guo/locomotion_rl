# 004: Targeted Gain/Force Follow-Up

## Goal

Only after the six-case gate, run targeted gain/force diagnostics around the
best pose/control candidate.

## Route

1. Confirm subtask 003 permits this route.
2. Choose one best pose/control candidate, not a cross-product.
3. Test named gain/force profiles one variable at a time:
   - Unitree-style leg gains;
   - higher damping;
   - stiffer knee/ankle;
   - higher force limit.
4. Stop when one profile passes or the gain/force hypothesis is falsified.

## Log

- 2026-05-11 Six-case gate permits this route because all six control/pose
  candidates failed.
- The next gain/force follow-up should anchor on `current + root_z=1.20`
  because it failed later than `unitree_gym`.
- First gate final rows showed `force_saturation_ratio=0.0`, so a pure
  force-limit increase is lower priority than damping/stiffness profile
  changes.

## Review

Status: pending follow-up implementation.
