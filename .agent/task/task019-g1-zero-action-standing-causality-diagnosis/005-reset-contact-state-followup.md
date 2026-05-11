# 005: Reset/Contact State Follow-Up

## Goal

Only if control, pose, and targeted gain/force do not explain the failure,
inspect reset-state and contact/asset dynamics.

## Route

1. Confirm subtasks 003 and 004 permit this route.
2. Test root-z, hard velocity zeroing, settle-before-eval, and target
   reapplication one variable at a time.
3. Record foot contact, body contact, force saturation, joint error, and root
   trajectory evidence.
4. Decide whether the next task must inspect asset/contact/inertia details.

## Log

- 2026-05-11 Six-case gate permits later reset/contact diagnostics if targeted
  gain/force follow-up does not produce a stable candidate.
- Current evidence still shows tilt/fall resets with hard-height termination
  counts at zero.

## Review

Status: pending earlier follow-up.
