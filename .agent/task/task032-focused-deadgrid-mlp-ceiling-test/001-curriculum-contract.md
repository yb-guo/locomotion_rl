# 001 Curriculum Contract

## Route

Define the focused MLP tuning experiment before running H200 training.

Focused joints:

- `left_hip_yaw_joint`
- `left_hip_roll_joint`
- `right_hip_pitch_joint`
- `right_knee_joint`
- `left_hip_pitch_joint`

Curriculum stages:

- Weak-focused: one focused motor degraded per episode, mostly weak or partial
  dead. This should teach compensation without immediate collapse.
- Mixed-focused: same focused set with higher dead probability and lower torque
  scale.
- Hard-focused: near Task031 forced dead-grid severity, only if weak/mixed
  improves eval without dynamic-switch regression.

Non-goals:

- No arbitrary mid-episode onset training.
- No history stack, GRU, transformer, or LocoFormer policy change.
- No mass/COM/inertia/link randomization.

## Log

- 2026-05-28 Contract opened with the user-approved goal: test whether focused
  tuning can solve or materially improve Level A forced persistent dead-grid.

## Review

Status: planned. Pass requires H200 env inspect evidence and a training/eval
decision.
