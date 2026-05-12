# 001: Repro Contract

## Goal

Define a deterministic zero-action standing loop that reproduces the task018
failure and can classify fixes or ablations.

## Route

1. Use G1 27DoF no-hand Genesis backend only.
2. Use standing commands only.
3. Use zero normalized action only.
4. Run seed-0, 1024 envs, 50 chunks, 32 policy steps per chunk.
5. Treat any `reset_count > 0` or `tilt_bad_count > 0` as failure.
6. Record enough low-level metrics to distinguish control, pose, gain, force,
   reset, and contact hypotheses.

## Log

- Contract created before coding.

## Review

Status: pending.
