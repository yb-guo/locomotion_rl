# 004: Dynamic Failure MLP Train

## Route

Train the same MLP PPO stack on the first dynamic failure distribution.

Initial training mix:

```text
clean episode                         20%
persistent weak/dead episode          20%
dynamic single-failure episode        45%
dynamic two-segment switch episode    15%
```

Training randomization:

- randomized onset time
- randomized duration
- randomized leg motor target
- randomized weak/dead severity
- at most two dynamic segments per episode

Do not change:

- policy architecture
- actor observation dimension
- action dimension/order
- reward stack except for scheduler-compatible bookkeeping if needed
- robot topology, link inertial values, or contact randomization

Pass:

- 64-env smoke proves the dynamic task can train from the selected checkpoint.
- H200 8192-env run produces checkpoints.
- Intermediate checkpoints are screened; final checkpoint is not blindly
  accepted.
- Clean and persistent robustness are not destroyed.

Fail:

- Training uses explicit actor fault labels.
- Training starts before subtask 002/003 evidence exists.
- A slower or stop-walking policy is accepted as dynamic adaptation.

## Log

- 2026-05-21 Opened.

## Review

Status: open. First target speed after training is fixed `1.6 m/s`; later
speed expansion belongs to subtask 005.
