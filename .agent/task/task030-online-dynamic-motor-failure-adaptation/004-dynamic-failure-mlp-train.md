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

Entry decision from subtask 003:

- Start from task029 accepted `Fast1p6 model_4700.pt`.
- First train/evaluate fixed `1.6 m/s`; do not advance to `1.8` or `2.0 m/s`
  until dynamic `1.6 m/s` passes.
- Bias the first dynamic distribution toward `left_knee_joint` dead/recovery,
  because isolated dynamic `single-left-knee` failed while isolated dynamic
  `single-right-hip-yaw` passed.
- Preserve the original mix shape, but make the dynamic single-failure bucket
  left-knee-heavy for the first smoke and H200 run.

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
