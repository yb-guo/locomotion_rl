# 004: Fall-Conditioned Retry Recovery Shaping

## Route

Align Task046 training semantics with its retry-after-fall claim.

`Task046PostResetRecoveryRewardVecEnvWrapper` currently selects final-trial
early/tail windows from trial index and trial step. It does not remember whether
the preceding inner reset was a fall or timeout. Add a per-env reset-reason
state machine:

- update state only on an inner reset;
- enable shaping in the following retry trial only when that reset reason was
  fall;
- do not shape a timeout retry or a final trial reached without a prior fall;
- clear state on outer reset;
- preserve hidden fault identity and keep the feature/reward default-off.

Use the same reset-reason constants as the multi-trial wrapper. Avoid deriving
fall state only from the current step's `reset_reason`, because later recovery
steps must retain the preceding reset reason.

## Acceptance

- A sequential fake-env test covers fall, timeout, no-prior-reset, and outer
  reset paths across multiple steps.
- Only the post-fall retry receives early/tail reward shaping.
- Timeout and no-prior-fall rows have exactly unchanged rewards.
- Outer reset clears stale fall state before the next episode.
- Debug JSON separates eligible post-fall samples, timeout retries, and skipped
  samples.
- Default-off behavior and actor-observation/fault-hiding contracts remain
  unchanged.
- A corrected local RTX 4090 consumer smoke records nonzero eligible post-fall
  samples.
- Any corrected quality comparison uses a newly trained checkpoint and reruns
  the same retry matrix; Task046 Stage1/Stage2 are retained as evidence for the
  old broader post-reset objective, not relabeled as fall-conditioned training.

## Log

- 2026-08-07 Opened from
  `Task046PostResetRecoveryRewardVecEnvWrapper._shape_rewards()`: masks depend
  on final trial and trial step only.
- 2026-08-07 The adjacent retry-context wrapper already demonstrates a
  per-environment previous-reset-reason contract, but the reward wrapper does
  not consume equivalent state.
- 2026-08-07 Added per-env previous inner-reset reason state to
  `Task046PostResetRecoveryRewardVecEnvWrapper`, cleared it on wrapper reset
  and outer reset, and shaped recovery/tail rewards only for post-fall retry
  windows.
- 2026-08-07 Added debug counters for eligible post-fall samples, skipped
  timeout retries, and skipped no-prior-fall windows.
- 2026-08-07 Updated the fake-env regression to cover post-fall recovery/tail,
  timeout retry, no-prior-reset, and outer-reset stale-state clearing. Local
  targeted pytest evidence is recorded in subtask 006 and passed.
- 2026-08-07 Runtime target changed to local RTX 4090. The corrected consumer
  smoke/retraining route is blocked locally by missing MJLab/task modules,
  missing configured G1 asset, no local checkpoint path, and current GPU memory
  pressure. No checkpoint or asset was downloaded.

## Review

Status: local code/regression fixed; local RTX 4090 consumer smoke/retraining
evidence blocked by missing runtime/assets/checkpoint and GPU memory pressure.

The existing Stage2 result remains a valid experiment for general post-reset
shaping. It is not evidence that the reward was specifically conditioned on a
fall.
