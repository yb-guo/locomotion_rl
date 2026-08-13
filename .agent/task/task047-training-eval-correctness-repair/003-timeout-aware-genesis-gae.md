# 003: Timeout-Aware Legacy Genesis GAE

## Route

Correct timeout handling in the repository's legacy Genesis PPO loop.

Current rollout collection retains `transition.terminated` and
`transition.truncated` only long enough to produce counts. `RolloutBatch`
stores combined `dones`, and `compute_gae()` uses `1 - done` for both value
bootstrapping and recursive GAE continuation.

The Genesis velocity environment auto-resets done rows before returning its
observation. Correct handling therefore needs two separate boundaries:

- the TD delta bootstraps a truncation from the value of its pre-reset terminal
  observation, but does not bootstrap a true termination;
- recursive GAE does not flow from a truncated transition into the next
  episode's reset transition.

Preserve the terminal observation/value before auto-reset or return an
equivalent per-transition bootstrap value. Do not use the reset observation as
the timeout bootstrap target.

## Acceptance

- `RolloutBatch` retains per-step termination/truncation information or an
  equivalent lossless mask contract.
- The environment/collector exposes a pre-reset terminal observation/value for
  auto-reset rows.
- An analytic test proves:
  - true termination uses no next-state value;
  - truncation includes `gamma * V(terminal_observation)` in its TD delta;
  - truncation does not carry the next reset episode's GAE backward;
  - ordinary non-done transitions preserve standard GAE recursion.
- A mixed vectorized batch proves masks are applied per environment.
- Timeout/fall/reset metric counts remain unchanged.
- Existing PPO smoke tests pass and no NaN/shape regression is introduced.
- The review explicitly scopes this fix to the local legacy Genesis PPO and
  does not rewrite conclusions from the upstream MJLab/RSL-RL baseline.

## Log

- 2026-08-07 Opened from `training/ppo_loop.py`: rollout collection sees both
  flags, while `compute_gae()` uses combined `batch.dones` for all boundaries.
- 2026-08-07 Additional static check found
  `G1VelocityTrackingVectorizedEnv.step()` resets done rows before returning
  `transition.observation`, so changing the mask alone would bootstrap from
  the wrong episode.
- 2026-08-07 Added `terminal_observation` to `G1VelocityTrackingStep` before
  auto-reset and propagated per-step `terminated`, `truncated`, and
  `terminal_values` through `RolloutBatch`.
- 2026-08-07 Updated `compute_gae()` so true terminations are value-terminal,
  truncations bootstrap from `V(terminal_observation)`, and recursive GAE uses
  the combined done mask to avoid crossing into reset episodes.
- 2026-08-07 Added an analytic mixed-vector regression proving true
  termination, timeout bootstrap, reset-boundary isolation, and ordinary
  recursion. Local targeted pytest evidence is recorded in subtask 006 and
  passed.
- 2026-08-07 Runtime target changed to local RTX 4090. A bounded Genesis
  runtime smoke is still blocked locally because `genesis` is not installed and
  the configured G1 MJCF asset is absent. No simulator asset or upstream repo
  was downloaded.
- 2026-08-07 User narrowed the local runtime route to MuJoCo-only training.
  Genesis is intentionally not used as a runtime gate for this task. The
  timeout-aware GAE repair remains covered by analytic/local PPO tests and is
  retained only as legacy code correctness evidence.

## Review

Status: local code/regression fixed; no Genesis runtime evidence required for
the current MuJoCo-only route.

This repair is necessary before using new legacy Genesis PPO results as
algorithm evidence, but no new Genesis result is part of the current local
training route.
