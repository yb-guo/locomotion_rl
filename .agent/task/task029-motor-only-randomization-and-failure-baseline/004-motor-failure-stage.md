# 004: Motor Failure Stage

## Route

Implement the first motor-failure training stage: episode-start persistent leg
motor degradation. This is the first acceptance failure mode for task029.

Failure contract:

- Sample at reset, not mid-episode.
- Affect only leg actuators in the first pass.
- Sample `0-2` failed/degraded motors per episode.
- Normal scale: `1.0`.
- Weak scale: `0.3-0.7`.
- Dead scale: `0.0-0.1`.
- Do not fail waist, arms, wrists, or grippers in this subtask.
- Do not train locked-joint or stuck-command failures in this subtask.

Preferred first implementation:

- Apply per-actuator strength or effort scaling.
- Keep failure persistent for the full episode.
- Record sampled actuator IDs, joint names, failure type, and scale.
- Expose failure data only to logs, eval summaries, and optionally critic
  privileged information.

## Minimal Closed Loop

Feedback loop:

1. Add `MotorFailure` task/stage.
2. Run a sampler diagnostic for many resets and save distribution stats.
3. Run a one-env replay/trace that forces one known motor weak/dead and proves
   the realized scale affects torque/effort.
4. Run 64-env, 2-iteration PPO smoke on H200.

Pass:

- Failure sampling matches the `0-2` motor contract.
- Only leg actuators are sampled.
- Forced single-motor weak/dead traces show the intended actuator-side effect.
- Actor does not receive failure labels or motor scales.
- The stage preserves action dim and actor obs contract.

Fail:

- Upper-body, waist, or gripper motors are failed in the first-pass stage.
- Failure is sampled mid-episode without an explicit later subtask.
- Failure mask leaks into actor obs.
- There is no forced-motor trace proving the selected actuator is actually
  degraded.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_failure_stage/`.

## Log

- 2026-05-19 Opened after the user approved persistent episode-start weak/dead
  leg motor failure as task029's first acceptance target.

## Review

Status: pending.

This subtask should prove the fault model is real and diagnosable before using
it for convergence claims. A passing PPO smoke alone is not enough; the forced
single-motor trace is required.
