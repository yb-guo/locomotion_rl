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
- 2026-05-19 Implemented
  `Unitree-G1-Gripper-Flat-MotorOnly-Failure` with a reset-mode
  `motor_failure` event. The event resolves leg targets by joint name, records
  joint names / ctrl ids / action indices / failure type / scale, restores
  default `actuator_forcerange` for reset envs, then applies weak/dead scales.
- 2026-05-19 Initial per-env reset diagnostic caught that the custom event had
  not declared `actuator_forcerange` as a domain randomization field, so model
  field writes were not isolated per env. Fixed by decorating
  `task029_motor_failure` with `@requires_model_fields("actuator_forcerange")`.
- 2026-05-19 Verification evidence saved under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_failure_stage/`:
  - `inspect/motor_failure_inspect_summary.json`: pass. Train/play configs have
    only `reset_base`, `reset_robot_joints`, and `motor_failure` events;
    `motor_failure` is reset mode; action dim is 31; actor obs is 104; actor
    terms do not expose motor/failure/fault/scale; critic has no failure
    privileged info.
  - `diagnostics/motor_failure_diagnostics_summary.json`: pass. Sampler proves
    0-2 failures per reset, leg-only targets, weak range 0.3-0.7, dead range
    0.0-0.1, and per-env reset default-restore plus scaled reapply.
  - `diagnostics/motor_failure_forced_trace.json`: pass. Same-action
    default/weak/dead trace shows the selected leg actuator force range scales
    and actuator force decreases default > weak > dead.
  - `smoke/motor_failure_smoke_summary.json`: pass. 64 env, 2 PPO iterations
    produced `model_1.pt`, params YAML, TensorBoard event, and no residual
    `scripts/train.py` processes.
- 2026-05-19 Actor leak probe covered train/play config terms, functions,
  params, actor shape, and action dim. Runner-level rollout observation hook
  probing was not performed in 004, so this is not a full robustness pass.

## Review

Status: passed for 004 stage smoke.

This only passes the 004 motor-failure stage smoke. It does not mark 005-007 or
full robustness complete. Runner-level actor observation plumbing remains a
later hardening check.
