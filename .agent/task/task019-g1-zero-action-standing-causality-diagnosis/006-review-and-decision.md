# 006: Review And Decision

## Goal

Review the probe, six-case evidence, stop-rule application, and final decision.

## Route

1. Read-only reviewer checks:
   - boundary compliance;
   - no PPO/update path;
   - control-mode correctness;
   - H200 evidence;
   - stop-rule correctness.
2. Fix blocking findings through coding subagent.
3. Record final causal decision.

## Log

- 2026-05-11 Pre-H200 read-only reviewer found no blocking implementation
  findings.
- 2026-05-11 Local full pytest passed: `229 passed, 12 skipped`.
- 2026-05-11 H200 focused tests passed: `38 passed in 1.67s`.
- 2026-05-11 H200 six-case gate completed; no candidate passed.
- 2026-05-11 Final read-only reviewer found no blocking findings and agreed
  task019 should remain in progress after the first gate.
- 2026-05-11 Targeted gain-force follow-up completed on H200. No strict pass
  occurred because every profile had an early full-env reset wave, but
  `unitree_leg_gains`, `global_kv_4x`, `global_kp_0_5x_kv_2x`, and
  `knee_ankle_kp_2x_kv_2x` recovered to final chunks with zero resets and zero
  tilt.
- 2026-05-11 Final read-only reviewer found no blocking findings after
  reset/contact follow-up. Reviewer confirmed no `VectorizedGenesisBackend`
  edits, no PPO/update path, strict pass semantics preserved, and docs do not
  claim task019 passed.

Interim decision:

- The first gate does not support Genesis target-hold or position-command
  frequency as the primary cause.
- The first gate does not support a simple switch from Genesis position control
  to custom torque PD as a fix.
- The Unitree Gym 12DoF-style pose is not a fix in this 27DoF no-hand Genesis
  asset; it fails earlier than the task018 current pose.
- Next diagnostic should stay below PPO and target gain/damping/stiffness or
  reset/contact state.
- Force limit is not supported as the primary cause: `force_limit_2x` matched
  baseline and final force saturation remained zero.
- Gain/damping affects recovery/final stability, but the remaining unexplained
  signal is the early all-env tilt/reset wave, so next diagnostic is
  reset/contact settling.
- Reset/contact follow-up found periodic real falls, not a false reset gate.
  Warmup-only, full pre-eval reset, selected-env pre-eval reset, current-pose
  root-z sweep, and `unitree_gym` pose root-z sweep all failed with
  `max_reset_count=1024`.
- Representative metrics show `unitree_leg_gains` bad chunks repeat every
  three chunks (`3, 6, 9, ... 48`), with full-env tilt/reset each time. This is
  a zero-action standing equilibrium failure, not a one-time initialization
  transient.

## Review

Status: reviewed; task019 has a causal decision but no pass.

- Final read-only reviewer found no blocking findings.
- It is correct not to mark task019 passed because no strict 1600-step
  zero-action pass exists.
- Next recommended task boundary: explicit standing-pose micro-sweep or
  asset/contact/inertia inspection. Do not move to PPO until zero-action
  standing has a clean equilibrium.
- Final read-only review found no blocking findings. Residual risk: reviewer
  relied on recorded H200 logs rather than rerunning all H200 experiments.
- Residual risk: reviewer relied on recorded local/H200 evidence instead of
  rerunning tests during review.
