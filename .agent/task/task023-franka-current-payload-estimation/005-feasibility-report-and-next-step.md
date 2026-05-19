# 005: Feasibility Report And Next Step

## Route

Review the continuous-estimation evidence and decide whether to proceed.

The decision should state whether to:

- move toward an online adaptation estimator;
- improve excitation/dynamics compensation first;
- revise the measurement source or simulator approach.

Required report sections:

- Genesis version/backend/timestep.
- Asset path and selected tool link.
- Payload attachment method: weld, pick-place fallback, or failed.
- Effort source: `get_dofs_control_force()` for estimator input.
- Diagnostic source: `get_dofs_force()` if recorded.
- Jacobian source: Genesis official `get_jacobian()`.
- Baseline replay method: same trajectory and step-aligned `0kg` trace.
- Mass set and leave-one-mass-out splits.
- Window size in steps and seconds.
- Metrics: MAE, RMSE, R2, slope/intercept, delay if measurable.
- Failure modes: tracking error, friction/damping, acceleration contamination,
  poor pose excitation, weld instability, Jacobian/API mismatch.

A pass means the Jacobian static-load estimator produces continuous
`mass_hat_kg` estimates with interpretable error under quasi-static replay. It
does not mean generic external-force estimation is solved.

## First-Pass Feasibility Result

Decision:

```text
traditional current/effort-based endpoint payload mass estimation is feasible
in the controlled quasi-static Franka setup.
```

Evidence:

- Simulator: Genesis `0.4.6`, CUDA backend, H200 target.
- Asset: `xml/franka_emika_panda/panda_nohand.xml`.
- Tool link: `link7`.
- Current proxy: `get_dofs_control_force()`.
- Diagnostic internal force: `get_dofs_force()`, not used as estimator input.
- Jacobian source: Genesis official `get_jacobian()`.
- Attachment: generated payload MJCF, welded to `link7`.
- Trajectory: explicit safe `q0`, `1s` hold, `12s` deterministic slow sweep,
  `dt=0.002`, `6500` steps per mass.
- Masses: `0`, `0.25`, `0.5`, `1.0`, `2.0kg`.
- Estimator: sliding-window Jacobian static-load least squares, `128` steps.

Numeric result:

```text
overall raw MAE  = 0.021341571457554224 kg
overall raw RMSE = 0.029435842384184258 kg
overall raw R2   = 0.9980711650685405
```

Per-mass means:

```text
0.25kg -> 0.24484318861482043kg
0.5kg  -> 0.49197420985602597kg
1.0kg  -> 1.0054029859689053kg
2.0kg  -> 2.0137248140389747kg
```

This supports the narrow claim:

```text
Given known endpoint payload location, known gravity direction, repeated slow
trajectory, and available actuator effort, endpoint payload mass is observable
from effort residuals.
```

It does not support these broader claims yet:

- arbitrary contact point estimation;
- arbitrary 6D external wrench estimation;
- fast-motion force estimation without dynamics/friction compensation;
- real hardware current-to-torque accuracy.

Recommended next task:

```text
Add dynamics/friction contamination tests and online adaptation buffer API.
```

The next clean extension is to keep the same estimator but vary speed,
trajectory pose coverage, friction/damping, and payload COM offset before adding
a learned residual estimator.

## Log

- 2026-05-13 Created in the task023 replanning pass.
- 2026-05-13 Added concrete feasibility report requirements and narrowed the
  pass claim to quasi-static endpoint payload mass estimation.
- 2026-05-13 First-pass feasibility report completed from H200 traces and the
  Jacobian static-load estimator.

## Review

Status: passed for first-pass feasibility.

Residual risk:

- Favorable setup: known tool payload, known direction, repeated trajectory.
- `2kg` saturation should be treated as stress-test evidence.
- The hardware mapping `current -> joint torque` remains unverified.
