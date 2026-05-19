# 000: Problem Contract And Observability

## Route

Define the exact continuous estimation problem before writing simulator code.

This subtask should answer:

- What is the estimated quantity: scalar endpoint payload mass, converted to
  vertical payload force.
- What assumptions make it observable: known tool attachment point,
  quasi-static motion, known gravity direction, sufficient multi-pose
  excitation, and a repeatable `0kg` baseline.
- Which signals are required from simulation and later hardware?

## Current Contract

Estimated target:

```text
mass_hat_kg
Fz_hat_N = mass_hat_kg * 9.81
```

The first pass does not estimate arbitrary contact location or a full 6D wrench.

Payload model:

```text
F_payload = [0, 0, -m * g]
tau_payload(q) = J_tool_trans(q).T * F_payload
```

Residual signal:

```text
delta_tau(t) = effort_payload(t) - effort_0kg(t)
```

Observability assumptions:

- tool link and payload attachment point are known;
- replay trajectory is deterministic and step aligned;
- trajectory visits multiple arm poses, not one static pose;
- acceleration is small enough that inertia residuals are secondary;
- friction and tracking-error effects are measured through diagnostics.

Stop if implementation starts claiming generic force estimation from this first
experiment. Passing this subtask only supports endpoint vertical payload mass
estimation under the stated assumptions.

## Log

- 2026-05-13 Created in the task023 replanning pass.
- 2026-05-13 Contract narrowed to quasi-static endpoint payload mass
  estimation using tool Jacobian and step-aligned control-effort residuals.
- 2026-05-13 User accepted the first-pass scope through the Q1-Q14 design
  review: Franka, endpoint payload mass, fixed numeric masses, deterministic
  slow replay, Genesis official Jacobian, and no neural estimator before the
  static-load baseline.

## Review

Status: complete for the first-pass experiment contract.

Evidence:

- Target is explicitly `mass_hat_kg`, with `Fz_hat_N` derived from it.
- Observability assumptions are listed and bounded.
- The contract rejects generic 6D force claims for this first pass.
