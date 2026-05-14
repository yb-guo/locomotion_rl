# 001: Trace Schema And Local Continuous Fixture

## Route

Create a task-local trace schema and a local continuous fixture.

The fixture must use continuous-valued regression labels. It must not become a
force-class dataset.

Expected row families:

```text
step, t, payload_mass_kg, payload_force_z_N
q, dq, q_target, tracking_error
effort_control, effort_internal_diagnostic
tool_link_name, jacobian_trans, jacobian_source
```

`effort_control` means Genesis `get_dofs_control_force()`.

`effort_internal_diagnostic` means Genesis `get_dofs_force()` and must not be
fed into the first estimator.

The first local fixture should support paired traces:

```text
same trajectory, payload_mass_kg = 0
same trajectory, payload_mass_kg = m
```

The paired traces must be step-aligned so downstream code can compute:

```text
delta_tau(t) = effort_control_m(t) - effort_control_0kg(t)
tracking_error_delta(t) = tracking_error_m(t) - tracking_error_0kg(t)
```

For continuous validation, use fixed numeric masses but train/test as
regression, not classification. The report must include leave-one-mass-out
splits.

## Log

- 2026-05-13 Created in the task023 replanning pass.
- 2026-05-13 Expanded schema around paired `0kg` and payload traces,
  `get_dofs_control_force()` primary effort, `get_dofs_force()` diagnostic,
  and continuous `payload_mass_kg` labels.
- 2026-05-13 Schema aligned to the accepted baseline method: step-aligned
  `0kg` replay, `q/dq/q_target/tracking_error`, control effort, diagnostic
  internal force, tool Jacobian, and continuous payload labels.

## Review

Status: complete for the first-pass trace contract.

Evidence:

- Required trace fields are listed before collector implementation.
- Primary and diagnostic effort channels are separated.
- Leave-one-mass-out regression is required to avoid hidden classification.
