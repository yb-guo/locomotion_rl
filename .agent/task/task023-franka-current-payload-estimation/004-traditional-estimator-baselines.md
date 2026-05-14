# 004: Traditional Estimator Baselines

## Route

Implement and compare traditional continuous estimators before any neural
estimator.

Primary baseline:

```text
delta_tau(t) ~= m * J_tool_trans(q_t).T * [0, 0, -g]
```

Estimate `m` by least squares over a sliding window:

```text
m_hat = argmin_m sum_t ||delta_tau(t) - m * a(q_t)||^2
a(q_t) = J_tool_trans(q_t).T * [0, 0, -g]
```

Use only `get_dofs_control_force()` for `delta_tau`. Record `get_dofs_force()`
only as a diagnostic/ablation channel.

The first nominal window is:

```text
128 steps at 500Hz, about 256ms
```

If the smoke uses another timestep, keep the duration near `250ms`.

Secondary baselines, only after the primary baseline runs:

```text
step-aligned residual-only scalar LS without Jacobian
windowed smoothing/MHE-style estimate
diagnostic ablation using get_dofs_force(), clearly marked as oracle-risk
```

Metrics must be continuous regression metrics such as MAE, RMSE, R2, correlation,
and delay.

Validation must include leave-one-mass-out splits, for example train on
`0.0, 0.25, 1.0, 2.0kg` and test on `0.5kg`. The estimator output is always
`mass_hat_kg`, not a mass class.

Task-local estimator:

```text
.agent/task/task023-franka-current-payload-estimation/jacobian_payload_mass_estimator.py
```

Output:

```text
outputs/task023/franka_current_force_estimation/summaries/jacobian_static_payload_estimator.json
```

## Log

- 2026-05-13 Created in the task023 replanning pass.
- 2026-05-13 Replaced generic residual/Ridge framing with Jacobian static-load
  least squares over sliding windows and leave-one-mass-out regression
  validation.
- 2026-05-13 Added task-local `jacobian_payload_mass_estimator.py`.
  It uses step-aligned full traces and estimates mass with:
  `delta_tau ~= mass_kg * J_tool_trans(q).T * [0, 0, +g]`.
  The positive gravity sign is intentional because `get_dofs_control_force()`
  measures actuator reaction to the downward payload.
- 2026-05-13 Syntax evidence:
  `python -c "import ast,pathlib; ast.parse(pathlib.Path('.agent/task/task023-franka-current-payload-estimation/jacobian_payload_mass_estimator.py').read_text(encoding='utf-8')); print('ESTIMATOR_AST_OK')"`
  -> `ESTIMATOR_AST_OK`.
- 2026-05-13 Estimator command:
  `python .agent/task/task023-franka-current-payload-estimation/jacobian_payload_mass_estimator.py --trace-root outputs/task023/franka_current_force_estimation/traces --output outputs/task023/franka_current_force_estimation/summaries/jacobian_static_payload_estimator.json --strict`
  -> `status=ok`.
- 2026-05-13 Estimator evidence:
  - window: `128` steps, after excluding first `500` hold steps;
  - per mass window count: `5873`;
  - overall raw continuous regression: `MAE=0.021341571457554224kg`,
    `RMSE=0.029435842384184258kg`, `R2=0.9980711650685405`;
  - `0.25kg`: mean `0.24484318861482043kg`, MAE
    `0.009767620327708268kg`;
  - `0.5kg`: mean `0.49197420985602597kg`, MAE
    `0.01755819336124124kg`;
  - `1.0kg`: mean `1.0054029859689053kg`, MAE
    `0.021893982441163144kg`;
  - `2.0kg`: mean `2.0137248140389747kg`, MAE
    `0.03614648970010424kg`, low-confidence due to saturation risk.
- 2026-05-13 Leave-one-mass-out linear calibration remained accurate:
  held-out MAE ranged from `0.01064076796034501kg` to
  `0.03558439873069835kg`.
- 2026-05-13 Compared default `2kg` trace against `2kg` with
  `--force-limit-scale 2.0`. Estimator output stayed almost unchanged:
  default `2kg` mean `2.013724814038456kg`, MAE `0.03614648969996734kg`;
  force-2x mean `2.012965473059285kg`, MAE `0.03718869291010696kg`.
  This suggests the first-pass mass estimate was not primarily an artifact of
  the default force saturation, although the default `2kg` trace remains a
  lower-quality control trace.

## Review

Status: passed for first traditional baseline.

Evidence:

- No neural estimator was used.
- Output is continuous `mass_hat_kg`.
- The estimator used the official Genesis Jacobian and control-effort residual.
- Leave-one-mass-out checks do not collapse to discrete classes.

Remaining risk:

- The current trace is a very favorable quasi-static endpoint payload setup.
- `2kg` has nontrivial force saturation and should not be used as clean
  calibration evidence without saturation handling.
- Raising simulated force limits helps diagnose saturation but is not a
  hardware-safe assumption.
