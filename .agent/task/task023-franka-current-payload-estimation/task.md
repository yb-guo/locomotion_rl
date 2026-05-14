# Task 023: Franka Current Force Estimation Feasibility

## Goal

Verify whether a Franka arm can estimate continuous external force or payload
changes from joint current/effort history.

The core question is:

```text
Can joint effort/current residuals provide enough information to continuously
estimate external force or payload-induced wrench on a manipulator?
```

This task is a feasibility and instrumentation task. It should produce evidence
for or against the signal chain before adding a learning estimator or policy.

## Boundary

This task is intentionally isolated from the existing G1 locomotion line.

- Robot: Franka arm.
- First simulator: Genesis.
- First asset: Genesis built-in `xml/franka_emika_panda/panda_nohand.xml`.
- Signal: joint effort/current proxy over time.
- Target: continuous force or payload estimate, not discrete force classes.
- Output root: `outputs/task023/franka_current_force_estimation/`.

Do not move prototype code into `src/` until the task has a passing feasibility
decision. Task-local scripts and notes may live in this task directory while the
experiment shape is still changing.

## First-Pass Experiment Contract

The first experiment estimates a scalar endpoint payload mass, not a generic
6D contact wrench.

Assumptions:

- Payload is rigidly attached at the Franka tool/end-effector link.
- Payload force is gravity only: `F = [0, 0, -m * g]`.
- Motion is quasi-static: slow deterministic joint sweep, low acceleration.
- The same trajectory is replayed for `0kg` and each payload mass.
- Residual effort is step-aligned:

```text
delta_tau(t) = effort_payload(t) - effort_0kg(t)
```

Primary simulated current proxy:

```text
get_dofs_control_force()
```

Diagnostic only:

```text
get_dofs_force()
```

Do not feed `get_dofs_force()` into the first estimator because Genesis documents
it as actual internal DOF force, including controller force plus other internal
effects such as collision and Coriolis terms.

Primary estimator:

```text
delta_tau(t) ~= m * J_tool_trans(q_t).T * [0, 0, -g]
```

Estimate `m` by least squares over a sliding window. The first nominal window is
`128` steps at `500Hz` when using `dt = 0.002`, about `256ms`. If the smoke uses
another timestep, keep the window duration near `250ms`.

Genesis kinematics source:

```text
franka.get_jacobian(tool_link, local_point=None)
```

The Genesis MJCF morph must enable Jacobian/IK support with
`requires_jac_and_IK=True`. The Jacobian row order must be verified in the smoke;
current Genesis source computes translation in rows `0:3` and rotation in rows
`3:6`.

Payload attachment route:

1. Try Genesis official rigid weld constraint between payload link and Franka
   tool link.
2. If weld is unstable or unsupported for this asset, fall back to an official
   pick-place style grasp/attach route.

Continuous validation:

- Use fixed calibration masses such as `0`, `0.25`, `0.5`, `1.0`, and `2.0kg`.
- The estimator output remains continuous: `mass_hat_kg`.
- Validate with leave-one-mass-out interpolation/extrapolation tests so the
  result cannot be a hidden mass-class lookup.

## Scope

- Define the physical estimation problem and the exact signals needed:
  - joint position `q`;
  - joint velocity `dq`;
  - commanded target or action;
  - control effort / current proxy;
  - optional total joint force diagnostic;
  - known external force or payload label from simulation.
- Build a Genesis Franka trace collector only after the trace contract is
  written.
- Use continuous-valued payload labels and regression outputs, not force bins.
- Implement traditional estimators first:
  - quasi-static Jacobian static-load least squares;
  - step-aligned current residual baseline;
  - short-window smoothing/MHE-style baseline after the static estimator.
- Measure whether the estimate tracks a continuous target:
  - MAE/RMSE;
  - R2/correlation;
  - response delay;
  - sensitivity by joint;
  - degradation under faster motion/noise/friction.

## Non-Goals

- No force-class classifier.
- No neural estimator before traditional baselines are measured.
- No PPO/training loop.
- No SONIC/G1 integration.
- No real EtherCAT control in this task.
- No arbitrary 6D wrench claim before contact point and observability are
  explicitly handled.
- No download of external robot assets.

## Architecture Fit

Current repo architecture has scalar runtime paths for focused smoke/probe work
and tensor paths for vectorized training. Task023 starts as a scalar probe.

The intended eventual flow is:

```text
Franka trace collector
  -> current/effort residual features
  -> traditional continuous force estimator
  -> feasibility report
```

Only if that report passes should code move toward:

```text
runtime/current_force_estimator.py
agents/adaptation_buffer or policy observation integration
```

## Diagnosis Loop

### Feedback Loop

First feedback loop:

```text
Given a trace with continuous known force labels and effort vectors, the
estimator emits continuous force estimates and stable metrics.
```

Pass signal:

```text
The traditional estimator tracks continuous force on synthetic or Genesis traces
with bounded MAE/RMSE and records the conditions under which it works.
```

Fail signal:

```text
Effort APIs are unavailable, effort residuals are not finite, continuous force
labels are not recoverable above noise/friction, or the report silently turns a
continuous target into classes.
```

### Ranked Hypotheses

1. **Quasi-static payload force is observable from effort residuals**
   - Prediction: under slow motion, residual effort changes smoothly with the
     applied continuous payload force.
2. **Excitation determines identifiability**
   - Prediction: one fixed pose can detect load but cannot robustly identify
     force/COM; multi-pose slow sweeps improve conditioning.
3. **Dynamics and friction contaminate fast estimates**
   - Prediction: fast trajectories increase error unless acceleration/friction
     compensation or windowed estimation is added.
4. **Genesis control effort is a usable current proxy**
   - Prediction: `get_dofs_control_force()` provides the cleanest simulated
     motor-current proxy; `get_dofs_force()` is useful only as a diagnostic
     channel because it may include non-actuator effects.

## Stop Rules

- Stop if the plan drifts into force classification.
- Stop if there is no continuous ground-truth force/payload label.
- Stop if Genesis Franka cannot expose finite effort-like signals.
- Stop if a single-pose experiment is being used to claim general external
  force estimation.
- Stop before learning if the traditional estimator signal is not measurable.
- Do not mark passed without numeric continuous-estimation evidence.

## Route

1. `000-problem-contract-and-observability.md`
2. `001-trace-schema-and-local-continuous-fixture.md`
3. `002-genesis-franka-effort-api-smoke.md`
4. `003-continuous-force-payload-excitation.md`
5. `004-traditional-estimator-baselines.md`
6. `005-feasibility-report-and-next-step.md`
7. `006-pickup-event-online-observer.md`

## Acceptance

- Task remains isolated from `src/` until feasibility is decided.
- Local fixture uses continuous force labels, not force classes.
- Genesis smoke records Franka asset, Genesis version/backend, timestep, joint
  names/indices, and effort API availability.
- Genesis smoke records whether `requires_jac_and_IK=True` enables
  `get_jacobian()` for the selected tool link.
- At least one slow deterministic multi-pose payload trajectory is evaluated.
- Validation includes at least one leave-one-mass-out split.
- Report includes:
  - continuous target definition;
  - trajectory/excitation description;
  - effort/current source;
  - Jacobian source and tool link name;
  - estimator type;
  - MAE/RMSE/R2 or explicit unidentifiability reason;
  - response delay where measurable;
  - joint sensitivity summary;
  - failure modes.
- Decision states one of:
  - traditional current-force estimation is feasible enough for online
    adaptation;
  - signal is visible but needs better excitation/dynamics compensation;
  - signal is too weak or unavailable and the approach should be revised.

## Log

- 2026-05-13 Created after deciding to study current/effort-based force
  estimation with Franka.
- 2026-05-13 Replanned after user clarified the target is continuous force
  estimation, not discrete force-level separation. Removed the task-local
  classification-oriented prototype and rewrote the route around observability,
  continuous trace labels, Genesis effort API smoke, excitation, traditional
  baselines, and feasibility reporting.
- 2026-05-13 Recorded first-pass design decisions: scalar endpoint payload
  mass, quasi-static slow deterministic replay, `0kg` step-aligned effort
  baseline, `get_dofs_control_force()` as the current proxy,
  `get_dofs_force()` as diagnostic only, Genesis official `get_jacobian()`,
  sliding-window least-squares mass estimate, leave-one-mass-out validation,
  and weld-constraint payload attachment with pick-place fallback.
- 2026-05-13 Started execution. Added task-local
  `genesis_franka_effort_api_smoke.py` to probe Genesis Franka
  `panda_nohand`, effort APIs, official Jacobian, and weld support without
  moving code into `src/`.
- 2026-05-13 Local Windows run produced a blocked environment result because
  Genesis is not installed in this Python environment:
  `outputs/task023/franka_current_force_estimation/local_genesis_api_smoke.json`.
- 2026-05-13 H200 guarded run passed under remote project
  `/root/agent_workspace/project/h200-locomotion-lab-task023-franka-current-payload-estimation`.
  Command used `CUDA_VISIBLE_DEVICES=1`, Genesis `0.4.6`, CUDA backend,
  Python `3.11.11`, asset `xml/franka_emika_panda/panda_nohand.xml`, and wrote
  `outputs/task023/franka_current_force_estimation/gpu_genesis_api_smoke.json`.
  The smoke reported finite `get_dofs_control_force()`, `get_dofs_force()`,
  `get_dofs_position()`, `get_dofs_velocity()`, and `get_jacobian()`, with
  `robot_n_dofs=7`, arm DOFs `[0..6]`, tool link `link7`, Jacobian shape
  `[1, 6, 7]`, and weld smoke `ok`.
- 2026-05-13 Added and ran task-local payload trace collector. H200 produced
  full finite traces for `0`, `0.25`, `0.5`, `1.0`, and `2.0kg`, each with
  `6500` steps at `dt=0.002`, explicit safe `q0`, `1s` hold, `12s` deterministic
  sweep, control effort, diagnostic internal force, and official Jacobian.
  Quick residual check excluding the hold found monotonic effort residuals and
  nonzero-mass `delta_effort_rms` versus mass fit `R2=0.9788405960737084`.
  `2kg` shows force saturation risk and should be treated as stress-test data.
- 2026-05-13 Ran task-local Jacobian static-load estimator on the H200 traces.
  With `128`-step windows, it produced continuous `mass_hat_kg` estimates with
  overall raw `MAE=0.021341571457554224kg`,
  `RMSE=0.029435842384184258kg`, and `R2=0.9980711650685405`.
  First-pass feasibility is passed for endpoint payload mass estimation under
  the controlled quasi-static assumptions.
- 2026-05-13 Added a simulator-only `--force-limit-scale` diagnostic and reran
  `2kg` with `2x` force limits. Saturation ratio dropped from about `5.17%` to
  `0.088%`, tracking-error RMS improved from about `0.052rad` to `0.039rad`,
  and the mass estimate stayed essentially the same (`2.0137kg` default versus
  `2.0130kg` force-2x).
- 2026-05-13 Added dependency-free SVG visualization script
  `visualize_genesis_collection.py` and generated collection dashboard,
  trace-schema, and force-limit comparison SVGs under
  `outputs/task023/franka_current_force_estimation/visuals/`.
- 2026-05-13 Added optional Genesis camera capture to the task-local payload
  trace collector and reran a full `0.5kg` H200 collection with camera enabled.
  Output:
  `outputs/task023/franka_current_force_estimation/renders/full_mass_0p5kg_camera.gif`
  with `131` frames at `640x480`, plus matching trace and summary.
- 2026-05-13 Added pickup-event extension plan
  `006-pickup-event-online-observer.md` after the user asked to measure payload
  estimation while picking up a heavy object and at a fixed pose before/after
  pickup.
- 2026-05-13 Executed the pickup-event first pass on H200. Added task-local
  `pickup_event_trace.py` and `pickup_event_estimator.py`. The return-home
  same-pose diff estimated `0.25/0.5/1.0kg` as
  `0.2581/0.4870/0.9778kg` (`R2=0.9975`). The known-path pickup transport
  estimator, using a pre-collected `0kg` trajectory baseline and a `256`-step
  causal window, detected `0.25/0.5/1.0kg` in about `102/110/102ms` and
  produced post-pickup means `0.2405/0.4760/0.9437kg` (`R2=0.9868`). `1.0kg`
  shows high saturation/tracking-risk and should not be used as proof of
  hardware margin.
- 2026-05-14 Replayed the pickup estimator at `50Hz` by adding
  `--sample-stride` and `--sample-method` to `pickup_event_estimator.py`. Raw
  point samples every `20ms` detect the event but are too noisy for stable mass.
  Causal `20ms` block-mean input with a `26`-sample window detected
  `0.25/0.5/1.0kg` in `100/100/100ms`, converged in `400/460/460ms`, and
  produced post-convergence MAE `0.0069/0.0130/0.0226kg`.

## Review

Status: first-pass passed; pickup-event extension first pass executed.

The narrow claim is supported: in Genesis Franka, with known endpoint payload
location, known gravity direction, repeated slow trajectory, and
`get_dofs_control_force()` as actuator-effort proxy, payload mass is
continuously observable from effort residuals.

Do not generalize this result to arbitrary contact forces, fast motion, or real
hardware current sensing without the next compensation/validation task.

Simulator force-limit scaling was used only as a diagnostic; real hardware must
respect motor, gearbox, thermal, and driver limits.
