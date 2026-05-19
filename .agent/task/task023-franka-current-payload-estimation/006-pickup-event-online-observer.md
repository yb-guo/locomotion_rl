# 006: Pickup Event Online Observer

## Goal

Extend the first-pass attached-payload experiment into a pickup event experiment.

The two questions are:

```text
1. Across an unloaded move -> pickup -> loaded move sequence, can the estimator
   emit a real-time continuous payload estimate?
2. If the arm starts at an initial pose, picks up a payload, and returns to the
   same initial pose, can the before/after effort difference estimate the
   payload mass?
```

This subtask is still a traditional-estimator task. It should not add a neural
estimator or policy loop.

## Boundary

- Robot remains Franka in Genesis.
- Payload remains a scalar endpoint mass at `link7`.
- Payload force remains gravity only.
- First pickup model is an instantaneous attach/weld event, not real grasp
  contact physics.
- Output remains continuous `mass_hat_kg`, not a class.
- Code, traces, and notes remain task-local until this extension passes.

## Route

## Diagnosis Feedback Loop

The feedback loop for this subtask is a deterministic Genesis episode runner
that emits one trace and one metrics JSON per run.

The loop should answer one question per invocation:

```text
Given a known payload mass and a known pickup event time, does the online
estimator output near 0kg before pickup and a continuous mass estimate after
pickup with bounded delay and bounded error?
```

Minimum viable loop:

```text
python pickup_event_trace.py \
  --scenario pickup_transport \
  --payload-mass-kg 0.5 \
  --seed 23 \
  --dt 0.002 \
  --window-steps 128

python pickup_event_estimator.py \
  --trace outputs/task023/.../pickup_transport_0p5kg.npz \
  --baseline outputs/task023/.../pickup_transport_0kg_baseline.npz
```

The exact script names may change, but the loop must stay deterministic and
agent-runnable on H200.

Do not count a run as reproduced unless the trace includes:

- `phase`;
- `pickup_event_step`;
- `payload_attached`;
- `payload_mass_kg`;
- `q`, `dq`, `q_target`;
- `effort_control`;
- `tool_pos`;
- `jacobian`;
- `mass_hat_kg`;
- `residual_norm`;
- `tracking_error`;
- `force_saturation_ratio`.

## Ranked Hypotheses

1. **Known-path moving pickup is observable after unloaded baseline subtraction**
   - Prediction: in `pickup_transport`, `mass_hat_kg` stays near `0kg` before
     pickup and converges near the true payload during loaded motion.
   - Falsifier: pre-pickup false positive is high, or post-pickup estimate does
     not move monotonically toward the true mass.

2. **Return-to-home same-pose difference is cleaner than moving estimation**
   - Prediction: `return_home_diff` has lower post-pickup mass error than
     `pickup_transport` because robot gravity and pose-dependent controller bias
     mostly cancel at `q_home`.
   - Falsifier: same-pose diff is no better than moving estimates, or changes
     sign across repeated runs at the same mass.

3. **Attach transient contaminates the first few hundred milliseconds**
   - Prediction: error spikes immediately after `pickup_event_step`, then
     settles after excluding a short transient window.
   - Falsifier: the estimate remains unstable long after the transient window.

4. **Tracking error and force saturation limit estimator reliability**
   - Prediction: large tracking error or actuator saturation correlates with
     mass error, especially at `2.0kg`.
   - Falsifier: mass error is uncorrelated with tracking/saturation metrics.

5. **Single home pose is insufficient for robust mass claims**
   - Prediction: one pose can detect a payload, but pose-conditioned error
     changes across two or three `q_home` choices.
   - Falsifier: estimates remain stable across home poses within tolerance.

### A. Moving Before/After Pickup

This is the real-time transport test.

The run is one continuous episode with phase labels:

```text
home_hold_before:
  hold q_home with no payload; estimator should output near 0kg.

move_to_pick_unloaded:
  move from q_home to q_pick with no payload; estimator should stay near 0kg.

pickup_event:
  attach the payload to link7 at the known event step.

move_with_payload:
  move away while carrying the payload; estimator should rise and track mass.
```

The estimator must emit `mass_hat_kg(t)` causally every control step. It may use
past data and pre-collected calibration data, but not future samples from the
same episode.

Keep estimator variants separate:

```text
trajectory_baseline:
  delta_tau(t) = effort_pickup_episode(t) - unloaded_baseline_by_phase(t)
  Purpose: first real-time moving test for a known/repeated path.
  Claim: causal if the unloaded baseline was collected before this episode.

oracle_reference:
  delta_tau(t) = effort_pickup_episode(t) - effort_0kg_same_trajectory(t)
  Purpose: debugging upper bound for signal visibility and timing.
  Claim: not deployable online by itself.
```

For the first implementation, use the same `128`-step sliding window as the
first-pass estimator unless pickup transients require a longer window.

Execution note: the `0.5kg` moving pickup run with `128` steps detected the
payload quickly but needed about `3.32s` to remain inside the `50g` convergence
band. Re-running the same trace with a `256`-step window passed the first
moving-pickup criterion. Treat `256` steps at `500Hz` as the current pickup
transport default until a better transient model is added.

A pass here means the signal is visible during transport and the estimator can
report the mass online on a calibrated path. It does not yet prove arbitrary
motion without a dynamics/unloaded-effort model.

First concrete episode:

```text
q_home:
  [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

q_pick:
  q_home + [0.18, -0.10, 0.14, 0.08, -0.12, 0.06, 0.10]

q_carry:
  q_home + [-0.16, 0.12, -0.12, -0.06, 0.10, -0.06, -0.08]
```

Use smooth joint interpolation between waypoints. Keep the first pass slow:

```text
home_hold_before:   1.0s
move_to_pick:       3.0s
pickup_event:       instant attach at q_pick
settle_after_pick:  0.5s
move_with_payload:  4.0s
final_hold_loaded:  1.0s
```

Run a matching unloaded baseline once:

```text
home_hold_before -> move_to_pick -> no pickup -> move_to_carry -> final_hold
```

Then run the pickup episode. The online estimator may use the unloaded baseline
up to each matching phase/time index, but it may not inspect future samples from
the pickup episode.

### B. Return-To-Initial-Pose Difference

This is the cleanest before/after difference test.

The run is one continuous episode:

```text
home_hold_before:
  hold q_home with no payload and record tau_home_before.

move_to_pick_unloaded:
  move from q_home to q_pick with no payload.

pickup_event:
  attach the payload to link7.

return_home_loaded:
  move back to q_home while carrying the payload.

home_hold_after:
  hold the same q_home with payload and record tau_home_after.
```

Estimate payload mass from the same-pose effort difference:

```text
delta_tau_home = mean(tau_home_after) - mean(tau_home_before)
delta_tau_home ~= m * J_tool_trans(q_home).T * [0, 0, +g]
```

This cancels most robot gravity, controller bias, and pose-dependent effects
because the pose before and after is the same. It is closer to a practical
payload-check routine:

```text
leave home -> pick object -> return home -> compare current/effort at home
```

Repeat with two or three `q_home` poses. A single home pose is enough to prove
that a payload changed effort, but not enough to claim robust mass estimation
across the workspace.

First concrete episode:

```text
q_home:
  [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

q_pick:
  q_home + [0.18, -0.10, 0.14, 0.08, -0.12, 0.06, 0.10]
```

Timing:

```text
home_hold_before:   2.0s
move_to_pick:       3.0s
pickup_event:       instant attach at q_pick
settle_after_pick:  0.5s
return_home_loaded: 3.0s
home_hold_after:    2.0s
```

Use only the stable center of the home holds for the same-pose diff:

```text
tau_home_before = mean(effort_control during last 1.0s of home_hold_before)
tau_home_after  = mean(effort_control during last 1.0s of home_hold_after)
```

Do not include move segments in `delta_tau_home`. The moving estimates are
evaluated separately in Scenario A.

### C. Event Implementation Check

Genesis may not support every runtime attach route. Try in this order:

1. Pre-create the payload in the scene, keep it parked/unwelded, then at event
   time move it to the tool and add the official rigid weld constraint.
2. If runtime weld is unsupported, use the official pick-place/grasp route.
3. If neither works, keep static before/after as separate same-pose runs and
   record moving pickup as blocked rather than faking a continuous event.

## Metrics

For each mass and scenario record:

- `mass_hat_kg(t)`;
- phase label;
- known `pickup_time_s`;
- detection delay in milliseconds;
- convergence time to `abs(mass_hat - mass_true) <= max(0.05kg, 0.10*m_true)`;
- post-convergence MAE/RMSE;
- pre-pickup false positive mean/max;
- tracking error RMS;
- force saturation ratio;
- residual norm or estimator confidence.

Start with `0.5kg`. Then test `0.25kg` and `1.0kg`. Keep `2.0kg` as a stress
case because the first-pass traces showed saturation risk under nominal force
limits.

## Experiment Matrix

Run in this order:

```text
smoke:
  scenario: return_home_diff
  mass: 0.5kg
  seed: 23
  purpose: prove runtime attach and before/after diff.

transport_smoke:
  scenario: pickup_transport
  mass: 0.5kg
  seed: 23
  purpose: prove moving real-time mass_hat_kg(t).

mass_sweep:
  scenarios: return_home_diff, pickup_transport
  masses: 0.25kg, 0.5kg, 1.0kg
  seed: 23
  purpose: continuous mass scaling.

pose_sweep:
  scenario: return_home_diff
  masses: 0.5kg
  q_home variants: home_a, home_b, home_c
  purpose: check whether same-pose diff generalizes across poses.

stress:
  scenarios: return_home_diff, pickup_transport
  masses: 2.0kg
  purpose: saturation/tracking-risk diagnostic only.
```

## Pass/Fail Criteria

First pass for `0.5kg`:

```text
return_home_diff:
  abs(mass_hat_home - 0.5kg) <= 0.05kg

pickup_transport:
  pre_pickup_abs_mean <= 0.05kg
  detection_delay <= 500ms
  convergence_time <= 1000ms
  post_convergence_MAE <= 0.10kg
```

Mass sweep:

```text
0.25kg, 0.5kg, 1.0kg:
  R2 >= 0.95 across masses
  no hidden class thresholds
  mass_hat_kg remains continuous
```

Stress case:

```text
2.0kg:
  report saturation and tracking error
  do not use as the primary pass/fail condition
```

Fail immediately if:

- payload attach silently fails;
- `effort_control` is non-finite;
- the estimator uses post-event future samples to estimate current time;
- a same-pose diff uses different `q_home` before and after;
- moving estimates are reported without stating which baseline was used.

## Execution Evidence

H200 path:

```text
/root/agent_workspace/project/h200-locomotion-lab-task023-franka-current-payload-estimation
```

Task-local scripts:

```text
.agent/task/task023-franka-current-payload-estimation/pickup_event_trace.py
.agent/task/task023-franka-current-payload-estimation/pickup_event_estimator.py
```

Local evidence copied back under:

```text
outputs/task023/franka_current_force_estimation/summaries/
outputs/task023/franka_current_force_estimation/visuals/
```

Return-to-home same-pose diff:

| Mass | Estimate | Abs error | Saturation ratio | Notes |
| ---: | ---: | ---: | ---: | --- |
| `0.25kg` | `0.2580617609412369kg` | `0.008061760941236895kg` | `0.0005714285714285715` | pass |
| `0.5kg` | `0.4870275431060155kg` | `0.012972456893984519kg` | `0.12038095238095238` | first-pass criterion passed |
| `1.0kg` | `0.9777893437222333kg` | `0.02221065627776675kg` | `0.46895238095238095` | high saturation/tracking-risk case |

Return-to-home mass sweep `R2 = 0.997508834698415` across
`0.25/0.5/1.0kg`.

Pickup transport with trajectory baseline and `256`-step causal window:

| Mass | Post-pickup mean | Post-convergence MAE | Detection delay | Convergence time | Saturation ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.25kg` | `0.2404791267595293kg` | `0.006329838686625727kg` | `102ms` | `410ms` | `0.0010526315789473684` |
| `0.5kg` | `0.47597650088913046kg` | `0.011530866412381066kg` | `110ms` | `774ms` | `0.11705263157894737` |
| `1.0kg` | `0.9436673152665057kg` | `0.021406256668836353kg` | `102ms` | `462ms` | `0.5168421052631579` |

Pickup transport post-pickup mean `R2 = 0.986830353464888` across
`0.25/0.5/1.0kg`.

Important caveat: moving pickup used a pre-collected `0kg` same-path baseline.
This is causal for the pickup episode, but it is not an arbitrary-motion
deployed estimator.

### 50Hz Observation Mode

The estimator was replayed on the same H200 traces with `50Hz` observation
input by reducing the original `500Hz` trace by `sample_stride=10`.

Point sampling every `20ms` is not enough for stable mass estimation. It detects
the event quickly, but produces large mass error:

| Mass | Detection delay | Post-pickup mean | Post-eval MAE |
| ---: | ---: | ---: | ---: |
| `0.25kg` | `0ms` | `0.25184613801574396kg` | `0.07912475365032462kg` |
| `0.5kg` | `0ms` | `0.49855100319248924kg` | `0.1649304861299149kg` |
| `1.0kg` | `18ms` | `0.986926972543701kg` | `0.30390998349078446kg` |

Using a causal `20ms` block mean before estimating is the current `50Hz` mode.
With `window_steps=26` at `50Hz` (about `520ms`), pickup transport remains
usable:

| Mass | Detection delay | Convergence time | Post-pickup mean | Post-convergence MAE |
| ---: | ---: | ---: | ---: | ---: |
| `0.25kg` | `100ms` | `400ms` | `0.2406389706429442kg` | `0.0069213785263301094kg` |
| `0.5kg` | `100ms` | `460ms` | `0.47627740790196704kg` | `0.013004007624819451kg` |
| `1.0kg` | `100ms` | `460ms` | `0.9437949445248706kg` | `0.022630923907771522kg` |

Return-home same-pose diff is unchanged under `50Hz` block mean because it uses
one-second home-hold averages:

| Mass | Estimate | Abs error |
| ---: | ---: | ---: |
| `0.25kg` | `0.25806176094123745kg` | `0.00806176094123745kg` |
| `0.5kg` | `0.4870275431060149kg` | `0.012972456893985074kg` |
| `1.0kg` | `0.9777893437222325kg` | `0.022210656277767526kg` |

Conclusion for this subtask: `50Hz` is acceptable for continuous payload
estimation if the input is a causal `20ms` averaged/low-pass effort signal. It
is not acceptable as raw instantaneous point samples if we need stable mass
values.

## Expected Interpretation

Moving before/after passing with `trajectory_baseline` means:

```text
On a calibrated path, the estimator can report near-zero before pickup and the
payload mass after pickup during motion.
```

Moving before/after passing only with `oracle_reference` means:

```text
The signal is visible during motion, but the implemented estimator still depends
on a same-trajectory reference and is not a deployable online method.
```

Return-to-initial-pose difference passing means:

```text
Returning to the same pose gives a clean before/after current-effort diff that
can estimate a continuous endpoint payload mass.
```

None of these proves arbitrary external-force estimation or real-hardware
current sensing.

## Log

- 2026-05-13 Created after the user reframed the next step as picking up a
  heavy object and measuring both moving pickup and fixed-pose before/after
  pickup behavior.
- 2026-05-13 Corrected the experiment shape after user clarification: measure
  real-time estimates across unloaded motion, pickup, and loaded motion; then
  measure same-pose before/after effort difference after leaving home, picking
  up the payload, and returning home.
- 2026-05-13 Expanded the subtask with a diagnose-style feedback loop, ranked
  falsifiable hypotheses, concrete episode timings, experiment matrix, required
  instrumentation, and pass/fail criteria.
- 2026-05-13 Implemented task-local pickup event trace and estimator scripts,
  ran H200 smoke plus `0.25/0.5/1.0kg` mass sweep for both return-home diff and
  pickup transport, and copied summaries/SVG evidence back to the local output
  tree.
- 2026-05-14 Added `50Hz` replay support to the estimator via
  `--sample-stride` and `--sample-method`. H200 replay shows raw point sampling
  detects pickup but is too noisy for stable mass; causal `20ms` block-mean
  input with a `26`-sample window preserves useful continuous estimates.

## Review

Status: first-pass executed; mass sweep passed under the stated assumptions.

Supported:

- same-pose return-home diff estimates continuous endpoint payload mass across
  `0.25/0.5/1.0kg`;
- known-path pickup transport estimates continuous payload mass online after
  subtracting a pre-collected unloaded trajectory baseline;
- `50Hz` block-mean observation input is sufficient for this known-path payload
  estimator with about a `520ms` stable-estimation window;
- generated traces and SVG plots show `mass_hat_kg(t)` across pickup events.

Not supported yet:

- arbitrary-motion online payload estimation without an unloaded-effort model;
- real grasp/contact pickup;
- real hardware current sensing;
- raw `50Hz` point samples as a stable mass signal without averaging/filtering;
- robust operation at high saturation. `1.0kg` already shows high saturation and
  tracking-risk in this trajectory.

Still pending from the full matrix:

- `q_home` pose sweep;
- `2.0kg` stress run.
