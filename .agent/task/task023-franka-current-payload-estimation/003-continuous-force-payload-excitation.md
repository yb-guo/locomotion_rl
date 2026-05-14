# 003: Continuous Force Payload Excitation

## Route

Design the continuous external-force or payload trajectory.

Start with a clean, observable setup:

- known endpoint payload at the Franka tool link;
- slow multi-pose sweep;
- multiple continuous-valued mass samples;
- no grasping ambiguity in the first run.

The output is a trace where the ground-truth target is continuous.

First mass set:

```text
0.0kg, 0.25kg, 0.5kg, 1.0kg, 2.0kg
```

Trajectory:

- deterministic slow sine or spline joint sweep;
- same command sequence for every mass;
- low acceleration so quasi-static gravity load dominates;
- no random trajectory in the first pass.

Payload attachment:

1. Preferred: Genesis official weld constraint between a known-mass payload
   link and the Franka tool link.
2. Fallback: official pick-place/grasp-style attachment if direct weld is not
   stable or unsupported.

The output labels are continuous numeric values:

```text
payload_mass_kg
payload_force_z_N = -payload_mass_kg * 9.81
```

Task-local collector:

```text
.agent/task/task023-franka-current-payload-estimation/genesis_franka_payload_trace.py
```

Safe Franka start pose:

```text
q0 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
```

Trajectory used for the first full trace set:

```text
dt = 0.002
hold = 1.0s
sweep = 12.0s
total = 6500 steps
q_target[i] = q0[i] + amp[i] * sin(2*pi*t/12.0 + phase[i])
```

## Log

- 2026-05-13 Created in the task023 replanning pass.
- 2026-05-13 Set first excitation plan to fixed numeric payload masses on a
  deterministic slow replay trajectory, with weld attachment first and
  pick-place fallback.
- 2026-05-13 Added task-local `genesis_franka_payload_trace.py`. It runs one
  mass per process to avoid Genesis singleton reuse, writes compressed NPZ
  traces plus JSON summaries, uses explicit `q0`, attaches positive-mass
  payload MJCF boxes to `link7` with Genesis weld constraints, and records
  `q`, `dq`, `q_target`, `tracking_error`, `effort_control`,
  `effort_internal`, `jacobian`, `tool_pos`, `payload_mass_kg`, and
  `payload_force_z_N`.
- 2026-05-13 Local syntax evidence:
  `python -c "import ast,pathlib; ast.parse(pathlib.Path('.agent/task/task023-franka-current-payload-estimation/genesis_franka_payload_trace.py').read_text(encoding='utf-8')); print('TRACE_AST_OK')"`
  -> `TRACE_AST_OK`.
- 2026-05-13 H200 short trace smoke passed for `0kg` and `0.5kg` with
  `hold=0.1s`, `sweep=0.2s`, `150` steps each. This verified trace fields,
  finite arrays, payload MJCF generation, and weld attachment. The short sweep
  is intentionally not used as quasi-static estimator evidence.
- 2026-05-13 H200 full trace set passed under remote project
  `/root/agent_workspace/project/h200-locomotion-lab-task023-franka-current-payload-estimation`
  with `CUDA_VISIBLE_DEVICES=1`, Genesis `0.4.6`, CUDA backend, explicit `q0`,
  `hold=1.0s`, `sweep=12.0s`, and `6500` steps per mass:
  - `full_mass_0kg`
  - `full_mass_0p25kg`
  - `full_mass_0p5kg`
  - `full_mass_1kg`
  - `full_mass_2kg`
- 2026-05-13 Full trace outputs were copied back locally under
  `outputs/task023/franka_current_force_estimation/traces/` and
  `outputs/task023/franka_current_force_estimation/summaries/`.
- 2026-05-13 Quick delta check excluding the first `500` hold steps produced
  `outputs/task023/franka_current_force_estimation/summaries/full_payload_trace_delta_summary.json`.
  Nonzero-mass `delta_effort_rms` versus mass fit:
  `slope=15.26558234460998`, `intercept=3.6377331142271365`,
  `R2=0.9788405960737084`.
  Per-mass `delta_effort_rms_vs_0kg`:
  - `0.25kg`: `5.902297154090056`
  - `0.5kg`: `11.5174888740859`
  - `1.0kg`: `21.24857378351316`
  - `2.0kg`: `33.12850643750679`
- 2026-05-13 Risk noted: `2.0kg` is a saturation/low-confidence case. Its
  control effort saturation ratio after hold is about `5.17%`, and
  tracking-error RMS is about `0.052rad`. Use it as stress-test data, not clean
  calibration data, unless the estimator explicitly models saturation.
- 2026-05-13 Added `--force-limit-scale` to the task-local trace collector so
  the simulator force range can be raised without changing the default
  experiment. H200 `2kg` rerun with `--force-limit-scale 2.0` applied force
  limits `[174,174,174,174,24,24,24]` and passed as
  `full_mass_2kg_force2x`.
- 2026-05-13 Force-limit comparison output:
  `outputs/task023/franka_current_force_estimation/summaries/force_limit_2kg_compare.json`.
  Raising the limit reduced `2kg` saturation ratio from about `5.17%` to
  `0.088%` and tracking-error RMS from about `0.052rad` to `0.039rad`.
- 2026-05-13 Added optional camera rendering to `genesis_franka_payload_trace.py`
  with `--render-gif`, `--render-every-steps`, camera resolution, pose, lookat,
  and FOV options. Default remains no camera, so numeric trace collection is
  unchanged unless rendering is requested.
- 2026-05-13 H200 camera rerun completed for `0.5kg`:
  `full_mass_0p5kg_camera`, `6500` numeric steps, camera resolution `640x480`,
  `131` rendered frames, render stride `50` steps, `12fps`. Output GIF:
  `outputs/task023/franka_current_force_estimation/renders/full_mass_0p5kg_camera.gif`.
  Summary copied locally:
  `outputs/task023/franka_current_force_estimation/summaries/full_mass_0p5kg_camera.json`.
  The run reported `status=ok`, RGB shape `[480,640,3]`, GIF bytes `7215141`,
  and tracking-error RMS `0.018920927354932345`.

## Review

Status: passed for first full payload trace generation.

Evidence:

- H200 generated finite traces for all planned masses.
- Positive payloads attached with weld to `link7`.
- Same deterministic trajectory and explicit safe `q0` were used for every
  mass.
- Continuous labels are present as numeric `payload_mass_kg` and
  `payload_force_z_N`.
- Effort residual magnitude increases monotonically with mass in the quick
  delta check.
- Camera rerun generated a real Genesis render GIF for a full `0.5kg` payload
  collection pass.

Remaining risk:

- `2kg` is partly force-limited and should be marked low-confidence in the
  estimator baseline.
- Simulator-only force-limit scaling is diagnostic. It must not be interpreted
  as a real-hardware safety recommendation.
