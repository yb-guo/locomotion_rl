# Task026: Official SONIC Direct MuJoCo Env

## Goal

Build the lowest-friction path for SONIC reproduction and later RL training on
H200: a single-process Python MuJoCo environment that matches the official
SONIC G1 plant and LowCmd torque contract, then use it for a small behavior
cloning smoke before any RL fine-tuning.

This task intentionally does not start by porting SONIC into mjlab. Task025
showed that mjlab position-actuator plant response diverges from official
SONIC under matched motion input. The new baseline should therefore be the
official MuJoCo plant first.

## Route

Phase A: direct official MuJoCo replay env.

1. Load the official G1 MuJoCo runtime asset used by the official Python sim:
   `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml`.
2. Avoid DDS and C++ deploy in the training/replay loop. Use one Python process.
3. Expose a Gymnasium-style single-env wrapper with:
   `reset(seed)` and `step(action) -> obs, reward, terminated, truncated, info`.
4. Use SONIC raw action as the first action contract:
   `q_des = default_angle + raw_action * g1_action_scale`.
5. Apply official-style LowCmd torque realization:
   `tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)`, then clip by motor
   effort limit before writing MuJoCo control.
6. Build a deterministic replay CLI that feeds official `action.csv` into this
   env and compares q, dq, base pitch, root height, and torque against official
   deploy logs.

Phase B: behavior cloning smoke on the verified env.

1. Use official deploy logs as the initial dataset, starting with `action.csv`
   as labels and logged robot state as observations.
2. Train a tiny BC policy only after Phase A replay metrics are close enough to
   trust the direct env.
3. Run the BC policy in the direct env for a short smoke and record rollout
   metrics.
4. Defer PPO/RL fine-tuning until BC replay confirms the observation/action
   pipeline and plant wrapper are coherent.

## Diagnose Loop

Feedback loop for Phase A:

- replay fixed official deploy `action.csv` through direct MuJoCo;
- compare direct env response to official deploy logs;
- fail if base pitch/root height/q residual is closer to the unstable mjlab
  replay than to official deploy.

Ranked hypotheses:

1. If the direct env correctly mirrors official plant and LowCmd torque
   contract, replaying official `action.csv` will produce base pitch, root
   height, q residual, and torque ranges close to official deploy logs.
2. If asset path/passive fields are wrong, the replay will diverge even with the
   correct raw action transform and torque formula.
3. If joint order/action scale/default angle mapping is wrong, q residuals will
   spike immediately and top-error joints will align with mapping-sensitive
   joints rather than contact events.
4. If official deploy logs depend on planner/internal state beyond raw action,
   action replay may match q targets but not measured q/base response; then the
   next loop must replay LowCmd q/kp/kd/tau directly instead of raw action.

## Grill Decisions

Recommended default decisions unless contradicted by evidence:

1. First action contract: raw SONIC action, not q target.
   Rationale: BC and RL should learn the same 29D command interface that SONIC
   deploy emits.
2. First replay fixture: official slow-walk `target_vel=0.5` deploy logs from
   task025.
   Rationale: this is the matched case already compared against mjlab.
3. First success threshold: qualitative closeness plus explicit numeric deltas,
   not an over-tight pass/fail gate.
   Rationale: official deploy and direct replay may differ in reset sequencing
   and logging alignment; the first loop should expose the deltas before setting
   hard tolerances.
4. First training method: BC smoke before PPO/RL.
   Rationale: BC isolates observation/action/data plumbing before adding reward
   design and exploration.
5. First RL vectorization target: not in scope.
   Rationale: correctness of the plant contract is the bottleneck; throughput
   work is premature until replay is trusted.

## Acceptance

- Direct official MuJoCo env can reset and step finite state locally when
  dependencies are available, and on H200 for the real official asset path.
- Replay CLI can consume official deploy `action.csv` and emit a JSON summary.
- H200 replay compares direct env against official deploy logs for:
  - base pitch p95/max;
  - root z final/min/mean;
  - q residual RMS and top joints;
  - torque range/top joints.
- Phase A review states whether the direct env is close enough to official
  deploy to start BC.
- Phase B only starts after Phase A passes or explicitly documents why action
  replay cannot be a valid feedback loop.

## Log

- 2026-05-18 Opened after task025 concluded that mjlab target/action semantics
  are not the primary blocker; plant/torque realization is. User selected route
  `1` then `3`: first build official SONIC direct MuJoCo env, then behavior
  cloning smoke.

## Review

Status: not started.
