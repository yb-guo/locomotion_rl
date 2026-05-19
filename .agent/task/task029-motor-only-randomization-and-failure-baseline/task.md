# Task 029: Motor-Only Randomization And Failure Baseline

## Route

Build the next benchmark after task028's fixed-topology G1-like whole-body
gripper environment: a motor-only randomization and persistent motor-failure
baseline that still uses the known-good MJLab + RSL-RL PPO/MLP stack first.

The first acceptance target is deliberately narrow:

- Fixed G1-like topology.
- Fixed link geometry, mass, COM, and inertia.
- Fixed reward, environment, action, and actor observation contract from
  task028.
- Actor does not observe `motor_scale`, `failure_mask`, or explicit fault
  labels.
- Critic may use privileged motor randomization/failure information.
- First policy is the existing MLP PPO baseline.
- First failure mode is episode-start persistent weak/dead leg motors.

Deferred from first acceptance:

- LocoFormer or other long-context policy replacement.
- Sudden mid-episode motor failure.
- Arm, waist, gripper, or upper-body motor failure.
- Locked-joint training.
- Stuck-command training.
- Link mass/COM/inertia randomization.
- Contact friction randomization.
- Encoder noise or observation corruption as a training randomization.

Planned slices:

1. `001-motor-only-contract.md`
   - Define allowed and forbidden randomization fields.
   - Confirm the inherited task028 action/actor-observation contract remains
     unchanged.
   - Define actor/critic information boundaries for motor fault data.

2. `002-motor-primitive-ranges.md`
   - Add motor-side primitive stages one group at a time:
     `kp/kd`, effort/strength scale, damping/friction,
     torque noise/bias, and deadband.
   - Require inspect evidence and short PPO smoke for each primitive.

3. `003-delay-bandwidth-step-response.md`
   - Add action/actuator delay and low-pass/bandwidth behavior.
   - Validate with a step-response harness before PPO training.

4. `004-motor-failure-stage.md`
   - Implement episode-start persistent leg motor failure.
   - Randomly choose `0-2` leg motors per episode.
   - Use weak scale `0.3-0.7` and dead scale `0.0-0.1`.

5. `005-mlp-baseline-train.md`
   - Train the existing MLP PPO baseline on the motor-only/failure stage.
   - Prove the baseline still walks under the first acceptance setting.

6. `006-ood-and-failure-grid-eval.md`
   - Run clean eval, motor-only randomized eval, doubled motor holdout, and
     a per-joint dead-motor grid.
   - Keep locked-joint and stuck-command cases as eval holdouts only.

7. `007-render-and-review.md`
   - Render the accepted checkpoint.
   - Review gait quality and detect reward hacking such as excessive shaking,
     dragging, or upper-body flailing.

## Minimal Closed Loop

Feedback loop:

1. Inspect the motor-only contract and prove forbidden link/contact/sensor
   randomization is disabled.
2. Add motor primitive stages one at a time and run short H200 PPO smokes.
3. Validate delay/bandwidth behavior with a deterministic step-response
   harness before using it in PPO.
4. Prove the persistent motor-failure sampler with reset statistics and forced
   single-motor traces.
5. Train the existing MLP PPO baseline on the first acceptance distribution.
6. Evaluate the same checkpoint on clean, in-distribution motor-randomized,
   doubled motor holdout, and dead-motor grid scenarios.
7. Render accepted and diagnostic cases from the same checkpoint used in eval.

Pass:

- The actor observation and 31-dim action contract remain compatible with
  task028.
- Training randomization is motor-only under the first acceptance setting.
- Episode-start persistent weak/dead leg motor eval has saved JSON evidence.
- Clean eval confirms the baseline walking behavior was not destroyed.
- Render evidence does not show obvious reward hacking in accepted cases.

Fail:

- Actor receives explicit motor failure labels or motor scales.
- Link mass/COM/inertia, contact friction, sensor corruption, or pushes are
  enabled in the first-pass training stage.
- Delay/bandwidth is accepted without a step-response timing trace.
- Training reward is used as the only evidence without closed-loop eval.
- The task is marked passed before H200 eval and render evidence exist.

Evidence:

- Planned root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/`.

## Log

- 2026-05-19 Opened after task028 passed fixed-topology G1-like whole-body
  gripper training/eval/render with the existing MLP PPO stack.
- 2026-05-19 User decision: task029 first acceptance is not sudden online
  fault adaptation. It is episode-start persistent weak/dead leg motor
  robustness with the existing MLP baseline.
- 2026-05-19 User decision: actor must not receive explicit failure labels or
  motor scales. Critic may receive privileged motor information for value
  learning and diagnostics.

## Review

Status: planned.

This task is designed as a diagnostic bridge between task028's environment
learnability proof and any later LocoFormer-style long-context policy work. If
the MLP baseline cannot handle persistent episode-start leg motor failures, the
next action is to diagnose motor randomization ranges, actuator wrappers,
reward stability, or failure sampling. It is not yet evidence that a larger
policy is required.

The first pass is intentionally motor-only. Any training result that mixes in
link mass/COM/inertia, contact friction, sensor corruption, or explicit actor
fault labels does not satisfy this task's core acceptance criteria.
