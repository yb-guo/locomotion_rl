# Task 028: Randomized Simplified Whole-Body Morphology Env

## Route

Build the next benchmark after the verified Unitree MJLab G1 baseline: a
simplified whole-body morphology environment with dynamics randomization, while
continuing to use the known-good MJLab + RSL-RL PPO/MLP training stack first.

The core decision for this task is to freeze an environment contract before
changing the policy. The first pass should answer whether the environment,
reward, randomization, and eval loop are learnable with the existing policy.
Only after that should we replace the MLP with a LocoFormer-style long-context
or morphology-token policy.

Planned slices:

1. `001-env-contract-and-randomization-scope.md`
   - Define the exact observation/action contract.
   - Keep a fixed action dimension for the first pass.
   - Decide which morphology and dynamics fields randomize at reset.
   - Define pass/fail eval metrics before implementation.

2. `002-g1-like-wholebody-gripper-asset-prototype.md`
   - Create or configure a G1-like whole-body robot asset with simplified
     grippers.
   - Include torso, legs, waist, arms, and policy-controlled grippers.
   - Avoid full dexterous manipulation in the first pass.
   - Keep contact geometry simple enough for large H200 vectorized training.

3. `003-existing-mlp-ppo-smoke.md`
   - Train with the current MJLab/RSL-RL PPO MLP stack.
   - Use small env counts first, then scale toward the task027 H200 pattern.
   - Do not change policy architecture in this slice.

4. `004-randomization-curriculum.md`
   - Add dynamics randomization gradually:
     mass, COM, inertia, friction, motor strength, PD gains, delay/noise.
   - Track which randomization breaks learning.
   - Keep a deterministic no-randomization eval as a control.

5. `005-closed-loop-eval-and-render.md`
   - Reuse the task027 style of JSON closed-loop eval.
   - Require fixed-command walking, randomized holdout eval, and render evidence.
   - Mark passed only with saved checkpoints and eval artifacts.

6. `006-policy-swap-readiness-review.md`
   - Only after the MLP baseline passes, decide whether the env contract is
     stable enough for a LocoFormer-style policy.
   - Document tokenization, padding/masks, history length, and action decoding
     requirements for the policy swap.

## Log

- 2026-05-19 Opened after task027 verified that upstream Unitree MJLab can
  train G1 velocity locomotion from scratch on H200 and pass closed-loop eval.
- 2026-05-19 Planning decision: task028 starts by changing the environment and
  randomization while preserving the existing MJLab/RSL-RL PPO MLP policy stack.
  A LocoFormer-style policy is deferred until the environment is learnable.
- 2026-05-19 User decision: first pass should not support variable topology or
  variable DoF. Keep topology and action dimension fixed, and randomize link,
  motor, contact, and sensor parameters inside that fixed contract.
- 2026-05-19 User decision: first fixed topology should be G1-like whole body
  plus simplified grippers, not a purely primitive toy morphology and not full
  dexterous hands.
- 2026-05-19 User decision: grippers are part of the first-pass action space.
  Expose one open/close action per gripper, two total gripper actions.
- 2026-05-19 User decision: first asset implementation should start by
  copying/extending the upstream Unitree MJLab G1 asset and registering a new
  task, not by building a generic robot asset generator.
- 2026-05-19 User decision: first-pass grippers do not need object contact or
  ground-contact capability. Keep gripper collisions conservative or disabled
  for locomotion-focused training.
- 2026-05-19 Diagnose audit: every task028 subtask must have a minimal closed
  loop with a concrete command/script, pass/fail criteria, and evidence path.
  Missing subtask docs for 003-006 were added.

## Review

Status: planned.

The main risk is confusing two experiments: environment learnability and policy
architecture. This task intentionally separates them. The first pass should
prove that a simplified whole-body randomized environment can be trained and
evaluated with the existing policy stack. If it cannot, the correct next step is
to diagnose the environment/reward/randomization loop, not to add a larger
policy.

First-pass scope is fixed-topology / fixed-DoF with a G1-like whole body plus
simplified grippers. Grippers are controlled by the policy with two total
open/close action dimensions. Variable topology, padding, masks, full
dexterous hands, action-token decoding, and a generic morphology asset
generator are explicitly deferred. Object manipulation and gripper contact
tasks are also deferred.
