# 002: G1-Like Wholebody Gripper Asset Prototype

## Route

Create the first fixed-topology task028 robot asset: a G1-like whole-body robot
with simplified grippers.

First-pass asset goals:

1. Stay close enough to the verified Unitree MJLab G1 baseline that the
   locomotion reward, observation terms, and contact sensors can be adapted
   instead of rewritten from scratch.
2. Add simplified grippers as fixed-topology end effectors.
3. Keep the grippers simple enough that locomotion remains the first feedback
   loop.
4. Keep the action and observation dimensions fixed so the existing MLP PPO
   stack can train it before any LocoFormer-style policy work.
5. Start from a copied/extended upstream Unitree MJLab G1 asset and register a
   new task instead of introducing a generic asset generator.

Recommended initial design:

- Start from the upstream G1 29-DoF style whole-body action contract.
- Register a new task name such as `Unitree-G1-Gripper-Flat`.
- Add two gripper commands total: left gripper open/close and right gripper
  open/close.
- Use mimic or mechanically coupled finger joints inside the MJCF when needed,
  but expose only one actuator per gripper to the policy.
- First locomotion reward should regularize grippers near a neutral/open pose.
- No object grasping or dexterous contact reward in this slice.
- No object contact requirement and no ground-contact requirement for grippers
  in the first pass.

User decision:

- The two gripper open/close commands are part of the first-pass policy action
  space, not passive joints.
- The first implementation should copy/extend the upstream G1 asset and task
  config. A procedural morphology generator is deferred.
- First-pass gripper collision should be conservative or disabled. The gripper
  should affect inertial dynamics and action/observation shape without turning
  walking into a contact-rich manipulation problem.

## Log

- 2026-05-19 Opened after the user selected G1-like whole body plus grippers
  for the first fixed-topology morphology.
- 2026-05-19 Inspected upstream Unitree MJLab G1 asset/config on H200:
  - Robot constants:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py`
  - MJCF:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml`
  - Task registration:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/tasks/velocity/config/g1/__init__.py`
- 2026-05-19 Compiled upstream G1 asset and confirmed current contract:
  `nq=36`, `nv=35`, `nu=29`, `njnt=30`. Non-free joint order is:
  left leg 6, right leg 6, waist 3, left arm 7, right arm 7. The last arm
  joints are `left_wrist_yaw_joint` and `right_wrist_yaw_joint`.
- 2026-05-19 Inspected wrist/hand MJCF structure. Each side already has:
  `*_wrist_yaw_link`, a visual `*_rubber_hand` mesh, `*_palm` site, and
  `*_hand_collision` capsule. The clean gripper attachment point is under
  `left_wrist_yaw_link` and `right_wrist_yaw_link`.
- 2026-05-19 Inspected `JointPositionActionCfg` implementation. Action targets
  are selected from actuated joints and ordered by the entity's natural joint
  order. Therefore if the gripper joints are actuated and inserted after each
  wrist yaw joint in MJCF, `actuator_names=(".*",)` will include them and the
  action dimension should become 31.
- 2026-05-19 Inspected `ActionManager`. Multiple action terms are supported
  and are concatenated in config dict order. This gives a cleaner first-pass
  action contract: keep the existing 29-DoF body action term first, then add a
  2-DoF gripper action term second, so policy action order is `[body29,
  gripper2]` even though the physical gripper joints live under the wrist
  bodies.

## Review

Status: planned.

The main risk is adding too much manipulation complexity before walking is
stable. This slice should produce an asset that can stand, walk, and survive
closed-loop eval with grippers included in the dynamics.

Implementation recommendation:

- Copy `unitree_g1` to a new asset/config namespace rather than modifying the
  upstream baseline in place.
- Add `left_gripper_joint` under `left_wrist_yaw_link` and
  `right_gripper_joint` under `right_wrist_yaw_link`.
- Keep one exposed actuated joint per gripper in the first pass.
- Do not require gripper-ground or gripper-object contacts to be active for the
  first smoke/eval loop.
- Target first-pass compiled contract: `nu=31` and fixed flat MLP action shape.
- Prefer two action terms for a stable policy contract:
  first `body_joint_pos` controls the original 29 G1 joints, second
  `gripper_joint_pos` controls `left_gripper_joint` and `right_gripper_joint`.
  This makes the raw policy action order `[original_g1_29, gripper_left,
  gripper_right]`.
- Register `Unitree-G1-Gripper-Flat` by copying the existing G1 velocity task
  registration and replacing `get_g1_robot_cfg()` / `G1_ACTION_SCALE` with the
  gripper variant.

## Minimal Closed Loop

Feedback loop:

1. Import the new task with `PYTHONPATH=.` from the upstream Unitree MJLab repo.
2. Compile the new robot asset.
3. Instantiate `Unitree-G1-Gripper-Flat` with 1 env on GPU.
4. Reset once and step 10 times with zero actions.
5. Print/write an inspect JSON.

Pass:

- New task appears in `scripts/list_envs.py`.
- Compiled robot has expected fixed contract: original G1 plus two actuated
  gripper joints.
- Env action manager reports two terms:
  `body_joint_pos` with 29 dims and `gripper_joint_pos` with 2 dims.
- Total action dim is 31.
- Actor observation shape is finite and stable across reset/step.
- Ten zero-action steps produce no NaN, no immediate crash, and no unexpected
  gripper/object/ground contact requirement.

Fail:

- `Unitree-G1-Flat` baseline is modified or broken.
- New task cannot import without local hacks.
- Action order is not `[body29, gripper2]`.
- Gripper collision creates startup self-contact failure.

Evidence:

- H200 inspect JSON under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/asset_contract/`.
