# 001: Contract Audit

## Route

Inspect the live mjlab G1 action and robot data objects before implementing the
adapter. The adapter must map by joint name and keep SONIC-specific policy order
inside the existing action bridge.

## Log

- Remote probe:
  `.agent/tmp/task025_inspect_mjlab.py`
- Environment:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab`
- Task:
  `Unitree-G1-Flat`

Findings:

- `robot.joint_names` has 29 joints.
- `action_manager.get_term("joint_pos").target_names` has the same 29 names.
- The target order matches SONIC `command_mujoco` order.
- `robot.data.default_joint_pos` differs from SONIC default angles, so the
  first adapter version should explicitly compute mjlab actions from the mjlab
  action term offset/scale instead of assuming SONIC defaults equal mjlab
  defaults.

## Review

Pass for first implementation. The backend can reject mismatched names at
construction time.
