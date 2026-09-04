# 003: Project Adapter Registration

## Route

Make the current repository's project-specific policy consumer available in a
fresh official Unitree MJLab checkout using the smallest repeatable patch path.

Required consumer boundary:

- Task044 hidden dynamic-fault training entry point;
- true-TXL/history runner integration;
- default-off Task046 retry context;
- no actor-visible fault identity leakage.

Do not reconstruct every historical experiment unless it is a dependency of
that minimal consumer. Record every upstream file changed by a patcher.

## Log

- 2026-08-18 Audited the tracked migration chain against the fresh official
  checkout. Task028 has a complete gripper generator. Task032 and later have
  incremental patchers, but the Task029 motor-failure implementation, Task030
  dynamic scheduler/config, and Task031 unified-speed/dead-grid environment
  were created cumulatively in the old external H200 checkout and are not
  represented by a complete tracked patcher.
- 2026-08-18 Parameterized the tracked Task028 generator with `--root`, then
  rebuilt the isolated `unitree_g1_gripper` asset and `g1_gripper` task package
  in the new checkout. Eight Gripper task IDs register successfully.
- 2026-08-18 The rebuilt `Unitree-G1-Gripper-Flat` passed a real RTX GPU
  inspect: MuJoCo contract `nq=38`, `nv=37`, `nu=31`, `njnt=32`; action terms
  `body_joint_pos=29` plus `gripper_joint_pos=2`; actor observation `104`; critic
  observation `119`; ten zero-action steps were finite with no done event.
  Local JSON evidence is in
  `.agent/tmp/task047/g1_gripper_flat_smoke.json`.
- 2026-08-18 Added `scripts/check_task044_migration.sh`. After Task028 restore,
  it exits `2` and names the exact remaining source boundary: Task029
  `_add_motor_failure_stage`, Task030
  `unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg`, Task031
  `unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg`, and the Task044
  runner/task registrations.
- 2026-08-18 Searched `/home/admin1/workspace/{proj,repo,run,store}` for another
  `g1_gripper/env_cfgs.py`, the missing environment anchors, `model_5349.pt`,
  and other historical `model_*.pt` files. No matching old source or checkpoint
  exists on this workstation.

## Review

Status: blocked for Task044/Task046 registration; passed for the Task028
31-action base.

Do not apply Task032-Task044 patchers to this checkout yet: they assume the
missing Task029-031 anchors and would produce an unverified hybrid environment.
The next valid input is either the matching old H200 `g1_gripper` source tree
and checkpoint, or a dedicated reconstruction task for Task029-031 followed by
their original contract/eval gates.
