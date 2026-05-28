# 001 Unified Speed Contract

## Route

Define the Task031 contract before touching H200 MJLab training code.

Scope:

- Use the accepted Task030 `model_5349.pt` checkpoint as warm start.
- Keep the actor/action contract unchanged: `104 -> 31`.
- Keep the MLP PPO architecture.
- Train one policy across forward `lin_vel_x = 0.4..2.0 m/s`.
- Evaluate at fixed bins: `0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`.
- Keep `vy` and yaw command expansion out of scope.

Information boundary:

- Actor may observe the existing command and proprioceptive observation terms.
- Actor may observe previous action as already defined by the current env.
- Actor must not observe explicit failure labels, motor scales, failure masks,
  active failure joint ids, current case ids, or speed-bin ids.
- Critic, JSON, traces, and debug artifacts may record hidden scheduling state.

Layer definitions:

- Level A: unified speed clean and persistent motor-failure robustness.
- Level B: unified speed plus specified Task030 dynamic switch.
- Level C: arbitrary per-joint mid-episode dead onset diagnostic only.

## Log

- 2026-05-21 Contract drafted with user-approved scope: A+B pass, C diagnostic.
- 2026-05-21 Added local patch artifact
  `artifacts/task031_create_unified_speed_stage.py` for H200 MJLab env/registration
  updates. The artifact is syntax-parse checked locally but not yet applied on H200.
- 2026-05-28 H200 contract inspect evidence exists for the staged Task031
  unified dynamic-switch env at `0.4` and `2.0 m/s`: actor obs/action remains
  `104 -> 31`, critic obs is `119`, `forbidden_actor_terms=[]`, and events are
  `dynamic_motor_failure`, `reset_base`, and `reset_robot_joints`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/unified_speed_contract/task031_unified_speed_contract_summary.json`.

## Review

Status: passed for the Task031 contract gate. The H200 staged env preserves the
actor/action contract and does not expose explicit fault labels or active fault
ids to actor observations.
