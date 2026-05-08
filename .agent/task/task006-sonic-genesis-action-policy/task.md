# Task 006: SONIC Genesis Action Replay Then Policy

## Goal

Move from SONIC reference joint-position replay to true action-driven Genesis G1
rollout.

The task order is fixed:

1. L1 action replay: drive the validated Genesis 29-motor G1 env with explicit
   29D normalized action sequences through `GenesisG1Env.step(action)`.
2. L2 SONIC policy: connect the real SONIC policy forward path only after L1
   passes with evidence.

Do not continue to L2 unless L1 passes on H200.

## Scope

- Genesis G1 29-motor action replay harness.
- Action CSV or deterministic action fixture loading.
- H200 smoke metrics:
  - finite state;
  - base height range;
  - min link height where available;
  - action range;
  - max qvel;
  - optional GIF/contact sheet.
- SONIC policy I/O inspection and rollout after action replay passes.

## Non-Goals

- No training loop yet.
- No PPO baseline yet.
- No Isaac Lab route.
- No new robot assets, checkpoints, datasets, or upstream repo downloads unless
  explicitly approved.

## Subtasks

- `001-genesis-action-replay.md`
- `002-sonic-policy-rollout.md`

## Current Known Inputs

- H200 run root:
  `/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline`
- Valid Genesis asset:
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml`
- Valid Genesis motor DOF order:
  `(6, 9, 12, 15, 19, 23, 7, 10, 13, 16, 20, 24, 8, 11, 14, 17, 21, 25, 27, 29, 31, 33, 18, 22, 26, 28, 30, 32, 34)`
- Existing reference replay evidence:
  `.agent/task/task004-genesis-g1-baseline/002-genesis-env-reset-step.md`

## Review

Status: L1 generic action replay passed; SONIC-compatible offline action bridge
has corrected smoke evidence. L2 decoder-only closed-loop 20-frame smoke now
passes after applying SONIC's official G1 motor kp/kv/force limits.

L1 pass evidence is recorded in `001-genesis-action-replay.md`. L2 may now
inspect and connect the SONIC policy path, but must not erase the distinction
between action replay and real policy rollout.

L2 route 1 evidence is recorded in `002-sonic-policy-rollout.md`: official C++
deploy emitted 1007 finite `obs_dict[994]` rows on H200, and the SONIC decoder
turned 50 real obs rows into 50 finite 29D actions. The earlier Genesis replay
pass was invalidated because the smoke read `robot.get_pos()`/spawn pose as base
height instead of the floating-base DOF state, and the Genesis MJCF entity pose
double-counted the pelvis height. Corrected dynamic-root instrumentation has a
20-frame partial smoke only; do not treat L2 as passed yet.

Documentation check found an additional action-bridge mismatch: official SONIC
deploy maps raw policy actions from IsaacLab order to MuJoCo order, then applies
per-joint `g1_action_scale` and `default_angles`. The current Genesis contract
still uses MuJoCo-order actions, a uniform `0.25` rad scale, clipping to
`[-1, 1]`, and often a reference-motion row as the nominal pose. Treat this as
valid only for synthetic Genesis action replay, not for replaying SONIC policy
outputs.

Correction implemented:

- `sonic_policy_raw` action mode maps raw decoder output using official
  `isaaclab_to_mujoco`, per-joint `g1_action_scale`, and `default_angles`.
- Root initialization can now use explicit `root_qpos` instead of abusing
  `MJCF(pos=...)`.
- Smoke logs now include Genesis contact count and max link contact force;
  `min_link_z` remains diagnostic only.

H200 corrected offline smoke:

- 20-frame numeric log:
  `/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge.log`
- 20-frame GIF:
  `/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge.gif`
- 5-frame contact-metric log:
  `/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_5f_sonic_bridge_contact.log`

L2 decoder-only closed-loop evidence:

- Implemented official 994D decoder observation layout:
  `token_state[64]`, 10-frame base angular velocity, 10-frame centered
  policy-order joint positions/velocities, 10-frame raw last actions, and
  10-frame gravity direction.
- Added Genesis online history recording and
  `genesis_sonic_policy_rollout_smoke`.
- H200 10-frame smoke with `--token-mode replay --history-init official_obs`
  passed: obs/action finite, base height final `0.6546086668968201`,
  height range `0.3..1.2`, action max abs `5.1907243728637695`.
- Initial H200 20-frame smoke with the same settings failed height: base height
  final `0.2644214928150177`, below the `0.3` threshold, action max abs
  `8.832411766052246`.
- Diagnose found the closed-loop tools had not applied SONIC's official G1
  motor kp/kv/force limits. With `MOTOR_CONFIG sonic_g1_kp_kv_force_range`,
  the same 20-frame smoke passes: base height final `0.7882418632507324`,
  base height min `0.7512305378913879`, action max abs `3.2313764095306396`,
  `GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_OK`.

Review: the original 20-frame closed-loop failure no longer reproduces. This is
still a decoder-only smoke with replayed token/history initialization; walking
quality is not passed.

Follow-up locomotion probes confirmed the 100/200-frame token replay was a
stable standing/settling behavior. A new official walking capture from
`walking_quip_360_R_002__A428` produced 922 finite 994D obs rows and a target
motion with about 10.13 m xy path length, but replaying those walking tokens in
Genesis still produced negligible displacement and no single-support gait.
Teacher-forced walking decoder actions fell within 50 frames, while direct
official walking reference joint-position replay moved about 0.14 m in 50
frames and stayed upright. The active blocker is now the SONIC policy-action
path under Genesis dynamics/contact, not the availability of a walking
reference clip.

Latest diagnose update: the official walking action replay failure was narrowed
to a root-pose initialization mismatch. The walking capture's `base_quat.csv`
starts near identity, but the failing Genesis replays used a quaternion from a
different capture. With official `action.csv` rows, measured q0 initialization,
and the walking capture base quaternion, the 50-frame H200 action replay now
passes height smoke with `BASE_HEIGHT_MIN 0.3062208294868469` and
`GENESIS_ACTION_REPLAY_SMOKE_OK`. L2 was reattempted under the same corrected
root pose and produced partial 100-frame locomotion evidence up to frame 80
(`disp_xy 0.243m`, single-support observed at frame 60), but SSH closed before
the final summary, so L2 remains unpassed until a complete run finishes.

L2 short walking pass: after adding heartbeat/progress controls, H200 completed
10/20/40-frame decoder-only SONIC walking probes with official walking obs,
corrected walking root quaternion, official q0 initialization, and SONIC motor
config. The 40-frame run reported `HORIZONTAL_DISPLACEMENT 0.1781279369041812`,
`SINGLE_SUPPORT_FRAMES 15`, `TOTAL_CONTACT_SWITCHES 6`,
`LOCOMOTION_OBSERVED True`, and `GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK`.
A 40-frame GIF was rendered at
`.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_walking_obs_40f_correct_root.gif`.
Treat this as a short L2 decoder-only walking pass; 80/100-frame scale-up and
full planner/encoder integration remain open.
