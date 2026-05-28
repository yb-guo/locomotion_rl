# 024: Lower-Body Plant Contact Actuator Audit

## Route

Diagnose why mjlab loses base posture under official SONIC motion-conditioned
decoder targets while official MuJoCo deploy stays comparatively upright.

Ranked hypotheses:

1. If startup randomization is the cause, disabling mjlab startup
   randomization for the fixed official-motion decoder replay will materially
   reduce done events, base pitch, contact impulses, or root-height collapse.
2. If actuator profile mismatch is the cause, official and mjlab hip/knee/ankle
   stiffness, damping, effort, armature, or control gains will differ on joints
   that dominate the pitch/contact failure, and matching those profiles should
   improve the replay without changing SONIC I/O.
3. If contact geometry/friction is the cause, official and mjlab foot/floor
   geoms, friction, condim, contact filters, timestep, or solver options will
   differ enough to explain the large mjlab contact impulses under the same
   lower-body commands.
4. If reset/base-state mismatch is the cause, mjlab replay will start from a
   different base pose, qpos, or qvel distribution than the official deploy
   logs and fail before control dynamics can be compared fairly.

## Log

- 2026-05-18 Opened after `023` showed fixed official-motion SONIC decoder
  replay terminates in mjlab with much worse pitch/contact response than
  official deploy, despite comparable aggregate joint-target residual.
- 2026-05-18 Corrected the official runtime source for this comparison. The
  official Python MuJoCo sim config
  `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`
  points to `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml`,
  which includes `g1_29dof_with_hand.xml`. It does not use the deploy
  directory's separate `g1/scene_29dof.xml` for the Python sim run.
- 2026-05-18 Official runtime plant findings:
  - floor/default geom friction is `1.0`;
  - each foot has one box sole geom under the ankle roll link;
  - lower-body joints carry passive XML fields around `damping=0.05`,
    `armature=0.01`, and `frictionloss=0.2`;
  - the official sim applies an external PD torque from LowCmd
    `q_des/dq_des/kp/kd/tau` and clips the resulting torque to the configured
    motor effort limit.
- 2026-05-18 Official deploy controller findings: C++ deploy computes motor
  targets as `default_angle + raw_action * g1_action_scale`, writes raw policy
  action into action history, and showed no target clamp or effective-history
  writeback. The deploy `kp/kd/action_scale/default_angle` contract is separate
  from the passive joint fields in the runtime XML.
- 2026-05-18 mjlab G1 plant findings:
  - runtime G1 XML comes from `src/assets/robots/unitree_g1/xmls/g1.xml`;
  - each foot uses seven capsule collision geoms rather than the official
    runtime's one box sole;
  - the flat task sets foot friction to `0.6`;
  - `create_position_actuator()` builds MuJoCo position actuators with force
    ranges, actuator-derived joint armature, and default joint frictionloss
    `0.0` unless the actuator config provides another value;
  - the current trace-only SONIC hip-pitch override fixes the controller
    `kp/kd/effort` profile, but it does not make mjlab match official passive
    joint armature/frictionloss or foot contact geometry/friction.
- 2026-05-18 Ran the focused H200 ablation with startup randomization disabled
  for the fixed official-motion SONIC decoder replay:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_motion_sonic_decoder_replay_slowwalk_0p5_400_no_randomization/sonic_decoder_motion_replay_slowwalk_0p5_400_no_randomization.json`.
  It still had 13 done events in 400 steps, `abs_pitch_p95=0.9449`,
  `root_z_final=0.6595`, `root_z_min=0.1989`, and
  `joint_error_rms_mean=0.7406`. The randomized baseline had 14 done events,
  `abs_pitch_p95=1.0294`, `root_z_final=0.6301`, `root_z_min=0.2480`, and
  `joint_error_rms_mean=0.7160`. Startup randomization is therefore not the
  primary cause.
- 2026-05-18 Current ranking: the remaining failure is most likely a lower-body
  plant contract mismatch, especially foot contact geometry/friction plus joint
  passive armature/frictionloss and actuator realization. It is no longer best
  explained by SONIC planner inputs, target range, deploy-side clipping, or raw
  versus effective action history.

## Review

Status: audit passed; plant-matching fix remains open.

The direct question "where is the problem" now has a narrower answer: mjlab is
not executing the same lower-body plant contract as official SONIC MuJoCo. The
strongest evidence is that official fixed-motion decoder input still destabilizes
mjlab even after startup randomization is disabled, while official deploy remains
upright under the same slow-walk motion family.

The next diagnostic should be trace-only and reversible:

1. Add an official-plant overlay for mjlab replay that changes only lower-body
   plant details, not SONIC I/O.
2. Split it into independent ablations:
   - passive joint fields: official-like `armature`, `damping`, and
     `frictionloss`;
   - contact: official-like foot sole geometry/friction;
   - both together.
3. Re-run the fixed official-motion SONIC decoder replay and compare done count,
   base pitch, root height, contact force, and joint residual.

Do not turn target clamping into production behavior yet. Official deploy also
emits out-of-range raw targets and appears to rely on plant dynamics and limits,
not deploy-side effective-history clipping, to realize the motion.
