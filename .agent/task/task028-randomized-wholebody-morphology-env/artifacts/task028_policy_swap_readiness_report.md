# Task028 Policy Swap Readiness Report

Date: 2026-05-19

## Verdict

Ready to open the next policy-only experiment on top of the fixed-topology
`Unitree-G1-Gripper-Flat-*` environment family.

This does not mean the environment is ready for variable topology,
full dexterous hands, deployment export, or object manipulation. It means the
current fixed-topology environment/reward/randomization loop is stable enough
that a LocoFormer-style policy task can be framed as a policy change rather
than an environment debugging task.

## Evidence

- Fixed action contract:
  `body_joint_pos` 29 dims followed by `gripper_joint_pos` 2 dims, total 31.
- Fixed observation contract in the flat task:
  actor obs 104, critic obs 119.
- Asset smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/asset_contract/g1_gripper_flat_smoke.json`
- MLP PPO smoke checkpoint-load eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/ppo_smoke/g1_gripper_env64_iter2_model1_short_eval.json`
- Randomization stage inspect artifacts:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/inspect/`
- Passing checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_eval_render_005/2026-05-19_16-31-34_gpu1_combined_env8192_seed53_full/model_600.pt`
- Deterministic eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_eval/model_600_forward_0p5_clean_eval.json`
- Randomized holdout eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_eval/model_600_forward_0p5_randomized_holdout_eval.json`
- Render evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_render/task028-g1-gripper-combined-model600-vx0p5.mp4`

## Stable Interfaces For The Next Policy Task

- Task family:
  `Unitree-G1-Gripper-Flat-Control`,
  `Unitree-G1-Gripper-Flat-Contact`,
  `Unitree-G1-Gripper-Flat-EncoderNoise`,
  `Unitree-G1-Gripper-Flat-MassComInertia`,
  `Unitree-G1-Gripper-Flat-MotorPd`,
  `Unitree-G1-Gripper-Flat-Combined`.
- First policy target:
  `Unitree-G1-Gripper-Flat-Combined`.
- Input:
  the existing flat actor observation vector of length 104.
- Output:
  31 raw actions in the same order as the current MLP:
  first the original 29 G1 body joint position actions, then
  left and right gripper actions.
- Baseline comparison:
  MLP checkpoint `model_600.pt` on deterministic and randomized holdout eval.

## Recommended Next Policy Task

Start with a fixed-topology policy swap, not variable topology:

1. Keep the env/task IDs unchanged.
2. Keep action order and action dimension unchanged.
3. Replace only the actor/critic model class or observation encoder.
4. Use the same train/eval/render harness and compare directly against
   `model_600.pt`.
5. Do not introduce morphology padding/masks until the fixed-topology policy
   beats or matches the MLP baseline.

For a LocoFormer-style first pass, use fixed tokens:

- one base token
- one command token
- one leg token per side
- one waist token
- one arm token per side
- one gripper token per side

The first pass can still decode into the existing flat 31-action vector. Token
padding and variable robot graphs should be a later task.

## Known Gaps

- ONNX/export metadata for multi-term actions is not fixed. The gripper task
  uses `MjlabOnPolicyRunner` because the velocity runner assumes a single
  action term named `joint_pos`.
- Delay/smoothing randomization is deferred. Actuator delay requires wrapping
  actuators with `DelayedActuatorCfg`; no generic action smoothing API was
  found for current position actuators.
- Gripper contact is intentionally disabled/conservative. This is a locomotion
  benchmark with gripper dynamics/actions, not a manipulation task.
- Variable topology, variable DoF, padding/masks, full dexterous hands, and
  generic morphology generation remain out of scope for this task.
- The Control checkpoint was not evaluated because the Combined checkpoint met
  the stronger eval/render gate first.
