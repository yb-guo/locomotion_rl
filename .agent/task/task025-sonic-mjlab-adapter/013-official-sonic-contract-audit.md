# 013: Official SONIC Contract Audit

## Route

Audit official `NVlabs/GR00T-WholeBodyControl` sources before turning the
trace-only clamp into a controller behavior. Focus areas:

- G1 joint limits, especially ankle pitch;
- action scaling and target construction;
- deploy-side action or target clipping;
- `last_action` / action-history semantics;
- whether clipped/effective commands are written back into history.

## Log

- 2026-05-17 Checked official repository entry points:

  - `gear_sonic_deploy`: official C++ inference/deployment stack.
    Source: <https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/gear_sonic_deploy>
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref`: G1 deploy reference.
    Source: <https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref>
  - `g1_29dof.xml`: deploy-side MuJoCo G1 asset.
    Source: <https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/g1/g1_29dof.xml>
  - `policy_parameters.hpp`: deployed action scale/default-angle constants.
    Source: <https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp>
  - `g1_deploy_onnx_ref.cpp`: control loop and motor command construction.
    Source: <https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp>

- 2026-05-17 Official deploy G1 ankle-pitch hard range is wider than mjlab's
  soft range but still much lower than the observed SONIC raw target:

  ```text
  official gear_sonic_deploy/g1/g1_29dof.xml:
    left_ankle_pitch_joint  range [-0.87267, 0.5236]
    right_ankle_pitch_joint range [-0.87267, 0.5236]

  mjlab observed soft range:
    left/right ankle pitch roughly [-0.8029, 0.4538]

  current SONIC raw target:
    left_ankle_pitch max 1.5653
  ```

  The mismatch is not only a mjlab soft-limit margin. The official hard high
  limit is still about `1.04 rad` below the observed raw target peak.

- 2026-05-17 Official deploy action contract matches this adapter's bridge:

  ```text
  target = action * action_scale + default_angle
  action_scale = 0.25 * effort_limit / stiffness
  ```

  `policy_parameters.hpp` defines the same mapping arrays, default angles, and
  action scales used by the local `ScalarActionBridge`.

- 2026-05-17 The official C++ deploy loop was searched for software clipping
  at the policy-output / q_target layer (`clip`, `clamp`, joint-limit clamp).
  I did not find deploy-side target clipping in the G1 deploy reference path.
  The visible command construction in `CreatePolicyCommand()` directly maps
  policy action to `q_target` with action scale and default angle.

- 2026-05-17 Official action history appears to store raw policy output, not
  clipped/effective command. In `CreatePolicyCommand()`, after computing each
  motor target from `action_value`, the code assigns `last_action[i] =
  action_value`. `StateLogger` then records that `last_action`, and the policy
  observation path gathers historical last actions from the logger.

## Review

Official SONIC does not provide evidence for a deploy-side clamp with effective
action-history writeback. The observed adapter behavior is therefore consistent
with the official contract in one important sense: history uses raw policy
action.

That does not make the current raw targets physically valid. Official G1's
ankle-pitch hard high is `0.5236 rad`, mjlab soft high is about `0.4538 rad`,
and the current raw SONIC target reaches about `1.5653 rad`. If this policy was
expected to stay within official deploy limits, then the current closed loop is
driving the decoder out of the intended target range. If official hardware or
SDK clamps silently below this layer, that behavior is not explicit in the
audited G1 deploy reference and would still need an action-history decision.

For this adapter, the conservative conclusion is:

- keep production `MjlabG1RobotBackend` unclipped until the contract is proven;
- keep target clamping as a trace-only diagnostic;
- test effective action history as a diagnostic, not as a formal controller
  change.

