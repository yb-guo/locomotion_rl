# Route

Task: task006-sonic-genesis-action-policy

Goal: Connect the real SONIC policy forward path to the Genesis 29-motor G1
backend after action replay passes.

Dependency:

- `001-genesis-action-replay.md` must pass first.

Pass condition:

- SONIC policy input/output contract is documented.
- One policy forward pass runs and returns a 29D action compatible with the
  Genesis action contract.
- A short H200 Genesis rollout runs from policy actions without non-finite state.
- Base height, action range, qvel, and termination/failure state are recorded.
- Produce a short GIF/contact sheet if the rollout smoke passes.

Fail condition:

- Missing or unusable policy artifact.
- Unclear observation/history/command contract.
- Policy output is not 29D or cannot be mapped to the G1 29-motor contract.
- Genesis rollout becomes non-finite or falls outside smoke range.

Implementation plan:

1. Locate SONIC deploy policy entrypoint and policy I/O definitions.
2. Run policy forward outside Genesis with a controlled dummy/recorded input.
3. Map the 29D policy output into `GenesisG1Env.step(action)`.
4. Run short H200 rollout.
5. Render GIF/contact sheet only after numeric smoke passes.

# Log

- 2026-05-07: Opened route as L2. Blocked until action replay L1 passes.
- 2026-05-07: L1 action replay passed. This route is now unblocked, but no
  SONIC policy inspection or rollout has been run yet.
- 2026-05-07: Located SONIC policy artifacts on the H200 target:

```text
Policy decoder ONNX: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
Policy encoder ONNX: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
Policy decoder TensorRT: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/policy_model_decoder.trt
Planner ONNX: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
Deploy binary: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```

- ONNX graph inspection on H200:

```text
model_encoder.onnx
  INPUT obs_dict [1, 1762] float32
  OUTPUT encoded_tokens [1, 64] float32

model_decoder.onnx
  INPUT obs_dict [1, 994] float32
  OUTPUT action [1, 29] float32

planner_sonic.onnx
  INPUT context_mujoco_qpos [1, 4, 36] float32
  INPUT target_vel [1] float32
  INPUT mode [1] int64
  INPUT movement_direction [1, 3] float32
  INPUT facing_direction [1, 3] float32
  INPUT random_seed [1] int64
  INPUT has_specific_target [1, 1] int64
  INPUT specific_target_positions [1, 4, 3] float32
  INPUT specific_target_headings [1, 4] float32
  INPUT allowed_pred_num_tokens [1, 11] int64
  INPUT height [1] float32
  OUTPUT mujoco_qpos [1, 64, 36] float32
  OUTPUT num_pred_frames [1] int64
```

- H200 runtime check:
  - Python `onnx` is installed.
  - Python `onnxruntime`, `tensorrt`, `cuda`, and `pycuda` are not installed.
  - `onnx.reference.ReferenceEvaluator` can run `model_decoder.onnx`, so no new
    runtime install was needed for this smoke.
  - A broad `find / -name trtexec` probe was started during runtime inspection,
    then killed after it was found to be too broad. Do not repeat broad
    filesystem searches on H200.

- Added:

```text
python -m h200_locomotion_lab.tools.sonic_policy_decoder_forward
```

This tool loads `model_decoder.onnx`, runs `obs_dict -> action` with
`onnx.reference.ReferenceEvaluator`, validates a 29D finite action, and can
write repeated action rows to CSV for the Genesis action replay harness.

- Local verification:

```text
Command: PYTHONPATH=src python -m pytest -p no:cacheprovider
Result: 23 passed
```

- H200 decoder tool verification:

```text
Command: PYTHONPATH=src python3 -m pytest -p no:cacheprovider tests/test_sonic_policy_decoder_forward.py
Result: 5 passed
```

- H200 SONIC decoder forward with zero `obs_dict` passed and wrote a 20-row
  action CSV:

```text
Log: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_decoder_zero_obs_forward.log
SONIC_POLICY_DECODER_FORWARD_MODE onnx_reference
DECODER .../gear_sonic_deploy/policy/release/model_decoder.onnx
OBS_SOURCE zero
OBS_DIM 994
OBS_FINITE True
ACTION_DIM 29
ACTION_FINITE True
ACTION_MIN_MAX -0.8099504113197327 0.7508559226989746
ACTION_MAX_ABS 0.8099504113197327
ACTION_FIRST10 (0.017019454389810562, -0.03419997915625572, -0.05208764970302582, 0.2511984705924988, -0.2821974754333496, -0.030565455555915833, 0.0014190655201673508, -0.020736368373036385, -0.6947729587554932, -0.06850829720497131)
OUTPUT_ACTIONS_CSV /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/sonic_decoder_zero_obs_action_20f.csv
OUTPUT_ACTION_ROWS 20
SONIC_POLICY_DECODER_FORWARD_OK
```

- H200 Genesis smoke using those decoder-produced actions passed:

```text
Log: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_decoder_zero_obs_action_20f.log
GENESIS_ACTION_REPLAY_MODE normalized_actions
ACTIONS_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/sonic_decoder_zero_obs_action_20f.csv
REPLAY_FRAMES 20
ACTION_DIM 29
ACTION_MIN_MAX -0.809950411 0.750855923
ACTION_MAX_ABS 0.809950411
ACTION_OUT_OF_RANGE_VALUES 0
BASE_POS (0.002389, 0.011728, 0.791166)
BASE_QUAT (0.711231, -0.00883, -0.004562, -0.702888)
DEFAULT_JOINT_POS_SOURCE .../walking_quip_360_R_002__A428/joint_pos.csv
DEFAULT_JOINT_POS_ROW 0
RESET_OBS_LEN 96
FRAME 0 base_z 0.7911660075187683 min_link_z 0.7911660075187683 action_min -0.809950411 action_max 0.750855923 max_abs_qvel 3.8745498657226562 obs_len 96
FRAME 19 base_z 0.7911660075187683 min_link_z 0.7911660075187683 action_min -0.809950411 action_max 0.750855923 max_abs_qvel 5.135611057281494 obs_len 96
FINITE_OK True
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
MIN_LINK_HEIGHT_MIN 0.7911660075187683
MIN_LINK_HEIGHT_FINAL 0.7911660075187683
MAX_ABS_QVEL 8.146235466003418
POLICY_STEPS 20
SIM_STEPS 80
GENESIS_ACTION_REPLAY_SMOKE_OK
```

- H200 GIF/contact-sheet verification for decoder-produced actions passed:

```text
Log: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_decoder_zero_obs_action_20f_gif.log
Remote GIF: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_decoder_zero_obs_action_20f.gif
Remote contact sheet: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_decoder_zero_obs_action_20f_contact.png
Local GIF: .agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_decoder_zero_obs_action_20f.gif
Local contact sheet: .agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_decoder_zero_obs_action_20f_contact.png
GENESIS_ACTION_REPLAY_GIF_MODE normalized_actions
FRAMES 20
ACTION_MAX_ABS 0.809950411
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
RENDERED_FRAMES 20
GIF_BYTES 47614
GENESIS_ACTION_REPLAY_GIF_OK
```

Remote `imageio` read-back:

```text
READ_FRAMES 20
DIFF_MIN 0.05914434523809524
DIFF_MAX 0.09008184523809523
DIFF_AVG 0.0731657268170426
CONTACT shape (320, 2100, 3)
```

# Review

Status: partial pass.

The SONIC decoder artifact is runnable on H200 without installing a new runtime:
`model_decoder.onnx` accepts `obs_dict[1,994]` and produces a finite 29D action.
Those decoder-produced actions can drive the validated Genesis G1 action replay
path without non-finite state or height failure.

This is not yet a full closed-loop SONIC policy rollout. The missing piece is
the real `obs_dict` construction/history/reference-motion contract for the
decoder input. The next step is to map Genesis state plus SONIC reference
context into the 994D decoder observation, or to reuse the official C++ deploy
input logging path to capture valid `obs_dict` rows.
