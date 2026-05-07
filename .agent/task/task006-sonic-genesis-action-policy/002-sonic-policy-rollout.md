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

- 2026-05-07: Started route 1: reuse the official C++ deploy path to capture
  valid policy `obs_dict` rows via `--policy-input-logfile`.

  Findings while building the H200 harness:

  - `scripts/setup_env.sh` must be sourced with `set +e`/`set +u`, matching
    `deploy.sh`; otherwise optional environment probes can abort the harness.
  - `TensorRT_ROOT` and `LD_LIBRARY_PATH` must explicitly prefer the task002
    TensorRT 10.13.3/CUDA 12.9 extraction:

```text
TensorRT_ROOT=/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr
LD_LIBRARY_PATH starts with:
/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr/lib/x86_64-linux-gnu
```

  - If the system TensorRT library is selected instead, the official deploy
    crashes before policy input capture:

```text
createInferRuntime: Error Code 6: API Usage Error (CUDA initialization failure with error: 35)
Segmentation fault
```

  - The official keyboard start key is right bracket `]`, not left bracket
    `[`. The previous harness label `sent_start_bracket` was ambiguous.
  - When using a FIFO for stdin, the write side must be opened and held before
    waiting for readiness. Otherwise the deploy process blocks while opening
    stdin and cannot emit the readiness log line.

  Successful official capture evidence:

```text
Command path:
/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref

Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/official_deploy_policy_input_capture.log

Policy input CSV:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_policy_input_capture.csv

Target motion CSV:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_target_motion_capture.csv

Rows: 1007
Dims: [994]
Finite: True
Min/max: -35.8944 / 42.5034
First 10 obs values:
(0.0, 0.125, 0.0, -0.125, 0.0625, -0.1875, 0.375, 0.1875, 0.125, -0.0625)
Deploy status: 0
```

  Official deploy log confirms CONTROL mode:

```text
[HARNESS] ready=1
[HARNESS] sent_start_right_bracket
Init Done
[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state
Loop timing ... Obs: ... Policy: ...
Playing motion 0 from frame 0 to end (497 total frames)
[InterfaceManager] EMERGENCY STOP triggered (O/o key pressed)
[DEBUG] Program exiting normally...
```

- Added batch decoding support to:

```text
python -m h200_locomotion_lab.tools.sonic_policy_decoder_forward
```

  It can now read multiple numeric obs rows from `--obs-csv`, optionally limit
  them with `--max-rows`, and write one 29D action row per obs row. The previous
  single-row `--repeat-rows` behavior is preserved.

  Local verification:

```text
Command: PYTHONPATH=src python -m pytest -p no:cacheprovider
Result: 26 passed
```

  Follow-up completed after SSH recovered:

```text
H200 test command:
PYTHONPATH=src python3 -m pytest -p no:cacheprovider tests/test_sonic_policy_decoder_forward.py
Result: 8 passed
```

- 2026-05-07: Real official obs rows decoded through `model_decoder.onnx` on
  H200 with the updated batch decoder path.

```text
Command:
PYTHONPATH=src python3 -m h200_locomotion_lab.tools.sonic_policy_decoder_forward \
  --decoder /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
  --obs-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_policy_input_capture.csv \
  --max-rows 50 \
  --output-actions-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_policy_input_decoder_actions_50f.csv

Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_decoder_official_obs_50f.log

OBS_ROWS 50
OBS_DIM 994
OBS_FINITE True
ACTION_ROWS 50
ACTION_DIM 29
ACTION_FINITE True
ACTION_MIN_MAX -3.584078311920166 5.846455097198486
ACTION_MAX_ABS 5.846455097198486
ACTION_FIRST10 (0.5058421492576599, 0.10938304662704468, -0.025784069672226906, -0.13401781022548676, 0.029130570590496063, 0.6880656480789185, -0.13201797008514404, -0.1565856784582138, -0.08554115146398544, 0.6643720269203186)
OUTPUT_ACTION_ROWS 50
SONIC_POLICY_DECODER_FORWARD_OK
```

  Note: the decoded action stream is not bounded to `[-1, 1]`; Genesis replay
  reports and clips out-of-range values through the existing action contract.

- 2026-05-07: H200 Genesis numeric smoke using the official-obs-decoded SONIC
  actions passed.

```text
Command:
PYTHONPATH=src python3 -m h200_locomotion_lab.tools.genesis_action_replay_smoke \
  --asset /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml \
  --actions-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_policy_input_decoder_actions_50f.csv \
  --frames 50 \
  --backend cuda \
  --logging-level warning \
  --base-pos 0.002389 0.011728 0.791166 \
  --base-quat 0.711231 -0.00883 -0.004562 -0.702888 \
  --default-joint-pos-csv /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428/joint_pos.csv \
  --default-joint-pos-row 0 \
  --min-base-height 0.3 \
  --max-base-height 1.2

Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_50f.log

REPLAY_FRAMES 50
ACTION_MIN_MAX -3.58407831 5.8464551
ACTION_MAX_ABS 5.8464551
ACTION_OUT_OF_RANGE_VALUES 465
FRAME 0 base_z 0.7911660075187683 min_link_z 0.7911660075187683 action_min -0.729726672 action_max 0.824930727 max_abs_qvel 2.802633285522461 obs_len 96
FRAME 49 base_z 0.7911660075187683 min_link_z 0.7911660075187683 action_min -1.0 action_max 1.0 max_abs_qvel 3.5462851524353027 obs_len 96
FINITE_OK True
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
MIN_LINK_HEIGHT_MIN 0.7911660075187683
MIN_LINK_HEIGHT_FINAL 0.7911660075187683
MAX_ABS_QVEL 3.6143109798431396
POLICY_STEPS 50
SIM_STEPS 200
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_ACTION_REPLAY_SMOKE_OK
```

  The local SSH wrapper timed out after the process had printed
  `GENESIS_ACTION_REPLAY_SMOKE_OK`; a follow-up process check found no
  `genesis_action_replay_smoke` or `run_sim_loop.py` process left running.

- 2026-05-07: H200 GIF rendering for the first 20 official-obs-decoded actions
  passed.

```text
Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_official_obs_decoder_actions_20f.gif

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_official_obs_decoder_actions_20f.gif

Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_20f_gif.log

FRAMES 20
ACTION_MIN_MAX -3.30871296 4.07460451
ACTION_MAX_ABS 4.07460451
ACTION_OUT_OF_RANGE_VALUES 156
FRAME 0 base_z 0.7911660075187683 rgb_shape (320, 420, 3)
FRAME 19 base_z 0.7911660075187683 rgb_shape (320, 420, 3)
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
RENDERED_FRAMES 20
GIF_BYTES 50474
GENESIS_ACTION_REPLAY_GIF_OK
```

  Local artifact check:

```text
Length: 50474 bytes
```

# Review Update

Status: route 1 previous pass invalidated; correction in progress.

The path `official deploy obs capture -> SONIC decoder -> Genesis G1 action
replay` has H200 evidence for the official obs capture and decoder action
export, but the Genesis replay height evidence above is invalid. The smoke used
`robot.get_pos()[2]` as base height; on this Genesis/MJCF path that value is the
entity spawn pose and stays constant. The dynamic floating-base state is in the
root DOFs before the first motor DOF.

Remaining limitation: this is still not a Genesis closed-loop SONIC policy. The
policy observations are captured from the official MuJoCo/deploy stack, decoded
offline, and replayed in Genesis. The next step for true closed-loop work is to
construct the 994D policy observation online from Genesis state, reference
tokens/history, and last-action buffers, then call the decoder each control
step.

- 2026-05-07: User challenged the visual/log evidence. Re-ran diagnosis and
  reproduced the core issue.

  Minimal H200 probe with no motor commands:

```text
N_DOFS 35
GET_POS_0 (0.002389000030234456, 0.011727999895811081, 0.7911660075187683)
DOFS_POS_0_FIRST10 (0.007086518686264753, 0.026773979887366295, 1.5840092897415161, ...)
STEP 200 GET_POS (0.002389000030234456, 0.011727999895811081, 0.7911660075187683)
STEP 200 ROOT_Q_FIRST6 (0.0435638502240181, 0.32732799649238586, 1.3173229694366455, ...)
```

  Interpretation:

  - `robot.get_pos()` is not the dynamic floating-base pose here.
  - The root qpos z changed from about `1.584` to `1.317`, proving the earlier
    constant `BASE_HEIGHT_* 0.791166` was a false pass.
  - `get_links_pos()` also includes a static/spawn-position z in the minimum,
    so `MIN_LINK_HEIGHT_*` is not a trustworthy foot/contact metric yet.

  The G1 MJCF itself starts the pelvis at z `0.793`:

```xml
<body name="pelvis" pos="0 0 0.793">
  <joint name="floating_base_joint" type="free" .../>
```

  Therefore passing `--base-pos ... 0.791166` to Genesis double-counted the
  pelvis height and initialized the floating root around z `1.584`. The corrected
  entity z for the reference pose is approximately:

```text
0.791166 - 0.793 = -0.001834
```

- Added dynamic-root height instrumentation to the Genesis smoke tools:

```text
BASE_HEIGHT_SOURCE floating_base_dof
```

  H200/local tests for the helper:

```text
Local: PYTHONPATH=src python -m pytest -p no:cacheprovider tests/test_genesis_action_replay_smoke.py
Result: 11 passed

H200: PYTHONPATH=src python3 -m pytest -p no:cacheprovider tests/test_genesis_action_replay_smoke.py
Result: 11 passed
```

  Re-running the same official-obs-decoded actions with the corrected height
  reader but the old double-counted `--base-pos ... 0.791166` fails as expected:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_20f_corrected_height.log

BASE_HEIGHT_SOURCE floating_base_dof
FRAME 0 base_z 1.584699034690857
FRAME 19 base_z 1.5525048971176147
HEIGHT_OK_RANGE 0.3 1.2 False
base height left smoke range during Genesis action replay
```

  Re-running 20 frames with corrected entity z `--base-pos 0.002389 0.011728
  -0.001834` gives a plausible dynamic root height:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_20f_corrected_init.log

BASE_HEIGHT_SOURCE floating_base_dof
FRAME 0 base_z 0.7931939363479614
FRAME 19 base_z 0.7633495330810547
FINITE_OK True
BASE_HEIGHT_MIN 0.7620419859886169
BASE_HEIGHT_MAX 0.7931952476501465
BASE_HEIGHT_FINAL 0.7633495330810547
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_ACTION_REPLAY_SMOKE_OK
```

  This corrected-init 20-frame result is a partial smoke only. It does not
  restore the previous route pass because the contact/min-link metric still
  needs replacement and the corrected 50-frame/GIF evidence has not been
  regenerated.
