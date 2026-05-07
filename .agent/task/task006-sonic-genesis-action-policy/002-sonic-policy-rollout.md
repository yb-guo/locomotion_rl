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

- 2026-05-07: Checked Genesis documentation and official SONIC deploy source
  after the user challenged the usage.

  Genesis findings:

  - Genesis docs for `gs.morphs.MJCF` say `pos` is a translational offset
    applied to the original floating-base pose, and `quat` is the entity
    baselink quaternion. This supports the diagnosis that passing the SONIC
    reference root z directly as `MJCF(pos=...)` double-counted the G1 MJCF
    pelvis `pos="0 0 0.793"`.
  - H200 has Genesis `0.4.6`. Its installed source shows `get_pos()` returns
    `self._solver.get_links_pos(self.base_link_idx, ...)`, while `get_qpos()`
    and `set_qpos()` are the generalized-coordinate APIs. For this MJCF/free
    joint path, dynamic root pose evidence should use qpos/free-joint state,
    not the old constant `get_pos()` smoke field.
  - Genesis control docs show `set_dofs_position` is a hard state set and
    `control_dofs_position` is the PD target API. The action replay path should
    keep using `control_dofs_position` for motor targets, but initialization
    should set root qpos explicitly instead of relying on an entity-level offset
    as a root-state substitute.

  SONIC action-bridge findings:

  - Official `policy_parameters.hpp` documents:
    `action_scale = 0.25 * effort_limit / stiffness` and
    `target = action * action_scale + default_angle`.
  - Official `g1_deploy_onnx_ref.cpp` maps policy output from IsaacLab order to
    MuJoCo/hardware order with `floatarr[isaaclab_to_mujoco[i]]`, then applies
    `g1_action_scale[i]` and `default_angles[i]`.
  - The current Genesis contract is not SONIC-compatible: it assumes action
    rows are already in MuJoCo order, applies a uniform `0.25` rad scale,
    clips raw policy outputs to `[-1, 1]`, and uses the reference motion first
    row as the action-zero pose in previous policy-action replays.
  - Therefore the previous decoder-action Genesis replays were invalid for
    SONIC action semantics even after the root-height fix. They only proved
    that Genesis can step finite under some numeric 29D commands.

  Next correction: add an explicit SONIC action bridge that takes raw decoder
  output in IsaacLab order and computes MuJoCo-order position targets exactly
  as official deploy does:

```text
target_mujoco[i] =
    default_angles[i]
    + raw_action_isaaclab[isaaclab_to_mujoco[i]] * g1_action_scale[i]
```

  Do not clip policy output unless the official deploy path is changed or a
  separate safety experiment is explicitly labelled as non-official. After that,
  rerun H200 numeric smoke and GIF with corrected root qpos initialization and
  a contact metric based on Genesis contact APIs or link contact forces.

- 2026-05-07: Implemented the corrected offline SONIC action bridge.

  Code changes:

  - Added `h200_locomotion_lab.sonic.g1_policy_bridge` with official
    `isaaclab_to_mujoco`, `default_angles`, and per-joint `g1_action_scale`.
  - Added `action_mode="sonic_policy_raw"` to `GenesisG1SceneBackend` and the
    action replay tools. In this mode raw decoder output is not clipped and is
    converted exactly as the official deploy path does:

```text
target_mujoco[i] =
    default_angles[i]
    + raw_action_isaaclab[isaaclab_to_mujoco[i]] * g1_action_scale[i]
```

  - Changed the default MJCF entity offset to `(0, 0, 0)` and added explicit
    `root_qpos` initialization/reset support.
  - Updated SONIC reference replay to set the floating root via qpos instead of
    using `MJCF(pos=ref_root)` / `robot.set_pos(ref_root)`.
  - Added Genesis contact metrics:
    `CONTACT_COUNT_*` and `MAX_LINK_CONTACT_FORCE_*`. `MIN_LINK_HEIGHT_*` is
    kept only as a diagnostic because it can read `0.0` from non-foot/static
    links on this MJCF import path.

  Verification:

```text
Local:
PYTHONPATH=src python -m pytest -p no:cacheprovider
Result: 36 passed

H200:
cd /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke
PYTHONPATH=src python3 -m pytest -p no:cacheprovider \
  tests/test_genesis_action_replay_smoke.py \
  tests/test_sonic_g1_policy_bridge.py \
  tests/test_genesis_adapter.py
Result: 26 passed
```

  Corrected 20-frame offline replay smoke using real official obs-decoded raw
  decoder actions:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge.log

ACTION_MODE sonic_policy_raw
ACTION_SCALE_MODE sonic_g1_per_joint
ACTION_SCALES_MIN_MAX 0.07450087032524835 0.5475464651829304
DEFAULT_JOINT_POS_SOURCE sonic_default_angles
BASE_POS (0.0, 0.0, 0.0)
ROOT_QPOS (0.002389, 0.011728, 0.791166, 0.711231, -0.00883, -0.004562, -0.702888)
BASE_HEIGHT_SOURCE floating_base_dof
FRAME 0 base_z 0.788633406162262 action_min -0.729726672 action_max 0.824930727
FRAME 19 base_z 0.7169569134712219 action_min -2.26649046 action_max 4.07460451
FINITE_OK True
BASE_HEIGHT_MIN 0.7169569134712219
BASE_HEIGHT_MAX 0.788633406162262
BASE_HEIGHT_FINAL 0.7169569134712219
MAX_ABS_QVEL 15.70450496673584
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_ACTION_REPLAY_SMOKE_OK
```

  Corrected 20-frame GIF:

```text
Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge.gif

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge.gif

Local contact sheet:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge_contact.png

GIF log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_20f_sonic_bridge_gif.log

BASE_HEIGHT_MIN 0.7169569730758667
BASE_HEIGHT_MAX 0.788633406162262
BASE_HEIGHT_FINAL 0.7169569730758667
GIF_BYTES 57402
GENESIS_ACTION_REPLAY_GIF_OK
```

  Short contact-metric smoke:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_obs_decoder_actions_5f_sonic_bridge_contact.log

FRAME 0 base_z 0.788633406162262 contact_count 0 max_contact_force 0.0
FRAME 4 base_z 0.7527632713317871 contact_count 2 max_contact_force 117.78065754789701
CONTACT_COUNT_MAX 2
CONTACT_COUNT_FINAL 2
MAX_LINK_CONTACT_FORCE_MAX 260.3147467625842
MAX_LINK_CONTACT_FORCE_FINAL 117.78065754789701
GENESIS_ACTION_REPLAY_SMOKE_OK
```

  Review: corrected offline bridge smoke passes numerically and produces
  visible motion, but this is still not a closed-loop SONIC-in-Genesis policy.
  The GIF/contact sheet shows posture/motion, not a validated walking rollout.
  Remaining work is online 994D observation construction from Genesis state,
  reference/history buffers, and previous raw actions, then decoder inference at
  each control step.

- 2026-05-07: Implemented decoder-only closed-loop SONIC policy smoke.

  Code changes:

  - Added `h200_locomotion_lab.sonic.g1_observation` for the official release
    decoder observation layout:
    `64 + 30 + 290 + 290 + 290 + 30 = 994`.
  - The builder follows official `StateLogger::GetLatest(...,
    newest_first=false)` ordering: oldest-to-newest, with leading zero padding
    when history is short.
  - Added MuJoCo-order motor state to SONIC policy-order `body_q/body_dq`
    conversion. `body_q` is centered by subtracting official `default_angles`.
  - Added Genesis online SONIC history recording on reset/step.
  - Added `genesis_sonic_policy_rollout_smoke`, which builds 994D obs from
    Genesis state, runs `model_decoder.onnx`, applies `sonic_policy_raw`, and
    records base height/contact/action metrics.
  - Added `--token-mode replay` and `--history-init official_obs` so the first
    decoder observation can match a captured official 994D row while later
    frames are progressively replaced by Genesis state.

  Verification:

```text
Local:
PYTHONPATH=src python -m pytest -p no:cacheprovider
Result: 44 passed

Local:
python -m ruff check src tests
Result: failed because ruff is not installed locally:
No module named ruff

H200, from /tmp with explicit PYTHONPATH to avoid a stale top-level package:
PYTHONPATH=/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/src \
python3 -m pytest -p no:cacheprovider \
  .../tests/test_sonic_g1_observation.py \
  .../tests/test_genesis_adapter.py
Result: 18 passed
```

  H200 10-frame decoder-only closed-loop smoke:

```text
Command shape:
python3 -m h200_locomotion_lab.tools.genesis_sonic_policy_rollout_smoke \
  --asset .../gear_sonic/data/robots/g1/g1_29dof.xml \
  --decoder .../gear_sonic_deploy/policy/release/model_decoder.onnx \
  --obs-csv .../task006-sonic-genesis-action-policy/actions/official_policy_input_capture.csv \
  --token-mode replay \
  --history-init official_obs \
  --frames 10 \
  --root-qpos 0.002389 0.011728 0.791166 0.711231 -0.00883 -0.004562 -0.702888

FRAME 0 base_z 0.7887014746665955 action_min -0.7297266721725464 action_max 0.824930727481842
FRAME 5 base_z 0.7348557114601135 action_min -1.8372162580490112 action_max 1.702573537826538
FRAME 9 base_z 0.6546086668968201 action_min -1.822142481803894 action_max 5.1907243728637695
OBS_FINITE True
ACTION_FINITE True
BASE_HEIGHT_MIN 0.6546086668968201
BASE_HEIGHT_MAX 0.7887014746665955
BASE_HEIGHT_FINAL 0.6546086668968201
MAX_ABS_QVEL 6.918185710906982
ACTION_MAX_ABS 5.1907243728637695
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_OK
```

  H200 20-frame decoder-only closed-loop smoke with the same settings failed:

```text
FRAME 0 base_z 0.7887014746665955 action_min -0.7297266721725464 action_max 0.824930727481842
FRAME 5 base_z 0.7348557710647583 action_min -1.8372153043746948 action_max 1.7025740146636963
FRAME 10 base_z 0.6256087422370911 action_min -2.0017192363739014 action_max 4.937207221984863
FRAME 15 base_z 0.4301317036151886 action_min -4.970677852630615 action_max 4.425835132598877
FRAME 19 base_z 0.2644214928150177 action_min -8.832411766052246 action_max 6.377843856811523
OBS_FINITE True
ACTION_FINITE True
BASE_HEIGHT_MIN 0.2644214928150177
BASE_HEIGHT_FINAL 0.2644214928150177
ACTION_MAX_ABS 8.832411766052246
HEIGHT_OK_RANGE 0.3 1.2 False
GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_FAILED
```

  Review: first-frame decoder output now matches the official captured obs
  action range, so the current issue is no longer the first-frame observation
  layout or action bridge. The remaining blocker is stable closed-loop
  Genesis feedback: after roughly 10-20 frames the base falls and decoder
  actions grow. L2 should be considered partially connected only; do not mark
  stable policy rollout passed.

- 2026-05-07: Diagnosed and fixed the 20-frame closed-loop fall.

  Diagnose loop:

  - Reproduced the failure with H200 20-frame decoder-only closed loop:
    `token-mode replay`, `history-init official_obs`, corrected `root_qpos`.
  - Added `genesis_sonic_policy_rollout_probe` to compare Genesis-generated
    994D obs against captured official obs by field, and to replace individual
    fields with official values.
  - Tested hypotheses:
    - replacing only base angular velocity did not recover height;
    - replacing only body joint velocities reduced action drift but did not
      recover height;
    - replacing only last-action history reduced action drift but did not
      recover height;
    - teacher-forcing all official obs still fell when the closed-loop tool used
      Genesis default motor config.
  - Compared with the passing action-replay path and found the difference:
    action replay applied `apply_sonic_g1_motor_config` but the closed-loop
    smoke/probe/GIF tools did not.

  Fix:

  - `genesis_sonic_policy_rollout_smoke.py`,
    `genesis_sonic_policy_rollout_gif.py`, and
    `genesis_sonic_policy_rollout_probe.py` now apply SONIC's official G1
    kp/kv/force limits by default.
  - `--no-sonic-motor-config` is retained for explicit negative-control debug
    runs.

  H200 verification after fix:

```text
MOTOR_CONFIG sonic_g1_kp_kv_force_range
FRAME 0 base_z 0.788633406162262 action_max_abs 0.824930727481842
FRAME 5 base_z 0.751387357711792 action_max_abs 1.555012583732605
FRAME 10 base_z 0.7731913924217224 action_max_abs 2.3254857063293457
FRAME 15 base_z 0.785487949848175 action_max_abs 2.797891616821289
FRAME 19 base_z 0.7882418632507324 action_max_abs 3.2313764095306396
OBS_FINITE True
ACTION_FINITE True
BASE_HEIGHT_MIN 0.7512305378913879
BASE_HEIGHT_FINAL 0.7882418632507324
MAX_ABS_QVEL 4.189242839813232
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_OK
```

  Review: the original 20-frame failure no longer reproduces. This closes the
  diagnosed smoke failure, but not the broader claim of long-horizon walking or
  a fully online encoder/planner token path.
