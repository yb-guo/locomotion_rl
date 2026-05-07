# Route

Task: task004-genesis-g1-baseline

Goal: Implement or wrap a Genesis env that can reset and step.

Scope:

- `src/h200_locomotion_lab/envs/genesis_adapter.py`
- tests for reset/step boundary where possible

Verify:

- Minimal script resets and steps without training.

Environment:

- Linux H200 target for real Genesis
- local stub tests allowed

No Hack:

- no simulator import at module import time if it breaks local tests
- no global mutable singleton scene
- no unbounded per-step Python logging

Hardware:

- avoid CPU/GPU sync in hot path
- batch envs when real training starts

# Log

- 2026-05-06: Replaced the placeholder `GenesisG1Env` with a simulator-independent
  reset/step boundary in `src/h200_locomotion_lab/envs/genesis_adapter.py`.
- The module does not import `genesis` at module import time.
- Added:
  - `GenesisG1Contract`
  - `StepResult`
  - `GenesisBackend` protocol
  - `ContractOnlyBackend` for local boundary tests
  - `GenesisG1Env.contract_only()` for local reset/step verification
- Added local tests in `tests/test_genesis_adapter.py`.
- 2026-05-06: Added `GenesisSceneConfig` and `GenesisG1SceneBackend`, a
  single-environment real Genesis backend for the validated 29-motor G1 asset.
  The backend:
  - loads Genesis lazily;
  - builds `gs.Scene(show_viewer=False)` with the G1 MJCF asset;
  - resolves the 29 policy motors through `robot.get_joint(name).dofs_idx_local`;
  - applies clipped normalized actions through `robot.control_dofs_position`;
  - steps Genesis for `decimation=4`;
  - returns the configured 96D observation boundary.
- H200 check:

```bash
python3 - <<'PY'
import importlib.util
print(importlib.util.find_spec('genesis'))
PY
```

Result: `None`; real Genesis package is not installed on the H200 target.

- 2026-05-06: Installed `genesis-world==0.4.6` plus its Linux/Python 3.11
  dependency wheelhouse on the H200 target under the base conda environment.
  The target has `torch==2.5.1+cu124`, so Genesis warns that `torch<2.8.0`
  is unsupported. `pip check` also reports dependency conflicts with the
  pre-existing GR00T environment and a `pygltflib` pin mismatch. This install is
  acceptable for smoke testing, but repeated SONIC/Genesis work should use a
  separate environment.
- H200 minimal Genesis CUDA plane smoke passed:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_plane_cuda_smoke.log
GENESIS_VERSION 0.4.6
Running on [NVIDIA H200] with backend gs.cuda
PLANE_CUDA_SMOKE_OK steps=20 elapsed_s=3.164
PLANE_CUDA_EXIT_STATUS=0
```

- H200 SONIC G1 MJCF attempt using the original contract inventory path failed
  because the mesh files below `gear_sonic/data/robots/g1/meshes` are still Git
  LFS pointer text files, not real STL meshes:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_sonic_g1_mjcf_cuda_smoke.log
Asset: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml
ValueError: decoder failed for mesh file .../left_hip_roll_link.STL
```

- H200 SONIC G1 MJCF smoke passed when using the already materialized SONIC
  `model_data/g1` asset path:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_sonic_g1_model_data_mjcf_cuda_smoke.log
Asset: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml
GENESIS_VERSION 0.4.6
Running on [NVIDIA H200] with backend gs.cuda
ROBOT_N_DOFS 49
ROBOT_N_LINKS 44
SONIC_G1_MODEL_DATA_MJCF_CUDA_SMOKE_OK steps=20 elapsed_s=3.861
SONIC_G1_MODEL_DATA_MJCF_CUDA_EXIT_STATUS=0
```

- 2026-05-06: Found the true 29-motor SONIC G1 MJCF in the deploy asset
  directory. It has real STL meshes already materialized:

```text
Asset: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml
XML motor tags: 29
Referenced meshes: 36 real, 0 pointer, 0 missing
```

- Filled the original contract inventory path by copying the 36 mesh files
  referenced by `gear_sonic/data/robots/g1/g1_29dof.xml` from
  `gear_sonic_deploy/g1/meshes`. The full `gear_sonic/data/robots/g1` directory
  still contains many unrelated LFS pointer meshes, but this XML's referenced
  mesh set now has `0` LFS pointers.

```text
Manifest: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/g1_29dof_mesh_fill_manifest.txt
mesh_refs 36 copied 36 already_real 0 missing_src 0 still_pointer 0
ORIGINAL_XML_REFERENCED_POINTER_MESHES 0
```

- H200 Genesis CUDA smoke for the original filled 29-motor G1 path passed:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_sonic_g1_29dof_original_filled_mjcf_cuda_smoke.log
Asset: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml
GENESIS_VERSION 0.4.6
Running on [NVIDIA H200] with backend gs.cuda
XML_JOINT_TAGS 30
XML_MOTOR_TAGS 29
ROBOT_N_DOFS 35
ROBOT_N_LINKS 31
SONIC_G1_29DOF_ORIGINAL_FILLED_MJCF_CUDA_SMOKE_OK steps=20 elapsed_s=3.823
SONIC_G1_29DOF_ORIGINAL_FILLED_MJCF_CUDA_EXIT_STATUS=0
```

`ROBOT_N_DOFS 35` is expected here: Genesis includes the 6-DoF floating base in
`n_dofs`, while the policy/action contract is the 29 motor actuators recorded by
`XML_MOTOR_TAGS 29`.

- H200 smoke for the repo `GenesisG1Env` real backend passed:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_g1_env_backend_reset_step.log
ASSET /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml
ENV_DESC Genesis G1 boundary: 96D obs, 29D action, 50Hz policy.
RESET_OBS_LEN 96
MOTOR_DOF_INDICES (6, 9, 12, 15, 19, 23, 7, 10, 13, 16, 20, 24, 8, 11, 14, 17, 21, 25, 27, 29, 31, 33, 18, 22, 26, 28, 30, 32, 34)
DEFAULT_MOTOR_POS_LEN 29
STEP_OBS_LEN 96
STEP_INFO {'backend': 'genesis', 'step_count': 1, 'asset_path': '...', 'robot_n_dofs': 35, 'motor_dof_count': 29}
GENESIS_G1_ENV_BACKEND_RESET_STEP_OK
GENESIS_G1_ENV_BACKEND_RESET_STEP_EXIT_STATUS=0
```

- 2026-05-06: Added
  `python -m h200_locomotion_lab.tools.sonic_reference_replay_smoke` for a
  SONIC reference replay smoke. This is not an ONNX policy-action test: it uses
  SONIC `reference/example/*/joint_pos.csv` rows as 29D position targets in
  MuJoCo/URDF order, sets the SONIC deploy Kp/Kd gains from
  `policy_parameters.hpp`, and drives the Genesis backend through
  `control_dofs_position`.
- A first non-decimated 120-frame run against
  `walking_quip_360_R_002__A428` did not reach frame output before the SSH
  session closed after high-face mesh SDF preprocessing under a heavily loaded
  H200. No replay/genesis process was left running afterward.
- A decimated 20-frame smoke passed on H200:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_g1_sonic_reference_replay_decimate_20f.log
SONIC_REFERENCE_REPLAY_MODE joint_pos_as_position_targets
REF_DIR .../gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428
JOINT_POS_ROWS 455
REPLAY_FRAMES 20
MOTOR_DOF_COUNT 29
FRAME 0 base_z 0.791166 mean_abs_err 0.003801 max_abs_err 0.061452 max_abs_qvel 6.86949
FRAME 19 base_z 0.791166 mean_abs_err 0.008670 max_abs_err 0.126555 max_abs_qvel 4.30685
FINITE_OK True
BASE_HEIGHT_MIN 0.791166
BASE_HEIGHT_FINAL 0.791166
MEAN_ABS_TRACKING_ERROR_AVG 0.0103434
MAX_ABS_TRACKING_ERROR 0.148632
MAX_ABS_QVEL 7.94898
SIM_STEPS 80
HEIGHT_OK_RANGE 0.2 1.5 True
SONIC_REFERENCE_REPLAY_GENESIS_SMOKE_OK
SONIC_REFERENCE_REPLAY_DECIMATE_20F_EXIT_STATUS=0
```

- 2026-05-06: Added
  `python -m h200_locomotion_lab.tools.sonic_reference_replay_gif` and rendered
  visual GIFs for inspection. The first dynamic-PD GIF showed the robot sinking
  into/behind the ground plane and is not treated as visual pass evidence. For
  visual asset/motion inspection, the GIF tool now defaults to `--mode
  kinematic`, which directly sets SONIC reference root position, root
  quaternion, and 29D joint positions per frame.

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_g1_sonic_reference_replay_kinematic_highcam_gif_40f.log
Remote GIF: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/videos/genesis_g1_sonic_reference_replay_kinematic_highcam_40f.gif
Local copy: .agent/task/task004-genesis-g1-baseline/artifacts/genesis_g1_sonic_reference_replay_kinematic_highcam_40f.gif
MODE kinematic
RES (360, 260)
RENDERED_FRAMES 40
GIF_BYTES 124927
BASE_HEIGHT_MIN 0.790827
BASE_HEIGHT_MAX 0.791459
BASE_HEIGHT_FINAL 0.790902
SONIC_REFERENCE_REPLAY_GIF_OK
SONIC_REFERENCE_REPLAY_KINEMATIC_HIGHCAM_GIF_40F_EXIT_STATUS=0
```

- 2026-05-07: Reworked the SONIC reference replay initialization to follow the
  Genesis control-demo shape more closely:
  - pass the SONIC reference root `pos` and `quat` into `gs.morphs.MJCF(...)`
    at entity creation time instead of relying only on a post-build pose set;
  - resolve motor DOFs through joint names and apply `set_dofs_kp`,
    `set_dofs_kv`, and `set_dofs_force_range`;
  - continue to drive dynamic replay through `control_dofs_position`;
  - record `body_quat.csv` and min-link-height diagnostics in the smoke output.

Genesis references checked:

- `gs.morphs.MJCF` accepts `pos` and `quat` and documents that those override the
  MJCF root/base pose.
- `RigidEntity` exposes `set_dofs_force_range`, `set_dofs_position`, and
  `control_dofs_position` on local DOF indices.

- H200 dynamic replay smoke with official-style initialization passed:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_g1_sonic_reference_replay_official_init_decimate_20f.log
SONIC_REFERENCE_REPLAY_MODE joint_pos_as_position_targets
REF_DIR .../gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428
REPLAY_FRAMES 20
ROOT0 (0.002389, 0.011728, 0.791166)
ROOT_QUAT0 (0.711231, -0.00883, -0.004562, -0.702888)
FORCE_LIMITS_MIN_MAX 5.0 139.0
MOTOR_DOF_COUNT 29
FRAME 0 base_z 0.7911660075187683 min_link_z 0.7911660075187683 mean_abs_err 0.007587959634948395 max_abs_err 0.062162503591537466 max_abs_qvel 3.953256130218506
FRAME 19 base_z 0.7911660075187683 min_link_z 0.7911660075187683 mean_abs_err 0.08002924763951451 max_abs_err 0.3317313223991394 max_abs_qvel 1.2765580415725708
FINITE_OK True
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
MIN_LINK_HEIGHT_MIN 0.7911660075187683
MIN_LINK_HEIGHT_FINAL 0.7911660075187683
MEAN_ABS_TRACKING_ERROR_AVG 0.05136528159935812
MAX_ABS_TRACKING_ERROR 0.3317313223991394
MAX_ABS_QVEL 6.954092979431152
SIM_STEPS 80
SONIC_REFERENCE_REPLAY_GENESIS_SMOKE_OK
```

- Attempted to start a dynamic 40-frame official-init GIF afterward, but both
  H200 SSH routes failed during banner/key-exchange before the Genesis command
  started:

```text
proxy myserver: Connection timed out during banner exchange
direct 116.198.70.4:22376: kex_exchange_identification / banner exchange error
```

Visual dynamic GIF verification is therefore still pending; do not treat the
dynamic visual route as reviewed until SSH is stable and the GIF is generated.

- 2026-05-07: H200 SSH stabilized and the dynamic visual replay was completed.
  Two GIFs were rendered; the second uses a better full-body camera and is the
  inspection artifact to use:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_g1_sonic_reference_replay_dynamic_official_init_fullbody_gif_40f.log
Remote GIF: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/videos/genesis_g1_sonic_reference_replay_dynamic_official_init_fullbody_40f.gif
Remote contact sheet: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/videos/genesis_g1_sonic_reference_replay_dynamic_official_init_fullbody_40f_contact.png
Local GIF: .agent/task/task004-genesis-g1-baseline/artifacts/genesis_g1_sonic_reference_replay_dynamic_official_init_fullbody_40f.gif
Local contact sheet: .agent/task/task004-genesis-g1-baseline/artifacts/genesis_g1_sonic_reference_replay_dynamic_official_init_fullbody_40f_contact.png
MODE dynamic
RES (420, 320)
RENDERED_FRAMES 40
GIF_BYTES 88434
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
SONIC_REFERENCE_REPLAY_GIF_OK
```

Remote `imageio` read-back of the full-body GIF:

```text
READ_FRAMES 40
DIFF_MIN 0.03939484126984127
DIFF_MAX 0.11858382936507937
DIFF_AVG 0.07370325854700854
CONTACT shape (320, 2100, 3)
```

Visual review of the 5-frame contact sheet shows the robot remains above the
plane; the previous "sunk under the floor" symptom is no longer present in this
official-init dynamic replay. The motion is small because this is still a short
reference joint-position PD replay, not a learned SONIC policy rollout.

- Local verification command:

```bash
PYTHONPATH=src python -m pytest -p no:cacheprovider
```

Result: `8 passed`.

# Review

Result: pass for reset/step smoke.
Syntax: pass.
Hack: pass; local contract-only backend remains explicit, and the real backend is
clearly labeled as a single-env Genesis smoke backend rather than a training env.
Scope: pass; adapter boundary, tests, and task notes only.
Efficiency: pass; no global scene singleton and no per-step logging.
Hardware: pass for raw Genesis CUDA import/build/step smoke on H200.
Verify: local reset/step tests passed; raw Genesis H200 plane, SONIC G1 MJCF
build/step, repo `GenesisG1Env` real backend reset/step smoke, and decimated
SONIC reference joint-position replay smoke passed. Official-style dynamic
Genesis replay smoke also passed numerically on H200, and the full-body dynamic
GIF/contact-sheet visual check no longer shows the robot sunk below the plane.
Findings: PPO training is still pending. The observation is contract-shaped and
minimal; reward, termination, vectorized envs, and locomotion reset curriculum
belong in the PPO baseline subtask. A true SONIC ONNX policy-action replay is
still separate from this reference joint-position replay.
