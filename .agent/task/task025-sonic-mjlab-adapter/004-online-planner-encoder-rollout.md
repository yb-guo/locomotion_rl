# 004: Online Planner Encoder Rollout

## Route

Run:

```text
mjlab state -> SONIC planner -> encoder -> decoder -> ScalarActionBridge
  -> MjlabG1RobotBackend -> mjlab step
```

Reuse task006 planner runner and ONNX wrapper behavior. Render only after
numeric finite/height checks pass.

## Log

- 2026-05-15 Not run. Current H200 workspace does not contain the SONIC ONNX
  artifacts or the C++ planner runner used by task006. The adapter CLI accepts
  these paths through:
  - `--planner`
  - `--planner-runner`
  - `--encoder`
  - `--decoder`
  - `--planner-work-dir`
- 2026-05-15 Rechecked H200 workspaces for existing SONIC artifacts:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace
  /mnt/workspace/users/guoyubo
  /mnt/workspace
  /root/h200-locomotion-lab-runs
  /root/agent_workspace
  ```

  Found only existing G1 XML copies:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/project/h200-locomotion-lab-task023/external_assets/task002_g1/g1_29dof.xml
  /mnt/workspace/users/guoyubo/h200-locomotion-lab-task023/external_assets/task002_g1/g1_29dof.xml
  ```

  Still missing:

  - `model_encoder.onnx`
  - `model_decoder.onnx`
  - `planner_sonic.onnx`
  - upstream reference motion directory

- 2026-05-15 Checked H200 for system C++ ONNX Runtime headers/libs. None were
  present under `/opt`, `/usr/local`, `/usr`, `/mnt/workspace/users/guoyubo`,
  or `/mnt/workspace`.

- 2026-05-15 Downloaded official ONNX Runtime C++ Linux x64 runtime locally and
  uploaded it to H200:

  ```text
  local:  .external_downloads/onnxruntime_cpp/onnxruntime-linux-x64-1.19.2.tgz
  remote: /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/onnxruntime-linux-x64-1.19.2.tgz
  sha256: eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33
  ```

- 2026-05-15 Built the planner runner from the task006 source on H200:

  ```bash
  g++ -O2 -std=c++17 sonic_planner_ort_runner.cpp \
    -I onnxruntime-linux-x64-1.19.2/include \
    -L onnxruntime-linux-x64-1.19.2/lib \
    -Wl,-rpath,onnxruntime-linux-x64-1.19.2/lib \
    -lonnxruntime \
    -o bin/sonic_planner_ort_runner
  ```

  Binary:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/bin/sonic_planner_ort_runner
  ```

  Dynamic link check:

  ```text
  libonnxruntime.so.1 => /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so.1
  ```

  Startup smoke:

  ```text
  SONIC_PLANNER_ORT_RUNNER_ERROR --planner is required
  ```

- 2026-05-15 Checked local agent/download paths for already-restored
  GEAR-SONIC artifacts before downloading. No local copies were found for:

  - `model_encoder.onnx`
  - `model_decoder.onnx`
  - `planner_sonic.onnx`
  - upstream `joint_pos.csv` reference motion

- 2026-05-15 User explicitly requested local download then upload. Downloaded
  official `nvidia/GEAR-SONIC` artifacts locally under:

  ```text
  .external_downloads/gear_sonic_artifacts
  ```

  Local artifact sizes and SHA256:

  ```text
  50,100,513  model_encoder.onnx  013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3
  40,900,688  model_decoder.onnx  c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed
  773,952,989 planner_sonic.onnx  39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea
  2,336       observation_config.yaml
  ```

- 2026-05-15 Uploaded the artifacts to H200:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/gear_sonic_artifacts
  ```

  Remote SHA256 verification:

  ```text
  013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3  model_encoder.onnx
  c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed  model_decoder.onnx
  39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea  planner_sonic.onnx
  ```

- 2026-05-15 Verified official ONNX model IO on H200:

  ```text
  model_encoder.onnx
  INPUT obs_dict (1, 1762) 1
  OUTPUT encoded_tokens (1, 64) 1

  model_decoder.onnx
  INPUT obs_dict (1, 994) 1
  OUTPUT action (1, 29) 1

  planner_sonic.onnx
  INPUT context_mujoco_qpos (1, 4, 36) 1
  INPUT target_vel (1,) 1
  INPUT mode (1,) 7
  INPUT movement_direction (1, 3) 1
  INPUT facing_direction (1, 3) 1
  INPUT random_seed (1,) 7
  INPUT has_specific_target (1, 1) 7
  INPUT specific_target_positions (1, 4, 3) 1
  INPUT specific_target_headings (1, 4) 1
  INPUT allowed_pred_num_tokens (1, 11) 7
  INPUT height (1,) 1
  OUTPUT mujoco_qpos (1, 64, 36) 1
  OUTPUT num_pred_frames (1,) 6
  ```

- 2026-05-15 Ran true online SONIC planner/encoder/decoder mjlab smoke on
  H200 for 40 steps, replan interval 10, EGL camera enabled, terminations
  disabled for adapter isolation:

  ```text
  provider online
  steps 40
  planner_calls 4
  done_steps []
  root_start_xyz [0.14208120107650757, -0.13886110484600067, 0.7968763709068298]
  root_end_xyz [0.3308540880680084, -0.0002928735048044473, 0.7456598877906799]
  root_delta_xyz [0.18877288699150085, 0.13856823134119622, -0.0512164831161499]
  video_bytes [183259]
  ```

  Remote video:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/online_smoke/online_smoke-step-0.mp4
  ```

- 2026-05-15 Ran a longer 160-step online smoke:

  ```text
  provider online
  steps 160
  planner_calls 16
  done_steps []
  root_start_xyz [0.3448217213153839, 0.20812548696994781, 0.7967156767845154]
  root_end_xyz [1.618209719657898, 0.19514200091362, 0.7064092755317688]
  root_delta_xyz [1.273387998342514, -0.01298348605632782, -0.09030640125274658]
  video_bytes [554603]
  ```

  Remote video:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/online_smoke_160/online_smoke_160-step-0.mp4
  ```

  Local copy:

  ```text
  outputs/task025/online_smoke_160/online_smoke_160-step-0.mp4
  ```

## Review

First true online rollout passed the adapter gate: official planner, encoder,
and decoder artifacts execute inside `unitree_rl_mjlab`, feed finite 29DoF
actions through the existing scalar runtime boundary, and render video evidence.

This is not yet a stable locomotion claim. The 160-step smoke moves forward
about 1.27 m, but root height drops by about 9 cm. Next work should inspect the
video, compare reset/context construction against official SONIC assumptions,
and then decide whether the remaining issue is initial state, command contract,
domain mismatch, or controller gains.
