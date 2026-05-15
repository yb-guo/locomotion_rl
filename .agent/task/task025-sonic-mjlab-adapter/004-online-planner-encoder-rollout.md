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

## Review

Planner runner dependency is now ready on H200. Full online rollout is still
blocked until the official SONIC ONNX artifacts and reference motion directory
are restored or explicitly downloaded.
