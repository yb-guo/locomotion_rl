# 018: Official Sim2Sim Runtime

## Route

Bring up the official SONIC runtime as the baseline needed by `015`:

1. Build the official `gear_sonic_deploy` C++ controller instead of the adapter
   runner.
2. Start the official MuJoCo sim loop from `gear_sonic/scripts/run_sim_loop.py`.
3. Run `gear_sonic_deploy/deploy.sh sim` or the equivalent built binary with
   the existing task025 ONNX artifacts.
4. Capture enough output to decide whether official SONIC target/action ranges
   match or contradict the mjlab adapter traces.

Dependency plan:

- reuse the official source staged in `017`;
- install TensorRT under the user workspace, not as an untracked repo file;
- do not download new SONIC checkpoints;
- if exact official deploy cannot run, record the first hard blocker with
  command output.

## Log

- 2026-05-17 Opened after the user asked to get the official normal runtime
  running.
- 2026-05-17 Preflight confirmed the local worktree is clean except for the
  existing unrelated `.test_tmp_task021/` untracked directory.
- 2026-05-17 H200 TensorRT search still found no existing install:

  ```text
  no NvInfer.h
  no NvInferVersion.h
  no libnvinfer.so*
  apt-cache search tensorrt/nvinfer: no usable results
  dpkg -l | grep tensorrt/nvinfer: no results
  ```

- 2026-05-17 Official build requirements from
  `gear_sonic_deploy/cmake/FindTensorRT.cmake`:

  ```text
  TensorRT_ROOT/include/NvInfer.h
  TensorRT_ROOT/include/NvInferVersion.h
  TensorRT_ROOT/lib/libnvinfer.so*
  TensorRT_ROOT/lib/libnvinfer_plugin.so*
  TensorRT_ROOT/lib/libnvonnxparser.so*
  ```

- 2026-05-17 Installed official deploy dependencies in H200 user-space only:

  ```text
  TensorRT root:
    /mnt/workspace/users/guoyubo/agent_workspace/official/tensorrt-10.16.1.11-cuda12.9/root
  Dev dependency extract:
    /mnt/workspace/users/guoyubo/agent_workspace/official/devdeps_jammy/extract
  GoogleTest source:
    /mnt/workspace/users/guoyubo/agent_workspace/official/googletest-v1.14.0/googletest-1.14.0
  ```

  The TensorRT root uses official NVIDIA deb contents for
  `libnvinfer10`, `libnvinfer-plugin10`, `libnvonnxparsers10`,
  `libnvinfer-headers-dev`, `libnvinfer-headers-plugin-dev`, and
  `libnvonnxparsers-dev`. The Jammy dev extract supplies
  `msgpack.hpp`, `zmq.h`, `zmq.hpp`, `Eigen3`, `nlohmann/json.hpp`,
  and `GTest`.

- 2026-05-17 Official `gear_sonic_deploy` CMake configure passed after
  setting:

  ```text
  TensorRT_ROOT=/mnt/workspace/users/guoyubo/agent_workspace/official/tensorrt-10.16.1.11-cuda12.9/root
  onnxruntime_ROOT=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/onnxruntime-linux-x64-1.19.2
  CUDAToolkit_ROOT=/usr/local/cuda-12.8
  CMAKE_INCLUDE_PATH=/mnt/workspace/users/guoyubo/agent_workspace/official/devdeps_jammy/extract/usr/include
  CMAKE_LIBRARY_PATH=/mnt/workspace/users/guoyubo/agent_workspace/official/devdeps_jammy/extract/usr/lib/x86_64-linux-gnu
  CMAKE_PREFIX_PATH=/mnt/workspace/users/guoyubo/agent_workspace/official/devdeps_jammy/extract/usr
  FETCHCONTENT_SOURCE_DIR_GOOGLETEST=/mnt/workspace/users/guoyubo/agent_workspace/official/googletest-v1.14.0/googletest-1.14.0
  ZMQ_LIBRARY=/usr/lib/x86_64-linux-gnu/libzmq.so.5.2.4
  ```

  `ZMQ_LIBRARY` must point to the system dynamic library. If CMake uses the
  extracted `libzmq.a`, linking pulls in unsatisfied static dependencies such
  as `sodium`, `pgm`, `gssapi`, and `norm`.

- 2026-05-17 Built the official runtime target successfully:

  ```text
  cmake --build build --target g1_deploy_onnx_ref --parallel 16
  [100%] Built target g1_deploy_onnx_ref
  ```

- 2026-05-17 Direct deploy startup without sim passed model/runtime loading.
  The official binary generated and cached TensorRT engines for:

  ```text
  policy/release/policy_model_decoder.trt
  policy/release/encoder_model_encoder.trt
  planner/target_vel/V2/planner_planner_sonic.trt
  ```

  It then waited at `LowState is not available`, which is expected when the
  MuJoCo sim publisher is not running.

- 2026-05-17 First official sim/deploy paired run exposed a Python DDS version issue.
  The `.venv_sim` had `cyclonedds 11.0.1`, while official
  `external_dependencies/unitree_sdk2_python/README.md` requires
  `cyclonedds==0.10.2`. With `11.0.1`, `lo` mode left deploy waiting for
  LowState, and `eth0` crashed the C++ deploy during DDS domain creation with:

  ```text
  ddsi_xt_type_init_impl with invalid type object
  timeout: monitored command dumped core
  ```

- 2026-05-17 Rebuilt and installed `cyclonedds==0.10.2` inside `.venv_sim`
  with `CYCLONEDDS_HOME` pointed at a user-space wrapper around the official
  bundled Unitree `libddsc.so`. Added both `libddsc.so` and `libddsc.so.0`
  symlinks under:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/official/cyclonedds_unitree_sdk2
  ```

- 2026-05-17 Verified the official Python sim publisher itself is live after
  the DDS fix. A temporary Python subscriber to `rt/lowstate` on `lo` received
  continuous messages while the official sim loop ran:

  ```text
  count=51
  count=150
  count=249
  count=347
  count=446
  count=545
  count=644
  ```

- 2026-05-17 Official sim/deploy paired run with `cyclonedds==0.10.2` now reaches the
  C++ deploy state machine instead of hanging at startup:

  ```text
  [DEBUG] G1Deploy object created successfully!
  Safety reset: Returned to reference motion at frame 0
  Init Done
  [ERROR] Lost LowState data connection from robot!
  [ERROR] Safety check failed, cannot start control.
  [DEBUG] Program exiting normally...
  ```

  The deploy CSV files remained empty, so this is not yet a valid control
  rollout. It proves that model loading, TensorRT runtime, and initial DDS
  contact work, but C++ deploy does not receive continuous LowState updates
  from the Python sim publisher. The Python-Python LowState probe rules out
  "sim is not publishing" as the primary cause.

## Review

Status: partial pass; official runtime build and model startup pass, full
official sim2sim control loop still blocked.

Evidence now separates the problem:

- Build/runtime dependencies are resolved without system install.
- `g1_deploy_onnx_ref` builds and loads decoder, encoder, and planner through
  TensorRT.
- Official Python sim loop publishes continuous `rt/lowstate` over `lo` when
  observed by a Python subscriber.
- C++ deploy receives enough initial LowState to finish `INIT`, but then its
  LowState timestamp goes stale beyond the official 500 ms safety threshold.

The next smallest diagnostic is a C++ LowState subscriber probe built against
the same bundled Unitree SDK as `g1_deploy_onnx_ref`. If that probe also sees
only intermittent Python sim samples, the blocker is cross-language Unitree
DDS/IDL compatibility. If the probe receives continuous samples, the issue is
inside the official deploy process after its full initialization path.
