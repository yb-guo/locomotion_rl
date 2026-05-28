# 017: Official Install Bootstrap

## Route

After explicit permission to install, bootstrap the smallest official
`NVlabs/GR00T-WholeBodyControl` environment needed for the remaining SONIC
target-contract diagnosis.

Install scope:

- fetch official source without pulling new model checkpoints;
- put it under the H200 official workspace path;
- reuse the existing task025 SONIC ONNX artifacts through symlinks;
- validate the official MuJoCo sim Python entry point;
- probe the C++ deploy build dependencies without installing system packages.

## Log

- 2026-05-17 H200 direct clone of
  `https://github.com/NVlabs/GR00T-WholeBodyControl.git` failed twice through
  the remote `gh-proxy.com` rewrite:

  ```text
  Failed to connect to gh-proxy.com port 443 after ~46s: connection timed out
  ```

- 2026-05-17 Created a local official shallow checkout at commit `0a87181`.
  Windows checkout hit long-path failures in deep asset/vendor directories, so
  the H200 install package was generated from the git object database instead
  of expanding the full tree on Windows.

- 2026-05-17 Uploaded and extracted an official source subset on H200:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/official/
    GR00T-WholeBodyControl/
  ```

  Included directories:

  ```text
  gear_sonic
  gear_sonic_deploy
  decoupled_wbc/control
  decoupled_wbc/data
  external_dependencies
  install_scripts
  ```

  Verified key official files exist:

  ```text
  gear_sonic/scripts/run_sim_loop.py
  gear_sonic_deploy/deploy.sh
  gear_sonic_deploy/g1/g1_29dof.xml
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
  ```

- 2026-05-17 Reused existing task025 artifacts by symlinking them into the
  official deploy layout:

  ```text
  gear_sonic_deploy/policy/release/model_encoder.onnx
    -> /mnt/workspace/users/guoyubo/agent_workspace/
       task025_sonic_mjlab_adapter/gear_sonic_artifacts/model_encoder.onnx
  gear_sonic_deploy/policy/release/model_decoder.onnx
    -> /mnt/workspace/users/guoyubo/agent_workspace/
       task025_sonic_mjlab_adapter/gear_sonic_artifacts/model_decoder.onnx
  gear_sonic_deploy/policy/release/observation_config.yaml
    -> /mnt/workspace/users/guoyubo/agent_workspace/
       task025_sonic_mjlab_adapter/gear_sonic_artifacts/observation_config.yaml
  gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
    -> /mnt/workspace/users/guoyubo/agent_workspace/
       task025_sonic_mjlab_adapter/gear_sonic_artifacts/planner_sonic.onnx
  ```

- 2026-05-17 Official `install_mujoco_sim.sh` initially failed because the
  extracted shell scripts had CRLF line endings. Converted official `.sh`
  files in place to LF.

- 2026-05-17 Official `install_mujoco_sim.sh` then failed because H200 could
  not reach the `uv` installer:

  ```text
  curl: (28) Failed to connect to astral.sh port 443 after ~46s
  ```

  A retry with elevated local permission showed the same remote network
  timeout.

- 2026-05-17 Built a compatible `.venv_sim` from the existing H200
  `unitree-rl-mjlab` conda Python with `--system-site-packages`, then uploaded
  a local Linux CPython 3.11 wheelhouse for missing sim dependencies. Installed:

  ```text
  gear_sonic-0.1.0 editable
  unitree_sdk2py-1.0.1 editable
  cyclonedds-11.0.1
  loguru-0.7.3
  msgpack-numpy-0.4.8
  pin-3.9.0
  ```

  This is a pragmatic runtime bootstrap, not a strict reproduction of the
  official uv-managed lock. Known pin differences:

  ```text
  unitree_sdk2py requests cyclonedds==0.10.2; installed cyclonedds 11.0.1
  gear_sonic requests numpy==1.26.4; installed venv-local numpy 2.3.5
  gear_sonic requests scipy==1.15.3; inherited scipy 1.16.2
  ```

- 2026-05-17 Python sim entry point verification passed:

  ```text
  cd /mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl
  .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --help
  ```

  The command prints the official `SimLoopConfig` CLI and imports the sim stack
  successfully.

- 2026-05-17 Import verification passed inside `.venv_sim`:

  ```text
  OK gear_sonic
  OK unitree_sdk2py
  OK cyclonedds
  OK mujoco 3.5.0
  OK tyro 1.0.13
  OK pinocchio 3.9.0
  OK zmq 27.1.0
  OK msgpack 1.1.2
  OK msgpack_numpy
  OK loguru 0.7.3
  OK cv2 4.11.0
  OK torch 2.8.0+cu128
  OK numpy 2.3.5
  OK scipy 1.16.2
  ```

- 2026-05-17 Official deploy script help works after LF conversion:

  ```text
  cd .../GR00T-WholeBodyControl/gear_sonic_deploy
  ./deploy.sh --help
  ```

- 2026-05-17 C++ deploy configure probe is blocked by missing TensorRT:

  ```text
  cmake -S gear_sonic_deploy -B /tmp/task025_official_deploy_build_probe \
    -Donnxruntime_DIR=/mnt/workspace/users/guoyubo/agent_workspace/
      task025_sonic_mjlab_adapter/onnxruntime-linux-x64-1.19.2/lib/cmake/onnxruntime
  ```

  Result:

  ```text
  CMake Error at cmake/FindTensorRT.cmake:46 (message):
    Fail to find TensorRT, please set TensorRT_ROOT. Include path not found.
  ```

  H200 search found the existing adapter ONNX Runtime but no TensorRT headers or
  libraries:

  ```text
  found:
    /mnt/workspace/users/guoyubo/agent_workspace/
    task025_sonic_mjlab_adapter/onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so
  not found:
    NvInferVersion.h
    libnvinfer.so*
  ```

## Review

Result: partial install complete.

The official source tree and Python MuJoCo sim entry point are now staged on
H200, and the official deploy layout points at the already verified task025
SONIC ONNX artifacts. This is enough to start building a Python-side official
sim loop probe, but it is not yet enough to run the official C++ deploy loop.

The direct official C++ deploy path is blocked by missing TensorRT, independent
of `just`: the CMake configure step fails before compilation. Installing
TensorRT should be treated as a separate explicit dependency task because it is
a large binary/system dependency and the H200 network path is unreliable.
