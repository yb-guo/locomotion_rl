# H200 Run Notes

## Machine

- Host: `nb-0p4ivpp1kj-0`
- Run dates on target: `2026-05-05T02:16:13-07:00` through `2026-05-05T22:36:04-07:00`
- Kernel: `Linux nb-0p4ivpp1kj-0 5.10.0-1.oe.jd_702.x86_64 #1 SMP PREEMPT Thu Dec 26 15:16:38 CST 2024 x86_64`
- GPU: 2x `NVIDIA H200`
- Driver: `570.195.03`
- CUDA: `12.8` reported by `nvidia-smi`; `/usr/local/cuda` resolves to CUDA `12.8.61`
- Python: `Python 3.11.11`
- Git: `git version 2.34.1`
- Git LFS: `git-lfs/3.0.2`
- TensorRT:
  - System packages initially present: `10.13.3.9-1+cuda13.0`
  - `TensorRT_ROOT=/root/TensorRT` points to system include/lib symlinks
  - Final L1 deploy used isolated runtime libs extracted from `10.13.3.9-1+cuda12.9` `.deb` packages, prepended through `LD_LIBRARY_PATH`
- Source snapshot:
  - H200 outbound access to GitHub failed, so source was transferred from local as a GitHub zip snapshot rather than cloned on target.
  - Source zip SHA256: `3CBCF78EA25EFEA7CB548E4CD35FB15176AD7D1B0FE72D5FEB0485D730245DAB`
  - No commit SHA was available because the zip snapshot has no `.git` directory.

## Environment Commands

```bash
date -Is
hostname
uname -a
nvidia-smi
python3 --version
git --version
git lfs version
nvcc --version
dpkg-query -W "*tensorrt*" "*nvinfer*"
```

Key output:

```text
Host: nb-0p4ivpp1kj-0
GPU: NVIDIA H200 x2
Driver Version: 570.195.03
CUDA Version: 12.8
Python 3.11.11
git version 2.34.1
git-lfs/3.0.2
libnvinfer10 10.13.3.9-1+cuda13.0
tensorrt 10.13.3.9-1+cuda13.0
```

## L1 MuJoCo Sim2Sim

- Result: pass on H200 headless path using MuJoCo under `xvfb-run`.
- L2 was not run until this L1 pass evidence was collected.

### Network and Transfer Workarounds

The H200 target could not reach GitHub, Hugging Face, PyPI, or several external apt HTTPS repositories.

```text
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
fatal: unable to access 'https://github.com/NVlabs/GR00T-WholeBodyControl.git/': Failed to connect to github.com port 443

python download_from_hf.py
[Errno 101] Network is unreachable
```

Workaround artifacts transferred from local machine:

- Upstream source zip: `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl-main.full.zip`
- LFS overlay for the L1-relevant paths:
  - `gear_sonic/data/robot_model/model_data/g1/`
  - `gear_sonic_deploy/`
  - `decoupled_wbc/control/robot_model/model_data/g1/`
- LFS result: `286` pointer files replaced from `168` unique LFS objects; selected L1 paths had `0` remaining LFS pointers.
- HF deployment files placed in `gear_sonic_deploy`:
  - `policy/release/model_encoder.onnx` (`48M`)
  - `policy/release/model_decoder.onnx` (`40M`)
  - `policy/release/observation_config.yaml` (`2.3K`)
  - `planner/target_vel/V2/planner_sonic.onnx` (`739M`)

### Build

Commands and logs:

```bash
cd /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
source scripts/setup_env.sh
just build
```

Important deviations:

- `scripts/install_deps.sh` initially failed because it tried to download `just` from GitHub.
- `just-1.43.0-x86_64-unknown-linux-musl.tar.gz` was transferred from local and installed to `/usr/local/bin/just`.
- `onnxruntime-linux-x64-1.16.3.tgz` was transferred from local and installed under `/opt/onnxruntime`.
- `just build` initially failed at CMake FetchContent for `googletest`; `googletest-v1.14.0.zip` was transferred from local and supplied via `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`.
- Final build passed.

Build logs on target:

```text
run_logs/install_deps_retry_after_local_release.log
run_logs/cmake_config_with_local_googletest.log
run_logs/just_build_after_cmake_cache.log
```

Build product:

```text
gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```

### MuJoCo Sim Env

Official installer:

```bash
bash install_scripts/install_mujoco_sim.sh
```

Result: failed while downloading `uv` from `https://astral.sh/uv/install.sh` because target outbound network timed out.

Manual offline env:

- Installed `python3.10-venv` and `python3.10-dev` from reachable JDCloud Ubuntu mirror.
- Created `.venv_sim`.
- Transferred and installed a local wheelhouse, including `torch-2.4.1+cpu`, `mujoco`, `unitree_sdk2py`, `cyclonedds`, `tomli`, and missing transitive dependencies.

Verification:

```text
pip check: No broken requirements found.
mujoco 3.8.0
torch 2.4.1+cpu
gear_sonic import OK
unitree_sdk2py import OK
```

Direct viewer startup without display failed as expected on the headless H200:

```text
GLFWError: X11: The DISPLAY environment variable is missing
ERROR: could not initialize GLFW
```

Headless viewer path passed under Xvfb:

```bash
timeout 60s xvfb-run -a -s '-screen 0 1280x720x24' python gear_sonic/scripts/run_sim_loop.py
```

Evidence:

```text
run_logs/run_sim_loop_xvfb_60s.log
run_logs/run_sim_loop_xvfb_pair.log
```

The long-running sim loop used for deploy was still alive during the controlled L1 run:

```text
494487 /bin/sh /usr/bin/xvfb-run -a python gear_sonic/scripts/run_sim_loop.py
494502 python gear_sonic/scripts/run_sim_loop.py
```

It was stopped after final evidence collection.

### TensorRT/CUDA Root Cause and Fix

First deploy attempt with system TensorRT failed:

```text
terminate called after throwing an instance of 'nvinfer1::APIUsageError'
what(): CUDA initialization failure with error: 35
```

Root cause check:

```text
CUDA 12 runtime with driver 570.195.03:
runtime 12080, driver_api 12080, devices 2

CUDA 13 runtime with driver 570.195.03:
runtime 13000, driver_api 12080, device_ret 35
```

Conclusion: system TensorRT `10.13.3.9-1+cuda13.0` pulled a CUDA 13 runtime path that is incompatible with the current driver/runtime capability. This caused the original CUDA error 35.

Fix tested without system downgrade:

- Transferred TensorRT `10.13.3.9-1+cuda12.9` runtime `.deb` files.
- Extracted them to:

```text
/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root
```

- Ran deploy with this library path first:

```bash
export TensorRT_ROOT=/root/TensorRT
source scripts/setup_env.sh
export LD_LIBRARY_PATH=/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Dynamic linker evidence from final run:

```text
libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12
libnvinfer.so.10 => /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr/lib/x86_64-linux-gnu/libnvinfer.so.10
libnvonnxparser.so.10 => /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10
```

Generated TensorRT engine files:

```text
gear_sonic_deploy/policy/release/policy_model_decoder.trt       40M
gear_sonic_deploy/policy/release/encoder_model_encoder.trt      48M
gear_sonic_deploy/planner/target_vel/V2/planner_planner_sonic.trt 712M
```

Planner engine build took `532.38` seconds on the first successful CUDA 12.x runtime run.

### Controlled Sim2Sim Run

Final controlled L1 command used a FIFO harness to send only:

```text
]  start control
T  play current reference motion
O  stop control and exit
```

Main log:

```text
run_logs/deploy_sim_controlled_keys.log
```

Command shape:

```bash
cd /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=/root/TensorRT
source scripts/setup_env.sh
export LD_LIBRARY_PATH=/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type manager \
  --output-type all \
  --zmq-host localhost \
  --disable-crc-check
```

Pass evidence from controlled run:

```text
✓ Policy model loaded successfully!
✓ TensorRT planner model loaded successfully!
[HARNESS] sent_start_bracket at 2026-05-05T22:35:47-07:00
[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state
Loop timing - LowState age: 8.149ms, ... Policy: 4153us, Obs 2 Motor Command: 7029us
Playing motion 0 from frame 0 to end (497 total frames)
[HARNESS] sent_play_T at 2026-05-05T22:35:52-07:00
Loop timing - LowState age: 10.427ms, ... Policy: 4152us, Obs 2 Motor Command: 7446us
...
[DEBUG] Program exiting normally...
[HARNESS] deploy_status=0
```

L1 pass criteria mapping:

- MuJoCo viewer starts: passed under `xvfb-run`; direct non-X11 startup fails on headless H200 as expected.
- Deploy binary starts: passed.
- Policy loads: passed.
- `]` starts policy: passed; log shows transition to `CONTROL`.
- `T` makes robot track/play current motion: passed by headless evidence; log shows `Playing motion 0 from frame 0 to end (497 total frames)` followed by continuous policy/control loop timing against MuJoCo low-state data.
- No TensorRT version warning: passed; final controlled run used TensorRT `10.13` runtime libraries and produced no TensorRT version mismatch warning.

Known caveat: this was a headless H200 run. There was no visual human inspection of the MuJoCo window; verification relies on Xvfb viewer startup, deploy/sim connectivity, low-state freshness, controlled key events, and policy/control loop logs.

## L2 Training Smoke

- Result: not run.
- Isaac Lab version: not checked.
- Command: not run.
- Metrics seen: none.
- Failure reason: user instruction was to execute L1 and not continue to L2 unless L1 passed. L1 pass was established at the end of this run; L2 was intentionally not started.

## L3 Finetune Smoke

- Result: not run.
- `num_envs`: not applicable.
- steps/sec: not recorded.
- GPU memory: not recorded for L3.
- Failure reason: L2 was not run.

## Decisions

- SONIC official sim2sim: validated on H200 headless MuJoCo path using transferred upstream snapshot, LFS/model overlays, and TensorRT `10.13` CUDA 12.x runtime libraries.
- SONIC training on H200: not attempted in this task. Next step can start L2 from a clean task/run only if sample data and Isaac Lab constraints are explicitly accepted.
- Operational finding: for this H200 host, do not use system TensorRT `10.13.3.9-1+cuda13.0` with driver `570.195.03`; prefer a CUDA 12.x TensorRT runtime path or upgrade the driver/runtime stack before CUDA 13 TensorRT.
