# Agent Execute: task002 SONIC MuJoCo Smoke

Run this on the H200/Linux target, not Windows.

## Goal

Reproduce official GEAR-SONIC release at the smallest useful level:

1. Build deployment stack.
2. Download release ONNX/checkpoints.
3. Run MuJoCo sim2sim.
4. Optionally run training smoke with sample data.
5. Record result back into this task.

## Sources

- SONIC page: `https://nvlabs.github.io/GEAR-SONIC/`
- Docs: `https://nvlabs.github.io/GR00T-WholeBodyControl/`
- Code: `https://github.com/NVlabs/GR00T-WholeBodyControl`
- Models/data: `https://huggingface.co/nvidia/GEAR-SONIC`

## Environment Record

Before installing, record:

```bash
date -Is
hostname
uname -a
nvidia-smi
python3 --version
git --version
git lfs version
```

Write output into:

```text
.agent/task/task002-sonic-mujoco-smoke/run_h200_notes.md
```

## L1: Deployment + MuJoCo Sim2Sim

### 1. Clone upstream

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull
```

### 2. Install TensorRT

Hard requirement:

```text
x86_64 TensorRT = 10.13
```

Set:

```bash
export TensorRT_ROOT=$HOME/TensorRT
```

Stop if TensorRT is missing or not `10.13`.

### 3. Download deployment models

```bash
pip install huggingface_hub
python download_from_hf.py
```

Expected layout:

```text
gear_sonic_deploy/policy/release/model_encoder.onnx
gear_sonic_deploy/policy/release/model_decoder.onnx
gear_sonic_deploy/policy/release/observation_config.yaml
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

### 4. Build deployment stack

```bash
cd gear_sonic_deploy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
source scripts/setup_env.sh
just build
cd ..
```

### 5. Install MuJoCo sim env

```bash
bash install_scripts/install_mujoco_sim.sh
```

Verify:

```bash
source .venv_sim/bin/activate
python -c "import mujoco; print(mujoco.__version__)"
deactivate
```

### 6. Run sim2sim

Terminal 1:

```bash
cd GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Terminal 2:

```bash
cd GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim
```

Controls:

```text
]  start policy
9  drop robot to ground in MuJoCo viewer
T  play current reference motion
N  next motion
P  previous motion
R  restart current motion
O  stop control and exit
```

Pass criteria:

- MuJoCo viewer starts.
- Deploy binary starts.
- Policy loads.
- `]` starts policy.
- `T` makes robot track motion.
- No TensorRT version warning.

Fail criteria:

- ONNX/TensorRT engine build fails.
- deploy process cannot connect to sim.
- robot does not move after control start.
- wrong TensorRT version.

## L2: Training Smoke With Sample Data

Only do after L1 passes.

### 1. Prepare Isaac Lab training env

Training official path uses Isaac Lab.

Requirements:

```text
Ubuntu 22.04+
Python 3.11
CUDA 12.x
Isaac Lab 2.3+
```

Verify:

```bash
python -c "import isaaclab; print(isaaclab.__version__)"
```

Stop if Isaac Lab fails due to Isaac Sim, RTX, Vulkan, or Kit startup.

### 2. Install training deps

```bash
pip install -e "gear_sonic/[training]"
```

### 3. Download sample data + training checkpoint

```bash
pip install "huggingface_hub[cli]"
hf download nvidia/GEAR-SONIC \
  --include "sample_data/*" \
  --include "sonic_release/*" \
  --local-dir .
```

### 4. Check env

```bash
python check_environment.py --training
```

### 5. Run 5-iteration smoke

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=16 headless=True \
  manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  ++algo.config.num_learning_iterations=5
```

Pass criteria:

- Training starts.
- Metrics print.
- No Isaac Sim / RTX / Vulkan / Kit startup failure.

Fail criteria:

- Isaac Lab cannot start on H200.
- Missing sample motion data.
- Hydra config path mismatch.
- Training does not reach metric print.

## L3: Finetune Smoke

Only after L2 passes.

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=512 headless=True \
  manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  ++algo.config.num_learning_iterations=100
```

Scale `num_envs` upward only after memory/throughput are recorded.

## Record Result

Create or update:

```text
.agent/task/task002-sonic-mujoco-smoke/run_h200_notes.md
```

Template:

```md
# H200 Run Notes

## Machine

- Host:
- GPU:
- Driver:
- CUDA:
- Python:
- TensorRT:
- Commit:

## L1 MuJoCo Sim2Sim

- Result:
- Commands:
- Logs:
- Robot moved:
- Failure reason:

## L2 Training Smoke

- Result:
- Isaac Lab version:
- Command:
- Metrics seen:
- Failure reason:

## L3 Finetune Smoke

- Result:
- num_envs:
- steps/sec:
- GPU memory:
- Failure reason:

## Decisions

- SONIC official sim2sim:
- SONIC training on H200:
- Next task:
```

