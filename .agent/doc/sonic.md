# GEAR-SONIC Plan

GEAR-SONIC is treated as the first external reproduction target.

## Official Path First

Start with the official repository and checkpoint:

- Upstream repo: `NVlabs/GR00T-WholeBodyControl`
- Checkpoint source: `nvidia/GEAR-SONIC`
- First validation: MuJoCo sim2sim
- Training validation: official headless training smoke test
- Official docs: `https://nvlabs.github.io/GR00T-WholeBodyControl/`
- SONIC page: `https://nvlabs.github.io/GEAR-SONIC/`

## Adapter Boundary

This repo should initially implement a thin adapter only:

- `observation_bridge`
- `reference_motion_bridge`
- `policy_runtime`
- `action_bridge`

Do not fork the whole upstream SONIC code until the official input/output shape
and runtime contract are confirmed.

## H200 Rule

SONIC MuJoCo sim2sim is the priority path on H200.

Isaac Lab training may be attempted only if explicitly requested, with a small
`num_envs` and tiny iteration count. If it fails due to Isaac Sim / RTX /
Vulkan startup, stop that route and keep training experiments in MuJoCo.

## Reproduction Levels

L1: Official MuJoCo sim2sim.

- Goal: released ONNX policy + C++ deployment stack controls G1 in MuJoCo.
- Hardware: H200 OK; RTX not required.
- Output: command log, screenshot/video optional, note whether robot moves.

L2: Training checkpoint smoke.

- Goal: released `sonic_release/last.pt` starts in training/eval path.
- Hardware: H200 OK if Isaac Lab starts headless.
- Output: `check_environment.py --training` result and 5-iteration smoke log.

L3: Finetune.

- Goal: resume from released checkpoint with sample or filtered motion data.
- Hardware: single H200 for experiment/debug, not full convergence.
- Output: rewards, tracking errors, throughput, memory.

L4: Full reproduction.

- Goal: train at paper scale.
- Hardware: not single H200; official docs recommend 64+ GPUs for reasonable convergence.
- Output: deferred until compute budget exists.

## Critical Versions

- x86_64 TensorRT: `10.13` required for deployment.
- Jetson/G1 TensorRT: `10.7` required with JetPack 6.
- Training Python: `3.11`.
- Training sim: Isaac Lab `2.3+`.
- Training CUDA: `12.x`.

Wrong TensorRT version can produce wrong motion. Treat this as a hard blocker.
