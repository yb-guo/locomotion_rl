# GEAR-SONIC Plan

GEAR-SONIC is treated as the first external reproduction target.

## Official Path First

Start with the official repository and checkpoint:

- Upstream repo: `NVlabs/GR00T-WholeBodyControl`
- Checkpoint source: `nvidia/GEAR-SONIC`
- First validation: MuJoCo sim2sim
- Training validation: official headless training smoke test

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

Isaac Lab training may be attempted once, with a small `num_envs` and tiny
iteration count. If it fails due to Isaac Sim / RTX / Vulkan startup, stop that
route and keep training experiments in Genesis or MuJoCo.

