# H200 Strategy

H100/H200 are CUDA training accelerators. They are not RTX GPUs.

## Default Split

| Runtime | Role |
| --- | --- |
| MuJoCo | Official GEAR-SONIC sim2sim and fast sanity checks. |
| Genesis | H200-friendly RL training and scalable headless experiments. |
| PyTorch | Policy, PPO, transformer, offline analysis. |
| Isaac Lab headless | Optional smoke test only. |
| Isaac Sim GUI / RTX sensors | Use an RTX workstation or cloud node, not H200. |

## Stop Rule

If an H200 Isaac Lab run fails in any of these areas, stop debugging it as the main path:

- Isaac Sim startup.
- Omniverse Kit startup.
- RTX renderer.
- Vulkan device selection.
- `carb.graphics`.

Move the experiment to Genesis or MuJoCo.

## Smoke Test Order

1. Python package import and agent inventory.
2. Official SONIC MuJoCo sim2sim.
3. Genesis import and empty scene smoke.
4. Genesis G1 reset/step smoke.
5. Optional SONIC Isaac Lab headless training smoke.
6. Small PPO baseline.
7. Transformer policy ablation.

