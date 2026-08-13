# H200 Strategy

H100/H200 are CUDA training accelerators. They are not RTX GPUs. The current
user-selected execution target is local RTX 4090, and training should use
MuJoCo only unless that route is explicitly reopened.

## Default Split

| Runtime | Role |
| --- | --- |
| MuJoCo | Official GEAR-SONIC sim2sim and fast sanity checks. |
| Genesis | Historical/optional adapter code only; not a current training dependency or gate. |
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

Move the experiment to MuJoCo.

## Smoke Test Order

1. Python package import and agent inventory.
2. Official SONIC MuJoCo sim2sim.
3. MuJoCo import/compile/step smoke.
4. Optional SONIC Isaac Lab headless training smoke only if explicitly requested.
5. Small PPO baseline.
6. Transformer policy ablation.
