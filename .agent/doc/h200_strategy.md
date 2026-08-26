# Hardware Strategy (RTX 5060 Ti Primary)

The active development target is the local RTX 5060 Ti (16 GB). H100/H200
remain documented only as optional historical/scale-out targets and are
disabled by default.

## Active Default Split

| Runtime | Role |
| --- | --- |
| MuJoCo | Official GEAR-SONIC sim2sim and fast sanity checks. |
| MuJoCo / MJLab | Primary headless RL training and simulator development on RTX 5060 Ti. |
| PyTorch | Policy, PPO, transformer, offline analysis. |
| Genesis | Optional backend when its local CUDA install is available. |
| Isaac Lab headless | Optional smoke test only on the RTX workstation. |
| H200 | Disabled; no default training, simulator, or deployment job. |

## Stop Rule

If an optional accelerator run fails in any of these areas, stop debugging it as
the main path:

- Isaac Sim startup.
- Omniverse Kit startup.
- RTX renderer.
- Vulkan device selection.
- `carb.graphics`.

Keep the experiment on the RTX 5060 Ti MuJoCo/MJLab path.

## Smoke Test Order

1. Python package import and agent inventory.
2. Official SONIC MuJoCo sim2sim.
3. RTX 5060 Ti CUDA/MJLab smoke.
4. Small PPO baseline.
5. Transformer policy ablation.

## H200 Pause

The H200 profile is retained for reproducibility but is not an active runtime
target. Re-enabling it requires an explicit project decision and a new
hardware/throughput review.
