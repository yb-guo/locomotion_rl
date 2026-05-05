# H200 Strategy

H100/H200 cards are training accelerators. They do not have RT cores, so the
full Isaac Sim/Omniverse/RTX stack is not the right default runtime.

## Recommended Split

| Machine | Use |
| --- | --- |
| H200 server | PyTorch RL, MuJoCo sim2sim, Genesis training, policy ablations. |
| RTX workstation or cloud node | Isaac Sim visualization, USD import/export, RTX sensors, videos. |

## Project Defaults

- MuJoCo for official GEAR-SONIC sim2sim validation.
- Genesis for scalable headless RL experiments.
- Optional Isaac Lab smoke test only when the environment is known to work.

## Stop Rule For Isaac Lab On H200

If the first headless smoke test fails in Isaac Sim, Vulkan, RTX, or Kit startup,
stop debugging that path and move the experiment to Genesis or MuJoCo.

