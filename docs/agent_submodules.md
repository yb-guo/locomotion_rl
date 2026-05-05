# Agent Submodules

This is the working decomposition for the agent code in this project.

## LocoFormer-Style Agent

| Submodule | Responsibility |
| --- | --- |
| `morphology_encoder` | Encodes robot morphology, DOF layout, body graph, and actuator metadata. |
| `proprio_tokenizer` | Converts joint positions, velocities, IMU, contacts, and commands into tokens. |
| `motion_context_encoder` | Encodes recent reference motion and task context. |
| `transformer_policy` | Long-context policy core. Start small before scaling context and robots. |
| `actor_critic_heads` | Produces action distribution and value estimate for PPO-style training. |
| `adaptation_buffer` | Maintains recent rollout context for online adaptation. |

## GEAR-SONIC Adapter

| Submodule | Responsibility |
| --- | --- |
| `observation_bridge` | Maps simulator state to SONIC deployment/training observations. |
| `reference_motion_bridge` | Loads or streams reference motion into SONIC-compatible input format. |
| `policy_runtime` | Calls PyTorch, ONNX, or TensorRT policy runtime. |
| `action_bridge` | Maps policy output to robot command or simulator actuator target. |

## Boundary

This repo should not fork all of upstream SONIC at the start. Keep the adapter
thin until the official MuJoCo sim2sim loop and input/output shapes are known.

