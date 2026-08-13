# Project

本项目目标是构建一个可在本地 RTX 4090 和训练服务器间迁移的 humanoid
locomotion RL 研究仓库。

核心方向：

- 跑通 NVIDIA GEAR-SONIC 的官方 MuJoCo sim2sim / deployment loop。
- 当前训练执行目标：本地 RTX 4090 + MuJoCo-only。
- 如果 Isaac Lab 在 H200 上因为 Isaac Sim / RTX / Vulkan / Kit 失败，立即停止该路线。
- 用 MuJoCo / PyTorch 搭建可持续迭代的 RL 训练闭环。
- 小规模复现 LocoFormer 的核心思想：多 embodiment、长上下文 policy、在线适应。

本项目不是一开始追求 full-scale 论文复现，而是先建立：

- 稳定环境。
- 可重复 smoke test。
- 清楚的 agent / policy / simulator 边界。
- 可比较的 RL baseline。
- 后续能继续扩展的任务系统。

## Current Code Boundary

- `src/h200_locomotion_lab/agents`: agent / policy decomposition.
- `src/h200_locomotion_lab/envs`: simulator adapter boundary.
- `src/h200_locomotion_lab/training`: training loop boundary.
- `configs`: experiment, agent, environment configs.
- `.agent`: planning, task decomposition, review, project memory.

## Main Output

- SONIC official sim2sim smoke result.
- H200 simulator compatibility report.
- MuJoCo G1 locomotion baseline.
- Minimal LocoFormer-style policy reproduction.
- Experiment logs with clear failure modes and hardware assumptions.

## Non-Goals

- Do not depend on Isaac Sim GUI on H200.
- Do not claim full SONIC reproduction before official sim2sim and training smoke tests pass.
- Do not claim LocoFormer reproduction before a small controlled benchmark improves over baseline.
- Do not download checkpoints, datasets, simulator assets, or robot assets without explicit request.
