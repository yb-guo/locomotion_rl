# Project

本项目当前主力是单卡 RTX 5060 Ti（16 GB）的 humanoid locomotion RL 研究
仓库。`h200_locomotion_lab` 是历史包名，暂不改名；H200 只保留为关闭状态的
可选 profile。

核心方向：

- 跑通 NVIDIA GEAR-SONIC 的官方 MuJoCo sim2sim / deployment loop。
- 在 RTX 5060 Ti 上验证 MuJoCo/MJLab 的 headless smoke 和训练闭环。
- 暂停 H200 训练与部署；除非显式重新启用 profile，不启动任何 H200 job。
- 用 Genesis / MuJoCo / PyTorch 搭建可持续迭代的 RL 训练闭环。
- 小规模复现 LocoFormer 的核心思想：多 embodiment、长上下文 policy、在线适应。

本项目不是一开始追求 full-scale 论文复现，而是先建立：

- 稳定环境。
- 可重复 smoke test。
- 清楚的 agent / policy / simulator 边界。
- 可比较的 RL baseline。
- 后续能继续扩展的任务系统。

## Current Code Boundary

- `src/h200_locomotion_lab/core`: framework-neutral RL contracts.
- `src/h200_locomotion_lab/tasks`: MDP semantics: observations, actions,
  rewards, resets, termination, and task metrics.
- `src/h200_locomotion_lab/policies`: task-independent action generators.
- `src/h200_locomotion_lab/algorithms`: task-independent update rules.
- `src/h200_locomotion_lab/experiments`: the only composition root.
- `src/h200_locomotion_lab/envs`: simulator/backend adapters only.
- `configs/{tasks,policies,algorithms,experiments}`: separately owned configs.
- `.agent`: planning, task decomposition, review, project memory.

Historical `agents/` and `training/` modules remain compatibility paths while
experiments migrate incrementally. The normative dependency rules and migration
map are in `component_architecture.md`.

## Main Output

- SONIC official sim2sim smoke result.
- RTX 5060 Ti simulator compatibility report.
- Genesis G1 locomotion baseline.
- Minimal LocoFormer-style policy reproduction.
- Experiment logs with clear failure modes and hardware assumptions.

## Non-Goals

- Do not depend on Isaac Sim GUI on H200.
- Do not select checkpoints or size the default run for H200-only throughput.
- Do not claim full SONIC reproduction before official sim2sim and training smoke tests pass.
- Do not claim LocoFormer reproduction before a small controlled benchmark improves over baseline.
- Do not download checkpoints, datasets, simulator assets, or robot assets without explicit request.
