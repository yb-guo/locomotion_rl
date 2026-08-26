# Task059 — RTX 5060 Ti 主力开发切换

## Route

将硬件决策从 H200-first 切换为单卡 RTX 5060 Ti（16 GB）优先：更新长期
项目文档、Agent 约束、whole-body 主配置和 Unitree baseline 元数据。保留
`whole_body_h200.yaml` 作为关闭的历史 profile，不删除历史任务证据，也不
改变 Python 包名。

## Log

- 2026-08-19：新增 `configs/experiments/whole_body_rtx5060ti.yaml`，默认
  4 topology shards × 64 env（256 env），扩展目标 1024 env。
- 2026-08-19：将 `whole_body_h200.yaml` 标记为 `status: disabled`，并写明
  使用 RTX 配置的迁移路径。
- 2026-08-19：更新 README、project/strategy/architecture 文档、AGENTS.md
  和包描述；历史 H200 task logs 保留为 provenance。
- 2026-08-19：本机 `nvidia-smi` 仅显示 `NVIDIA GeForce RTX 5060 Ti`，未发现
  H200 设备或本地 H200 runtime 可关闭。

## Review

验收：

- 默认 whole-body experiment 是 RTX 5060 Ti profile。
- H200 profile 明确 disabled，不会被主线配置误选。
- 历史 H200 文件和结果不删除、不改写。
- `pytest`、`git diff --check` 和 agent inspection 通过。

状态：implemented; RTX 5060 Ti 是后续主力开发和训练目标。
