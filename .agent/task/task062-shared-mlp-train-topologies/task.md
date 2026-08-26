# Task062 — 64 Train Topology Shared MLP

## Route

使用 32 个 train biped + 32 个 train quadruped topology，RolloutMux 固定二/四足
各 50% 权重；连续 physics randomization 分五阶段打开。actor 不接收 family/vendor
ID 或物理参数，reward 按质量、尺寸和 active actuator 数归一化。

## Log

- 2026-08-19：排队，等待 Task061 两个 specialist 的 quality gate。

## Review

family survival 各 ≥0.90，normalized tracking error 各 ≤0.30，heldout worst-10%
survival ≥0.75。失败优先检查 mask、reward normalization 和采样比例。
