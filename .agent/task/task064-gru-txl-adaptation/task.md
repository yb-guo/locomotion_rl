# Task064 — GRU 到 Transformer-XL

## Route

先用 GRU 验证 trial/context reset、padding 和 loss mask，再训练 canonical TXL：
6 layers、hidden 256、8 heads、128 segment/memory。比较 shared MLP、GRU、TXL、
reset-memory baseline 和 explicit morphology conditioning。

## Log

- 2026-08-19：排队，依赖 Task063。

## Review

在线 motor change 后至少 70% rollout 在 2 秒内恢复正向运动且不跌倒；TXL 在
5 秒 adaptation 后相对 reset-memory baseline 的 normalized return 提升 ≥10%，
paired multi-seed bootstrap CI 排除零提升。
