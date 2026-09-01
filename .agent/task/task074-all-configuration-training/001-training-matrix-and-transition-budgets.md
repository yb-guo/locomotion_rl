# 001 — Training matrix and transition budgets

状态：**blocked / waiting_for_Task073_handoff**。

## Route

1. 读取唯一 Task073 18-case handoff，逐 case 绑定 asset/contact/stance/adapter SHA。
2. 在 RTX 5060 Ti 上做 capacity smoke，冻结 `num_parallel_envs`、`rollout_steps_per_env`、batch 与
   simulator backend；不同 capacity 是不同 lineage。
3. 所有 checkpoint、预算和停止条件用累计 transitions 定义；updates 由
   `ceil(target_transitions / (num_envs * rollout_steps))` 反推并记录实际 transitions。
4. G1 参考 progression 为 10M/20M/40M，最大 `63,897,600` transitions；其他 case 在训练前登记自己
   的 transition checkpoints/max budget，禁止运行后按结果改预算。
5. 固定随机初始化、command/eval、optimizer、video 和 verifier contract；checkpoint resume 只能在
   同一 lineage 内用于故障恢复，不能加载其他 case 或 Task048 权重。

## Log

- 2026-08-30：合同已列出；尚未执行 capacity smoke 或预算矩阵。

## Review

通过条件：18/18 都有完整 transition-based budget 与 RTX-safe execution config；任何只写 iterations、
缺 env/rollout 数或无法反算 transitions 的 case 均不得训练。
