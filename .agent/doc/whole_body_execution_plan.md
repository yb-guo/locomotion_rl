# Whole-Body 执行计划（RTX 5060 Ti 主线）

Task060 已证明 2000 个程序化本体可以进入 MuJoCo 训练环境。后续严格按
“先 specialist、再 shared、再 hidden fault、最后 OOD/扩规模”的顺序执行。

## 阶段队列

| Task | 目标 | 进入条件 | 退出条件 |
| --- | --- | --- | --- |
| 061 | 二足/四足 MLP specialist | Task060 通过 | 两个 family 的 100×10s gate 或明确诊断 |
| 062 | 64 train topology shared MLP | specialist 不因 generator/reset 失败 | family survival ≥0.90，tracking error ≤0.30 |
| 063 | hidden online motor process | shared MLP gate 通过 | clean gait 降幅 ≤10%，trace 与设定一致 |
| 064 | GRU → Transformer-XL | motor process gate 通过 | reset/sequence/KV cache 与 adaptation gate 通过 |
| 065 | heldout/OOD/named robot | 冻结 TXL checkpoint | heldout、Berkeley、ANYmal、G1、Go2 指标完成 |
| 066 | 扩规模与 flow matching 独立实验 | Task065 通过 | 仅在有真实 likelihood 时接 PPO |

## 固定运行规则

- 主力硬件：单卡 RTX 5060 Ti 16 GB；默认从 256 env 开始，测量后再扩展。
- 不为 2000 个本体训练 2000 个 policy；训练样本来自 train topology 和物理随机化。
- validation 可用于 checkpoint 选择；heldout/OOD 只能用于最终报告。
- 每个 Task 必须保存 JSON/ checkpoint/ 命令和失败诊断，不能用 smoke 代替质量 gate。
