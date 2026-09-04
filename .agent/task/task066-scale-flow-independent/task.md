# Task066 — 扩规模与 Flow-Matching 独立实验

## Route

Task065 通过后，将 structural blueprint 扩到数百/数千、physical instances
扩到约 100k，再加入 rough terrain 和 pushes。MIP/JiT/flow matching 不改变
TaskFamily、45D schema、split 或验收指标。

## Log

- 2026-08-19：排队，依赖 Task065。

## Review

只有有真实 tractable `log_prob` 才接 PPO；否则使用 advantage/Q-weighted flow
matching。所有新算法必须与 PPO/GRU/TXL 做同一 split 的对比。
