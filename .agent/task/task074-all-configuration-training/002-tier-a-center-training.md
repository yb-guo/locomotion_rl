# 002 — Tier A center training

状态：**blocked / waiting_for_001**。

## Route

1. 按 G1、PM01、Spot、Go2、Lite3 顺序从随机初始化训练；每个 case 只消费自己的 Task073 profile。
2. 先做 smoke，再按 001 的累计 transition checkpoints 训练和 screen；未达到 walking gate 的早期
   checkpoint 只记 progression。
3. G1 必须重新训练并达到不低于 Task072 003c 的 fixed-command walking/contact/video gate，不能复用
   003c checkpoint。
4. biped/quadruped 使用各自 logical-terminal gait gate；不得用同一相位模板强套所有中心构型。
5. 任一 center 在最大预算内失败即保留 evidence 并阻塞 Tier B，不删除 denominator。

## Log

- 2026-08-30：合同已列出；尚未训练 Tier A。

## Review

通过条件：5/5 center 都有 from-scratch transition progression 和完整 walking/eval/video/verifier pass。
