# 003 — Tier B wheel training

状态：**blocked / waiting_for_Tier_A_5_of_5**。

## Route

1. 按 wheel G1、PM01、Spot、Go2、Lite3 顺序训练，预算仍按累计 transitions。
2. 每个 case 绑定自己的 rolling contact、wheel axis、velocity/torque actuator 和 active-balance contract。
3. 训练与 eval 同时检查命令跟踪、跌倒、轮地接触、打滑、轮速/力矩和 active balance；foot gait gate
   不得替代 wheel gate。
4. 每个 case 产出 deterministic eval、视频和独立 verifier；失败即阻塞 Tier C。

## Log

- 2026-08-30：合同已列出；尚未训练 Tier B。

## Review

通过条件：5/5 wheel composition 有 from-scratch progression，并通过各自 rolling-contact locomotion
gate；仅能站立或轮子空转不算通过。
