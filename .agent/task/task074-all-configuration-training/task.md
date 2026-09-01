# Task074 — All-configuration locomotion training and testing

状态：**blocked / waiting_for_Task073_frozen_18_case_handoff**。

## 目标与边界

消费 Task073 冻结的 18 个 contact-aware RobotAssets，对全部 center、wheel 和 candidate 构型依次做
from-scratch training、fixed-command evaluation、contact/gait diagnosis、视频与独立 verifier。Task072
003c 的 G1 checkpoint 不能复用；Task074 必须对由通用生成器重新产出的 G1 再从随机初始化训练，证明
资产重构没有回归。

所有训练预算统一使用累计 environment transitions：

`total_transitions = num_parallel_envs * rollout_steps_per_env * completed_updates`

不得以 iteration/update count 比较不同并行环境的 run。每个 case 先通过 RTX 5060 Ti capacity smoke
冻结并行环境数，再由预注册 transition budget 反推 updates。禁止 Task048 checkpoint、外部下载、H200
和训练中临时改写 Task073 asset/contact/stance。

## Denominator

- Tier A center：G1、PM01、Spot、Go2、Lite3；
- Tier B wheel：wheel G1、PM01、Spot、Go2、Lite3；
- Tier C candidate：AgiBot X1、AgiBot X2 Ultra、EngineAI T800、T800Pro、LimX HU_D04、
  Booster T1-23、Booster T1-29、RobotEra STAR1。

## Subtasks

1. `001-training-matrix-and-transition-budgets.md`
2. `002-tier-a-center-training.md`
3. `003-tier-b-wheel-training.md`
4. `004-tier-c-candidate-training.md`
5. `005-fleet-eval-render-verifier.md`

## Route

严格按 `Task073 handoff -> 001 -> 002 -> 003 -> 004 -> 005` 执行。每个 case 都先 smoke、再按累计
transitions 分段训练；失败 case 保留在 denominator。Tier A 未通过不得进入 Tier B，Tier B 未通过不得
进入 Tier C。训练可在 checkpoint 已通过完整 case gate 时停止，但不能因并行度较低而减少总 transitions。

## Log

- 2026-08-30：根据用户纠正，把正式全构型训练放到 Task074，位于 Task073 资产迁移之后。当前没有
  Task073 handoff，没有启动任何训练。

## Review

只有 18/18 case 都绑定 exact Task073 asset/contact/stance lineage，并有 from-scratch progression、
数值 gate、接触/运动模式 gate、视频和独立 verifier 证据，Task074 才能 passed。compile、stance 或
one-update smoke 不能代替 locomotion training proof。
