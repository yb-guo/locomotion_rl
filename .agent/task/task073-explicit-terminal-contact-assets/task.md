# Task073 — Explicit terminal-contact asset generation and fleet migration

状态：**blocked / waiting_for_Task072_003c_walk_and_004_freeze**。

## 目标与边界

只有 contact-aligned G1 已经真正走起来并由 Task072 004 冻结后，才把试点中验证过的 contact 表达
整理成通用 RobotAsset contract，并迁移后续训练 denominator 中全部 18 个构型。每个 logical
terminal 都必须声明自己的 contact profile；G1 的 7-capsule 只是一个已验证 profile，不得强制复制
给其他机器人。

统一资产边界为：

```text
RobotAsset
├── structure          body / joint / axis
├── inertial           mass / COM / inertia
├── actuation          motor / limit / PD
├── terminal_contact   logical terminal -> contact primitives
└── visual             mesh or anonymous primitives
```

示例：

```text
left_foot
  body: anon_limb0_ankle_roll_link
  geoms: capsule_00 ... capsule_06
  reference_site: [...]
  material: condim / priority / friction
```

Task073 负责 contract、source intake、资产生成、stance/contact/no-update/one-update admission 和 frozen
handoff；正式的 18 构型 locomotion training/eval 由 Task074 消费 frozen assets 执行。

本任务禁止 Task048 checkpoint、外部下载、H200 和长训练。source evidence 不足的 case 不得伪装成
source-aligned：可以使用显式登记的 generated contact prior，但必须记录 provenance 和 claim boundary。

## Denominator

- center：G1、PM01、Spot、Go2、Lite3；
- wheel：wheel G1、PM01、Spot、Go2、Lite3；
- candidate：AgiBot X1、AgiBot X2 Ultra、EngineAI T800、T800Pro、LimX HU_D04、
  Booster T1-23、Booster T1-29、RobotEra STAR1。

## Subtasks

1. `001-robot-asset-terminal-contact-contract.md`
2. `002-source-contact-profile-intake.md`
3. `003-fleet-asset-regeneration.md`
4. `004-contact-aware-env-stance-admission.md`
5. `005-freeze-and-task074-training-handoff.md`

## Route

严格按 `Task072 003c walk -> Task072 004 freeze -> 001 -> 002 -> 003 -> 004 -> 005 -> Task074`
执行。不得一边迁移 contact contract 一边长训，也不得用 G1 试点成功替代 18-case 资产 denominator。
任何 case 缺 profile、编译失败、stance 失败、logical-terminal shape 错误或 provenance 不闭合，都保留
在 denominator 并阻塞 Task074。

## Log

- 2026-08-30：根据用户纠正，把原先倒置的 Task074→Task073 链路重排为 Task073 资产迁移、Task074
  正式训练。本任务尚未启动；G1 003b/003c/004 尚未通过，没有迁移资产或运行训练。

## Review

Task073 只有在 18/18 case 都使用显式 terminal-contact profile、旧 implicit-footpad 不再作为新 contract
隐藏默认、非接触物理字段保持 lineage、stance/no-update/one-update gates 全部通过并发布 frozen
handoff 后才能 passed。Task073 passed 只表示资产可以进入 Task074，不表示 18/18 已学会 locomotion。
