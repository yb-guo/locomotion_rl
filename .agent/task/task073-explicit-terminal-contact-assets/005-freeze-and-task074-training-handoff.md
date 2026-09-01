# 005 — Freeze and Task074 training handoff

状态：**blocked / waiting_for_004_18_of_18**。

## Route

1. 冻结 18-case RobotAsset、contact profile、source mapping、stance、environment adapter、compiled
   audit、admission artifact、source commit 和 lockfile SHA。
2. 生成唯一 handoff matrix；每个 case 明确 terminal/primitive count、profile provenance、Task074
   training id、allowed material randomization 与禁止变化字段。
3. Task074 只能消费 matrix 中的 frozen asset；训练时不得临时重写 contact shape、stance 或 grouping。
4. 训练/eval artifact 必须反向引用 Task073 asset/contact/stance SHA；asset drift 使 run 失效。
5. 只有 handoff denominator 为 18/18 且 Task074 contract 已绑定，才发布
   `task074_training_admission=true`。

## Log

- 2026-08-30：合同已列出；当前没有 frozen handoff。

## Review

通过条件：存在唯一 18-case frozen handoff，Task074 能 fail closed 拒绝错 asset、contact、stance 或
adapter。正式训练结果只能在 Task074 记录。
