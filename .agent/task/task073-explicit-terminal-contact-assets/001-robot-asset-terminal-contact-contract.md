# 001 — RobotAsset terminal-contact contract

状态：**blocked / not_started**。

## Route

1. 定义 logical terminal 对多个 collision primitives 的显式 schema；字段至少包含 terminal id、owner
   body、reference site、primitive type、local pose/fromto、size、contact bits、material 和 provenance。
2. 支持 sphere、capsule、box 与 rolling wheel primitive；schema 不编码具体机器人或固定 geom 数量。
3. `structure/inertial/actuation/terminal_contact/visual` 分层序列化并分别计算 identity；visual 变化不得
   改 physical identity，contact 变化必须改 asset/contact identity。
4. logical terminal count 是 policy/reward/sensor contract；primitive count 只属于 collision 实现。
5. 新 contract 禁止未声明的 implicit box fallback；旧 `foot_size` 仅保留为 legacy-v2 读取路径。

## Log

- 2026-08-30：合同已列出，尚未实现。

## Review

通过条件：无歧义表达 G1 7-capsule、Unitree 4-point、四足 foot、wheel rolling contact 和显式 box；
缺字段、重复 geom、未知 owner body 或 terminal/primitive 混淆必须 fail closed。
