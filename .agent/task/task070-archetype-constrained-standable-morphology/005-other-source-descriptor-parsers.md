# 005 — Other Source Descriptor Parsers

## Route

1. Extend the descriptor parser to the already authorized local PM01, Spot, Go2,
   and Lite3 source files; do not download new assets.
2. Emit per-source v2 descriptors with expected accounting:
   PM01 `23 -> 23`, Spot `12 -> 12`, Go2 `12 -> 12`, Lite3 `12 -> 12`.
3. Preserve source parent-child edges, joint semantic order, local axes, ranges,
   limb/module grouping, and terminal/contact attachments.
4. Add tests for PM01 elbow-yaw not being collapsed into elbow-pitch, and for
   quadruped hip ab/ad, hip flex/ext, knee order per leg.

## Log

- 2026-08-25：只读取 R0 已授权的本地 PM01、Spot、Go2、Lite3 文件；没有下载新资产。
  新增通用 URDF source-tree parser，并复用 MJCF tree parser，逐 joint 保留 XML 顺序、
  parent/child link、origin position/RPY quaternion、local axis、range、module 和 anonymous edge。
- Accounting：PM01 `23 -> 23`，Spot/Go2/Lite3 各 `12 -> 12`。PM01
  `J16/J21_ELBOW_PITCH` 与 `J17/J22_ELBOW_YAW` 保持为四条不同 source edge；因冻结的
  whole-body schema 没有 elbow-yaw slot，后者显式落到对应 arm 的 axial `wrist_yaw` slot，未折叠
  到 elbow-pitch。
- 四足每腿严格按 source 顺序映射为 `hip_roll`（hip ab/ad）、`hip_pitch`（hip flex/ext）、
  `knee_pitch`。Go2 使用 source site terminal offset，Lite3 使用 fixed ankle child offset；Spot
  URDF 没有 terminal frame，descriptor 明确记录 same-source upper-span fallback，不伪造 exact
  terminal claim。
- 验证：`tests/test_task070_morphology.py` 覆盖全部四个新 descriptor、PM01 elbow axial edge、
  四足 motor order/axis/range/module/terminal；最终 focused 结果 `18 passed`。

## Review

- Pass only when each descriptor has exact source motor accounting and branch
  coverage.
- Fail on any source motor deletion, semantic swap, ambiguous branch merge, or
  unlicensed/non-local source dependency.
- 2026-08-25 scoped review：**PASS for descriptor-parser microtask**。这不是 Task070 v2
  overall pass，也没有 stance claim。
