# Task060 — 2000 程序化本体可用性 Gate

## Route

把“本体能否进入训练环境”与“策略是否已经学会走路”分成两个独立验收。
对每个二足/四足生成体执行：确定性重放、MJCF 编译、nominal reset、193D
observation 编码、45-slot mask/gather/scatter、actuator-to-joint 一对一
映射、target-joint generalized-force influence，以及 2 秒 bounded action
trace。

## Log

- 2026-08-19：新增
  `src/h200_locomotion_lab/tools/whole_body_usability_gate.py`。
- 每条记录保存 structural hash、actuator 数、reset 误差、最小 target-force
  变化、off-target force、最大 qpos/qvel、最小 contact distance 和失败原因。
- passive generator smoke 仍保留；它不再被解释为完整 usability gate。

## Review

通过条件：2000/2000 记录 `passed=true`，且每条记录同时满足：

- compile/reset/observation 通过；
- active mask 与 actuator 数一致；
- 每个 active slot 只映射到一个 actuator 和对应 joint；
- bounded action trace 运行 2 秒无 NaN、无状态爆炸、无严重穿模。

该 gate 只证明本体可以作为训练环境使用，不证明 policy 已经正常行走。

2026-08-19 实测结果：

- `usability_gate_2000x2s.json`：二足 1000 + 四足 1000，`passed_records=2000`，
  `failed_records=0`，2000 个 structural hash 全部唯一。
- 每个本体执行 100 个 control steps / 1000 个 physics steps（2 秒，50/500 Hz）。
- deterministic replay、MJCF compile、nominal reset、193D observation、45-slot
  mapping、actuator target influence、bounded action trace 全部通过。
- active actuator 数范围 7–27；最大 `|qpos|=3.127`、最大 `|qvel|=34.45`；
  最小 contact distance `-0.0789`，优于 `-0.25` 阈值；off-target force delta 为 0。

结论：2000 个本体已通过“可作为训练环境使用”的 gate。该结论仍不等价于
2000 个本体已经学会正常行走；行走质量 gate 留给 specialist/shared-policy 训练。
