# 001 — G1 motor-tuple-derived per-slot action scale

状态：**passed**。Owner：action-adapter owner；review owner：Task072 verifier owner。

## Route

1. 从 exact-bound G1 descriptor 读取每个 active slot 的 transmission-resolved coherent motor
   tuple，至少绑定 control mode、effective effort limit、velocity limit、`kp`、`kd`、armature、
   friction 和 provenance SHA；unknown 或不适用字段不得从别的型号猜补。
2. 对 position-controlled joint 先由同一 tuple 计算
   `motor_delta_i = 0.25 * effective_effort_i / kp_i`；并联 transmission 必须先解析为同一 joint-side
   effective tuple，不能把单个 physical motor 与 joint-space gain 混用。
3. 再从 compiled actuator range `[lower_i, upper_i]` 与 stance target `q0_i` 计算
   `margin_i=max(0.05*(upper_i-lower_i),1e-4)`、
   `neg_headroom_i=q0_i-(lower_i+margin_i)`、
   `pos_headroom_i=(upper_i-margin_i)-q0_i`。任一 headroom `<=0` 必须 fail closed；最终
   `delta_neg_i=min(motor_delta_i,neg_headroom_i)`、
   `delta_pos_i=min(motor_delta_i,pos_headroom_i)`。
4. 执行 piecewise target：`action_i<0` 时
   `q_target_i=q0_i+action_i*delta_neg_i`，否则
   `q_target_i=q0_i+action_i*delta_pos_i`。compiled-range clamp 只作防御；对 finite
   `action_i in [-1,1]`，`would_clamp` 必须为 false。inactive slot 始终为零。
5. 以最小、显式、versioned 的 action-adapter 配置接入；未选择该配置的 frozen legacy 路径输出
   必须字节级/数值级保持原行为。
6. 该 adapter 必须是 shared、tuple-driven 实现。在 G1 训练前加载 exact-bound Go2 的 12 个 tuple，
   生成其逐 slot scale，并执行 config/no-update target smoke；不得使用 G1 数值，也不得在此启动
   Go2 PPO。这样 004 freeze 时 Go2 所需 adapter 与 config 已完整存在。
7. 输出 G1/Go2 逐 slot 表，记录 joint、module、motor class、tuple、transmission、`motor_delta_i`、
   margin、正/负 headroom、正/负 amplitude、clamp diagnostics 与 source SHA；测试 tuple 或 stance/range
   改变都会进入 config/checkpoint identity，且 source mismatch
   fail closed。

## Log

- 已知旧 G1 `0.35 * half_range` 与 `0.25 * effort/kp` 的比例约为 `0.22x–9.01x`，说明统一
  range fraction 同时低估部分平衡关节、放大部分低扭矩腕关节。
- 2026-08-27：v1 已实施但因 nominal_v2 clamp 实证 rejected；R1 重开，v2 尚未实施。
- 2026-08-27：在 recovery worktree 实施 `action_amplitude_by_slot` optional path；G1 29/29 与
  Go2 12/12 motor tuple 均由 Task071/Task070 bound metadata 解析并生成
  `artifacts/nominal_v2/action_contract.json`。`smoke-action` 对 G1/Go2 no-update target smoke 均为
  passed；targeted pytest `motor_tuple or action_amplitude or legacy_scalar` 为 2 passed。
- 2026-08-27：在 recovery worktree `/home/admin1/workspace/run/locomotion_rl/task071-1`
  branch `codex/task072-bound-walk-proof` 实施 `motor_tuple_headroom_residual_v2`。命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py smoke-action
  --case unitree_g1 --case unitree_go2 --output
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/action_contract.json`。
  结果 `smoke_action_passed=true`，G1 `29/29`、Go2 `12/12` residual bounds 均生成，zero-action ctrl
  error 均为 `0`。`action_contract.json` raw SHA
  `ea50671d5614f20887ad13dda96d72425bedb0ddeda9f3561c8e0233094370f9`。两份指定测试：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python -m pytest -q
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py` 为 `28 passed`。

## Review

R1 状态：**passed**；v1 evidence 不得作为 R1 pass evidence。版本为
`motor_tuple_headroom_residual_v2`。定义 `motor_delta=0.25*effective_effort/kp`，
`safety_margin=max(0.05*(upper-lower),1e-4)`，`neg_headroom=max(q0-lower-safety_margin,0)`，
`pos_headroom=max(upper-q0-safety_margin,0)`，再取 `delta_neg=min(motor_delta,neg_headroom)`、
`delta_pos=min(motor_delta,pos_headroom)`；`action<0` 用 neg、否则用 pos，最后 compiled-range
clamp。新增 asymmetric bounds API 与 would/actual clamp diagnostics；±1 target 必须安全、正常
路径 clamp=0、legacy scalar 数值等价。任一 slot/SHA/finite/inactive-zero 失败停止 002。

通过条件已满足：G1 29/29 active motor slots 均有一一对应、可追溯且 finite positive 的动作尺度；Go2
12/12 tuple 能在同一实现中加载、生成尺度并通过 no-update target smoke。尺度只由该 slot 同一
transmission/motor tuple 与 joint safety headroom 推导；inactive、unknown、legacy 路径和 contract
hash 均通过回归。只改变一个全局 scalar 或只调 G1 颜色/几何不通过本 subtask。

历史 v1 状态：**passed**，证据为 `action_contract.json` SHA
`b0abe7b01fd66250be3d6ab846c3cdbfbf6af35e7ab611a3505de9e3698cbd01` 与上述 pytest/smoke。

## Code implementation

### Production API

在 `WholeBodyMuJoCoShardConfig` 新增 optional
`action_residual_bounds_by_slot: Mapping[str, tuple[float, float]] | None = None`。tuple 顺序固定为
`(negative_magnitude, positive_magnitude)`；mapping 必须恰好覆盖 active position slots，两个值均须
finite positive。它只与 `action_amplitude_by_slot` 互斥；`action_scale` 字段继续存在，但仅在两个
optional mapping 都为 `None` 时使用。这样不改变构造函数和 frozen legacy 默认值。

`_set_targets()` 分支顺序固定为：bounds v2 -> historical symmetric amplitude v1 -> scalar legacy。
若 bounds 与 amplitude 同时非空立即 `ValueError`。v2 按 action sign 选择 magnitude；写入 ctrl 前同时
累计 per-slot `target_would_clamp` 与 `actual_clamp`。不要改变 action mask、motor fault process、wheel
semantics 或 frozen legacy 默认值。

### Task-local resolver

在 `task072_locomotion_proof.py` 新增 frozen dataclass `MotorTuple`，字段固定为
`semantic_slot, control_mode, effective_effort, velocity_limit, kp, kd, armature, friction,
transmission_group, provenance_sha256`，并新增：

- `resolve_motor_tuples(case_id, parent_artifacts) -> tuple[MotorTuple, ...]`：从 Task071
  `official_sim_physics_overlay_v1.json` 对应 record 的 `motor_mapping` 读取 `force_range`、
  `position_kp/kd`、armature/friction 和 slot；用 Task070 actuation stack 补 transmission id 与
  velocity provenance。Task071 overlay 中 joint-side official compiled `force_range=[lo,hi]` 是 effort
  的唯一量化来源，固定计算 `effective_effort=max(abs(lo),abs(hi))`；Task070 manifest 的
  `blueprint_manifest.profile_metadata.motor_configuration` 提供 control mode/velocity hint，其
  `final_compiled.kp/kd` 必须数值等于 Task071 record 的 `position_kp/kd`，而
  `actuation_stack.coherent_motor_config` 提供 transmission/family identity。禁止反向用 Task070
  companion effort 猜 official bound effort。对每个 slot 同时保存两份 raw source record 与各自
  canonical payload SHA；
  slot/order/control mode/transmission 无法一一对齐即 fail closed。
- `derive_position_action_amplitudes(tuples) -> dict[str, float]`：v1 对每个 position slot 计算
  `0.25 * max(abs(force_range)) / position_kp`；`kp<=0`、unknown quantitative tuple 或非 position
  mode fail closed。该 v1 formula 已因 clamp 实证 rejected；R1 使用 asymmetric bounds 与 v2
  headroom formula，不再把 joint headroom 留给最终 clamp。
- `derive_position_action_bounds(tuples, stance_ctrl, actuator_ranges)`：按 Route 中的公式返回
  `dict[str, tuple[float,float]]`；slot order、stance slot 和 actuator range 必须完全一一对应，range
  或 stance 缺失、stance 不在 range、headroom 非正均 fail closed。
- `action_contract_payload()`：保存 task.md 表中的六个 Task070 frozen input path/raw SHA、四个 JSON
  canonical payload SHA，四个 Task071 parent path/raw/payload SHA，逐 slot 两侧 source record SHA、tuple、
  motor_delta、margin、headroom、asymmetric bounds、公式版本
  `motor_tuple_headroom_residual_v2` 与 payload SHA。还必须校验 Task071 `frozen_input` 的 descriptor、
  manifest、XML path/raw SHA 精确等于 Task070 表，manifest 内 descriptor/XML SHA 闭合；任一漂移不得
  只写 warning。

G1 必须得到 29/29、Go2 必须得到 12/12。新增 CLI 子命令：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
env PYTHONPATH="$PWD/src" "$TASK_PY" \
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py \
  smoke-action --case unitree_g1 --case unitree_go2 \
  --output .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/action_contract.json
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q \
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py \
  -k 'motor_tuple or residual_bounds or action_amplitude or legacy_scalar'
```

测试必须覆盖 exact slot count/order、v2 formula/headroom、±1 safe target、normal no-clamp、clamp
diagnostics、tuple/SHA 漂移拒绝、inactive zero，以及
两个 optional mapping 均为 `None` 时与修改前 scalar path 数值等价。任何失败都停止 002。
