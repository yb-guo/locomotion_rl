# 005 — Tier C variable-DoF and provenance gate

## Route

1. 对 AgiBot X1、X2 Ultra、EngineAI T800/T800Pro、LimX HU_D04、Booster T1-23/T1-29、
   RobotEra STAR1 逐 case 审计 source joint tree、actuator count、axis/range、transmission、physical
   motor count 与 coherent motor config provenance。
2. 建立 versioned variable-DoF observation/action schema 与显式 mask/adapter；不允许静默裁剪、
   合并或把 extra slots 冒充 frozen `whole_body_v1_45`。
3. 对 physical motor count 与 generalized joint count 不相等的并联或末端机构，分别记录
   transmission mapping；source 不足时保持 unknown，不猜造 joint 或 Jacobian。
4. 只有完成 002 的 mass/COM/inertia、friction、motor/action binding，并通过 runtime gate 的 case
   才能进入 006。

## Log

- 当前八个 candidate 只有 structural descriptor/anonymous witness/actuator smoke；其
  `policy_adapter_compatible=false` 与 candidate motor evidence fail-closed 尚未被本任务解决。
- 2026-08-27：尚无 candidate 晋级 nominal training。

## Code implementation

采用一个明确方案：**topology-local dynamic schema**，不建立 64-slot union，也不修改 frozen
`whole_body_v1_45`。新增 `src/h200_locomotion_lab/robots/topology_local_adapter.py`：

- `TopologyLocalSchema(case_id, semantic_slots)`；`D=len(semantic_slots)`，action dim 为 `D`，actor obs
  layout 固定为 base linear 3 + angular 3 + gravity 3 + command 3 + `q,D` + `qd,D` + previous action
  `D` + active mask `D` + trial-start 1，因此 `obs_dim=13+4D`；schema hash 覆盖 case id、source joint
  order、slot names 和 layout。
- `TopologyLocalEmbodiment` 精确实现 shard 使用的 `action_mask`、`gather_action()`、
  `scatter_joint_values()`、`gather_action_batch()`、`scatter_joint_values_batch()`、
  `encode_actor_observation()` 与 `validate_observation()`；但只在 local order 内映射，不
  padding/truncate。另暴露只读 `action_dim=D`、`obs_dim=13+4D`。
- `WholeBodyMuJoCoShard` 接受 optional `embodiment_adapter`；未传时仍构建 45-slot
  `BoundEmbodiment`。把内部硬编码 `45/193` 仅替换为 adapter 的 `action_dim/obs_dim`，默认值与 frozen
  tests 不变。
- `WholeBodyPPOConfig` 新增默认 `obs_dim=193, action_dim=45`；`WholeBodyPPOTrainer.__init__()` 用这两个
  config 值构造 `WholeBodyMLPConfig`，并以同一值校验 reset observation 和 action mask；rollout buffer
  不再引用 module constant。checkpoint manifest 把 schema hash/action dim/obs dim 纳入 lineage。

八个 candidate 的 source generalized joint order 必须原样进入 local schema；physical actuator count
通过 transmission map 单独保存，不得为了相等而增删关节。执行：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q \
  tests/test_task073_variable_schema.py tests/test_whole_body_extended.py
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" bind --case agibot_x1_serial
```

输出 `binding/variable_schema.json`，含 D、ordered slots、transmission map、layout/hash 和 checkpoint
identity。测试覆盖 23/29/31/43/55 DoF、round trip、无截断、hash drift，以及 frozen 45-slot exact
regression。缺 motor/transmission evidence 的 case 可以完成 structural schema，但仍保持 blocked，不能进 006。

## Review

通过条件：8/8 各有无歧义 actuator/transmission accounting、versioned adapter、物理 provenance 和
runtime evidence。若某 case 仍 unknown，则保留在 denominator 并标 fail；不得缩小为“可完成的子集”。
