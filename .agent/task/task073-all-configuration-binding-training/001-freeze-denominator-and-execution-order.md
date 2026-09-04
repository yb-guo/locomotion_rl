# 001 — Freeze denominator and execution order

## Route

1. 读取 Task070 最终 18-case registry 与 Task072 G1/Go2 freeze，逐项保存 case id、family、source
   descriptor SHA、actuator count、canonical-root contract、license/provenance role 和当前 gate。
2. 固定 18-case denominator，不因 source unknown、训练失败或 adapter 不兼容删除 case。
3. 固定执行顺序：Tier A remaining 3 -> Tier B wheels 5 -> Tier C candidates 8；同一 tier 内逐 case
   完成 physical binding 与 nominal proof，不以 tier 平均值代替 individual pass。
4. 建立可机读 state machine：`registered -> physics_bound -> runtime_ready -> nominal_passed ->
   randomized_passed`；任何失败保留状态与 reason，不可跳级。
5. 将 Task072 的 G1/Go2 nominal binding/evidence 标记为只读 imported baseline；Task073 只能为其增加
   versioned randomization overlay。任何 nominal descriptor/physics/motor/action/reward identity 改变都
   必须使 imported pass 失效并回到 Task072 重跑。

## Log

- 2026-08-27：固定分母为 center 5、wheel 5、candidate 8；Task073 新增工作量为除 G1/Go2 外的
  16 个 case。

## Code implementation

新增 task-local `task073_case_registry.py`：

- module constant `CASE_ORDER` 固定为
  `unitree_g1, unitree_go2, engineai_pm01, spot_base, deeprobotics_lite3,
  unitree_g1_wheeled, engineai_pm01_wheeled, spot_base_wheeled, unitree_go2_wheeled,
  deeprobotics_lite3_wheeled, agibot_x1_serial, agibot_x2_ultra, engineai_t800,
  engineai_t800pro, limx_hu_d04, booster_t1_23, booster_t1_29, robotera_star1`；registry row order、
  `case_index` seed 派生与 final matrix 都只能用这个 0-based 顺序。
- frozen `CaseRecord` 字段固定为 `case_id, family, tier, source_dir, xml_path/xml_sha256,
  descriptor_path/descriptor_sha256, manifest_path/manifest_sha256, actuator_count,
  canonical_root_contract, license_role, source_allowlist_sha256, quantitative_source_status,
  physics_status, motor_status, adapter_version, imported_read_only, state, blocked_reason`；unknown value
  用 JSON `null`，不能省略 key。G1/Go2 的 `imported_read_only=true` 冻结 nominal identity，但允许在
  007 用新 overlay evidence 做唯一的 `nominal_passed -> randomized_passed` state transition。
- `build_registry(task070_root, task072_freeze, source_allowlist)` 必须构造 exact 18 id；G1/Go2 从
  Task072 manifest 导入 `nominal_passed` 只读状态，其余 16 的 state 始终从 `registered` 开始。当前
  缺口只写 `blocked_reason`/`physics_status`，不创建另一个 `blocked` state。
- `transition(case_id, expected_state, next_state, evidence_sha)` 只允许
  `registered -> physics_bound -> runtime_ready -> nominal_passed -> randomized_passed`；CAS-style
  expected state 不匹配、跳级、`blocked_reason != null` 或缺 evidence SHA 全部拒绝。
- `set_blocked_reason(case_id, reason, evidence_sha)` 不改变 state；
  `clear_blocked_reason(case_id, expected_reason, resolution_evidence_sha)` 只有在对应 validator 已通过、
  source allowlist 已更新为新 version 且 expected reason 精确匹配时才清空。这样 unknown 可在后续获得
  合法 source 后恢复，而不能绕过 gate。

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_case_registry.py" build \
  --task070-root .agent/task/task070-archetype-constrained-standable-morphology/artifacts/preview_task070_v2_descriptor_driven_attempt010 \
  --task072-freeze .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v2/freeze/task072_freeze_manifest.json \
  --source-allowlist "$TASK073/task073_physical_source_allowlist_v1.json" \
  --output "$TASK073/artifacts/v1/registry.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_case_registry.py" validate \
  --registry "$TASK073/artifacts/v1/registry.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q tests/test_task073_registry.py
```

测试覆盖 18 unique、`5+5+8` denominator、输入 SHA 漂移、G1/Go2 read-only、非法跳级、blocked 时拒绝
transition，以及带 resolution evidence 的可恢复 clear。缺 case、Task072 未 pass、source allowlist
未闭合或 attempt010 混入其他 attempt 时停止 002。

## Review

通过条件：registry 与 Task070/Task072 SHA 一致、18 个 case 无缺失/重复、G1/Go2 imported baseline
只读、state transition fail closed，且 Task072 未通过时所有 Task073 execution state 保持未开始。
