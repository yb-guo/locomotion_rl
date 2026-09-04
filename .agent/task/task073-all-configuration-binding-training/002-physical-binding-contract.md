# 002 — Per-case physical binding contract

## Route

本 subtask 的新绑定对象严格限定为 Task073 remaining 16。Task072 的 G1/Go2 nominal binding 只能做
SHA/provenance 审计和 schema 对照，不得修改；若发现必须修改，则停止 Task073、废弃对应 imported
baseline，并回到 Task072 在新 lineage 上重跑两者。

1. 对每个 body/link 绑定质量、local COM 和正定惯量；保持总质量、左右对称/非对称来源与
   geometry scale 一致，不能只设置 total mass 或让 COM 随 link scale 漂移。
2. 对 load-bearing collision geom 绑定有 provenance 的 friction/contact tuple，并明确 foot、wheel、
   body 的不同接触角色；统一 placeholder 不能作为真实 quantitative prior。
3. 对每个 generalized joint 绑定唯一 transmission group，再绑定同一 source-supported coherent
   motor tuple：control mode、effort、velocity、gear/effective reduction、`kp/kd`、armature、friction
   与 delay。未知字段保持 null；禁止跨型号拼接或彼此独立无条件采样。
4. 从同一 transmission-resolved tuple 推导 per-slot action scale 与 actuator limits，保存单位、frame、
   source/config SHA 和 source-to-anonymous mapping。
5. 在训练前验证 exact actuator accounting、compile/reset、canonical root、nonpenetration、load-
   bearing contact、paired actuator response、finite energy/force 和 family-specific stance/active balance。

## Log

- Task071 已为 G1/Go2 建立较完整的 official-simulator physics prior；其余 case 不能自动继承这些
  数值，只能复用 schema 和验证方法。
- 2026-08-27：尚未执行其余 16 个 case 的 complete physical binding。

## Code implementation

新增 production module `src/h200_locomotion_lab/robots/physical_binding.py`，但不修改
`archetype_morphology.py` 的 frozen descriptor 或 `procedural_morphology.compile_mjcf()` 默认输出：

- `LinkPhysicsBinding(link, mass_kg, com_m, inertia_full_kg_m2,
  friction_3, contact_role, source_path, source_sha256)`；
- `JointMotorBinding(slot, control_mode, transmission_id, effort_limit, velocity_limit, kp, kd,
  armature, frictionloss, damping, action_amplitude, source_path, source_sha256)`；
- `PhysicalBinding(case_id, source_xml_sha256, links, joints, schema_version)`；
- `apply_physical_binding(source_xml, binding) -> bound_xml` 只产生 derived XML overlay，写入 inertial、
  geom friction、joint dynamics 和 actuator parameters；inertia 用 MuJoCo `fullinertia` 六分量写出，
  source XML 保持不变；
- `validate_physical_binding()` 检查每个 link/joint 一一覆盖、mass/effort finite positive、inertia 三轴
  positive、COM/单位、actuator count、source SHA 与 control mode。

每个 `LinkPhysicsBinding` 另保存 `source_body, frame_quat_wxyz, geometry_transform_3x3`。transform 只能由
Task070 source-to-anonymous frame 与该 link 已实现的 primitive scale 求得，不能从名字猜。对 source
inertial `(m,c,inertia_diag,inertia_quat)`，先把 diagonal inertia 按 quaternion 旋到 source body frame
得到 `I`；nominal named-center mass 保持 `m`，再令 source second-moment matrix
`J = 0.5*trace(I)*Identity - I`，计算 `c' = A*c`、`J' = A*J*A^T`、
`I' = trace(J')*Identity - J'`；其中 `A=R*geometry_transform_3x3`。这保证 uniform scale `s` 时 COM
乘 `s`、惯量乘 `s^2`。对称化后的 `I'` 按 `(Ixx,Iyy,Izz,Ixy,Ixz,Iyz)` 写
`inertia_full_kg_m2`，避免重复 eigenvalue 导致不稳定的 principal-axis quaternion。这样不会出现
“连杆缩放了但 COM/惯量仍停在旧坐标”的漂移。若 link 没有唯一
source-body mapping、scale transform 不可逆或结果非正定，该 link 保持 unknown 并阻断 binding；不得
退回 total-mass 均分。local wheel engineering link 使用 `_compose_terminal_wheels` 已声明的 mass/geometry
和 identity source transform，并明确不是 named parity。

pipeline `bind --case` 只能从 registry、Task070 actuation stack 和
`task073_physical_source_allowlist_v1.json` 的 exact locator 组装 schema；不允许运行时搜索“相似 config”
或自行选 source。每个解析值都保存 allowlist record SHA、raw source SHA、locator 和单位；locator 未命中
恰好一个值即视为 unknown。任何所需 quantitative field unknown 时仍输出 `physical_binding.json`，但 `binding_passed=false` 并保持
`blocked_reason`，不能生成假数值或进入 runtime smoke。成功时编译 derived XML，执行 exact accounting、
canonical root、reset/nonpenetration/load-bearing contact 与 paired actuator response，写
`binding_verifier.json`。具体状态转换分两步：binding schema、source SHA 与 derived XML 静态验证通过后，
以 `physical_binding.json` SHA 执行 `registered -> physics_bound`；随后 pipeline 读取 `bound.xml` 文本和
SHA，调用 Task072 已有的
`WholeBodyMuJoCoShard(case.blueprint, physical=case.physical, num_envs=num_envs, config=env_config,
motor_config=motor_config, model_xml=bound_xml, model_xml_sha256=bound_xml_sha256,
stance_solution=bound_stance)`，完成全部 runtime gate 后，再以 `binding_verifier.json` SHA 执行
`physics_bound -> runtime_ready`。不得再新增第二个 XML override 入口；stance solution 必须针对
derived XML 求解并绑定同一 XML SHA。

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" bind \
  --registry "$TASK073/artifacts/v1/registry.json" --case engineai_pm01
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q tests/test_task073_binding.py
```

输出固定为 `<artifact-root>/<case>/binding/{physical_binding.json,bound.xml,
binding_verifier.json}`。测试包含 unknown fail-closed、SHA 漂移、非正惯量、跨型号 source 拒绝和
Task070 XML byte-unchanged；另覆盖 bound XML/stance SHA 不一致时 shard 构造失败，以及
`model_xml=None` 的 legacy compile path 数值不变。

## Review

通过条件：remaining 16 中每个晋级 case 都有 body-level mass/COM/inertia、contact friction、
transmission、motor tuple、action scale 和 provenance 的一一对应 artifact；G1/Go2 imported
baseline identity 未改变。任一 quantitative field unknown 或 lineage mismatch 时，该 case 保持
fail-closed，不能进入 nominal training。
