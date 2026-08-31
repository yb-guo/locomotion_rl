# 003b — G1 MJLab terminal-contact alignment

状态：**passed / alignment_only / training_unblocked_for_003c_only**。

Owner：G1 bound asset、logical-foot contact contract、MuJoCo environment contact aggregation 与
static stance owner。

本 subtask 只把当前 anonymous G1 的脚底碰撞对齐到本地 MJLab G1 示例；不修改质量、COM、惯量、
关节、actuator、电机参数、policy、reward 或 PPO，也不运行任何 optimizer step。现有 Task070、
Task071 与 Task072 artifacts 全部保持 immutable；新结果必须使用新的 contact-profile、bound-asset、
stance 和 artifact version，不能覆盖 `official_sim_physics_overlay_v1` 或 nominal_v3/v4。

冻结输入：

- parent bound G1 XML SHA-256：
  `c622754f2bdd01f68877873f6dfb70e55b37c29c8ba8c1cd052352b41072066d`；
- 本地 MJLab checkout commit：
  `1425b15f73bd4095f0df53709d7c389c3eb9e790`；
- MJLab G1 XML SHA-256：
  `56539bc76eadb05dd439c47de94df52130ea8fa243d08bdddd9cbc32dd4c78a0`；
- contact profile id：`mjlab_g1_7capsule_v1`。

禁止 Task048 checkpoint、外部下载、H200、训练、resume 和对旧 artifact 的原地修改。

## Route

1. **R0 — preserve parent and declare the only allowed asset delta**
   - 记录 parent XML、Task071 overlay、descriptor、stance 和 contact geometry 的 raw SHA；
   - 编译确认 parent 为每个 logical foot 一个 box、`condim=3`、`priority=0`；
   - allowlist 只允许 terminal contact geoms、logical-foot grouping、foot reference site，以及由新接触
     高度必然导致的 stance solution/root height 变化。
2. **R1 — emit `mjlab_g1_7capsule_v1`**
   - 在左右 anonymous ankle-roll body 下，各生成 7 个匿名 capsule；删除该 variant 的旧大 box，
     二者不得叠加；
   - 使用 MJLab source body-local `fromto` 和 `radius=0.01 m`，经已登记的 source→anonymous local
     frame transform 与对应 morphology scale 变换；
   - 编译后的 capsule 必须为 `condim=6`、`priority=1`，接触 bit 与 nominal friction 必须显式记录；
   - collision primitive 使用零密度/不参与惯性推导，body 的显式 inertial 保持 parent 值。
3. **R2 — logical foot aggregation**
   - contact contract 表达 `left_foot/right_foot -> tuple[geom] + reference_site + material`；
   - environment 对同一 logical foot 的任一 geom 接地取 `contact=any(...)`，normal force 求和；
   - airtime/touchdown 必须按 logical foot 维护，输出仍精确为 2 feet，不能把 14 个 geoms 当成
     14 只脚；
   - foot height、planar/vertical velocity 与 slip 使用 logical-foot reference site/body，不对 7 个
     geoms 重复计权；non-foot contact 必须把组内全部 geom 识别为 foot。
4. **R3 — generic support geometry for stance**
   - static stance/support code 支持 sphere、capsule、box，不再只读取 box 四角；
   - support points、sole height、support polygon、contact load 与 center of pressure 均先按 geom
     计算，再按 logical foot 聚合；
   - 新 contact profile 必须重新求 stance；旧 R1 stance solution/hash 不得复用或改写。
5. **R4 — compiled invariants and no-update admission**
   - parent 与 candidate 的 `nq/nv/nu`、body/joint/actuator order、joint axis/range、body mass、COM、
     inertia、actuator transmission/limits/PD 必须逐项相等；
   - candidate 必须为 `2 logical feet / 14 foot collision geoms / 0 legacy foot boxes`；
   - reset、finite forward、logical contact shape、stance/load、non-foot classification 和 no-update
     parameter delta 必须通过；任何 optimizer `step()` 调用都使本 subtask 失败；
   - 输出新的 bound XML、contact profile、stance、mapping、compiled audit 和 verifier artifact，全部
     互绑 SHA。

## Log

- 2026-08-30：只读比较确认当前 Task072 bound G1 为每脚一个 box，MJLab 示例为每脚 7 个
  capsule；nominal friction 相同，但当前为 `condim=3/priority=0`，MJLab 为
  `condim=6/priority=1`。现有 environment 与 stance solver 都假定一脚一个 box。
- 2026-08-30：根据用户要求建立本 alignment-only contract；尚未修改实现、生成新资产、求新 stance
  或运行 smoke/training。
- 2026-08-30：完成 `mjlab_g1_7capsule_v1` alignment-only 资产。新 XML：
  `.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/contact_alignment/mjlab_g1_7capsule_v1/unitree_g1_mjlab_g1_7capsule_v1.xml`
  SHA `bd06eff122d35044018f3867a9d227346af4df847a8c56ce1df3f4cd074faf36`；contact profile SHA
  `304e464577636d45322e98547db6ef8557585c2bd1c3d254ee898e440b41156d`、payload SHA
  `2523a11840ae28cc1d2402c02d341a4965819925814ed986315e1661e351d857`。原 precision-rounding
  candidate 已移入
  `artifacts/contact_alignment/mjlab_g1_7capsule_v1_rejected_invariant_rounding_20260830T000000Z/`，
  未覆盖旧 Task071/Task072 asset。
  生成命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py align-contact --output-root .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/contact_alignment/mjlab_g1_7capsule_v1`。
- 2026-08-30：机器验收通过。`compiled_invariant_diff.json` SHA
  `11f2911f3eab726bb877f00397ca64924a47635af1ad0b464250c99c2febe129`，checks
  `nq/nv/nu/body_order/joint_order/actuator_order/joint_axis_range/body_mass_com_inertia/actuator_transmission_limits_pd`
  全为 true；`compiled_contact_geometry.json` SHA
  `0ec19a7335fab7ed528761ec407e52c6525c7f0cbc73c5c608fc2fbf5cbc3c17`，`2 logical feet / 14 capsule
  geoms / 0 legacy foot boxes`，`condim=6/priority=1` 全为 true。
- 2026-08-30：新 stance 已按 contact-aligned XML 重算。`stance_solution.json` SHA
  `3671c9335e58591a5d9252c9aa38d02689579c222ada81432851b9d327030e19`、payload SHA
  `7218694acd65774e253a6d4d6900ed302fd6a735817a8fec41b26acb678f9e2b`，绑定
  `model_xml_sha256=bd06eff122d35044018f3867a9d227346af4df847a8c56ce1df3f4cd074faf36`、
  `base_height=0.8534139176251306`、solver contract
  `whole_body_static_stance_v3_actual_dynamics_feedforward`。
- 2026-08-30：environment logical-foot aggregation 与 generic stance support 已接线并通过 focused
  tests：`tests/test_whole_body_extended.py::test_stance_support_points_cover_box_capsule_and_sphere`
  和
  `tests/test_task072_locomotion_proof.py::test_contact_aligned_g1_groups_14_capsules_as_two_logical_feet`
  返回 `2 passed`。`py_compile` 覆盖 task-local CLI、MJLab runner、environment、stance、actual-stance
  与测试文件，返回 passed。
- 2026-08-30：no-update smoke artifact
  `.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/contact_alignment/mjlab_g1_7capsule_v1/no_update_smoke.json`
  SHA `670b6a4ceccbee2a20d2c44c7d1314962e511f4d89b0b306bc97f550f6611f0c`，`optimizer_step_calls=0`，
  `foot_contact_shape=[2,2]`，`touchdown_shape=[2,2]`，`foot_normal_force_shape=[2,2]`，
  `non_foot_contact_fraction=[0,0]`，finite true。
- 2026-08-30：汇总 gate
  `.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/contact_alignment/mjlab_g1_7capsule_v1/alignment_gate.json`
  SHA `6cc7b105362ac09a2fd2ababa22050da142cb824ed3d96970972b5c3caf5d8af`，`passed=true`，
  `claim_boundary.optimizer_steps_allowed=false`、`training_started=false`。独立 verifier
  `contact_alignment_verifier_after_eval_config.json` SHA
  `c47304991b595e4f5ef746a6901ccfc365f8318dc97600e8aa72c2cfd5c39cfe`，checks
  `asset_reproducible/asset_sha_bound/contact_geometry/invariants/no_update/stance_bound/stance_hash`
  全为 true。
  验证命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py verify-contact-alignment --output-root .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/contact_alignment/mjlab_g1_7capsule_v1 --output .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/contact_alignment/mjlab_g1_7capsule_v1/contact_alignment_verifier_after_eval_config.json`。

## Review

状态：**passed**。新 variant 的唯一物理差异已由 terminal-contact profile 限定；logical-foot 输出仍为
2；全部非接触物理字段逐项不变；新 stance 与 no-update gate 均通过。

`003b passed` 只授权执行 `003c`；不证明 locomotion，也不改变旧 Task072 `not_passed` 结论。
