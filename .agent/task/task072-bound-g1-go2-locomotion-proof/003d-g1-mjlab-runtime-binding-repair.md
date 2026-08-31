# 003d — G1 MJLab runtime binding repair

状态：**rejected / runtime_binding_mismatch_double_ground**。

003c 的旧 `proof_1024x24_seed720301` 保持 immutable；本 route 建立了新的
`mjlab_g1_7capsule_task_v2` contact asset/profile/stance。后验编译模型审计发现 frozen XML 自带
`floor`，而 Unitree-G1-Flat scene 又创建 `terrain`，因此本 route 的 runtime binding 不成立。
原合同要求 XML、CollisionCfg 与 compiled runtime 一致：
两只 logical feet、每脚 7 capsules、零 legacy foot boxes，foot `condim=3`, `priority=1`,
MuJoCo friction vector `[0.6, 0.005, 0.0001]`, `contype=1`, `conaffinity=1`，declared non-foot
collision geoms count 与 compiled runtime count 必须一致且若存在则 `condim=1`。

`build_task_cfg` 必须将 stance `joint_qpos` 用作 runtime init/default joint position，将 `root_pose_eq`
用作 root pose，并将独立的 `actuator_ctrl_eq` 用作 zero-action processed target；29 个 semantic joint
到 anonymous joint/action target 必须显式一对一映射。禁止训练、optimizer update、Task048 checkpoint 和外部下载。

## Route

003c rejected runtime binding mismatch -> 003d rejected double-ground runtime ->
003e rejected contaminated training -> 003f single-ground runtime repair。

## Log

- 2026-08-31：建立 `mjlab_g1_7capsule_task_v2` lineage，未覆盖
  `mjlab_g1_7capsule_v1`。v2 XML
  `artifacts/contact_alignment/mjlab_g1_7capsule_task_v2/unitree_g1_mjlab_g1_7capsule_task_v2.xml`
  SHA `c41bfe757fbeb51f094a08457258d17004989948be6eb1fac5bbf3eafa644f93`；contact profile SHA
  `de1fcd515052afe488f3b769fad3649eeb3d509b8ac4ab6f4558cb45137d4f21`、payload SHA
  `dc4e9d9ac898ad0f9e837f5e89125e0f4c6e547ffa9e2824a10abfa145a69f46`；runtime material SHA
  `6179293ff3186429a5e4a21727ba55a2989801c3538bc7021203d3f9dc037804`；重新求解 stance SHA
  `b2cf38c891bdc5e6b7bf5c4eaaed3cb42fee5b45e40ac773bc4b339dd261aac4`、payload SHA
  `85d29988805b4ef82d15bf575280b8d20ff304cfb14a3317cf277ee3783cc492`、内部
  `stance_solution_sha256=cc522b67380713954480c3e9781be01fc6ad96445fb133d410f213f551f5ce9a`。
- 2026-08-31：`verify-runtime-binding` no-update gate passed。artifact:
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2/runtime_binding_verifier.json`，SHA
  `26b93587a11cbc12c0de32e8ea0bef23020351e47f367901ce83afe5b13ff36b`。
  `joint_mapping_count=29`、`unmapped=0`、`duplicate=0`；`default_qpos_error_max=1.4767210787525187e-08`；
  `action_offset_error_max=2.2015378742246128e-08`；`zero_action_processed_target_error_max=2.2015378742246128e-08`；
  `stance_qpos_vs_actuator_ctrl_min_abs_delta=3.810971428740453e-06`；
  reset root pose max error `4.416348237798657e-08`；`joint_pos_rel_reset_error_max=0.0`；
  actor/critic observation layouts exactly match Unitree-G1-Flat flat terms and shapes `(98,)` / `(113,)`；
  compiled material checks passed with 14 declared foot geoms, zero declared non-foot collision geoms,
  foot `condim=3`、`priority=1`、friction `[0.6, 0.005, 0.0001]`、`contype=1`、`conaffinity=1`；
  `optimizer_step_calls=0`、`parameter_delta_max_abs=0.0`。Zero-action 2 s hold had `done_count=0`,
  base-height drift `3.3855438232421875e-05`, max gravity-XY `0.001529861823655665`.
- 2026-08-31：后验 exact compiled-model audit 推翻上述 pass。旧 runtime 同时包含
  `robot/floor` 与 `terrain` 两个 z=0 collision planes；在冻结 stance 上 `mj_forward` 得到 56 个
  foot-plane contact records，其中 28 个对 `robot/floor`、28 个对 `terrain`。与此同时
  `feet_ground_contact.secondary` 只匹配 `terrain`，即动力学约束与 contact telemetry/reward 所见地面不一致。
  原 verifier 未检查 compiled ground topology，因此其 `passed=true` 仅作为 rejected audit history 保留。

## Tombstone

- 2026-08-31：应用户请求，使用 `gio trash` 移除无效目录 `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2`（9 个 `.pt`，51,918,322 file bytes，约 50M）及其全部内容。目录路径现已不存在；本文历史 SHA 仅作 audit-only 记录。

## Review

状态：**rejected / runtime_binding_mismatch_double_ground**。003d 证明了 joint/pose/action/material 的
局部绑定，但没有证明完整 runtime binding；双地面足以否定该 route。未运行的项目仍包括本 subtask 内的
optimizer step、正式训练、eval checkpoint 与 passing video。
