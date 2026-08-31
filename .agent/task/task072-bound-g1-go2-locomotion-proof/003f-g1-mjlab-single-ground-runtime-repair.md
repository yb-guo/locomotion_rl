# 003f — G1 MJLab single-ground runtime repair

状态：**passed / no-training-only**。

保留 immutable contact asset/profile/stance `mjlab_g1_7capsule_task_v2`，新增 runtime lineage
`mjlab_g1_7capsule_task_v3_single_ground`。运行时从传给 MJLab 的内存 XML 中删除 asset worldbody 的
唯一有效 `floor`，由 Unitree-G1-Flat scene 唯一创建 `terrain`；不得改写 frozen v2 XML。

编译后 verifier 必须 fail closed：有效 plane 恰好一个且名为 `terrain`，不存在 `robot/floor`，14 个
foot capsule 均只与 `terrain` 接触，不存在 hidden foot-plane contact pair。`contype` 或
`conaffinity` 任一非零的 plane 都按有效 plane 处理。本 subtask 禁止 optimizer step、GPU training、
checkpoint 初始化、外部下载和 walking pass 声明。

## Route

003d rejected double-ground runtime -> 003e rejected contaminated training ->
003f v3 single-ground no-update verifier -> separately authorized v3 from-scratch walking run -> 004。

## Log

- 2026-08-31：`task072_mjlab_contact_runner.py` 将 contact profile ID 与 runtime lineage 分离：
  `CONTACT_PROFILE_ID=mjlab_g1_7capsule_task_v2`，
  `LINEAGE_ID=mjlab_g1_7capsule_task_v3_single_ground`。v3 capacity/runtime evidence root 与 v2
  分离，旧 v2 capacity artifact 不能通过 lineage consumption gate。
- 2026-08-31：runtime XML 只删除 direct worldbody collision-enabled `floor`；若源 asset 不恰好包含
  一个该 plane 则 fail closed。Frozen v2 XML SHA 保持
  `c41bfe757fbeb51f094a08457258d17004989948be6eb1fac5bbf3eafa644f93`；runtime spec SHA
  `9216ef486aa9b535412c65b198e5a168d76d595763e88cf900057a65aa33874a`。
- 2026-08-31：CPU `verify-runtime-binding` passed。artifact
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v3_single_ground/runtime_binding_verifier.json`
  SHA `8ced901ec7b4eb5b69d29e2098d2411e0db84f27c33529e7061d9df0cd467dc1`，runner source SHA
  `742ece071c4c77e652a3d09971f2a78b0a53ccd7d99ce6eef1f7fca4e3f2a33b`。Compiled ground audit：
  `collision_enabled_plane_names=[terrain]`，14/14 foot geom names observed，当前 frozen stance
  `foot_ground_pairs=28`，`hidden_plane_pairs=0`；joint/pose/action/material/observation/finite/2 s hold
  checks 全部通过，`optimizer_step_calls=0`、`parameter_delta_max_abs=0.0`、CUDA unavailable。
- 2026-08-31：聚焦回归覆盖 runtime XML 不改 frozen asset、正常 single-ground、duplicate plane 以及
  `contype=0, conaffinity=1` 的半启用隐藏 plane；结果 `6 passed`。

- 2026-08-31：配置修复后以 `CUDA_VISIBLE_DEVICES=''` 重跑 CPU verifier，现存 v3 verifier SHA `377daa19e8d84950208f8b0b6f820ffd5360a9e00e10ef1d752c9d48299f27a1`，runner source SHA `30d18f0c07d105d5f65de65010df49fef88bf304d8d221f1d425ce4b1c6f2a5e`；`passed=true`、`optimizer_step_calls=0`、`parameter_delta_max_abs=0`。
- 2026-08-31：reviewer fix 后最终 runner source SHA `b323fce3d90889f2836512be6888021f343a00c3e06354b39c4a64242708a57c`，v3 verifier SHA `551d0743a029ca033b02d049a88cb48f5c36cd01d00bb58a3d737f4bffdf420f`；registration 的 `lineage_id`、`run_name`、`experiment_name` 均为 `mjlab_g1_7capsule_task_v3_single_ground`，`passed=true`。

## Review

状态：**passed / no-training-only**。003f 证明 v3 的 compiled runtime ground topology 与 contact sensor
目标一致，并保留 v2 contact asset 不变；没有运行 capacity smoke、optimizer update、正式训练、checkpoint
eval 或 video，因此 Task072 仍为 not_passed。旧 v2 checkpoints 对 v3 不具备可继承的 pass 权限。
