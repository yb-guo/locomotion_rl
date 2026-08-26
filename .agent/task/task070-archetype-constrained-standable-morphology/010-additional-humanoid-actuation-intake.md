# 010 — Additional Humanoid Actuation Intake

## Route

1. Freeze official fixed commits and licenses for AgiBot X1/X2, EngineAI
   T800/T800Pro, LimX HU_D04, Booster T1, and RobotEra STAR1.
2. Retain only source description, transmission, motor, and controller evidence
   explicitly authorized by the user; exclude mesh, texture, checkpoint,
   dataset, and motion files.
3. Record structural-model DoF separately from physical-motor/config DoF,
   transmission completeness, coherent motor-config coverage, and every
   promotion blocker.
4. Keep all entries candidate/fail-closed until a descriptor and anonymous
   instantiation pass the Task070 source-tree gates.

## Log

- 2026-08-25：已固定七个官方 source root/commit/license，最小文件清单与逐文件 SHA 保存于
  `artifacts/task070_v2_additional_humanoid_candidate_source_inventory.json`；未下载 mesh、texture、
  logo、checkpoint、dataset 或 motion data。
- 已登记 structural source inventory：X1 serial MJCF `29` joints（official control config 另含两只
  claw motor，共 `31` physical motors）、X2 Ultra `31`、T800 `25`、T800Pro `43`、HU_D04 `31`、
  T1 `23/29`、STAR1 `55`。X1 的 model/config 差异不得静默合并。
- X1 nonlinear ankle lookup、T800Pro palm `.mnn` 未纳入；X2/T1/STAR1 只有 limit-level motor
  evidence。所有型号保持 candidate/fail-closed，不进入 sampler/R4 denominator。
- Inventory SHA-256：`c363182d8fae259048fc783cda5712f73fcfd7a6a98e1a55e3129e9aceca8c03`。

## Review

- Scoped source/config intake: **complete**。固定来源、许可证、DoF 与缺口均可审计。
- Quantitative-prior promotion: **not approved**。缺失或未对齐的 transmission/motor evidence
  继续 fail closed；这不是 Task070 v2 overall pass。
