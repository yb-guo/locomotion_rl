# 003g — G1 MJLab single-ground walking proof

状态：**in_progress / not_passed**。

在 003f 已通过的 `mjlab_g1_7capsule_task_v3_single_ground` runtime lineage 上，使用 RTX 5060 Ti
从随机初始化运行 v3 single-ground G1 正式训练。训练沿用 003f 的 immutable v2
contact asset/profile/stance，不修改 reward、command、PPO cadence、asset、stance 或 contact
material，不使用旧 checkpoint、Task048 checkpoint、外部下载、H200、Task073 或 Task074。

## Route

003f single-ground CPU verifier -> v3 2048/4096/6144 CUDA capacity smoke -> 4096x24 one-update
smoke -> 4096x24x650 random-init proof training -> fixed-command checkpoint eval -> numeric gate
通过后再生成 8 s no-reset video。任一 lineage、hash、single-ground audit、capacity、OOM、NaN/Inf 或
GPU lock 证据失败则停止；4096 capacity 未通过不得降级训练；numeric gate 未通过不得生成伪 passing
video、不得 freeze、不得调 reward 或参数搜索。

## Log

- 2026-08-31：建立 003g 执行记录。初始状态为 `in_progress / not_passed`；等待重新运行 CPU
  `verify-runtime-binding`、v3 capacity smoke、one-update smoke、正式训练和固定命令 eval。
- 2026-08-31：训练前以 CPU/no-update 重跑 `verify-runtime-binding`，artifact
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v3_single_ground/runtime_binding_verifier.json`
  SHA `8ef6762545ae3ee5e0336684af95ed66c69774d9497e775b939537a6332c1c7f`。`passed=true`，
  compiled plane 只有 `terrain`，14/14 foot geoms 只接触 terrain，`hidden_plane_pairs=0`，
  `optimizer_step_calls=0`；registration 的 `lineage_id`、`run_name`、`experiment_name` 均精确等于
  `mjlab_g1_7capsule_task_v3_single_ground`。Runner SHA
  `d7560de48175aece01d626e59ff13a8e28c66829c42cdcfb6e2d1aa61219656e`，runtime spec SHA
  `9216ef486aa9b535412c65b198e5a168d76d595763e88cf900057a65aa33874a`，asset XML SHA
  `c41bfe757fbeb51f094a08457258d17004989948be6eb1fac5bbf3eafa644f93`。
- 2026-08-31：准备修改已用显式 pathspec 提交为
  `41e98278f25c0741de8cdc8f668964aac2086c09`；v3 capacity/training/eval 均使用该 source commit。
  聚焦回归 `tests/test_task072_locomotion_proof.py -k 'mjlab_runner_records_003f_only_for_runtime_verifier
  or mjlab_runtime_defaults_are_v3_single_ground_paths or mjlab_runner_capacity_defaults_require_4096_equivalence
  or mjlab_capacity_evidence_is_fail_closed'` 为 `4 passed`。
- 2026-08-31：在持有 `/home/admin1/workspace/run/.gpu.lock` 的祖先进程下完成 v3 capacity smoke。
  Artifact `capacity_smoke_2048_4096_6144.json` SHA
  `16e24b010c9c48b7b6698473ab4c16dddf445ef7a81fcd347c31c2dc548d2419`，`passed=true`，选中
  `4096 x 24 = 98,304` transitions/update；`gpu_lock.held_by_ancestor=true`。
- 2026-08-31：完成 `one_update_4096x24_seed720301`，manifest SHA
  `f721fd68b6c9f71c7331bd6c82510facf2db5a60c5b6e0612de1fb4011cb15a5`，`passed=true`，
  observed transitions `98,304`，wall time `3.3279764652252197 s`，产出 `model_0.pt`；capacity
  consumption 与 GPU lock checks 均通过。
- 2026-08-31：从随机初始化完成正式训练 `proof_4096x24x650_seed720301`，manifest SHA
  `b86e6efd3003b2527b8c494183a00507e78935b685471d9fd12299d98826c131`，`passed=true` 仅表示训练与
  checkpoint 产出完成，不代表 walking pass。Observed transitions `63,897,600`，manifest wall time
  `1534.6346380710602 s`，wrapper wall time `1562 s`。指定 checkpoint SHA：`model_100.pt`
  `8ee404697cee22f20197465872dbd997482a28c80e11ed7229795e17cbd96216`，`model_200.pt`
  `23b3e49afcadf34e5f8e86f6c2333109b41b8aae4860fe5a6b8a5e733e5a0a9f`，`model_400.pt`
  `8d495f223af24265fde8ab055347ca2aea2f4d39fa759c3ce521ff892f350b5a`，`model_649.pt`
  `36d153a49328c0a4fdd6f65eed253097c93c57c7e99b88c459ad5fe3cdc7f0aa`。
- 2026-08-31：固定命令 independent reload eval 已对 `model_100.pt`、`model_200.pt`、`model_400.pt`、
  `model_649.pt` 完成；全部 `passed=false`，均有 `gpu_lock.held_by_ancestor=true`。`model_100`：
  zero-fall `1.0`，mean vx `0.0000175372`，+x `0.0003507431`，planar error `0.5002220869`，
  yaw `0.0242850855`，gravity XY `0.0218535811`，touchdown L/R `0/0`，single support L/R `0/0`，
  alternation `0`。`model_200`：zero-fall `1.0`，mean vx `-0.0000487382`，+x `-0.0009747641`，
  planar error `0.5006250739`，yaw `0.1261544824`，gravity XY `0.0137992054`，touchdown L/R `0/0`，
  single support L/R `0/0`，alternation `0`。`model_400`：zero-fall `1.0`，mean vx `-0.0013752637`，
  +x `-0.0275052749`，planar error `0.5085780025`，yaw `0.0820983648`，gravity XY `0.0301792882`，
  touchdown L/R `319/90`，single support L/R `90/63`，alternation `86`。`model_649`：zero-fall
  `1.0`，mean vx `0.0064487740`，+x `0.1289754808`，planar error `0.4939913750`，yaw
  `0.0923699662`，gravity XY `0.0118293408`，touchdown L/R `6203/256`，single support L/R
  `0/8836`，alternation `256`。
- 2026-08-31：numeric walking gate 未通过；未生成 8 s no-reset passing video，未 freeze，未启动 004。
  GPU monitor artifacts：`003g_gpu_monitor_attempt2.tsv` SHA
  `418448e6f8c949eeeee897b32be00e0454c59ba4398bdb60abb5adad0796ecea`，
  `003g_eval_gpu_monitor_attempt3.tsv` SHA
  `8cc309a82d68d346a0f253175079dc619bc20016df324b6c57aec910ada2f2df`。

## Review

状态：**failed / not_passed**。v3 single-ground runtime binding、capacity smoke、one-update smoke 和
650-update training 均完成，但四个 fixed-command eval 全部未通过 walking gate。主要失败项是
forward velocity、+x displacement 和 planar tracking；部分 checkpoint 还缺少 touchdown、single support
或左右交替。按停止条件不生成视频、不 freeze、不调 reward、不继续参数搜索。
