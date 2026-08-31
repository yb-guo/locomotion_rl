# 003e — G1 MJLab repaired walking proof

状态：**rejected / runtime_binding_mismatch_double_ground**。

Owner：历史上使用 003d `mjlab_g1_7capsule_task_v2` asset/profile/stance 的 MJLab/RSL-RL
from-scratch baseline training、checkpoint screen、正式 eval、video 与 independent verifier。由于 003d
后验发现 double-ground，本 route 的训练与评估证据全部保留但不得解释为正确 single-ground binding 下的结论。

003e 只能在 003d runtime binding verifier passed 后运行。禁止 Task048 或任何 checkpoint 初始化、外部下载、
H200、Go2、Task073、Task074，以及为通过验收而降低 walking gate。

## Route

1. **R0 — capacity and batch equivalence**
   - 在 `/home/admin1/workspace/run/.gpu.lock` 下测试 `2048/4096/6144` envs，`rollout_steps_per_env=24`；
   - `capacity-smoke` artifact 必须记录 `gpu_lock.held_by_ancestor=true`，且 `selected` 必须为 passed
     的 `4096 x 24 = 98,304` transitions/update row；`2048` 单独通过或未包含 `4096` 均 fail closed；
   - `one-update-train` 必须消费上述 capacity artifact，且自身 `num_envs=4096`、`rollout_steps=24`；
   - 选择标准为 finite、无 OOM、吞吐稳定、optimizer smoke 正确；
   - 若 `4096` env 可用，正式 baseline 必须为 `4096 x 24 x 650 = 63,897,600` transitions；
   - 若 `4096` 不可用，停止并报告，不自行换成非等价 PPO cadence。
2. **R1 — repaired baseline**
   - 使用 003d v2 asset/profile/stance，从随机初始化训练，seed 预先登记；
   - `track_linear_velocity` weight 保持官方 `1.0`，其余 reward、PPO、command curriculum 保持
     Unitree-G1-Flat 配置；
   - 在约 `10M/20M/40M/63,897,600` transitions checkpoint，并记录 actual observed transitions。
3. **R2 — gate telemetry**
   - 每个 checkpoint 记录 fixed-command mean vx、x displacement、planar/yaw error、zero-fall ratio、
     gravity XY、left/right touchdown、single support、alternation、reward decomposition、action stats、
     PPO LR、adaptive KL、approx KL、clip fraction、episode reward/length、GPU VRAM、throughput、wall time。
4. **R3 — decision points**
   - 约 `20M` transitions 若 LR 长期钉在 `1e-5` floor、KL/LR 异常或 optimizer telemetry 不完整，则归入
     optimizer/batch correctness 并停止 baseline；
   - 约 `40M` transitions 若 forward velocity 明确增长则继续；若仍近零且 optimizer、stance、command、
     contact、reward coupling 均机器验证正常，才允许单变量 reward B。
5. **R4 — conditional reward B**
   - 只有 baseline 满足触发条件仍未形成足够前进速度时，才运行独立 B lineage；
   - 唯一变化：`track_linear_velocity` weight `1.0 -> 1.5`；不得试 `2.0/3.0` 或继续搜索。
6. **R5 — final walking gate**
   - fixed command `vx=0.5, vy=0, yaw=0`，20 s、256-env independent reload eval；
   - zero-fall `>=0.95`，mean vx `>=0.30 m/s`，mean +x displacement `>=6.0 m`，planar/yaw error
     `<=0.35`，gravity XY `<=0.35`；
   - 左右脚均有非初始 touchdown 和 single support，存在可重复左右交替；
   - 8 s no-reset video +x displacement `>=2.4 m`，不是原地踏步、拖脚、双支撑滑行或摔倒位移。

## Log

- 2026-08-31：003d no-update verifier passed 后创建本 repaired walking proof 合同；尚未运行 GPU
  capacity smoke、optimizer smoke、baseline training、reward B、checkpoint eval 或 video。
- 2026-08-31：尝试在 `/home/admin1/workspace/run/.gpu.lock` 下运行 `2048/4096/6144` capacity smoke，
  但 lock 已由 PID `2056863/2057703/2057729`（另一个 docker compose task）持有，`flock -n` 立即失败；
  未绕过 GPU lock，未创建 capacity artifact，未启动 optimizer smoke 或训练。
- 2026-08-31：runner 已加 fail-closed guards：CUDA `capacity-smoke` 需要 inherited shared GPU lock；
  `capacity-smoke` 缺少 `4096 x 24` 或未持有 lock 会在 env 创建前失败；`one-update-train` 需要消费
  passed 的 4096×24 capacity artifact。Negative CLI probes 写到 `/tmp` 并均按预期 failed。
- 2026-08-31：在 inherited `/home/admin1/workspace/run/.gpu.lock` 下完成 003e capacity smoke：
  `2048/4096/6144` candidates 均 passed，selected 为 required `4096 x 24 = 98,304`
  transitions/update；artifact
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2/capacity_smoke_2048_4096_6144.json`
  SHA `836a125164d185cd8a78226218f8e592dbfc5342a88864968b664d93ade65c37`，
  `gpu_lock.held_by_ancestor=true`。
- 2026-08-31：capacity-consuming one-update smoke passed under the same lock:
  `one_update_4096x24_seed720301/task072_mjlab_one_update_smoke.json` SHA
  `a423f2ccf2cbbd59c9c893b63e74bfb0ebaa4f7263c650f048828f465e4c1321`，checkpoint
  `model_0.pt` produced，`4096 x 24`、`98,304` transitions/update。
- 2026-08-31：正式 repaired baseline 从随机初始化完成
  `4096 x 24 x 650 = 63,897,600` transitions，seed `720301`，run dir
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2/proof_4096x24x650_seed720301/`。
  Training manifest `task072_mjlab_one_update_smoke.json` passed，SHA
  `88de879629cb83d7cba8bad34c9d919c16d44a6d0d0721636106d26060f0110b`，wall time
  `1965.1720781326294 s`，`h200_used=false`、`external_downloads_performed=false`、
  `task048_checkpoint_used=false`。Checkpoints: `model_100.pt`
  `bf0a3c99d96fd07bca82f8fb9b5e476e8c8a5061b760f34f5a5014530dfc996d`，`model_200.pt`
  `9e0dd9ba6700c6915d6162a7af06bc7997d67992519e3b9839a8c5ec79072512`，`model_400.pt`
  `7a5487f2e6204ec641281a702a2b6f3813ae1f4f639f0fe8e70757c8ed7d92ba`，final
  `model_649.pt` `f3828137d2f8056824fe2aab41dbe09c454169c78d202ceb238b68ad553bd18f`。
- 2026-08-31：fixed-command independent reload evals at `10M/20M/40M/final` all failed walking gate。
  `model_100` eval SHA `24da523dfa00a98ae61d66edab1434d771d22e92f44c03541c418718b513c4a8`:
  zero-fall `1.0`，mean vx `-0.0001093650`，x `-0.0021873005`，planar error `0.5005537271`，
  failed `mean_forward_velocity/mean_x_displacement/planar_tracking_error/right_touchdown/left_single_support/alternating`。
  `model_200` eval SHA `68536f63da90131afb2e9905f2c3c49087c499434aaab669706f424efef6d07b`:
  zero-fall `1.0`，mean vx `-0.0001918841`，x `-0.0038376828`，planar error `0.5002169609`，
  failed the same six checks。`model_400` eval SHA
  `9b9e07e9048265d7cb3facd349885b27c4f075deb1f79d2a1d04fd10422e64c1`: zero-fall `1.0`，
  mean vx `-0.0030932274`，x `-0.0618645474`，planar error `0.5080574155`，touchdown/single
  support/alternation passed but forward velocity, +x displacement and planar tracking failed。
  `model_649` eval SHA `02458b2803e709244a205cb1faa63e01ce3dfca911c473e73b5a0758a7d98f57`:
  zero-fall `1.0`，mean vx `-0.0006511346`，x `-0.0130226929`，planar error `0.5035098791`，
  yaw `0.0302346162`，gravity XY `0.0306854099`，touchdowns `left=531/right=6`，single support
  `left=6/right=646`，alternation `6`；failed forward velocity, +x displacement and planar tracking。
  No 8 s passing video was generated.
- 2026-08-31：后验 compiled-model audit 发现本 route 全程使用重合的 `robot/floor` 与 `terrain`。
  动力学同时施加两套 foot-plane contact constraints，而 foot contact sensor/reward 只匹配 `terrain`。
  因此 capacity、one-update、63,897,600-transition training、checkpoints 与 eval 数值只作为污染证据保留；
  training manifest 的 `passed=true` 只表示 run/checkpoint 产出完成，不表示 walking gate 或 runtime binding
  通过。上述 checkpoint 不得 freeze，也不得复用为 v3 single-ground 初始化权重。

## Tombstone

- 2026-08-31：应用户请求，使用 `gio trash` 移除无效双地面目录 `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2`（9 个 `.pt`，51,918,322 file bytes，约 50M）及其全部内容。目录路径现已不存在；本文历史 SHA 仅作 audit-only 记录。

## Review

状态：**rejected / runtime_binding_mismatch_double_ground**。003e 只证明被污染的双地面 runtime 能在
RTX 5060 Ti 上以 `4096 x 24` cadence 产出 full-budget run；它既不证明正确 single-ground binding 下
无法行走，也不产生 walking proof。003f 必须先独立修复并验证 single-ground runtime；004 freeze、
Task073 和 Task074 继续 blocked。
