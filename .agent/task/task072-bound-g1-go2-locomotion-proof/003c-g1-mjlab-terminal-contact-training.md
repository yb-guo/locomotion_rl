# 003c — G1 MJLab-contact-aligned walking proof

状态：**rejected_runtime_binding_mismatch / not_passed**。

003c 的旧 `proof_1024x24_seed720301` 证据被 reclassified：
`rejected_runtime_binding_mismatch`。数值结果仍是真实证据，但不得继续解释成“匿名 G1 在正确配置下无法行走”。
拒绝理由：

- 003b stance 没有进入 MJLab init/default pose；
- 29 个 runtime `default_joint_pos` 实际全为零；
- runtime root height 为 `0.8`，而新 stance 为 `0.8534139176251306`；
- zero action target 没有使用 stance `actuator_ctrl_eq`；
- `1024` env run 与官方 `4096` env 控制组的 PPO batch/update cadence 不等价；
- XML/CollisionCfg/runtime material 不一致，effective foot contact 未对齐 Unitree-G1-Flat task；
- action/default pose 绑定依赖匿名名称无法验证的一般匹配，而非 29/29 显式 semantic mapping。

旧目录与 `proof_1024x24_seed720301` 必须保持 immutable。

Owner：本地 MJLab G1 registration、from-scratch training、evaluation、render 与 verifier owner。

本 subtask 只在 `003b` 全部 asset/no-update gate 通过后执行。目标是把
`mjlab_g1_7capsule_v1` bound anonymous G1 注册到本地已验证的 MJLab velocity task，使用官方
MJLab/RSL-RL 训练路径，从随机初始化把 G1 实际训练到稳定前进行走。只有完整数值、交替 gait、视频
和独立 verifier gate 通过才算完成；“训练能启动”“loss finite”或短 smoke 都不能作为通过。

训练预算与进度只用 **total environment transitions** 表达：

`total_transitions = num_parallel_envs * rollout_steps_per_env * completed_updates`

不得用 `iteration/update count` 单独比较不同并行度的 run。实际并行环境数由 RTX 5060 Ti smoke 在
VRAM-safe 范围内确定并冻结；目标 transitions 固定后，所需 updates 由上式反推。例如
`4096 envs * 24 steps * 650 updates = 63,897,600 transitions`，若并行环境减少，则相应增加 updates，
不能因此减少总样本预算。

同一最大预算的换算示例：

| parallel envs | rollout steps/env | transitions/update | derived updates for 63,897,600 |
| ---: | ---: | ---: | ---: |
| 4096 | 24 | 98,304 | 650 |
| 2048 | 24 | 49,152 | 1,300 |
| 1024 | 24 | 24,576 | 2,600 |
| 32 | 64 | 2,048 | 31,200 |

表中的 updates 只用于执行换算；训练进度、checkpoint 和不同 run 的比较全部使用
`transitions/update * completed updates` 得到的累计 transitions，不用“跑了多少轮”下结论。

禁止 Task048 checkpoint、任何其他 checkpoint 初始化、外部下载、H200，以及修改 `003b` 已冻结的
asset/contact/inertial/actuation。固定 `vx=0.5, vy=0, yaw=0` 用于正式 eval；训练使用被绑定的 MJLab
G1 command/curriculum contract，不能把训练 curriculum 与 contact delta 混写成单变量结论。

## Route

1. **R0 — registration and lineage without training**
   - 不修改本地 external MJLab checkout；在项目侧注册独立 task id；
   - 绑定 `003b` bound XML/contact profile/stance SHA、MJLab checkout commit、environment config、
     runner config、RSL-RL version、lockfile、seed、GPU 与命令；
   - actor/critic observation、logical-foot sensors、action mapping、default pose 与 termination 必须
     通过 reset/no-update contract；缺一项即停止。
2. **R1 — parallel-capacity and one-update smoke**
   - 只有用户明确授权后运行最短 GPU smoke；必须 finite、无 OOM、optimizer step 数与配置一致；
   - 从 VRAM-safe 候选中确定正式 `num_parallel_envs` 与 `rollout_steps_per_env`，记录实际 batch
     transitions；正式训练中不得无 lineage 地动态改变并行度；
   - smoke artifact 与正式训练目录分离，不得 resume；失败即停止。
3. **R2 — transition-budgeted from-scratch progression**
   - smoke 通过并再次获得训练授权后，从随机初始化运行；不加载 Task048 权重；
   - checkpoint 与停止条件按累计 transitions 定义，至少在 10M、20M、40M 和最大
     `63,897,600` transitions 前后保存；updates 只作为实现计数，不作为科学预算；
   - 每到一个 transition gate 就做固定命令 screen；只有 checkpoint 已通过完整 walking gate 才可
     提前停止，稳定但尚未行走的 10M/20M checkpoint 只能记为 progression，不能提前宣告失败；
   - 每个 checkpoint 保存 command tracking、fall、orientation、foot contact/airtime/touchdown、
     single/double support、action、KL/clip、reward decomposition 与 wall-clock/VRAM。
4. **R3 — G1 must actually walk**
   - 对预注册 checkpoint 做 `vx=0.5, vy=0, yaw=0` deterministic evaluation；
   - 产出 20 s、256-env full evaluation 和 8 s/400-frame 视频；至少满足 zero-fall ratio
     `>=0.95`、mean forward velocity `>=0.30 m/s`、mean +x displacement `>=6.0 m/20 s`、planar
     velocity error `<=0.35 m/s`、yaw error `<=0.35 rad/s`、projected-gravity XY norm `<=0.35`；
   - 左右脚都必须出现非初始 touchdown 和 single-support，存在可重复的左右交替序列；persistent
     double support、原地摆动、拖脚或仅靠倾倒产生位移一律不算 walking；
   - 8 s 视频必须无 reset、+x displacement `>=2.4 m`，并通过人工视觉检查；同时保存 paired
     random-init/zero-action baseline；
   - verifier 必须从 checkpoint 独立加载重跑，不能只读取 progression JSON。
5. **R4 — freeze admission for Task072 004**
   - locomotion 数值 gate、交替 gait/contact gate、视频人工审核、checkpoint/config/source/asset
     lineage 全部通过后，才发布 `task072_g1_contact_walking_passed=true` 并允许执行 Task072 004
     freeze；
   - 任一 gate 失败时保留 artifacts，Task072 004 与后续资产任务继续 blocked；不得为过 gate 在同一 lineage 顺手改
     reward、asset、command、runner 或 eval。

## Log

- 2026-08-30：根据用户要求建立“先 asset 对齐、后训练”的独立 subtask。本轮没有注册 MJLab task、
  没有运行 smoke 或训练，也没有使用 checkpoint。
- 2026-08-30：003b passed 后，注册 task-local MJLab task
  `Task072-G1-MJLab-7Capsule-Flat`，基于 `Unitree-G1-Flat`，不修改 external MJLab checkout，不下载外部
  资源。runtime spec SHA `a3db856ed16685a1cd372a0f7d7b3bd92566c754ffd6e71e60f58f441c4a4da3`；
  asset XML SHA `bd06eff122d35044018f3867a9d227346af4df847a8c56ce1df3f4cd074faf36`；contact profile
  SHA `304e464577636d45322e98547db6ef8557585c2bd1c3d254ee898e440b41156d`；stance SHA
  `3671c9335e58591a5d9252c9aa38d02689579c222ada81432851b9d327030e19`。
- 2026-08-30：RTX 5060 Ti capacity smoke passed for `256/512/1024` envs；选择 `1024` envs、
  `24` rollout steps/env，`transitions_per_update=24576`。Capacity artifact
  `.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/mjlab_contact_training/g1/r1_capacity_smoke.json`
  passed true，记录 GPU lock `/home/admin1/workspace/run/.gpu.lock`、`h200_used=false`、
  `external_downloads_performed=false`、`task048_checkpoint_used=false`，action scale SHA
  `f106dcb86f73427797a505f880dcde4b57184a7c83b01385f262acdf6cde0d76`。旧 scalar action-scale 0.05
  artifacts 已移入 `artifacts/mjlab_contact_training/g1/rejected_scalar_action_scale_005/`。
  Capacity 命令：
  `flock -n /home/admin1/workspace/run/.gpu.lock env PYTHONPATH="$PWD/src" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python .agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py capacity-smoke --candidates 256 512 1024 --rollout-steps 24 --steps 2 --device cuda:0`。
- 2026-08-30：正式训练从随机初始化开始，run dir
  `.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/mjlab_contact_training/g1/proof_1024x24_seed720301/`。
  命令：
  `flock -n /home/admin1/workspace/run/.gpu.lock env PYTHONPATH="$PWD/src" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python .agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py one-update-train --run-dir .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/mjlab_contact_training/g1/proof_1024x24_seed720301 --num-envs 1024 --rollout-steps 24 --updates 2600 --save-interval 407 --device cuda:0`。
  Training manifest `task072_mjlab_one_update_smoke.json` SHA
  `96781895798e0c8c1a2e2ae89368ac8c10a9be427106df192ef914fa8584b6de`，`observed_transitions=63897600`，
  `wall_time_s=3343.7972259521484`，`h200_used=false`，`external_downloads_performed=false`，
  `task048_checkpoint_used=false`。
- 2026-08-30：transition accounting: `1024 envs * 24 rollout steps/env = 24576 transitions/update`。
  Checkpoints and observed transitions: `model_407.pt -> 10027008` transitions, SHA
  `7ac16a655fad723a5f460412095e179a84133ad1f12a2dc5d68c6a700e913eac`; `model_814.pt -> 20029440`,
  SHA `bbb9b2d488f5276b76c2e9c4cda8f256355409a3243d110a96fb85ce9db5fdd6`;
  `model_1628.pt -> 40034304`, SHA
  `70942b02d8591ef59db4f7d538b94c4dc2090d1bd47454e58db51c78b59f51b0`; `model_2599.pt -> 63897600`,
  SHA `5e6f614114a1e4d1e3e78f2b0d63824664254e132d7c3d8f55ddfb2a74699ff4`。
- 2026-08-30：TensorBoard training scalars at those checkpoint steps: `407/10027008` mean reward
  `-6.19351`, train xy/yaw errors `0.343544/0.53608`, fell_over `3.5`, mean std `0.690245`;
  `814/20029440` mean reward `-6.81898`, xy/yaw `0.79855/1.20915`, fell_over `1.79167`, mean std
  `0.69717`; `1628/40034304` mean reward `-5.06848`, xy/yaw `0.671423/1.03302`, fell_over `1.25`,
  mean std `0.693133`; `2599/63897600` mean reward `-1.66894`, xy/yaw `0.904601/1.07719`, fell_over
  `1.25`, mean std `0.672505`。Event file:
  `.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/mjlab_contact_training/g1/proof_1024x24_seed720301/events.out.tfevents.1788088651.admin1-B860M-K.1483996.0`。
- 2026-08-30：独立重载 verifier/eval 用固定 command `vx=0.5, vy=0, yaw=0`，`256` envs、`20` s，
  eval seed `720400`，禁用 non-reset training events 和 curriculum。四个 checkpoint 均 fail closed：
  `eval_model_407_20s_256.json` SHA
  `060971ad0edb83c2354b610ab4cb86f5f601a26068547a3fed575fda78f8f264`，zero-fall `1.0`，mean vx
  `0.0001045351`，x displacement `0.0020907023`，planar error `0.5133395195`，yaw error
  `0.1586990654`，gravity XY `0.0746598765`；`eval_model_814_20s_256.json` SHA
  `863a133f66bba034c9ca2688dbfb20de0d91d81e2354c11e16e7a91b27cda879`，zero-fall `1.0`，mean vx
  `0.0000345447`，x displacement `0.0006908933`，planar error `0.5425215960`，yaw error
  `0.1602293998`，gravity XY `0.0434840545`；`eval_model_1628_20s_256.json` SHA
  `b94391ee470eda5c1b612912a7d61689034711e4a2aa826717893513cf80137b`，zero-fall `1.0`，mean vx
  `0.0005213788`，x displacement `0.0104275765`，planar error `0.5364387035`，yaw error
  `0.1363560110`，gravity XY `0.0294358730`; `eval_model_2599_20s_256.json` SHA
  `05ca1bcbd2b9b113374d69936aa3f94fd95d176b7ca1f763060f180bc6a684f9`，zero-fall `1.0`，mean vx
  `0.0001334263`，x displacement `0.0026685263`，planar error `0.5329391360`，yaw error
  `0.1241449267`，gravity XY `0.0982930735`。
  Eval 命令模板：
  `flock -n /home/admin1/workspace/run/.gpu.lock env PYTHONPATH="$PWD/src" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python .agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py evaluate --checkpoint <model_N.pt> --output <eval_model_N_20s_256.json> --eval-envs 256 --eval-seconds 20 --device cuda:0`。
- 2026-08-30：所有 evaluated checkpoints 都满足 finite、zero-fall、yaw、gravity、left/right touchdown、
  left/right single-support 和 alternating-contact checks，但都失败 `mean_forward_velocity >= 0.30`、
  `mean_x_displacement >= 6.0`、`planar_tracking_error <= 0.35`。因此没有 checkpoint 可进入 video gate；
  未生成新的 8 s passing video，未执行 004 freeze，未启动 Go2、Task073、Task074 或 18-config 训练。

## Tombstone

- 2026-08-31：应用户请求，使用 `gio trash` 移除无效目录 `artifacts/mjlab_contact_training/g1`（10 个 `.pt`，62,819,574 file bytes，约 60M）及其全部内容。目录路径现已不存在；本文历史 SHA 仅作 audit-only 记录。

## Review

状态：**failed / not_passed**。003b candidate 已在本地 MJLab 路径上从随机初始化跑满
`63,897,600` transitions，但固定 `0.5 m/s` 独立评估显示 mean forward velocity 约为
`0.00003-0.00052 m/s`，20 s +x displacement 最高仅 `0.01043 m`，planar tracking error 为
`0.513-0.543`，不满足 walking gate。虽然存在左右 touchdown、single-support 和 alternating-contact
事件，但这些事件没有产生持续向前位移，因此不算 G1 walking。Task072 004、Task073 和 Task074 继续
blocked；旧 rejected custom-PPO artifacts 仍保持 rejected，不得补判。
