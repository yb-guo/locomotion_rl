# Task071 — Task070 v2 多形态训练就绪门

状态：**active / G1+Go2 representative R0–R3 passed / Tier A 2/5 / Task071 not passed**。

进入条件：Task070 v2 的最终输入 artifact、manifest/descriptor SHA 和 case registry 已冻结；
Task070 的用户视觉验收为前置 gate。2026-08-26 用户已通过 append-only overlay 接受 attempt010，
因此本任务可执行 R0–R3 gate。当前只有 G1+Go2 通过 representative bounded smoke；其余 Tier A、
Tier B、Tier C 仍不得由这两个 case 的结果暗示为 train-ready。

## 目标与边界

把 Task070 v2 attempt010 的匿名 primitive-link morphology、`canonical_root`/state 和
actuation contract 接入 RTX 5060 Ti-first 的 WholeBody/MuJoCo 训练路径，建立逐实例、分层、
fail-closed 的 train-readiness gate。当前基线是：18/18 compile/reset/paired actuator response，
`canonical_root` 已接入 WholeBody；Task070 generic stance 仍为 0/18、没有 gait，8 个 candidate humanoid
的 `policy_adapter_compatible=false`，candidate transmission/motor evidence 仍 fail-closed。因此
本任务不得把现状描述为“全部 18 个可训练”。Task071 后续已为 G1+Go2 增加独立、instance-bound
stance，不能反向改写 Task070 generic 结果，也不能替代 Tier A 其余 3 个 case 的 gate。

本任务不改变 Task070 的视觉验收；`user_visual_acceptance` 仍由 Task070 管理，不由本任务代签。
除用户逐次显式授权并固定 commit 的官方 simulator asset intake 外，不下载数据集、checkpoint、
策略或 motion，不启动长训练。所有 unknown provenance 保持 unknown，不猜补真实电机参数。

分层 case registry：

| 层 | case | 训练口径 |
| --- | --- | --- |
| Tier A | Unitree G1、EngineAI PM01、Boston Dynamics Spot、Unitree Go2、DeepRobotics Lite3 的 5 个非轮 center | 先 G1+Go2 最小 representative smoke，再扩至 5/5；这是本任务最小 pass 目标 |
| Tier B | 对应 5 个 wheel composition | 独立 wheel smoke；wheel 仅允许 continuous velocity/torque-compatible semantics，不能复用 position target，也不与静态 stance claim 混淆 |
| Tier C | AgiBot X1 serial、X2 Ultra、EngineAI T800/T800Pro、LimX HU_D04、Booster T1 23/T1 29、RobotEra STAR1 | excluded/fail-closed；最高约 55 actuator，禁止静默裁剪/合并进 frozen 45-slot，须先完成显式 variable-DoF adapter 和 transmission/config provenance |

## Route

### R0 — 冻结输入与训练 contract

- 冻结 Task070 attempt010 artifact、manifest/source SHA、许可证/provenance 和上述 case denominator；
  默认使用其 canonical root/state 与匿名 primitive-link 输出，不复制 vendor identity。
- 冻结 `whole_body_v1_45`：45-slot observation/action contract 不得静默漂移，inactive slots 必须
  有明确 mask/zero 语义，现有 193D observation contract 同样不得隐式改维。Tier C 另行做
  versioned variable-DoF padded action schema 或
  topology-sharded policy heads 的显式决策，不能截断、合并或借用 45-slot 名义通过。
- 明确 morphology、transmission model、coherent motor config、runtime fault process 四层边界：
  fault 是 nominal stack 之上的后续过程，不得用 fault rollout 冒充 nominal motor evidence。

### R1 — 每实例 reset/contact/stance gate

- 为每个纳入 case 建立 instance-bound、contact-aware reset/stance；generic stance 0/18 不得标记为
  已解决。检查 nonpenetration、有效接触/support、fall/reset/trial/context 语义，并采用至少现有
  Task067/Task070 约定的验证时长。
- biped/quadruped 分别验证 load-bearing contacts；wheel biped 使用 active balance，不能以静态
  两脚 stance 替代 wheel 控制验证。失败 case 必须保留在 R0 冻结的 gate denominator 中并记为
  fail，只能从实际训练采样池移除，不得通过缩小 denominator 洗掉失败。

### R2 — WholeBody environment contract

- actor observation、velocity/yaw reward、upright/fall 判定和 reset 均只能通过
  `read_canonical_root_state`/`canonical_root_frame_v1`；不得把各 source 的 native free root 直接
  当跨型号统一状态。
- 覆盖 observation/mask/action/reward/fall/reset/trial/context、finite 值、inactive slot 为零和
  action scaling；至少回归旋转 root。T1/STAR1 腰链只在 Tier C adapter 路线开启后加入回归。

### R3 — RTX 5060 Ti training smoke

- 先对 G1+Go2，再对 Tier A 5/5 执行 bounded no-update rollout 和至少一个真实 PPO update；记录
  finite observation/action/reward/loss/gradient/parameter delta、无 NaN、reset/trial 语义、吞吐和
  RTX 5060 Ti VRAM。不得以编译或零动作 actuator response 代替 PPO update。
- 先通过 exact motor accounting、canonical root、R1、R2，才允许进入短 smoke；不启动长训练，不
  宣称会站、会走、LocoFormer reproduction、motor parity、sim2real 或性能。

### R4 — wheel 独立 smoke

验证 5 个 wheel composition 的 wheel axis/frame、连续速度/扭矩控制、contact/reward、active balance、
reset 和 bounded rollout；分别记录 wheel denominator 与 failure reason，不能由 Tier A 5/5 pass 暗示。

### R5 — candidate variable-DoF 与 motor evidence

对 Tier C 逐 case 完成 source parent/child/joint order/axis/range/module、actuator count、transmission
和 coherent motor config provenance 审计。未知保持 null/unknown；只有 schema、adapter 和 evidence
齐全才可升级为训练 case。不得凭产品图、相似型号或 31 physical-motor config 猜造 serial MJCF
不存在的关节。

### R6 — readiness matrix 与独立 review

按 case 输出 exact denominator、gate 状态、命令、硬件/环境、commit/source SHA 和失败原因，并由独立
只读 reviewer 检查 claim 边界、root/action contract、tier denominator 和 artifact provenance。

## 预期产物

建议写入 Task071 artifact 目录：`r0_training_case_registry.json`、`r1_reset_stance_matrix.json`、
`r2_env_contract_smoke.json`、`r3_ppo_update_smoke.json`、`r4_wheel_training_smoke.json`、
`r5_candidate_adapter_decision.json`、`r6_training_readiness_matrix.json` 及必要日志。每个 artifact
必须包含执行命令、RTX 5060 Ti 环境/硬件、commit 或 source SHA、case denominator 和 failure reason。

## 退出条件与 claim 边界

Task071 的最小 pass 是 Tier A 5/5 通过 exact accounting、canonical root、instance-bound reset/stance、
environment contract、no-update rollout 和 one-update PPO smoke；Tier B、Tier C 必须有各自独立的
denominator/status，未通过不得被 Tier A pass 暗示。Tier C 即使完成结构编译也仍是 excluded，直到
variable-DoF/action adapter 与 coherent motor/transmission evidence 明确通过。

允许的最终 claim 仅为：指定 Tier A case 在指定 frozen contract 下通过 bounded training-readiness
smoke。不得 claim all 18 train-ready、会站/会走、LocoFormer reproduction、motor parity、sim2real、
真实部署或性能结论。

## Log

- 2026-08-27：用 versioned `task071_instance_bound_inverse_static_position_hold_v1` stance profile
  完成 G1+Go2 fresh R1，取代下方历史 `0/2` 结果。profile 只覆盖逐实例 leg nominal pose 与固定
  contact penetration；在 free root、原重力、无 equality/hidden support/external wrench 下，使用
  MuJoCo `mj_inverse(qvel=0,qacc=0)` 计算 position target feedforward。没有改冻结 morphology、
  joint order/axis/range 或 physics overlay。compile/accounting/lineage/reset/paired actuator response/
  stance 均 `2/2`，每 actuator response `32` steps，stance 为 `1000 × 0.002 s = 2.0 s`。G1
  max roll/pitch `0.04197 rad`、qvel norm `0.13934`、height drift `0.000478 m`、minimum terminal
  load `171.67 N`、max effort fraction `0.12199`；Go2 分别为 `0.03558 rad`、`0.12871`、
  `0.001826 m`、`26.09 N`、`0.10689`。R1 admission=true。Task070 helper 默认参数已按旧精确
  metrics 回归，generic stance 结果不变。
- 2026-08-27：R2 让 `WholeBodyMuJoCoShard` 显式消费 exact bound XML 与上述 instance-bound
  `StanceSolution`；预编译入口现在要求 expected XML SHA，且同一 SHA 是 `StanceSolution` manifest/
  solution hash 的不可分离字段；跨 XML 复用 stance 会失败。入口并对 timestep、joint/
  actuator accounting、free root、position transmission type/gear/ctrlrange/KP/KD、joint type/axis/range
  和 canonical site parent/local transform fail-closed，mass/COM/contact 等其余物理字段由 exact XML
  SHA 绑定。缺 joint 不再以 `mj_name2id=-1` 误索引最后一个 joint。G1+Go2 对
  `whole_body_v1_45` 的 `45 action / 193 observation`、active mask `29/12`、inactive position/
  velocity/previous-action 零值、inactive action control 隔离、stance-centered action scaling、finite
  observation/reward/metrics、旋转 root 的 observation/reward/fall canonical reader、canonical-height
  fall/reset、2-step trial/2-trial context 均为 `2/2`。fall probe 现在实际降低 canonical root 后调用
  `step()`，断言 fall/trial_done、stance/control/counter reset；fault context probe 以 seed
  `7100→15019` 验证事件改变，并由第二个同 seed shard 精确复现。同时修正 terminal observation 为
  post-action/pre-reset state，并让新 context 使用递增 context seed。R2 admission=true，未在 R2 启动 PPO。
- 2026-08-27：在 RTX 5060 Ti 上完成 representative R3。每个 case 使用 exact R1 XML/stance、
  nominal fault disabled、`4 env × 8 control steps = 32 env steps`；先做 no-update rollout，参数 L1
  delta 精确 `0`、inactive action 精确 `0`、finite tensor 与 2 trial/1 context reset 通过、fall `0`；
  再用同一真实 batch 做 `1 epoch × 1 minibatch` 的一次 clipped-PPO update。G1/Go2 parameter L1
  delta 分别约 `7.54/4.85`，gradient/loss/advantages/returns/更新后参数均 finite；通过 wrapper 对
  optimizer 的真实 `step()` 调用计数，两 case 都精确为 `1`，峰值 allocated VRAM 最大约
  `65.61 MiB`。没有 checkpoint、长训练或 walking/quality claim。representative R3 admission
  `2/2`；Task071 仍为 Tier A `2/5`，不得标记 passed。

- 2026-08-27：收紧最终 R1–R3 证据链。R1 artifact 记录 exact command、Python/MuJoCo/platform、git
  HEAD、overlay probe 与 Task070 stance helper source SHA；R2 验证并继承这些 SHA，R3 再绑定完整 R2
  payload/R1/overlay。overlay/model/R1/R2/R3 发布改为同目录 atomic replace，
  `write_artifact=false` 不再写 overlay、model 或 R1 文件。新增 actuator type/gear/ctrlrange、joint axis/range、canonical frame、
  mass/contact 与 XML/stance SHA 的正负回归。运行代码 commit 为
  `f7ba641b74b956ca765b804b12a5fd0124a49e32`；最终 artifact SHA256：overlay
  `dc954b9229df4ddd1d7a2b7556e1b2efa53ac680b066256d8ab12cfbe8b199c7`，R1
  `3614ed08e17d01effd64c564bdbc0e7c3c78d106ce06cbc27bbebb26ad38b6bf`，R2
  `1194cafa6ce80d4256a5113c20a7377620c068eeeebe69c0a6134ae9240d9571`，R3
  `ca7c048b116460a97324143aacfc8de3404856297b7816e16b7632944c0e7044`。targeted Task071
  `22 passed`；加入 WholeBody contract 后为 `39 passed / 1 failed`，唯一失败仍是既有 legacy-v2
  random biped seed0 strict static-stance solver，不由 G1/Go2 instance-bound route 掩盖。

- 2026-08-27：完成 versioned `official_sim_physics_overlay_v1` 正式绑定。工具只消费冻结
  attempt010 descriptor/manifest/XML，不调用当前 generator；逐 body 使用官方 Unitree commit
  `4134cb5dc7ff1ba7f484deda48b5274b58694519` 的 nominal mass、local COM、inertial quaternion、
  diagonal inertia，并按冻结 morphology scale 应用 `COM × s`、`inertia × s²`。joint
  damping/armature/frictionloss 与 actuator force range 来自 MuJoCo 3.12 编译后的官方有效值，
  position KP/KD 保持冻结 companion config，terminal friction 来自官方接触 geom；runtime fault
  process 未叠加。G1 映射 `30 body / 29 joint / 29 actuator / 2 terminal`，Go2 为
  `13/12/12/4`；结构 signature、canonical root、匿名 primitive geometry、joint order/axis/range、
  position-actuator 语义均 `2/2` 保持，冻结 raw SHA 未改变。bound compile 为 G1
  `nq/nv/nu=36/35/29`、Go2 `19/18/12`，finite `2/2`。证据
  `artifacts/official_sim_physics_overlay_v1.json` SHA256
  `dc954b9229df4ddd1d7a2b7556e1b2efa53ac680b066256d8ab12cfbe8b199c7`。
- 历史记录（已被上方 instance-bound stance R1 取代，2026-08-27）：使用同一冻结
  blueprint/physical manifest 直接加载 bound XML，完成当时的 fresh R1：
  compile/accounting/lineage/reset/paired actuator response 均 `2/2`，response 每 actuator `32`
  steps；stance `1000 × 0.002 s` 仍为 `0/2`，故 R1 admission=false、next gate=false，未进入
  R2/PPO。G1 主要表现为翻倒（max roll/pitch `3.13 rad`、base height drift `1.66 m`）；Go2
  support gate 失败且出现 terminal unload（minimum terminal load `0 N`）。证据
  `artifacts/r1_g1_go2_bound_official_sim_physics_overlay_v1.json` SHA256
  `e1c6a979ebea892343721db4b260677f02ec7b9aac24123a9be94f4552366b36`；本轮已消除旧
  `frozen descriptor lineage mismatch`，但没有消除 stance 动力学失败。官方参数仍只是 nominal
  simulator prior，`real_system_identified=false`，Task071 未 passed。

- 2026-08-27：按用户显式授权，把官方完整仿真资产稀疏检出到 ignored
  `.external/task071_full_sim_assets/`：Unitree `4134cb5`、DeepRobotics `e6753d2`、Booster
  `508cbee`、LimX `02adfbd`、EngineAI `335c60e`，5/5 固定 commit、官方 origin、detached clean。
  未检出 `.mnn/.npz/.npy/.csv/.onnx/.pt/.pth/.ckpt`，Booster motions 及 EngineAI policy/
  trajectory 明确排除。locked/offline MuJoCo 3.12 编译审计 8/8：G1 `nu=29`、Go2 `12`、
  Lite3 `12`、T1 23DoF `23`、HU_D04 `31`、PM01 EDU `24`、T800 `25`、T800Pro `43`；
  actuator count 8/8 exact、finite physics 8/8、jointed-body inertial gate 8/8。EngineAI 三型还取得
  enabled/sign/offset、PD stand KP/KD 和 parallel-ankle transform 的官方 YAML，slot count 3/3
  与 MJCF `nu` 一致；它们仍不是电气参数系统辨识。PM01 EDU 24DoF（含 head yaw）不得替换
  Task070 的 23DoF PM01；T800Pro palm YAML 引用的 `forward_net.mnn` 因禁用策略/checkpoint
  范围未下载，故完整 palm transmission 仍 fail-closed。证据：
  `artifacts/r0_official_sim_asset_intake.json`，SHA256
  `cb5aeefbda305ec79b48db5cbb34380b8893847c5168990abefe66be5873d4fb`。本轮没有训练；
  official asset intake pass 只表示 nominal prior 可编译，不改变 R1 stance `0/2`、frozen
  descriptor lineage mismatch 或 Task071 not passed。

- 2026-08-26：locked/offline `mujoco==3.12.0` CPython 3.11 import 已从 shared UV cache 成功；消费并
  校验 fresh arena artifact `artifacts/r1_g1_go2_fresh_arena.json`（exact G1 biped + Go2 quadruped，
  `stance_steps=1000`、`response_steps=32`、source SHA）。dynamic run 已完成但 stance `0/2`，因此
  R1 admission=false，next gate=false，未启动 R2/PPO。旧 Task070 arena 仅保留为 historical evidence，
  不作为 fresh R1。fresh XML 可绑定当前重编译输出，但与冻结 Task070 descriptor lineage 不匹配，故不是
  validated frozen Task070 R1 run。此次仅更新 probe/log evidence，未改 source code、controller 或 stance 参数。

- 历史记录（当时状态，2026-08-26）：按用户要求继续执行下一 gate。仅修改 motor-DoF-preserving v2 sampler：COM 现在按
  `base COM × global_scale × link_scale` 缩放；G1/Go2 只在完整 audited companion motor config、
  exact family/group slot coverage 且非 candidate 时，按 shared motor-family effort/bandwidth latent
  与 shared transmission-group efficiency latent 组合采样 strength/KP/KD，并采样全局 `0–40 ms`
  baseline delay。没有逐 slot 独立噪声，不声称 exact physical transmission mapping；Spot 等证据
  不完整 center 与 Tier-C candidate 保持 identity/fail-closed。nominal sampled strength 仅缩放
  `compile_mjcf` actuator force range；v2 `MotorProcess` baseline strength 固定为 `1`，只负责后续
  runtime fault，避免重复缩放 position target。R0 probe 现为 static `2/2`、physical-stack
  admission `2/2`，并通过同一 runtime helper 验证 50 Hz delay-step 量化；这不是 train-ready
  claim。G1 COM 最大编译舍入残差 `2.13e-10 m`，两 case failure reasons 均为空。新增 4 个
  Task071 静态回归 case；连同 frozen R0 compatibility baseline 共 targeted `5/5` 通过，targeted
  Ruff 通过。完整 `tests/test_task070_morphology.py` 为 `26 passed / 13 failed`：其中 12 个失败由
  当前 locked dev 环境缺少 MuJoCo 直接或间接引起，另 1 个是既有 Task070 visual artifact status
  断言；不将其记录为 full-suite pass。R1 仍未执行：locked Python 3.11
  `mujoco==3.12.0` wheel 不在共享 cache；最新在线获取经 4 次 retry 后在 `127.7 s` timeout，精确
  offline probe exit1。因此按 gate 停止，没有使用旧版/项目 `.venv` 绕过 lock，没有 no-update
  rollout 或 PPO update，Task071 未 passed。

- 历史记录（当时状态，2026-08-26）：用户视觉 gate 已满足；R0 输入冻结。G1/Go2 v2 physical probe static integrity
  `2/2`，training physics admission `0/2`。两者的 geometry/mass/friction 都有确定性随机变化，
  但 metadata 要求的 motor strength、KP、KD、delay 均未随机化；G1 的非零 COM 还没有随随机
  link geometry 一致缩放。历史记录中的 fresh dynamic rerun=`blocked_environment`（locked MuJoCo 3.12 wheel
  不在共享 cache；精确 locked/offline CPython 3.14 probe exit1；此前在线下载两次卡住后
  exit130 的 transcript 未保留、仅作历史背景），Task070 旧 generic stance 仅 `0/2`。按
  fail-closed 停止，没有 PPO/长训练；不得标记 passed。

- 历史记录（当时状态，2026-08-26）：由 Task070 当前结论建立；未开始训练或 artifact 生成。已知基线为 18/18
  compile/reset/paired actuator response、generic stance 0/18、无 gait、8 candidate humanoids
  `policy_adapter_compatible=false`。
- 待执行：Tier A 的 PM01/Spot/Lite3 R1–R3、Tier B R4、Tier C R5 与最终 R6 readiness matrix；逐阶段
  继续记录命令、硬件、SHA、denominator、失败原因和是否允许进入下一 gate。

## Review

- 2026-08-27：最终高风险只读复审无 P0/P1/P2/P3 findings。首轮提出的 precompiled XML/stance
  可分离 SHA、R1 source identity、未实际 step 的 fall reset、未验证的 context reseed、自报 optimizer
  update count 与 no-write 落盘问题均已以运行时 contract 和负例闭环；末轮 positional API 兼容问题
  也通过单字段 keyword-only 与 legacy positional regression 修复。Task071+WholeBody extended 为
  `34 passed`，targeted Ruff 与 `git diff --check` 通过。Task070 exact test 在挂载冻结 artifacts 后为
  `38 passed / 1 failed`，唯一失败是冻结 visual status 仍为 false 的既有断言；未修改或代签 Task070
  visual artifact。representative 结论仍严格限制为 G1+Go2 Tier A `2/5`，Task071 not passed。

- 2026-08-27：最新 targeted R1/R2/R3 回归已执行；G1+Go2 representative gate 通过，但 generic
  procedural biped `solve_static_stance` 的既有 seed-0 test 仍可能因 strict equilibrium 失败，这与
  Task070 generic `0/18` 风险一致，不由本次 instance-bound 解法掩盖。Tier A 的 PM01/Spot/Lite3、
  Tier B wheel、Tier C variable-DoF 仍未完成 R1–R5，Task071 not passed。

- 历史复审记录（后续已由 instance-bound R1 取代，2026-08-27）：overlay fail-closed mapping 与 bound R1 targeted regression `6 passed`（含篡改
  bound XML path/SHA 的两个负例），targeted Ruff 通过。R1 每次从冻结输入重建并校验 persisted
  overlay，caller 自报路径/SHA 不能进入动力学证据。fresh bound evidence 证明 lineage `2/2`，
  因此旧 lineage mismatch 不再是当前 G1/Go2 阻断；当前决定性阻断为 stance `0/2`。未执行
  rollout/PPO，不得声称 train-ready。最终只读复审无 P0/P1/P2/P3 findings；残余风险为 ignored
  官方资产与 MuJoCo 3.12 环境依赖，以及尚未通过的 stance gate。

状态：**G1+Go2 representative R0–R3 passed；Tier A 2/5；Task071 not passed**。

- 2026-08-27：官方资产 intake readback 确认 5/5 repo pin/origin/clean、8/8 MuJoCo compile 与 exact
  actuator count、禁用后缀 0 命中；artifact 明确 vendor mesh 只留在 ignored `.external`、不复制进
  anonymous witness，并把 PM01 版本差异、Lite3 welded point-mass feet、LimX helper-body 极小惯量、
  T800Pro palm 外部模型缺口保留为 warning。该审计未审核站立、步态、rollout 或 PPO，不能作为
  Task071 admission/pass。

- 2026-08-26：fresh R1 arena artifact 已由 probe 做 exact denominator、metadata、response/stance
  duration、source path/SHA 校验；locked/offline dependency available，dynamic completed，stance
  `0/2`，但 frozen Task070 lineage mismatch。R1 admission=false，未进入 R2/PPO。旧 blocked-environment
  记录仅为历史背景，不代表当前状态。补强 canonical-audit manifest SHA link 后，最终独立只读 reviewer
  报告无 P0/P1/P2。

- 历史记录（当时状态，2026-08-26）：独立只读 reviewer 首轮指出动态阻断命令不可复现、static gate 未逐项核对 compiler
  输出两个 P2。补入 exact locked/offline 命令、returncode/output/SHA，以及 compiled
  mass/friction、resolved motor config、effort/KP/KD、geometry hash 断言后复核；无剩余
  P0/P1/P2。结论仍为 static `2/2`、training physics `0/2`、R1
  历史记录中的 `blocked_environment`，不得进入训练。

- 历史记录（当时状态，2026-08-26）：本轮独立只读 reviewer 首轮发现 v2 sampled strength 同时进入 compiled actuator
  force range 与 runtime position target、以及 probe 未验证 runtime delay step 两个 P1。修复为
  v2 nominal strength 只归 compiler、`MotorProcess` identity baseline 只保留 fault ownership，并由
  runtime helper 验证 50 Hz delay 量化；同时把 admission 命名收窄为
  `r0_physical_stack_admission`。复核提出旧 profile helper regression 的一个 P2，补入 legacy、
  paper-faithful、archetype-constrained 三 profile 回归。最终复核 P0/P1/P2 均无。残余阻断仍是
  locked MuJoCo wheel 缺失；未执行 rollout/PPO，Task071 未 passed。

通过标准：独立只读 reviewer 确认 Task070 未被覆盖、`whole_body_v1_45` 未漂移、Tier A/B/C 分母未
混淆、所有 unknown evidence fail-closed，并确认最终报告没有超出本任务 claim 边界。Task071 未经
这些证据不得标记 passed。
