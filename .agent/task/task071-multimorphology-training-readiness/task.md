# Task071 — Task070 v2 多形态训练就绪门

状态：**active / r0_physical_passed_r1_environment_blocked**。

进入条件：Task070 v2 的最终输入 artifact、manifest/descriptor SHA 和 case registry 已冻结；
Task070 的用户视觉验收为前置 gate。2026-08-26 用户已通过 append-only overlay 接受 attempt010，
因此本任务可执行 R0/R1 gate；在具体物理属性与 fresh dynamic gate 未通过前，不得把任何 case
加入训练采样池或启动 PPO smoke。

## 目标与边界

把 Task070 v2 attempt010 的匿名 primitive-link morphology、`canonical_root`/state 和
actuation contract 接入 RTX 5060 Ti-first 的 WholeBody/MuJoCo 训练路径，建立逐实例、分层、
fail-closed 的 train-readiness gate。当前基线是：18/18 compile/reset/paired actuator response，
`canonical_root` 已接入 WholeBody；但 generic stance 为 0/18、没有 gait，8 个 candidate humanoid
的 `policy_adapter_compatible=false`，candidate transmission/motor evidence 仍 fail-closed。因此
本任务不得把现状描述为“全部 18 个可训练”。

本任务不改变 Task070 的视觉验收；`user_visual_acceptance` 仍由 Task070 管理，不由本任务代签。
不下载新资产、数据集、checkpoint，不启动长训练。所有 unknown provenance 保持 unknown，不猜补
真实电机参数。

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

- 2026-08-26：按用户要求继续执行下一 gate。仅修改 motor-DoF-preserving v2 sampler：COM 现在按
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

- 2026-08-26：用户视觉 gate 已满足；R0 输入冻结。G1/Go2 v2 physical probe static integrity
  `2/2`，training physics admission `0/2`。两者的 geometry/mass/friction 都有确定性随机变化，
  但 metadata 要求的 motor strength、KP、KD、delay 均未随机化；G1 的非零 COM 还没有随随机
  link geometry 一致缩放。fresh dynamic rerun=`blocked_environment`（locked MuJoCo 3.12 wheel
  不在共享 cache；精确 locked/offline CPython 3.14 probe exit1；此前在线下载两次卡住后
  exit130 的 transcript 未保留、仅作历史背景），Task070 旧 generic stance 仅 `0/2`。按
  fail-closed 停止，没有 PPO/长训练；不得标记 passed。

- 2026-08-26：由 Task070 当前结论建立；未开始训练或 artifact 生成。已知基线为 18/18
  compile/reset/paired actuator response、generic stance 0/18、无 gait、8 candidate humanoids
  `policy_adapter_compatible=false`。
- 待执行：R0–R6 逐阶段记录命令、硬件、SHA、denominator、失败原因和是否允许进入下一 gate。

## Review

状态：**R0 physical probe independently reviewed；Task071 not passed**。

- 2026-08-26：独立只读 reviewer 首轮指出动态阻断命令不可复现、static gate 未逐项核对 compiler
  输出两个 P2。补入 exact locked/offline 命令、returncode/output/SHA，以及 compiled
  mass/friction、resolved motor config、effort/KP/KD、geometry hash 断言后复核；无剩余
  P0/P1/P2。结论仍为 static `2/2`、training physics `0/2`、R1
  `blocked_environment`，不得进入训练。

- 2026-08-26：本轮独立只读 reviewer 首轮发现 v2 sampled strength 同时进入 compiled actuator
  force range 与 runtime position target、以及 probe 未验证 runtime delay step 两个 P1。修复为
  v2 nominal strength 只归 compiler、`MotorProcess` identity baseline 只保留 fault ownership，并由
  runtime helper 验证 50 Hz delay 量化；同时把 admission 命名收窄为
  `r0_physical_stack_admission`。复核提出旧 profile helper regression 的一个 P2，补入 legacy、
  paper-faithful、archetype-constrained 三 profile 回归。最终复核 P0/P1/P2 均无。残余阻断仍是
  locked MuJoCo wheel 缺失；未执行 rollout/PPO，Task071 未 passed。

通过标准：独立只读 reviewer 确认 Task070 未被覆盖、`whole_body_v1_45` 未漂移、Tier A/B/C 分母未
混淆、所有 unknown evidence fail-closed，并确认最终报告没有超出本任务 claim 边界。Task071 未经
这些证据不得标记 passed。
