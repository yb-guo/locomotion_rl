# Task070 — 成熟运动学先验约束的可站立程序化形态

状态：**redesign_required_motor_dof_preserving_v2 — 用户已完成 attempt010 视觉验收，但旧 R0–R5 仅为历史执行证据，未满足用户最终确认的 motor-DoF-preserving structural prior contract**。

进入条件：Task069 已由独立只读 reviewer 批准其“四家族程序化形态包络”窄 claim，
并冻结 `legacy_v2` 与 `locoformer_paper_faithful_morphology_v1` 的 manifest、XML、contract、
cache/checkpoint identity。Task067 的 stance artifact 仍只对其原 contract 有效，不得直接作为
本任务新 profile 的站立证据。

退出条件：新增一个独立版本化的、由来源和许可证可审计的多厂商成熟机器人运动学先验
约束的本地 morphology profile；覆盖 biped、quadruped、wheeled biped、wheeled quadruped，
输出形式与 LocoFormer 公开材料所证明的方向一致：四 family、程序生成、primitive-link、成组
多样化 morphology，而不是单个 named robot asset。每个被选中的成熟 reference 必须先转换为
**motor-DoF-preserving anonymous linkage**：保留全部 actuated motor DoF、body/joint parent-child
tree、joint semantic sequence、normalized joint local-axis vector（或有证据的 canonical-frame
变换）和 load-bearing attachment，只允许把复杂
mesh/外形替换成匿名 box/capsule/cylinder 连杆，并对 normalized geometry、mass/inertia 和
actuator parameter 做受约束随机化。生成体在结构、比例、attachment、joint chain、terminal
contact 和 actuator scaling 上通过 fail-closed gate，并在明确声明的 stance-hold 控制器下完成
接触可行、静力诊断和不少于 2 秒的站立验证。旧 profile 不漂移，named-robot reference 与
held-out evaluation 身份不混用，生成结果不携带厂商 mesh、纹理、Logo 或型号身份。

本任务允许的最终 claim：

> 本仓库实现了一个 LocoFormer-style 四家族程序化 primitive-link morphology 分布；其离散
> topology center 来自来源和许可证可审计的成熟机器人 motor-DoF-preserving anonymous
> linkage，其连续 geometry/physical 参数经过受约束随机化，并具有已验证接触站姿。

这里的“可站立”只表示：生成器同时给出与该实例绑定的 contact-aware stance，并在文档化的
position feedforward、biped base-attitude hold、wheeled-biped active wheel balance 或
wheeled-quadruped zero-velocity hold 控制下通过规定时长的站立 gate。
不得把它解释为零力矩被动站立、动态行走能力或训练后策略性能。

不得扩展为以下 claim：

- LocoFormer 官方 morphology generator/source reproduction；
- G1、Spot 或任何 named robot 的精确参数、URDF/MJCF、几何或像素 parity；
- 使用过的 named embodiment 仍是严格 unseen/OOD；
- LocoFormer policy、TXL、长上下文训练或大规模 RL 复现；
- 动态行走、sim2real 或真实机器人部署。

## 用户最终口径：motor-DoF-preserving linkage（优先于旧执行记录）

2026-08-24 用户明确纠正 Task070 的结构目标：不是从多个成熟机器人取“共同最小关节集合”，
也不是只抽一个连续 feature vector；而是把每个成熟构型的**全部电机运动自由度**保留下来，
将其转换成匿名 primitive 连杆结构，再在该结构上做程序化随机化并输出 LocoFormer-style 四
family 群体。

因此以下规则为硬约束；与后文旧 log/artifact 冲突时，以本节为准：

- source 中每个进入 selected motor set 的 actuated joint 必须在 anonymous linkage 中存在且
  恰好出现一次；不得为了统一 action dim、插值方便、站立容易或视觉简化而删除/合并 motor；
- 必须保留 motor joint 的 parent body、child body、joint type、normalized local-axis vector、
  semantic sequence 和所属 limb/module；允许匿名重命名和有证据的 local-frame canonicalization，
  但必须记录 source→anonymous frame transform，不允许只保留粗 axis class 或交换语义顺序；
- “primitive-link 简化”只允许替换 visual/collision geometry、匿名化 body identity，以及对
  normalized link/attachment ratios、mass/COM/inertia、limit/effort hints 做冻结范围内的变换；
- G1 `g1.xml` center 应保留 29 个 actuated DoF；EngineAI PM01 center 应保留 23 个 actuated
  DoF；Spot、Go2、Lite3 center 各保留 12 个 actuated DoF。23/27DoF G1 variant 只有在具体
  source/派生规则已登记并逐 motor accounting 后才能成为额外 structural center；
- `12DoF lower_body_biped` 只能作为显式 local ablation，且
  `counts_toward_mature_biped_prior_claim=false`；不得代表 G1/PM01 center；
- wheeled family 默认是在完整 non-wheel anonymous linkage 末端组合本地 wheel module：biped
  增加 2 个 wheel motor、quadruped 增加 4 个 wheel motor。除非 R0 登记了成熟 wheel-legged
  source 及其 replacement semantics，否则 wheel 不得替换或删除原 reference motor；
- 输出应呈现 LocoFormer-style 的四 family 程序生成群体和 topology/ratio 多样性，但不得声称
  官方 generator/source reproduction、Figure 6 pixel parity 或 named-robot parameter parity。

## Route

### 1. 设计原则

本任务不再让每个 link、attachment 和 joint order 彼此独立乱采。生成过程采用六层分离：

1. **reference intake**：仅接收来源、许可证、角色和允许提取字段已经登记的原型；
2. **motor inventory**：逐 source 冻结全部 actuated motor joint 和 source-to-anonymous semantic
   mapping，完成 motor count、parent-child edge 和 local axis audit；
3. **structural topology center**：把每个成熟 reference 转换成独立、匿名、motor-DoF-preserving
   body/joint graph；不同 DoF/graph 不得先压成同一个连续向量；
4. **normalized primitive geometry**：在所选 graph 上用 box/capsule/cylinder 实例化 body/link，
   按躯干尺度采样 attachment、肢段和 terminal 比例；
5. **physical parameters**：再采样 mass、COM、inertia、friction、joint limit 和 motor；
6. **stance realization**：为完整实例求解并验证 contact-aware stance。

### 1.1 Actuation 四层契约

Task070 v2 对每个 structural center 同时保存以下四层，但不得把这一本地工程分层描述成
LocoFormer 官方实现：

```text
structural descriptor
    -> transmission model
        -> coherent motor config
            -> runtime fault process
```

1. `structural descriptor` 决定 generalized joint/body tree：逐 source 保留 selected actuated
   joint 的 parent/child、axis、range、semantic order、module 和 attachment；这一层的 joint DoF
   accounting 仍是 Task070 的 motor-DoF-preserving gate。
2. `transmission model` 决定 physical actuator coordinate 与 generalized joint coordinate 的映射：
   至少区分 `simple_reduction`、`parallel_coupled`、`continuous_wheel_direct` 和
   `source_unspecified_joint_space_proxy`。并联踝、腰、腕必须按 source-supported group 记录 joint
   slots、physical actuator count、方向和 mapping provenance；若只知道“存在并联机构”而不知道
   几何/Jacobian，只允许记录 nominal joint-space approximation，不得伪造精确 transmission。
3. `coherent motor config` 保存同一真实 motor family/模块内相互关联的一组参数：motor class、
   gear/ratio evidence、peak/declared effort、velocity limit、rotor/reflected inertia、kp/kd、friction、
   delay 和 control mode。未知字段保持 `null`；URDF 中明显统一的 placeholder limit 必须拒绝作为
   quantitative prior。不得把 torque、speed、gear 和 armature 独立无条件采样成物理不可能组合。
4. `runtime fault process` 只在 nominal topology/transmission/motor config 之上施加 weak、dead、
   latency、lock 或 derating 等时变事件。Task070 preview 只绑定该接口并记录
   `not_applied_preview_only`；实际 fault rollout/policy training 仍由 Task055/063 或后续独立任务
   负责，不在视觉 witness 中伪造适应结果。

覆盖规则：每个 anonymous generalized joint 必须恰好进入一个 transmission group；每个 resolved
motor config 必须绑定一个 transmission group；wheel motor 使用显式 torque/velocity-compatible
control class，不能沿用 position-target joint 的语义。source 没有公开 transmission/config 时，
manifest 必须 fail closed 地标记 unknown/fallback，不能从相似机器人补值。

当前已清理的 G1、PM01、Spot、Go2、Lite3 及其本地 wheel composition 必须生成完整四层
`actuation_stack` manifest。用户后续授权下载的 AgiBot X1/X2、EngineAI T800/T800Pro、LimX
HU_D04/Oli、Booster T1、RobotEra STAR1 先进入 additional candidate intake；只有 source、license、
selected motor inventory、transmission 和 config provenance 逐项清理后，才能升级为新的
`quantitative_training_prior` structural center。

candidate 可以先生成 `counts_toward_task070_v2_pass=false` 的完整 structural descriptor、匿名
primitive-link witness 和 flat-arena actuator diagnostic，以便审计 source tree 与可编译性；这不等于
prior promotion。缺 transmission、motor family、controller 或 source-model/config 对齐证据时，
manifest 必须继续保留 `candidate_fail_closed`，不得进入 sampler、R4 denominator 或旧 45-slot
policy adapter。尤其 X1 official control config 的 31 physical motors 与当前 serial MJCF 的 29 个
结构关节必须分开记录；不得凭配置中的两只 claw motor 猜造 source tree 中不存在的关节。

建议的结构 center/组合规则：

- `generic_humanoid_biped`
  - 不是固定 12DoF 共同最小树，而是一个包含多个 discrete structural center 的 family；
  - G1 29DoF center：保留 12 leg motor、3 waist motor、左右完整 shoulder/elbow/wrist motor；
  - EngineAI PM01 23DoF center：保留 12 leg motor、1 waist-yaw motor，以及每臂
    shoulder pitch/roll/yaw + elbow pitch + elbow yaw/axial 共 5 个 motor；
  - pelvis、waist、torso、left/right leg、left/right arm 必须是可审计的 body tree，而不是把
    upper body 合并成无关节 trunk；
  - G1/PM01 在本任务中作为 `quantitative_training_prior`，用于提炼完整 anonymous motor
    joint tree、attachment 关系和 normalized ratio prior；
  - 至少一个许可证和来源均已清理的众擎（EngineAI）人形描述作为第二个 biped
    `quantitative_training_prior`；具体型号和文件必须在 R0 冻结，不能仅凭产品图片猜参数；
  - 23/27/29DoF、point-foot/no-ankle 或 lower-body-only 都只能作为 source-supported 或显式
    local-ablation sub-archetype；任何 motor 删除必须有 variant identity、来源/角色和 accounting。
- `generic_mammal_quadruped`
  - trunk；
  - front/rear、left/right 成对 attachment；
  - Spot 在本任务中作为 `quantitative_training_prior`，用于提炼简化 joint tree、front/rear
    attachment 和 normalized ratio prior；
  - 至少一个许可证和来源均已清理的 Unitree 四足描述，以及至少一个云深处
    （Deep Robotics）四足描述，分别作为额外 `quantitative_training_prior` center；具体型号
    和文件同样必须在 R0 冻结；
  - Spot、Go2、Lite3 各自保留 12 个 motor DoF；每腿保留 hip ab/ad、hip flex/ext、knee
    flex/ext 的 source parent-child 次序、local axis/sign 和 hip-offset/thigh/shank linkage；
  - 三个 12DoF graph 可以在 semantic correspondence 经审核后共享匿名 archetype grammar，
    但每个 source center 的 attachment/link ratios 和 axis convention 仍须单独可追溯。
- `terminal_wheel_module`
  - 在对应 load-bearing limb 末端增加真实 continuous hinge、contact cylinder、actuator、slot；
  - wheel axis 由 limb local frame 推导，不写死为与任意世界轴重合；
  - wheel radius、width、front/rear clearance 与 leg/trunk 比例联动。
  - 默认是 `local_engineering_module`，不是成熟 wheel-legged prior；组合后总 actuator count 必须
    等于完整 non-wheel structural center motor count 加 declared wheel motor count。

primitive box/capsule/cylinder 可以继续用于轻量 MuJoCo 编译；本任务要求的是运动学树、轴向、
attachment、比例、惯量和 contact topology 合理，不要求复制复杂 visual mesh。primitive-link
替换不得改变 motor joint count、motor semantic multiset 或 parent-child graph。

### 1.2 统一 canonical root 坐标契约

不同 source 的 MuJoCo free root 可能落在 pelvis、torso、trunk、base 或腰链上，且重排 source
body tree 会反转 parent/child、改变关节方向和动力学语义。因此 Task070 v2 不把所有 source
物理重根到同一 body；它保留原生 `root_free` 作为动力学自由根，并为每个生成模型增加唯一、
匿名且可查询的虚拟 site `canonical_root`。下游不得把原生 free-root qpos 直接当作跨型号统一
状态。

`canonical_root_frame_v1` 的可机读约定如下：

- 右手坐标，`+X` forward、`+Y` left、`+Z` up，四元数顺序 `wxyz`；
- biped 原点是左右第一髋关节 attachment 的中点；quadruped 原点是 front-left、front-right、
  rear-left、rear-right 四个第一髋 attachment 的质心；wheeled family 继承对应 non-wheel 根；
- site 挂在这些髋 attachment 的共同 parent body。T1/STAR1 等 source 即使把 free root 放在
  trunk，canonical site 仍跟随腰/骨盆 parent，而不破坏原 source tree；
- manifest 同时保存 `anchor_body_from_canonical` 与 `canonical_from_anchor_body` 可逆变换。
  translation 使用 blueprint length，MJCF 编译时乘 `physical.global_scale`；运行时位置单位为 m；
- 运行时统一调用 `read_canonical_root_state(model, data[, site_id])`，返回 site-to-world pose、
  canonical local angular/linear twist，以及 world `-Z` 在 canonical frame 中的 projected gravity；
- `WholeBodyMuJoCoShard` 的 actor observation、velocity/yaw reward、upright/fall 判定均读取该
  canonical state。没有该 site 的 frozen legacy/Task069 模型走原行为 fallback；
- `canonical_root_frame` 与 `canonical_root_v1` 必须进入 Task070 v2 contract/hash、blueprint/
  instance identity 和 preview manifest，防止旧 cache/checkpoint 把不同根语义视为同一 embodiment。

### 2. Reference 与 held-out 隔离

任何成熟机器人资料在使用前必须写入 `artifacts/r0_reference_registry.json`，至少包含：

- 厂商、型号、family、来源 URL 或已有本地路径、版本/commit/date；
- source 是否为厂商官方、上游维护、第三方转换或反向工程；
- 许可证标识、许可证文件 URL/path/SHA、适用范围、notice 要求、仿真/硬件/再分发限制；
- 是否实际读取了参数，还是只用于人工识别 joint semantics；
- 允许提取的字段与禁止复制的字段；
- `design_reference_only`、`quantitative_training_prior` 或 `heldout_evaluation` 三选一角色；
- 本地文件 SHA-256（如有）、获取方式及是否为用户逐来源明确授权下载。

R0 的最小多厂商候选池如下；“候选”不等于已经获准读取、复制或再分发：

| family | 原型候选 | Task070 目标角色 | 进入条件 |
| --- | --- | --- | --- |
| biped | Unitree G1 | `quantitative_training_prior` | 已有本地 pinned source，复核逐文件许可证 |
| biped | 众擎（EngineAI）人形 | `quantitative_training_prior` | R0 选择一个官方且许可证明确的具体型号/source |
| quadruped | Boston Dynamics Spot | `quantitative_training_prior` | 只在许可证允许的软件仿真范围内使用官方描述 |
| quadruped | Unitree 四足 | `quantitative_training_prior` | R0 选择至少一个官方且许可证明确的具体型号/source |
| quadruped | 云深处（Deep Robotics）四足 | `quantitative_training_prior` | R0 选择至少一个官方且许可证明确的具体型号/source |
| either | 其他厂商或型号 | candidate only | 按同一 registry/许可流程逐项加入，不自动继承许可 |

规则：

- 用户本次只批准更新 Task070 的多厂商设计，不等同于批量下载授权。未经用户逐来源明确
  授权，不下载 URDF、MJCF、mesh、checkpoint、数据集或上游仓库；优先盘点仓库已有、
  许可证明确的描述文件。
- 用户已明确指定 G1 与 Spot 为本任务的 `quantitative_training_prior`。它们分别进入 biped
  与 quadruped archetype 的设计/范围提炼，因此在 Task070 及使用该 profile 的下游实验中
  都属于 seen/reference，不得再列入 held-out/OOD 结果。
- 所有 quantitative prior 只允许提取经过 registry 声明的 motor-DoF-preserving anonymous
  topology、joint semantics、parent-child attachment 关系和 normalized ratios；“anonymous/
  simplified”不得删除、合并或重排 selected source motor。不得把 mesh、纹理、Logo、控制
  权重或精确型号身份带入生成结果，也不得声称精确参数 parity。若执行所需 source 尚不在
  本地，必须停在 R0 等待用户另行授权下载或提供文件。
- 公开可访问不等于许可证已清理。许可证缺失、只覆盖 SDK/代码但未覆盖模型、存在第三方
  notice，或来源为未验证反向工程时，默认只能是 `design_reference_only` 或排除，不能进入
  定量 prior。
- Spot 的官方 SDK/URDF 若进入本任务，必须在 registry 中显式保存其自定义许可约束；本任务
  只允许在软件仿真中提炼 prior，不把 Spot-derived asset 或 SDK 软件传播到其他硬件路径。
- 厂商顶层许可证不能替代逐文件检查；每个选中的 URDF/MJCF、mesh 和转换产物都必须记录
  copyright/notice 与许可来源。Task070 生产输出默认只使用本仓库生成的 primitive geometry。
- 同一 embodiment 一旦被用于定量拟合拓扑、比例或参数范围，必须标为 seen/reference；之后
  不得在同一实验中宣称严格 held-out/OOD。
- held-out registry 必须改选未参与 topology、ratio、range 或 gate calibration 的其他
  embodiment；具体名单在 R0 冻结，不得在看到 Task070 结果后调整。
- 若执行 leave-one-vendor-out，某一 fold 的 topology、ratio、range、distance normalization
  和 gate 阈值必须完全不读取该 fold 的厂商；使用全体厂商拟合过的全局范围不能再把其中
  任一家称为该结果的 held-out vendor。
- 通用 family 应包含 source-audited discrete structural centers；每个 center 保留其 motor graph，
  再从多项成熟设计原则中抽象 normalized ratios，不保留厂商名称、精确尺寸或唯一识别参数。
  若只有单一 reference，必须在 artifact 中披露该限制。
- LocoFormer 论文仍只用于证明四 family 和程序生成/参数随机化方向；本地 archetype 与范围
  全部属于 repository design choice。

### 3. 版本与兼容性

- 修正版 profile 建议命名为 `motor_dof_preserving_archetype_morphology_v2`，使用新的 embodiment
  contract。旧 `archetype_constrained_morphology_v1` 与其 R0–R5 artifact 保留历史 provenance，
  但不得承载 v2 语义或作为 v2 structural pass evidence。
- 不得修改以下已冻结默认或显式 profile 的确定性输出：
  - `legacy_v2`；
  - `locoformer_paper_faithful_morphology_v1`。
- structural descriptor SHA、structural center、source motor count、anonymous motor count、body-tree
  edge set、module mask、archetype、topology variant、symmetry policy、terminal type、wheel topology、
  stance contract 必须进入 blueprint manifest、structural hash、instance key 和
  cache/checkpoint identity。
- `reference_registry_hash`、`source_license_matrix_hash`、`prior_set_id`、prior contribution、
  sampling region、nearest-prior distance 和 clone-guard verdict 必须进入 manifest；其中会改变
  生成分布或许可边界的字段必须进入 contract/cache/checkpoint identity，不能只写在旁路日志。
- 连续 geometry/physical identity 与 topology identity 分离；相同 archetype/topology 的不同
  连续采样不得伪装成新 topology。

### 4. 随机化原则

- 每个定量 prior 先转换为两份绑定 artifact：不含厂商/mesh identity 的 discrete structural
  descriptor，以及只作用于该 descriptor 的 normalized continuous feature vector。structural
  descriptor 至少包含 body tree、motor inventory、joint semantics/order、parent-child edge、
  normalized local-axis vector/canonical-frame transform、limb/module grouping、terminal/contact type；
  continuous vector 至少包含
  attachment ratios、link ratios，以及可合法提取时的 mass/effort scaling hints。
- sampler 必须区分并记录三个区域：`prior_neighborhood`、多个 prior center 之间的
  `interpolation_band`，以及其外侧但仍在工程硬边界内的 `bounded_outward_band`。不得把无界
  extrapolation 或 rejection 后偶然存活的样本称为“往外派发”。
- R0 必须在运行结果前冻结 feature normalization、distance metric、各维权重、每个区域的
  距离上下界、mixture weight、hard feasibility envelope 和每区域 expected denominator。
- 每个生成实例记录 selected structural center、structural descriptor SHA、source/anonymous motor
  accounting、continuous prior contribution、nearest prior、normalized distance、sampling region
  和 clone-guard 结果。输出不得与任何 named prior 的 topology+ratio+parameter fingerprint
  在冻结容差内完全相同；该 guard 只防止意外复制，不能被描述成法律结论。
- 左右默认成对对称；只有显式 `bounded_asymmetry` profile 才能引入有限不对称，并记录幅度。
- attachment 必须由 trunk/pelvis 实际表面、link radius、最大 physical scale 和 clearance 推导；
  禁止固定世界坐标导致躯干尺寸变化后穿插或悬离。
- link-length、trunk aspect、hip/shoulder spacing、wheel radius/width 使用 normalized ratio，
  范围在 R0 先冻结再生成 R3，不得看失败结果后放宽。
- joint order 只能从 archetype 白名单采样；joint axis 必须在 parent/child local frame 中有明确
  语义，不允许任意轴排列制造结构多样性。
- mass、COM、inertia 必须与生成 geometry 相容；motor effort、gear、kp/kd 范围应随质量、
  尺度和关节杠杆臂缩放，并保留未缩放与最终值。
- actuator randomization 必须先按 transmission class 和 source motor family 采样 coherent tuple，
  再施加小范围 module/左右 variation；gear ratio、torque/speed envelope、reflected inertia、kp/kd
  不得作为彼此无关的独立均匀变量。parallel group 的 fault 作用于 physical actuator 后再通过
  transmission mapping 影响多个 generalized joints，不能伪装成单一 joint strength scalar。
- 约束失败必须保留 seed、阶段、错误、尝试次数和 expected denominator；不得无限 rejection
  sampling，也不得只输出成功样本。若采用 bounded deterministic retry，retry 序列必须进入
  contract 并可完全复现。
- topology 只能在 source-supported structural center 或显式 local-ablation 白名单内做离散变异；
  continuous outward distance 不得通过删除任意 motor joint、交换语义轴或移动 terminal 到
  非承重链来伪造。不同 motor count/body graph 的 center 不得逐维连续插值；只有 graph-isomorphic
  center 或显式 module correspondence 通过审核后才能做 structure-aware interpolation。

### 5. “可站立”的契约定义

R0 必须在运行最终矩阵前冻结数值阈值和理由。至少验证：

- 所有 declared support terminal 可同时到达地面容差带；
- reset 无 terminal 穿地、无非预期 self-contact、无断链；
- 非轮式 support polygon 与 COM projection/接触力解满足预先冻结的 margin/residual gate；
- wheeled family 明确使用 wheel brake 或 zero-velocity hold，报告轮轴速度、轮端接触和所需
  actuator effort，不把摩擦锁死假装成轮子；
- stance qpos、ctrl、qacc、contact wrench、constraint force 和 actuator force 全部有限；
- 以 2 ms timestep 运行不少于 1000 step（2 秒），全过程无 solver fatal/warning、NaN/Inf、
  非 terminal 支撑、terminal 丢失、轮间接触或超阈值穿透；
- base height、roll/pitch、qvel、terminal load 和 actuator saturation 使用 R0 冻结阈值判定；
- 单列 zero-control 结果作为诊断，但 zero-control 失败不等同于 stance-hold 失败，二者不得混写。

旧 Task067 stance artifact 只能作为算法参考；新 profile 的 stance 必须重新求解、重新绑定
instance key，并产生新的 artifact/contract。

## 允许修改范围

- `src/h200_locomotion_lab/robots/procedural_morphology.py`，或新增小型
  `src/h200_locomotion_lab/robots/archetype_morphology.py`；共同的 manifest/compile/physical
  逻辑必须复用，不复制第二套基础 generator；
- 可新增小型 `src/h200_locomotion_lab/robots/structural_prior.py` 或等价模块，负责读取、验证和
  实例化 R0 motor-DoF-preserving structural descriptor；不得把 source-specific graph 再散落成
  family 分支里的硬编码 `joint_suffixes`；
- 可新增小型 `src/h200_locomotion_lab/robots/actuation_stack.py` 或在现有
  `archetype_morphology.py` 的 v2-only 路径中实现 transmission/coherent-config manifest；不得改变
  legacy/Task069/Task070-v1 的 actuator 输出；
- `src/h200_locomotion_lab/robots/whole_body_slots.py`（仅在新语义确有缺口时）；
- stance/contact realization 的小型独立模块，不得覆盖 Task067 contract；
- `src/h200_locomotion_lab/robots/__init__.py`；
- 新增 `src/h200_locomotion_lab/tools/task070_*` 验证和 gallery 工具；
- `tests/test_task070_*.py` 及直接相关 whole-body contract tests；
- `.agent/doc/locoformer.md`、本 task 的 `task.md` 与 `artifacts/`。

若需要修改 policy、reward、PPO/TXL、Task061/062 训练链、Task067 artifact 或其他任务源码，
必须拆成独立 task，不属于本任务。

## 禁止项

- 不在没有用户明确授权时下载 robot asset、mesh、checkpoint、数据集或上游仓库。
- 不启动 H200；默认本地 RTX 5060 Ti + MuJoCo/headless，gallery 可使用本地 renderer。
- 不复制许可证不明的 URDF/MJCF 数值或 mesh。
- 不把 G1/Spot 等 named robot 的精确参数硬编码为匿名 archetype 后继续声称它们 unseen。
- 不以匿名化、primitive geometry、统一 slot schema、站立控制器或 LocoFormer-style 输出为理由
  删除/合并成熟 reference 的 selected motor DoF。
- 不把 G1 29DoF、PM01 23DoF 等不同 graph 先压成相同 12DoF lower-body tree 后再称为多中心
  structural prior。
- 不修改或复用 Task069/legacy contract 名称承载新语义。
- 不用放宽 contact/solver/站立阈值、增加随机 restart、丢弃失败 seed 或筛图制造通过。
- 不把“100 step finite smoke”“reset 看起来直立”或“fall flag 未触发”写成稳定站立。
- 不把 stance-hold 通过扩展成动态行走、policy、sim2real 或真实机器人结论。

## Closed units

### R0 — Reference registry、claim contract 与冻结 baseline

#### Route

1. 重新盘点已有本地 robot descriptions，不下载新资产；把 G1、Spot、Unitree 四足、云深处
   四足、众擎人形及后续候选逐项登记到 `artifacts/r0_reference_registry.json`。
2. 生成 `artifacts/r0_source_license_matrix.json`，逐 source 冻结官方性、commit、license、
   notices、允许提取字段、禁止传播字段、角色和 clearance verdict；未 clear 的 candidate 不得
   进入定量 feature extraction。
3. 新增 `artifacts/r0_structural_prior_descriptors.json`；逐 quantitative prior 保存：
   - source actuated motor inventory 和 exact count；
   - source joint name 到 anonymous semantic slot 的一对一 mapping；
   - body nodes、parent-child edges、joint type、normalized local-axis vector/axis class，以及任何
     source→anonymous canonical-frame transform；
   - leg/waist/arm/wrist/wheel 等 limb/module grouping；
   - load-bearing terminal、attachment frame 和 normalized link/attachment ratios；
   - source SHA、registry entry SHA、descriptor canonical SHA 和 extraction method/version。
4. 结构解析必须区分同侧不同 branch；不得把 G1/PM01 的 leg 与 arm 合成一个 limb。任何
   `semantic=other`、重复 source motor、遗漏 actuated motor、未解释的同一 limb semantic collision、
   断链或 parent-child cycle 均 fail；PM01 `ELBOW_YAW` 不得再次映射成 `elbow_pitch`。
5. 冻结 expected motor accounting：G1 `29→29`、PM01 `23→23`、Spot/Go2/Lite3 各
   `12→12`；额外 G1 23/27 variant 只有在 descriptor 中绑定具体 source/派生 motor mask 后可用。
6. 在任何生产修改前重新绑定 legacy v2、Task069 v1 和 Task070 v1 的固定 seed
   manifest/XML/contract/cache baseline，写 `artifacts/r0_compatibility_baseline.json`。
7. 重建 `artifacts/r0_design_contract.json`：区分 paper facts、各 prior 实际派生的 engineering
   abstractions、local ratio/range/distance choices、license constraints 和 unknown claims。
8. 冻结 discrete structural center plan、graph-isomorphism/module-correspondence policy、normalized
   ratio ranges、continuous feature distance、三个采样区域、clone guard、symmetry policy 和全部
   stance/gate 阈值。structure distance 与 continuous distance 必须分开，不得用 joint-count scalar
   代替 graph。
9. 旧 v1 R0–R5 artifact 必须保留 provenance但标为 superseded；新 artifact 显式记录 predecessor
   SHA。旧 v1 的 12D biped clearance/standing evidence 不得作为 v2 structural pass evidence。

#### Log

- 2026-08-21：用户确认采用“成熟机器人连杆构型简化后再受约束 random”的方向。Task070
  仅创建任务文档；本次未下载、读取或复制任何新 robot asset，也未修改生成器。
- 2026-08-24：执行 R0 本地盘点。已确认 pinned local checkout
  `.external/unitree_rl_mjlab` 中存在 Unitree G1 `g1.xml`，注册为 biped
  `quantitative_training_prior` / seen-reference，并仅允许提炼简化 joint tree、attachment
  和 normalized ratios。工作区及 pinned external checkouts 中未发现 Spot 的 URDF/MJCF 或
  运动学描述；`.venv` 中的 `mujoco_warp` `spot.obj` 仅为测试碰撞 mesh，不具备 prior 资格。
  已冻结 held-out identity 为 Berkeley Humanoid 与 ANYmal C，二者未参与 Task070 calibration。
  已生成 `artifacts/r0_reference_registry.json`、`r0_compatibility_baseline.json` 和
  `r0_design_contract.json`；legacy/Task069 identity 未改动。由于缺少 Spot source，未进入
  R1、未修改 production generator、未下载或读取新机器人资产；等待用户提供授权本地描述或
  明确授权获取 Spot description。
- 2026-08-24：用户批准把 Task070 更新为多厂商原型池：G1、Spot 保持既有
  `quantitative_training_prior` / seen-reference；新增许可证明确的 Unitree 四足、云深处四足
  和众擎人形作为目标 quantitative prior center，并采用匿名 archetype、多中心插值与有界
  outward band。此次授权仅覆盖任务设计文档更新，不是任何外部仓库或资产的下载授权。
  因设计 contract 已扩展，先前双 prior `r0_reference_registry.json`、
  `r0_compatibility_baseline.json` 和 `r0_design_contract.json` 不再足以通过当前 R0，必须重建并
  绑定 predecessor SHA；本次仍未下载或读取新 robot asset，也未修改 production generator。
- 2026-08-24：按用户要求清理过期文件。旧双 prior 的三个 R0 JSON 已从 active
  `artifacts/` 移入 `artifacts/superseded/`，内容和原 SHA 保留，并新增 README 说明其
  predecessor-only 身份。未删除证据、未移动其他任务文件；新的多厂商 R0 仍必须在 active
  路径重建，且显式绑定这些 predecessor SHA。

#### Review

- reference 角色、许可证或 held-out 身份不明确时，R0 不通过。
- 任何 named robot 若同时进入 quantitative prior 和 held-out list，fail closed。
- 未在运行结果前冻结范围/阈值时，不得进入 R1。
- 任一 quantitative prior 的 anonymous motor count 不等于 source selected actuated motor count，
  或 motor semantic/parent-child/local-axis mapping 不完整时，R0 不通过。
- 只提供连续 8D/12D feature center 而没有 structural descriptor/body graph，R0 不通过。
- 2026-08-24 用户最终口径更新后，以下旧 R0 `execution_verified` 记录仅为 v1 历史 provenance；
  新 v2 R0 尚未执行，不得引用旧 SHA 宣称 motor-DoF-preserving R0 passed。
- 以下为旧 v1 R0 记录：2026-08-24 R0 重新执行并通过；独立审核 remediation 后重建
  active artifacts：
  - `r0_reference_registry.json`
    SHA-256 `931bf5346fe379b3fd1c25e91d39007704e06208a4b2c93b38b96293a58922f1`；
  - `r0_source_license_matrix.json`
    SHA-256 `1d1c243797cb721942a88445a2ea8c4f9ee30b4237eb4c6e9214850fa1c3465a`；
  - `r0_compatibility_baseline.json`
    SHA-256 `81f80ee296172a2ae7e7702670e1a0be7c2b2bdff615e7af31e1fc0662c786c5`；
  - `r0_design_contract.json`
    SHA-256 `5d9bc169681984d5c9682cec6bbaa2e031e82c9eda4439ee716f95d28ae2cf7d`。
- cleared quantitative prior pool：biped `unitree_g1`、`engineai_pm01`；quadruped
  `spot_base`、`unitree_go2`、`deeprobotics_lite3`。Spot 保留 SDK software simulation
  restriction；生产输出仅使用本仓库 primitive geometry。Task067 artifact guard 明确为
  `accepted_as_task070_stance_evidence=false`。
- remediation 版 R0 明确冻结实际执行的 12D prior centers、`FEATURE_LIMITS`、
  attachment realized envelope、biped foot/wheel geometry contract、stance support/static
  residual/6D contact-wrench residual/self-contact/wheel-wheel gate，并在 registry/source matrix
  中记录 limited R0 source-file authorization evidence；不再把 source parser 的 8D observation
  当作 distance center。

### R1 — Archetype topology 与版本隔离

#### Route

1. 实现显式 `motor_dof_preserving_archetype_morphology_v2` profile/contract；保持
   Task069/legacy/Task070 v1 baseline byte/hash stable。
2. generator 先选择 R0 discrete structural center，再从 descriptor body roots 递归创建匿名
   primitive links、joint 和 actuator；不得由 family 分支写死统一 `joint_suffixes`。
3. 每个 selected source motor 必须产生且只产生一个 anonymous actuated joint；保留 semantic
   sequence、parent-child edge、normalized local-axis vector，或按 descriptor 中显式
   source→anonymous frame transform 验证等价轴向。source/non-wheel/added-wheel/total actuator count
   全部写入 manifest。
4. 实现 G1 29DoF、PM01 23DoF、Spot 12DoF、Go2 12DoF、Lite3 12DoF structural center witness；
   每个 witness 均只使用 primitive geometry，且不能携带 named body/model identity。
5. 实现 terminal wheel local module：完整 biped center 增加 2 wheel motor，完整 quadruped center
   增加 4 wheel motor；除非 descriptor 明确声明 replacement，不删除原 motor。
6. attachment 随 pelvis/trunk/parent body 尺度变化；symmetry 默认成对，bounded asymmetry
   显式化；shoulder、waist、hip、front/rear attachment 都必须来自 graph 中正确 parent body。
7. manifest 完整记录 registry/license/descriptor hash、selected structural center、motor accounting、
   body-tree hash、module mask、continuous prior contribution、sampling region/distance、clone guard、
   reference role、archetype、variant、axes/order、ratios、wheel 和 identity。

#### Log

- 2026-08-24：新增独立 `archetype_constrained_morphology_v1` profile 与
  `procedural_archetype_constrained_morphology_v1` contract。实现匿名
  `generic_humanoid_biped`、`generic_mammal_quadruped` 和 terminal wheel module，manifest
  记录 R0 registry/license/design hash、prior set、sampling region、nearest prior/distance、
  clone guard、stance contract、primitive-geometry-only 和 seen/reference 边界。
- 2026-08-24：`legacy_v2` 与 `locoformer_paper_faithful_morphology_v1` 的 profile/contract
  名称、XML/cache identity 未复用新语义；R5 compatibility 复核 256/256 passed。

#### Review

- 固定 attachment、任意 joint permutation、左右独立乱采或新旧 cache identity 冲突均 fail。
- 只有外观变化而无 topology/ratio contract，不通过。
- source motor 被删除/合并、leg/arm branch 丢失、joint semantic/order/local axis vector 漂移、
  anonymous motor count 与 source accounting 不相等，均 fail closed。
- `12DoF lower_body_biped` 即使 compile/stance 通过，也不能满足 G1/PM01 structural center gate。
- 当前旧 R1 结论：**historical_v1_execution_only**；v2 R1 尚未执行。

### R2 — Constrained geometry、physical 与 actuator scaling

#### Route

1. 实现分层 deterministic sampler；discrete structural-center selection、graph-compatible reference
   mixture、sampling region、topology、geometry、physical 分别 hash。
2. 不同 motor count 或非同构 body graph 不做逐维 topology interpolation；G1 29 与 PM01 23
   保持独立 center，只有 R0 明确审核的 module correspondence 才能生成新离散 variant。
3. 在 selected descriptor 上应用 R0 冻结的 normalized ratios、continuous prior distance、bounded
   outward limits、clearance、inertia 和 motor scaling；每个采样维度必须有 realized geometry/
   physical consumer，禁止 distance 中存在未消费 feature。
4. 对所有约束失败保留 expected denominator、seed、region、nearest prior、stage、error 和
   retry trace。
5. 验证同 seed 全字段稳定、不同 physical seed 不改变 topology identity；验证 prior、
   interpolation、outward 三个区域均有预先冻结 denominator，而非只保留成功区域。
6. 验证 perturb 每个 attachment/link-ratio feature 会改变其声明的 realized body/link 字段；尤其
   quadruped `lower_link_fraction` 必须改变 shank length，不能只影响 distance/metadata。

#### Log

- 2026-08-24：实现 deterministic multi-center sampler。每 family seeds 0..31 的 region
  denominator 固定为 `prior_neighborhood=8`、`interpolation_band=16`、
  `bounded_outward_band=8`；distance bands 和 clone guard 写入 manifest 与 focused tests。
- 2026-08-24：physical randomization 独立于 topology identity；不同 physical seed 改变
  physical hash/cache key，不伪装成新 topology。wheel width、biped support、quadruped
  wheel spacing 和 actuator scaling 均约束在 Task070 新 profile 内，未改 Task069/legacy。
- independent-review remediation 后，actuator scaling 增加
  `task070_morphology_aware_mass_lever_arm_scaling_v1` metadata，保存 raw motor/kp/kd、
  morphology mass/lever factors 与 final values；同一 physical seed、不同几何会改变至少一个
  共同 actuator slot 的 final scaling。

#### Review

- mass/inertia 与 geometry 不相容、motor 不随尺度变化或失败样本被隐藏时 fail。
- continuous sampler 改变 motor count/body graph/joint semantic，或 feature 未被 geometry/physical
  消费时 fail。
- 当前旧 R2 结论：**historical_v1_execution_only**；v2 R2 尚未执行。

### R3 — Contact-aware stance realization

#### Route

1. 为每个完整 motor-DoF-preserving instance 生成与 instance key 绑定的 stance
   qpos/ctrl/contact target；waist/arm/wrist 等非承重 motor 也必须有 finite nominal qpos/ctrl，
   不得从模型删除来简化站立。
2. 非轮式验证全 support terminal、support polygon、COM/contact wrench 和 residual。
3. 轮式验证 brake/velocity-hold、wheel speed、接触力和 effort，不锁死 joint。
4. 四 family 按 R0 冻结的 structural-center × sampling-region denominator 运行不少于
   1000×2 ms stance-hold；每个 G1/PM01/Spot/Go2/Lite3 center 都必须有预先冻结的 witness，不能
   只让最易站立的 topology 进入矩阵。
5. 单独保存 reset、zero-control、stance-hold、扰动前后诊断，不合并结论。
6. flat-arena actuator response 作为独立前置诊断：在带重力和平地的 MuJoCo 中逐 actuator reset、
   施加有界小幅命令，并验证对应 position joint 的 qpos response 或 continuous wheel 的 qvel
   response、finite state 和 solver warning。该 gate 只证明 actuator wiring/response，不证明站立，
   更不证明 walking；stance 仍必须独立满足上方 1000-step contract。

#### Log

- 2026-08-24：新增 task-scoped CLI
  `h200_locomotion_lab.tools.task070_morphology_verification`。remediation 后 stance 使用与实例
  绑定的 contact-aware reset qpos/ctrl：biped 为 base-attitude hold；quadruped 为 position
  feedforward；wheeled_biped 为 active wheel pitch/pitch-rate balance；wheeled_quadruped 为
  wheel zero-velocity hold，`wheel_velocity_hold_gain=4.0`。zero-control 与 disturbance 只作为
  诊断保存，不并入 stance-hold 结论。
- 2026-08-24：remediation 后正式 R4 matrix 运行
  `--seed-range 0:32 --steps 1000 --timestep 0.002 --render`；四 family stance-hold、
  support gate、final static contact residual、final 6D contact-wrench force/torque residual、
  nonterminal-support exclusion、self-contact exclusion 和 wheel-wheel contact exclusion 均为
  32/32。阈值覆盖 finite qpos/qvel/qacc/ctrl/qfrc_constraint/efc_force/actuator force/contact
  wrench、solver warning、floor/self penetration、terminal missing、terminal load、support
  polygon/area、roll/pitch、qvel、base height drift、actuator saturation、wheel speed/effort。

#### Review

- 任一冻结 structural-center × sampling-region 的 expected denominator 未全部通过，不得声称该
  v2 profile 全分布“可站立”。
- Task067 旧 artifact、短 finite smoke 或图片不能替代本阶段证据。
- 站立通过但 motor accounting/structural descriptor gate 失败，仍不得进入 R4 passed records。
- 当前旧 R3 结论：**historical_v1_execution_only**；v2 R3 尚未执行，旧 12DoF biped stance 不
  是完整 G1/PM01 linkage 的站立证据。

### R4 — 结构矩阵与视觉 gallery

#### Route

1. 每个 seed 输出一份可独立 compile 的 primitive-linkage MJCF/XML 与 manifest；同时输出新的
   v2 morphology matrix，逐 seed 保存 structural descriptor SHA、selected anonymous
   structural center、source/non-wheel/wheel/total motor accounting、body-tree hash、joint semantic
   sequence by limb/module、parent-child/axis coverage、ratios、continuous prior contribution、nearest
   prior/distance、sampling region、clone guard、contact、stance、solver、actuator、error 和 artifact
   SHA。matrix record 必须绑定对应 MJCF/XML SHA，不能只保存 feature vector 或渲染图。
2. 输出 LocoFormer-style 四 family 群体：每 family 按冻结 denominator 输出 montage，至少包含
   oblique/side/front/contact closeup；biped 视图必须看见 pelvis/waist/torso、双臂和腿链，轮式
   增加 axle/ground/wheel-clearance 近景。
3. 同时渲染 raw reset 与 verified stance，避免把相机姿态误认作 stance 修复。
4. 执行 agent 必须用本地 image viewer 逐张打开最终 montage/closeup，并用逐图 manifest
   记录 family/seed/view、腿数、attachment、比例、terminal、轮轴、接触、穿插和裁切。
5. 每个 family 的矩阵和 gallery 必须先按 structural center，再按 `prior_neighborhood`、`interpolation_band`、
   `bounded_outward_band` 分层报告 expected/built/stance/visual denominator，并证明所有已 clear
   prior center 对应的目标 family 均实际进入预先冻结的 sampling plan。
6. biped 必须至少展示一个 G1-derived 29DoF anonymous primitive witness 和一个 PM01-derived
   23DoF anonymous primitive witness；wheeled 对应 witness 总 actuator count 分别为 31 和 25。
   quadruped 三个 12DoF center 及其 16-actuator wheeled composition 都必须有 witness。
7. additional humanoid candidate gallery 必须按 source model 的完整 actuated inventory 分开输出；
   head、hand/finger 或 source-specific axial joint 不得为了适配旧 45-slot schema 被删除或错映射。
   candidate gallery 只用于结构/视觉和 arena wiring 审计，保持
   `user_visual_acceptance=false`、`counts_toward_task070_v2_pass=false`，不能补入 R4 denominator。

#### Log

- 2026-08-24：remediation 后输出 `artifacts/r4_archetype_morphology_matrix.json`
  SHA-256 `fbfbe166ef779b4cb40473bb5623eb55f6bc888bab80c10b706413e654681493`。
  summary：biped/quadruped/wheeled_biped/wheeled_quadruped 均 built/compiled/deterministic/
  finite/slot/wheel/reset/stance/support/contact-residual/contact-wrench-residual/
  nonterminal-support/self-contact-clear/wheel-wheel-clear/region-band/clone-guard `32/32`，
  各 region denominator `8/16/8`。
- 2026-08-24：输出 `artifacts/gallery_task070/`，每 family 32-sample montage、per-region
  closeup、轮式 contact+wheel-axis closeup、raw-vs-verified comparison；本地 image viewer 查看
  四 family montage 与关键 contact/wheel-axis closeup。`r4_visual_observations.json`
  SHA-256 `88515a588c9893788f6cea001c107f876fe46181de47f7ee348f81a5e0027818`，
  逐图 manifest 覆盖 174 张 gallery PNG，记录 family/seed/view/stage/region、腿数、
  attachment、比例、terminal、轮轴、接触、穿插、裁切、SHA 和 visual denominator。视觉复看
  明确记录：biped full foot length / leg length 最大 `0.78`；wheeled_biped 为大轮
  active-balance wheel-legged archetype，旧 `x=+0.50/-0.50` 悬离 attachment 已移除；R4 matrix
  `visual_inspection.status=passed_execution_agent_full_gallery_local_image_viewer_review`。

#### Review

- 机器人过小、关键结构裁切、轮/腿重叠、足端悬空但未披露，或只做自动像素检查时 fail。
- 视觉只证明结构 sanity，不证明 named-robot 或 Figure 6 parity。
- 任一 record 的 motor count、required semantic multiset、parent-child edge、normalized local-axis
  vector/canonical-frame equivalence 或 module accounting 不精确匹配 selected descriptor，R4 fail
  closed。
- 只有同一个 12DoF biped topology、没有 waist/arms 的 montage，不满足 LocoFormer-style 成熟
  biped linkage 输出 gate，即使 stance/visual decode 全通过也必须 fail。
- 当前旧 R4 结论：**historical_v1_execution_only**；旧 128/128 matrix/gallery 不满足 v2 gate。

### R5 — Compatibility、回归与独立 claim gate

#### Route

1. 复核 legacy v2、Task069 v1、Task070 v1、motor-preserving v2 profile 的
   contract/cache/checkpoint 四方隔离，并验证更换 prior registry、license matrix、structural
   descriptor 或 distance contract 会 fail closed 或产生 distinct identity。
2. 运行 Task070 focused tests、whole-body contract tests、Ruff、agent inspection 和全量 pytest。
3. 写新的 v2 final verification artifact，绑定 R0–R4 artifact/source/structural-descriptor SHA 和
   全部命令结果。
4. 更新 `.agent/doc/locoformer.md`，明确三层区别：Task069 family envelope、Task070
   multi-vendor engineering morphology/stance、仍未完成的 policy/scale/sim2real。
5. R5 从每条 blueprint manifest 重新计算而不是信任 summary：
   - exact source→anonymous motor bijection；
   - exact source/non-wheel/wheel/total actuator accounting；
   - required joint semantic multiset/order；
   - required parent-child edge set 和 normalized local-axis vector/canonical-frame equivalence；
   - module mask/DoF sum；
   - structural descriptor SHA/identity binding。
6. 由独立只读 reviewer 复核 reference leakage、motor-DoF preservation、结构合理性、LocoFormer-style
   四 family 输出、站立证据、gallery 和 claim。

#### Log

- 2026-08-24：round-2 remediation 后 focused pytest：
  `.venv/bin/python -m pytest -q tests/test_whole_body_contract.py tests/test_whole_body_extended.py
  tests/test_whole_body_usability_gate.py tests/test_task069_*.py tests/test_task070_*.py`
  结果 `52 passed in 44.03s`；日志
  `artifacts/logs/task070_focused_pytest.log` SHA-256
  `ffc49358f249debca3fffba0215b53a68f4a41de2f8275ac4ac80fdc09dc804b`。
- 2026-08-24：round-2 remediation 后 Ruff：
  `.venv/bin/ruff check src/h200_locomotion_lab/robots
  src/h200_locomotion_lab/tools/task070_*.py tests/test_task070_*.py`
  结果 `All checks passed!`；日志 `artifacts/logs/task070_ruff.log` SHA-256
  `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`。
- 2026-08-24：agent inspection：
  `.venv/bin/python -m h200_locomotion_lab.tools.inspect_agent` 通过；日志
  `artifacts/logs/task070_inspect.log` SHA-256
  `9c14487abe74aff4e4e4680a12ea268c7c0c7c1f1234eabffd7eafd141ebf003`。
- 2026-08-24：full pytest：`.venv/bin/python -m pytest -q`
  结果 `839 passed, 35 warnings in 242.14s`；warnings 为既有
  `torch.jit.script` deprecation；日志 `artifacts/logs/task070_full_pytest.log` SHA-256
  `dfab2e0794ac2dbe5299c5d12ce19b00d4e9ba8467f5f3785970a0c4ef66b7e1`。
- 2026-08-24：R0 compatibility CLI 复核 legacy v2 与 Task069 profile baseline `256/256`
  passed。remediation 后输出 `r5_final_verification.json`
  SHA-256 `051b4d0ad3d7b9a5f2d4452ca9f5e580d13542afacbd9f9104a277b90519aa00`，
  `matrix_passed=true`，`commands_passed=true`，`visual_review.passed=true`，且 R5 artifact
  绑定 `r4_visual_observations.json`、全部命令日志 SHA、exit code 和日志 verdict。

#### Review

- 未通过独立只读 review 前，状态不得标为 `passed`。
- 最终报告必须明确 stance-hold 使用的 controller，不能写成被动站立或行走能力。
- 以下任一负向反例必须令 `matrix_passed=false`：G1/PM01 被压成 12DoF、缺 waist、缺一侧 arm、
  source motor 重复/遗漏、PM01 elbow-yaw 被误映射成 elbow-pitch、hip semantic order/axis 交换、
  descriptor 内容与 SHA 不符、module mask 与
  actuator count 不符、quadruped ratio feature 未改变 realized link、local wheel augmentation 被
  误标为成熟 wheel prior。
- 当前旧 R5 结论：**superseded_by_motor_dof_preserving_v2_contract**。v2 未执行、未通过独立
  只读 review 前不得标记 passed。

## Redo Microtasks

为避免再次把“计数能编译”误当成“成熟构型匿名化正确”，v2 重做按以下小任务推进。每个小任务
独立保留 `Route / Log / Review`，通过前不得进入后续 passed claim：

视觉产物必须走双层验收：execution agent 先用本地 image viewer/manifest 完成逐图视觉检查并
记录 `agent_visual_check_passed=true`；只有 agent check 通过后，停止推进并把图片、manifest 和
失败点摘要交给用户人工检查。用户未明确确认 `user_visual_acceptance=true` 前，不得进入后续
passed claim，也不得把 preview/gallery 计入 R4/R5 成功证据。

1. `001-preview-failure-freeze.md` — 冻结当前失败 preview，防止被后续 R4/R5 误计入。
2. `002-g1-source-tree-parser.md` — 从 pinned G1 MJCF 解析 29 motor source tree descriptor。
3. `003-g1-descriptor-driven-primitive-witness.md` — 用 descriptor 生成真正可检查的 G1 primitive witness。
4. `004-g1-preview-visual-review.md` — 本地逐图复核 G1 witness 是否可审计。
5. `005-other-source-descriptor-parsers.md` — 扩展 PM01、Spot、Go2、Lite3 descriptor parser，并为
   G1/PM01/Spot/Go2/Lite3 及 wheel composition 绑定 transmission/coherent-motor/fault-interface
   `actuation_stack`；下载的其他人形 source 先生成 fail-closed candidate inventory。
6. `006-wheel-module-composition.md` — 在完整 non-wheel descriptor 末端增加 wheel module。
7. `007-v2-sampler-preview-gallery.md` — 做 descriptor-driven sampler 和小规模 preview gallery。
8. `008-v2-stance-matrix-r5.md` — 重新跑 v2 stance/R4/R5 和独立只读 review。
9. `009-source-motor-config.md` — 对当前 G1、PM01、Spot、Go2、Lite3 与本地 wheel module
   绑定 source motor hints、transmission group、coherent config 和 runtime-fault interface。
10. `010-additional-humanoid-actuation-intake.md` — 对用户授权下载的 AgiBot X1/X2、EngineAI
   T800/T800Pro、LimX HU_D04/Oli、Booster T1、RobotEra STAR1 逐 source 固定 commit/license、
   motor inventory、transmission evidence 和 config coverage；未满足 quantitative-prior gate 的型号
   只保留 candidate artifact，不进入 sampler/R4 denominator。
11. `011-additional-humanoid-primitive-witnesses.md` — 对 X1 serial 29DoF、X2 Ultra 31DoF、
    T800 25DoF、T800Pro 43DoF、HU_D04 31DoF、T1 23/29DoF、STAR1 55DoF 生成完整 source-tree
    descriptor 与匿名 primitive witness；保留 passive attachment、head/hand branch 和全部 actuated
    joints，继续按 candidate fail closed。
12. `012-v2-flat-arena-actuator-smoke.md` — 把当前五个 center 的 non-wheel/wheel 共十种构型以及
    additional candidate witness 放入本地 MuJoCo flat arena，分开记录 compile/accounting、reset/
    contact、逐 actuator response 和 1000-step controller-assisted stance；任何 actuator-response
    结果均不得写成 walking claim。
13. `013-booster-t1-visual-chain-redo.md` — 冻结 attempt006 的 Booster T1 23/29 arm-chain
    视觉假阳性和 attempt007 partial fix；只用源中真实 shoulder-roll/elbow-yaw 审计姿态把共线
    shoulder/elbow/wrist 链转出平面，并在独立 attempt008 重新完成逐图 agent review。
14. `014-all-configuration-visual-review.md` — 撤销 attempt008 因 torso/pelvis box 重叠造成的完整
    witness 假阳性；以 root→paired-waist 的累计 source attachment 距离只补足小净间隙，并在不
    覆盖旧 artifact 的 attempt009 统一输出和逐图检查 18 个现有构型及 flat-arena review bundle。
15. `015-canonical-root-coordinate-contract.md` — 保留各 source 原生动力学 free root，同时为 18 个
    构型统一生成髋 attachment 中心 `canonical_root`；绑定 manifest、公共 state reader、WholeBody
    observation/reward/fall、18-case 几何审计与不覆盖旧 artifact 的 attempt010 用户 review bundle。

## Verification commands

执行时可增加更窄命令，但最终至少运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_whole_body_contract.py \
  tests/test_whole_body_extended.py \
  tests/test_whole_body_usability_gate.py \
  tests/test_task069_*.py \
  tests/test_task070_*.py

.venv/bin/ruff check \
  src/h200_locomotion_lab/robots \
  src/h200_locomotion_lab/tools/task070_*.py \
  tests/test_task070_*.py

.venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
.venv/bin/python -m pytest -q
```

最终 morphology/stance matrix 命令必须在实现时写成 task-scoped CLI，显式提供 input/output、
structural descriptor、structural-center plan、seed range、timestep、steps 和 v2 profile；不得只
存在于交互式脚本或 task log。focused tests 必须包含 motor inventory/parser、source→anonymous
bijection、body-tree/axis coverage、23/29DoF biped、12DoF quadruped、25/31DoF wheeled-biped、
16DoF wheeled-quadruped和 12DoF-biped fail-closed 负向测试。

## Log

- 2026-08-21：Task069 的独立审核确认其窄 claim 成立，但视觉与 LocoFormer Figure 6 的成熟
  机器人结构仍有明显差距；其 reset 仅保证无穿地/自碰撞，不证明稳定站立。
- 2026-08-21：用户提出从 G1、Spot 等成熟连杆构型提炼简化骨架后 random。本任务采用
  “generic archetype + normalized constrained randomization + separate held-out registry”方案，
  避免直接复制 named robot 或污染 OOD 结论。
- 2026-08-21：用户进一步指定 G1 与 Spot 先作为 `quantitative_training_prior`。因此二者在
  Task070 及其下游实验中正式视为 seen/reference，不再具备 held-out/OOD 身份；执行阶段仍需
  先完成 source/license registry，且本次没有下载或读取新资产。
- 2026-08-24：用户批准把 Unitree 四足、云深处四足、众擎人形及未来许可证明确的厂商描述
  纳入 Task070 多厂商原型候选池，并从匿名化 archetype center 做分层受约束 random。任务新增
  逐 source/license clearance、多 prior interpolation、有界 outward band、clone guard、最近
  prior/distance 记录和 leave-one-vendor-out 防泄漏规则。该消息只授权更新 Task070 设计；未
  授权下载任何外部资产，未启动 H200 或训练。
- 2026-08-24：后续执行阶段在用户要求完整执行 Task070 并授权 Task070 R0 limited official
  source/license files 的范围内读取/保留五个 quantitative prior center 的最小源文件；该授权不
  覆盖 checkpoint、dataset、mesh/texture/logo、上游仓库批量导入、H200 或训练任务。
- 2026-08-24：独立只读 reviewer 返回 `needs_changes_with_findings`。remediation 已处理：
  biped footpad 不再使用 `0.46 m` half-length 下限，R4 最大 full foot/leg ratio `0.78`；
  wheeled_biped 不再使用固定 `x=+0.50/-0.50` attachment，改为 trunk-derived mirrored
  attachment 和大轮 active balance；R0 冻结实际 12D prior centers/FEATURE_LIMITS；
  stance pass gate 纳入 support polygon/contact residual/finite qfrc_constraint/efc_force/
  nonterminal-support exclusion；R5 绑定 visual/log SHA。
- 2026-08-24：第一轮 Task070 remediation 后 R0–R5 执行完成并保持
  `execution_verified_pending_independent_readonly_review`；随后第二轮独立只读 reviewer 继续提出
  contact-wrench、sampler、visual/R5 等 blocking findings，因此第一轮 evidence 已被下方
  round-2 remediation supersede，不作为当前通过证据。
- 2026-08-24：第二轮独立只读 reviewer 返回 `needs_changes_with_findings`。round-2
  remediation 已处理：stance gate 增加 6D contact-wrench force/torque residual、contact wrench
  finite、self-contact/wheel-wheel contact exclusion；constrained sampler retry 从 4 提高到 6，
  retry exhaustion 改为 fail-closed，四 family seed `0..9999` 探针无 region/clone 违规；R4
  visual manifest 扩展到 174 张 gallery PNG 并由 `finalize-visual` 闭合；actuator scaling 绑定
  sampled mass scale 与 global scale；R5 命令证据增加 exit code 和日志 verdict。R4 matrix
  `128/128` records_passed/contact-wrench/self/wheel/region/clone gate passed，focused pytest
  `50 passed`，full pytest `837 passed`。最终状态仍仅为
  `execution_verified_pending_independent_readonly_review`，等待新的独立只读 reviewer。
- 2026-08-24：第三轮独立只读 reviewer 返回 `needs_changes_with_findings`，唯一 P1 为 R5
  matrix gate 对空 `summary`/`records` 可因 `all([])` fail-open。round-3 remediation 已处理：
  R5 `matrix_gate` 显式检查 task/families/steps/timestep、`expected_denominator=128`、四 family
  summary 精确集合、128 条 record、每 family seed `0..31` 精确集合、summary fields、
  region denominators、record status/region-band/clone-guard，并在 R5 artifact 中保存逐项布尔
  证据。新增负向测试复现“保留 gallery、置空 summary/records”的只读反例，确认
  `visual_review.passed=true` 时 `matrix_passed=false`。R4 matrix 仍 `128/128` 通过，focused
  pytest `52 passed`，full pytest `839 passed`。最终状态仍仅为
  `execution_verified_pending_independent_readonly_review`，等待新的独立只读 reviewer。
- 2026-08-24：新的独立只读 structural-prior 审核指出，旧 v1 将 G1 29DoF、PM01 23DoF
  都压成唯一 12DoF lower-body biped，且 R4/R5 没有 mature-reference motor/tree coverage gate。
  用户随后明确最终口径：从成熟构型中保留全部电机运动自由度，把构型匿名化为 primitive 连杆
  结构，并输出 LocoFormer-style 四 family 程序生成群体。Task070 因此升级为
  `motor_dof_preserving_archetype_morphology_v2` 设计；旧 v1 R0–R5 artifact、stance、gallery、
  pytest/ruff 仅保留历史 provenance，不能证明 v2。此次只更新 task contract，未修改 generator、
  artifacts/tests，未运行训练、仿真或下载新资产。
- 2026-08-24：按用户要求先生成一个可检查样本。新增独立
  `motor_dof_preserving_archetype_morphology_v2` profile/contract 登记和 G1-derived first
  inspection preview generator；输出
  `artifacts/preview_task070_v2/unitree_g1_seed000/`，包含 29 source actuated motor
  → 29 anonymous primitive actuator accounting、可编译 MJCF 和 front/side/oblique 三视角 sheet。
  该 preview 明确标记 `first_inspection_preview_not_task_pass` 与
  `stance_claim=not_run_preview_only`，不作为 R0–R5 passed evidence。验证：
  `.venv/bin/python -m pytest -q tests/test_task070_morphology.py::test_task070_v2_g1_preview_preserves_29_motor_dofs_and_compiles`
  `1 passed`；Ruff targeted `All checks passed!`。artifact SHA-256：XML
  `25ea1fa678627e37e18bcaf005c951414ca139bfb6795ad6154f84bef266701e`；manifest
  `fb8799efb1e23cd3713a7ca1886d027d25c6324fa1c9029ee0e49f8ed86c2c09`；sheet PNG
  `b525d0d0ae87e2fd4fde2b35211f0cf3fcaff8a2bcf7012558652dd6e23ff7be`。
- 2026-08-25：按用户要求将失败 preview 拆成小任务重做。001 已冻结旧
  `preview_task070_v2/unitree_g1_seed000` 为负例并写入
  `counts_toward_task070_v2_pass=false`。002 新增 parsed G1 source-tree descriptor：
  递归解析 pinned local G1 MJCF，逐 motor 保留 source joint/body edge、body-local `pos`、
  joint-local `pos`、axis/range/type、module 和 anonymous parent/child mapping。003 使用该
  descriptor 生成新的 anonymous primitive witness，输出
  `artifacts/preview_task070_v2_descriptor_driven/unitree_g1_seed000/`。新 artifact SHA-256：
  descriptor `e0b44aad94001ba7252fd00d1d2229a46f7e6023754f6c7e388eff434251019a`；
  XML `526ed69350067fd329f4ed9fa3fed03d3d0879e7dc6345607c8c0730bdad1de3`；
  manifest `6e62f9899baf3f76de883602e7b08c475e4de94c7460113ac8cbe82330fa1906`；
  sheet PNG `967e8d2cd031c0f7b525cfa06966f3b329f9a34c95f9aba08857e3da1bfe56d2`；
  visual observation
  `31c33a0dff6cef410fda30fc5c63920c30f8571f8509491ee614a13c37ce8dd1`。Focused pytest
  `3 passed` and targeted Ruff `All checks passed!`。004 已完成 execution-agent local image
  viewer check 并记录 `agent_visual_check_passed=true`，但 `user_visual_acceptance=false`；
  当前停止在用户视觉验收门，预览仍 `counts_toward_task070_v2_pass=false`，不作为 R4/R5
  passed evidence。
- 2026-08-25：用户要求对照 LocoFormer Figure 6 重新判断视觉，指出上一版“一眼连杆不对”。
  已查 arXiv 官方 HTML/PDF 记录：Figure 6 展示 quadrupeds、wheeled quadrupeds、bipeds、
  wheeled bipeds 四类 procedural samples；biped/wheeled-biped 图是高对比彩色 primitive modules、
  directed limb segments、terminal feet/wheels 的可读连杆群体。接受用户判断并撤销
  `preview_task070_v2_descriptor_driven/unitree_g1_seed000` 的 agent visual pass，manifest status
  改为 `descriptor_driven_preview_rejected_after_locoformer_figure6_visual_comparison`。
  修复：v2 preview metadata 增加 `capsule_local_fromto` 与 `link_visual_rgba`，`compile_mjcf()`
  仅在这些 v2 metadata 存在时画 directed colored primitive links；legacy profile dataclass 未变。
  生成 replacement attempt002：
  `artifacts/preview_task070_v2_descriptor_driven_attempt002/unitree_g1_seed000/`。Artifact SHA-256：
  descriptor `e0b44aad94001ba7252fd00d1d2229a46f7e6023754f6c7e388eff434251019a`；
  XML `d4d9c0c5e35913bc58542ce4c334d385fbd5e0e6c3e56b90a602ace87a1f972e`；
  manifest `5e207ddb966a63b117d1d4347e9873ed7b65b12003382ac3086ceff52700b805`；
  sheet PNG `6d566d9667a36d1658b2c1edf7eee14cf72c59358209a5ebc15c3737973f5e85`；
  visual observation
  `292230ac4173702f76c3d56d4e6b421bf0c23b8c9e2b73680f4f798b499b5261`。
  Attempt002 agent visual check 已通过，仍等待用户视觉验收；`user_visual_acceptance=false`，
  `counts_toward_task070_v2_pass=false`。验证：focused v2 tests `2 passed`，Ruff
  `All checks passed!`，完整 `tests/test_task070_morphology.py` `15 passed`。
- 2026-08-25：用户判定 attempt002 的连杆结构仍不对，原 execution-agent visual pass 作废。
  本地重检确认并修复结构根因：旧实现用 incoming body offset 放大 outgoing capsule（例如约
  `1.8 cm` ankle-pitch→roll edge 被画成约 `31 cm`，约 `4 cm` shoulder-pitch→roll edge
  被画成约 `27 cm`）；parser 丢失 source body-local quaternion；torso box 覆盖 pelvis/waist；
  torso→shoulder 与 pelvis→hip 无显式 attachment primitive。v2-only 修复保留 parsed body
  quaternion，以当前 motor origin→下一 descriptor joint origin 画 directed capsule，并增加局部
  torso/pelvis offset、shoulder/hip connector、ankle/footpad attachment 和已记录的 elbow visual
  audit pose；legacy/Task069 输出未进入这些 metadata 分支。新 attempt003 输出到
  `artifacts/preview_task070_v2_descriptor_driven_attempt003/unitree_g1_seed000/`。SHA-256：
  descriptor `d7933388ee454ed4fb3f76a7f9b52859637104ea7fb93e1669422ee75a844c26`；
  XML `6ede5976ea1135bd049e1148d223d5156aedd67f21f9fa97174b8514e83b77f1`；
  sheet `3ef629e63db83caa6d0e83f6165aae0ef9658e66a3c6e75456669c1b2050f070`；
  manifest `111470c4b2ce492adb513ed75806749b37f0c8385b99dcc7ef6022b4b7760bd5`；
  agent visual observation `dcd96fecec5b1414cc2b103dc4e588800b1b601dc85d7bd18f18f7849bef95bb`。
  execution agent 已用本地 viewer 打开最终 sheet 和 front/side/oblique/contact 单图，记录
  `agent_visual_check_passed=true`；仍保持 `user_visual_acceptance=false`、
  `counts_toward_task070_v2_pass=false`、`stance_claim=not_run_preview_only`。验证：指定 focused
  pytest `15 passed in 1.74s`，指定 Ruff `All checks passed!`，legacy/Task069 compatibility
  `256/256` passed。未下载资产、未训练、未运行 H200；Task070 状态未标为 passed。
- 2026-08-25：按用户后续要求，用同一 descriptor-frame/exact-outgoing-edge 方法修复四足和轮腿。
  新增 PM01、Spot、Go2、Lite3 local source parser；四足 witness 显式画 trunk→hip attachment、
  hip ab/ad 短段、hip flex/ext 长段、knee bend、shank 和单一 terminal。Wheel composition 只在完整
  source limb 末端追加 local-lateral continuous wheel：G1 `29+2=31`、PM01 `23+2=25`、三种
  quadruped 各 `12+4=16`，不删除 source motor。Attempt003 新增八组四视图 witness 和 aggregate
  visual observation
  `artifacts/preview_task070_v2_descriptor_driven_attempt003/quadruped_wheel_leg_agent_visual_observation.json`。
  execution agent 已逐图确认结构可审计并记录 `agent_visual_check_passed=true`；所有 manifest 仍为
  `user_visual_acceptance=false`、`counts_toward_task070_v2_pass=false`、
  `stance_claim=not_run_preview_only`。Focused pytest `18 passed in 1.83s`，targeted Ruff passed，
  frozen compatibility `256/256` passed。007 完整 sampler/region/negative gate 与 008 stance/R5
  尚未执行，Task070 未标为 passed。
- 2026-08-25：按用户确认，将电机层设计写成 v2-only 四层契约：`structural descriptor ->
  transmission model -> coherent motor config -> runtime fault process`。v2 contract 升级为
  `procedural_motor_dof_preserving_archetype_morphology_v2_actuation_stack_v1`；legacy v2、
  paper-faithful v1、archetype v1 contract/hash 未改。当前五个 source center 及其 wheel composition
  均生成 `actuation_stack`：每个 generalized slot 必须且只能属于一个 transmission group；motor
  参数按 source motor family 分组，禁止把 strength/effort/speed/Kp/Kd/armature 当成互不相关的逐关节
  scalar noise；runtime fault 层明确记录现有 `MotorProcess` 只工作在 generalized action slot，
  physical parallel-motor fault projection 尚未实现。G1 三组并联踝/腰按 `2 DoF -> 2 physical
  motors` 计数，总数保持 29，不再把每个 pitch/roll 各误算成两台独立物理电机。PM01 另绑定
  EngineAI official native SDK 的 PM01-Edu motor sign、parallel-ankle 和 RL gain config；由于该
  related variant 有 24 enabled motors（多 `J23_HEAD_YAW`），而当前 descriptor 是 23 DoF，配置
  仅作为 topology/config evidence，未静默写入 resolved actuator prior。
- 2026-08-25：生成不覆盖旧 artifact 的 actuation-only attempt005：
  `artifacts/preview_task070_v2_descriptor_driven_attempt005/`，包含 G1/PM01/Spot/Go2/Lite3 原构型
  与轮式组合共 10 份可编译 XML、descriptor、manifest；aggregate observation SHA-256
  `d27bff9fd920803268185fe5bbe82b8c16f5e3f885c54071cb5e9c0b23441f45`。本轮未改变几何、未
  render 新图，因此严格保持 `agent_visual_check_passed=false`、`user_visual_acceptance=false`、
  `counts_toward_task070_v2_pass=false`，不借用 attempt003 视觉结论制造新 attempt 假阳性。
- 2026-08-25：按用户明确授权，从官方固定 commit 只下载 license、README、URDF/MJCF、
  transmission/motor/controller config：AgiBot X1 `9e0b8188`、AgiBot X2 Ultra `77f43eb0`、
  EngineAI PM01-Edu/T800/T800Pro `335c60e8`、LimX Oli/HU_D04 description `02adfbdd` 与 deploy
  `6d8771cd`、Booster T1 `508cbee6`、RobotEra STAR1 `e8660e66`；未下载 mesh、texture、logo、
  checkpoint、dataset 或 motion data。候选 source/config/license/tree SHA 与 motor/transmission
  completeness 记录在 `artifacts/task070_v2_additional_humanoid_candidate_source_inventory.json`
  （SHA-256 `c363182d8fae259048fc783cda5712f73fcfd7a6a98e1a55e3129e9aceca8c03`）。X1 nonlinear ankle
  lookup 大文件和 T800Pro palm `.mnn` 未纳入；X2/T1/STAR1 只有 limit-level evidence；所有新增
  型号保持 candidate/fail-closed，未进入 sampler 或 R4 denominator。验证：Task070 focused
  pytest `26 passed in 1.81s`；指定 Ruff `All checks passed!`；frozen legacy/Task069 compatibility
  `256/256` passed；candidate 7 source roots 和 attempt005 10 manifests 的本地 SHA/coverage audit
  全部通过。未训练、未运行 H200，Task070 未标为 passed。
- 2026-08-25：整理新增人形构型为 microtask 010/011，并生成不覆盖旧 artifact 的
  `artifacts/preview_task070_v2_descriptor_driven_attempt006/`：AgiBot X1 serial `29`、X2
  Ultra `31`、EngineAI T800 `25`、T800Pro `43`、LimX HU_D04 `31`、Booster T1
  `23/29`、RobotEra STAR1 `55`。每个 witness 从完整 parsed source tree 出发，保留 actuated
  joint order/parent/child/axis/range/module，并把 passive head/sensor/palm/finger branches 画成
  anonymous primitive；没有复制 mesh/texture/logo。execution agent 打开八张 sheet 和 aggregate
  gallery 后记录 `agent_visual_check_passed=true`；visual observation SHA-256
  `ab544466b88241d44886d96f1b29bf125406d82be270b788909f82fe98f82f4c`。所有新增型号仍
  candidate/fail-closed，`user_visual_acceptance=false`、
  `counts_toward_task070_v2_pass=false`。
- 2026-08-25：microtask 012 将旧五个 center 的原/轮式 10 case 与新增八个人形 candidate
  一起放入 MuJoCo 20 x 20 平地。修复空 contact diagnostic 的 TypeError，并对每个 actuator 从
  identical reset 同时运行 nominal-hold baseline 与单 actuator pulse，只用目标关节的轨迹差值
  判定响应，避免把重力/自由基座下落误算成电机响应。最终 `18/18` compile、exact accounting、
  contact-aware reset、all-actuator paired-baseline response
  通过；arena evidence SHA-256
  `c460eef106c024fdf026e171ebfb403b56439dc0ba94365cc355c95ce8206f3a`。通用 nominal
  joint-hold 下 stance 为 `0/18`，且没有 locomotion policy，因此明确保持
  `walking_claimed=false`，不能把电机响应写成会站或会走。Candidate extra-slot fail-closed
  负例覆盖 status/adapter/counts/declared-set 四类违规。Focused pytest
  `30 passed in 3.31s`，targeted Ruff passed，frozen compatibility `256/256` passed。
- 2026-08-25：用户要求 execution agent 重新逐张检查后，撤销 attempt006 的 aggregate visual
  pass：Booster T1 23/29 的源 arm offsets 基本共线，T1 29 六个 arm DoF 在旧图中塌成一根横杆，
  shoulder/elbow/wrist 不可审计。Attempt006 aggregate observation 改为 false（SHA-256
  `1352dd423f7e470b571e0b8d627f34cf257401495f47d97f141aea291402cdcf`）；只含
  shoulder-roll 的 attempt007 仍因 wrist chain 重叠被拒绝（observation SHA-256
  `a569fc0764ef4fcf122eb880bbd3e3cc9d25d39b1f2b577d193747a306eb35cb`），没有覆盖旧图。
- 2026-08-25：microtask 013 的最终 attempt008 只为 Booster T1 23/29 设置源 range 内的镜像
  visual-audit nominal：shoulder-roll `-0.42/+0.42`、elbow-yaw `+0.58/-0.58`，保留
  parsed source positions/parent-child/axis/range/order/module 和 23/29 accounting，不新增虚构
  link。execution agent 打开两张 sheet 与全部 front/side/oblique/contact 单图后，确认 T1 23
  shoulder/elbow terminal chain 与 T1 29 shoulder/elbow/wrist/hand chain 可读，记录
  `agent_visual_check_passed=true`；observation SHA-256
  `7acb7838bac9d39da1522af5856a4e4e8fa92d02421dbf170404b66185901779`。
  T1-only arena 为 `2/2` compile/accounting/reset/all-actuator paired-baseline response，
  stance `0/2`、`walking_claimed=false`；arena SHA-256
  `86248d19063e750cf8d842b7488ebcabd69db93299f8bb0ae2a16d4db326d95f`。Focused pytest
  `31 passed in 3.28s`，targeted Ruff passed，frozen compatibility `256/256` passed。
- 2026-08-26：用户指出 attempt008 身体上仍有两个重叠方块；full-witness recheck 确认旧 candidate
  builder 把 root 与 branching waist hub 的 box 都画在各自 body origin，随后一次固定大偏移预检又
  把 T1/STAR1 修成断腰。Microtask 014 改为 source-attachment-driven 最小补足：由 root→最近
  branching waist 的累计 world-Z attachment 距离计算所需 outward shift，按 `0.4/0.6` 分给两块，
  再经累计 source quaternion 转回 body-local geom center；只改 anonymous visual box center/size，
  不改 source body pos/quat、joint parent/child/order/axis/range/module 或 DoF。最终 8 个新增人形
  nominal torso/pelvis gap 为 `0.0101–0.0202 m`。Attempt008 已改记 rejected；新 attempt009 统一
  生成现有 18 构型的 XML/descriptor/manifest/四视图/sheet/gallery，execution agent 已逐图确认
  `agent_visual_check_passed=true`，aggregate observation SHA-256
  `922ec777fcbd7a21609d73c7d8964d9dccf7eb1e0452c198426d8822cd0dce46`。18-case flat arena 为
  `18/18` compile/accounting/reset/all-actuator paired-baseline response，但 stance `0/18`、
  `walking_claimed=false`；evidence SHA-256
  `d9dcc45142026043aa23532f14324e23fb24dd03c084f59f0d2dfb1296af23eb`。Focused pytest
  `31 passed in 3.31s`，targeted Ruff passed，frozen compatibility `256/256` passed。所有 attempt009
  manifest 仍保持 `user_visual_acceptance=false`、`counts_toward_task070_v2_pass=false`；Task070
  未标记 passed。
- 2026-08-26：Microtask 015 统一了所有当前 v2 构型的 root state contract。原生 `root_free`
  继续保留 source tree/dynamics，新增透明 runtime site `canonical_root`；人形取左右第一髋
  attachment 中点，四足取四髋质心，统一右手 `+X forward/+Y left/+Z up`。公开
  `read_canonical_root_state` 返回 site pose、local twist 和 projected gravity；WholeBody actor
  observation、reward、upright/fall 已改用该 state，且修复 descriptor-driven 匿名腿链无法被
  stance solver 识别的问题。新 contract identity 为
  `procedural_motor_dof_preserving_archetype_morphology_v2_actuation_stack_v1_canonical_root_v1`。
  Attempt010 在不覆盖 attempt009 的前提下重新生成 18 份 XML/descriptor/manifest/四视图/sheet
  和 review gallery；canonical audit 为 `18/18`，site→髋中心最大残差
  `4.999999969612645e-09 m`，最大正交误差 `0`，最小 forward/left/up 对齐 `1.0`；另以
  `q=0.31 rad`、`qvel=0.8 rad/s` 主动转动 T1/STAR1 的中间 waist joint，`2/2` 仍保持当前髋中心、
  parent orientation、local twist 和 projected gravity exact。Audit SHA-256
  `f96da04079f8155221b4067cac6af31968182209f809a08ee1face32d28b8547`，agent visual observation
  SHA-256 `0251b4b8e3cbcdf7984676a659669a8860c6d13664f846c8ff06c307451c141a`。
  Attempt010 flat arena 仍只证明 `18/18` compile/accounting/reset/all-actuator response；stance
  `0/18`、`walking_claimed=false`，evidence SHA-256
  `8a8d281d9ede3d32713ceb92c96976f8eacab864d35d6feba1c31fb2db52436d`。Focused Task070 pytest
  `35 passed`，WholeBody contract/extended pytest `30 passed`，targeted Ruff passed。所有 manifest
  保持 `user_visual_acceptance=false`、`counts_toward_task070_v2_pass=false`；follow-up high-risk
  read-only reviewer 在动态腰链补测后报告无 P0/P1/P2；Task070 未标记 passed。
- 2026-08-26：用户明确表示“目前我认证过了，感觉还行”。通过 append-only
  `016-user-visual-acceptance.md` 将 attempt010 的有效视觉验收记录为
  `user_visual_acceptance=true`；不改写 frozen JSON，且
  `counts_toward_task070_v2_pass=false`、stance `0/18`、walking `false`、Task070 not passed
  保持不变。
- 硬件假设：RTX 5060 Ti-first、本地 MuJoCo/headless；H200 disabled。

## Review

- 2026-08-26 user acceptance overlay：用户明确表示“目前我认证过了，感觉还行”；`user_visual_acceptance=true`，绑定 attempt010 audit/visual/arena SHA，且 `counts_toward_task070_v2_pass=false`。

状态：**redesign_required_motor_dof_preserving_v2**。

旧 v1 已证明其 12/14/12/16-actuator 简化四 family 可以编译并在声明 controller 下通过原
stance gate，但没有证明成熟 reference motor-DoF-preserving linkage，因此旧
`execution_verified_pending_independent_readonly_review` 状态已被本 task contract supersede。

v2 必须重新完成：R0 structural descriptor/source motor audit；R1 descriptor-driven primitive
linkage；R2 graph-compatible discrete/continuous sampler；R3 完整 23/29DoF biped、12DoF
quadruped及轮式组合 stance；R4 LocoFormer-style 四 family matrix/gallery；R5 exact motor/tree/
axis/module/SHA fail-closed gate和新的独立只读 review。完成前不得标记 `passed`，也不得使用旧
128/128 matrix、174 张 gallery 或 839 passed pytest 扩展成 v2 claim。
