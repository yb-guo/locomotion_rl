# Task067 — 二足 Static Stance Contract（generator 根因修复）

进入条件：Task061 诊断完成（见 task061 Log 与 `artifacts/stance_*`）。
退出条件：下面的 stance gate 通过；只有通过后才允许重跑 Task061 pilot，
Task062 shared MLP 仍然被阻塞。

## Route

诊断结论：二足 specialist 站不住的主要根因在 generator 的 stance contract，
不在 controller / reward / PPO。三条最小修复按顺序执行，每条独立可验证，
每条都不改 45D action、193D actor observation、mask、task/env 公共接口。

### R0 — stance / checkpoint 契约修正（先于行为 patch）

`MorphologyBlueprint` 继续只拥有离散 topology，不能保存 `stance_qpos` 或
`stance_height`。静态 stance 依赖完整的连续物理实例：`global_scale`、每段
`link_scale`、mass/COM、shifted joint range、`nominal_offsets` 和 payload 都可能
改变可行解。因此：

- `MorphologyInstanceKey` 对完整 blueprint manifest、完整 `PhysicalParams.manifest()`
  和当前 embodiment contract version/hash 做 SHA-256；stance cache 只能按该
  exact key 复用，禁止只按 `structural_hash` / topology 复用。
- `StanceSolution` 保存 source `MorphologyInstanceKey`、absolute `base_height` 和
  absolute compiled-model joint qpos。使用前必须 `validate_for(blueprint, physical)`；
  不同 physical sample 即使 topology 相同也必须拒绝。
- `PhysicalParams.nominal_offsets` 已反映在 compiled joint range 与求解边界中；
  `StanceSolution.joint_qpos` 是最终绝对坐标。reset 和 action midpoint 使用时不得
  再加一次 `nominal_offsets`。
- 一个固定 `blueprint + physical` shard 的多个 `MjData` 可共享一次求解；若
  `context_done` 真正重采样 physical，则必须切换/重编译到对应 instance 并重新
  查 cache 或求解，不能沿用旧 topology 的 stance。

Checkpoint compatibility 分成三层，不混用 tensor schema：

1. `schema_hash` 只描述 45D slot / 193D observation layout，本任务不改它；
2. `embodiment_contract_version/hash` 描述 contact geometry、reset stance 和
   action-midpoint 语义；R1/R2 每次改变这些语义都必须升版/换 hash；
3. `manifest_hash` 描述具体训练 realization/distribution。single-topology specialist
   至少包含完整 blueprint + physical instance key；shared policy 则 hash 完整 train
   split、physical distribution config 和 stance solver contract。

Loader 必须同时比较预期的 embodiment contract 与 manifest；缺少新字段的旧
Task061 `.pt` fail closed。当前尚未改行为的 capsule/fixed-nominal runtime 明确标为
`procedural_whole_body_v1_capsule_nominal`，R1/R2 落地时必须更新，不能提前把未实现
语义标成已生效。

### R1 — 有面积的脚（最小 patch）

问题：末端 leg link 的碰撞体是 capsule 圆端，`{link}_foot` 只是 site
（MuJoCo site 不参与碰撞），所以二足支撑多边形恒为 ≤2 个点、面积 0。
32/32 二足 `hull_area=0`、COM 投影 `0/32` 在支撑内。

最小 patch 设计（`robots/procedural_morphology.py`）：

- `LinkBlueprint` 增加 `foot: bool = False`（terminal leg link 置 True），
  以及 `foot_size: tuple[float, float] | None`（half-length, half-width）。
- `compile_mjcf` 对 `foot=True` 的 link 追加一个 box geom
  `{link}_footpad`，`pos=(0,0,-(length+radius))`，尺寸随 `global_scale`
  和 link scale 缩放；把该 link 原 capsule 的 `contype/conaffinity` 置 0，
  使 footpad 成为唯一足底碰撞体。
- `WholeBodyMuJoCoShard._foot_geoms` 改为收集 `{link}_footpad`，这样
  `non_foot_contact` 才真正只统计非足底接触（当前把整根小腿算成脚）。
- 足底长宽由腿长派生（例如 `0.16 * leg_total_length`、`0.09 *`），
  不引入新的独立随机维度。

### R2 — 每个本体求解可支撑 nominal stance（最小 patch）

问题：`nominal_height` 是 family 常数（1.05 / 0.52），与采样腿长
（reach 0.96–1.08 m）无关；nominal 关节角是固定模板
（`knee_pitch/elbow_pitch=0.55`，其余 0），2000 个本体共用。
`ground_nominal_pose` 只平移最低 geom 并留 0.015 m 间隙，所以 reset
`ncon` 中位 0、贴地脚数中位 1、左右脚底高度差中位 0.062 m。

最小 patch 设计：

- 在 `robots/whole_body_stance.py` 的 R0 contract 上新增
  `solve_static_stance(model, data, blueprint, physical, ...) -> StanceSolution`：
  对每条腿的 pitch 链 + 可选 ankle 做小规模投影坐标下降，目标
  `sum(foot_bottom - margin)^2 + sum(foot_tilt)^2 + ||com_xy - support_centroid_xy||^2`，
  返回 absolute `base_height` 与每个 semantic slot 的 absolute nominal 角。
  参考实现已存在于诊断工具 `tools/whole_body_stance_isolation.py`
  （`solve_optimized_stance_pose`）；每个 exact compiled realization 只解一次，
  cache key 由 `stance_cache_key(MorphologyInstanceKey)` 同时 hash instance 与
  stance solver contract version/hash。
- `MorphologyBlueprint` 不增加 stance 字段；`WholeBodyMuJoCoShard` 在 model compile
  之后持有并校验 `StanceSolution`。solution manifest/hash 随运行 artifact 保存，
  但不改变 topology split 用的 `structural_hash`。
- `WholeBodyMuJoCoShard._reset_env` 用 solution 的 absolute `base_height` / qpos
  代替 `nominal_height` / `joint.nominal + nominal_offset`；`_set_targets` 的 midpoint
  直接使用同一个 absolute qpos，禁止再次加 `nominal_offset`。
- `ground_nominal_pose` 的 `margin` 默认改为 0.0，并且只在 stance 解
  失败时作为有显式 failure reason 的诊断兜底；兜底样本不得进入训练集。

### R3 — 二足最小机械可行性约束（最小 patch）

问题：grammar 允许结构上无法完成任务的二足。1000 个二足中
`42.2%` 双腿都有 ankle、`52.6%` 双腿都有 roll 权限、`22.6%` 至少一条腿
只有 hip_pitch+knee_pitch 两个自由度（纯平面腿，却要跟踪 vy±0.2、
yaw±0.5）、`77.3%` 左右腿关节数不相等。四足没有这些问题
（最小 3 自由度、0% 出现 2 自由度腿），也正好 0/32 跌倒。

最小 patch 设计（`MorphologyGenerator._append_chain` / `generate`）：

- biped leg 采样时把 `distal_count` 下界从 0 提到 1，且必选
  `ankle_pitch`；`proximal` 必选 `hip_pitch` 之外再必选一个 roll
  （`hip_roll`），其余轴仍随机。
- biped 左右腿改为镜像采样：只采一次腿结构，右腿复用同一轴序列与
  分段数，只镜像 attachment 的 y 与 roll/yaw 轴符号。左右连续物理参数
  仍独立采样，保留 morphology 随机性。
- 四足 grammar 不改（当前已通过 zero-action 站立）。
- `MorphologyGeneratorConfig` 增加 `require_biped_ankle: bool = True`、
  `mirror_biped_legs: bool = True`，默认开启，便于回退对照。

### R4a — 动态平衡因果诊断（诊断与 artifact，不做行为 patch）

问题：R1–R3 后二足 reset 几何已经接近 gate 目标，但 zero-action 2s 仍
8/8 跌倒；关闭重力 8/8 存活，四足原始控制 8/8 存活。简单
`qfrc_bias/Kp` 常量或动态预载也 8/8 跌倒，因此不能把无接触的
`qfrc_bias[joint]` 当成精确静态力矩。浮动基座 + 足底接触下，足底接触力也承担
重力，必须显式诊断接触一致平衡。

最小诊断设计：

- 新增 `tools/whole_body_dynamic_balance_diagnosis.py`，直接在私有 MuJoCo rollout
  上诊断，不改 reward、actuator `kp/kv`、45D action、193D observation、mask、
  motor process 或 env 公共接口。
- 对固定 4 个二足 seed、`range_fraction=0.0/0.5` 跑：
  baseline、zero-gravity、root-locked、constant `qfrc_bias/Kp` preload、
  dynamic `qfrc_bias/Kp` preload、contact-equilibrium hold；四足跑 baseline
  与 contact-equilibrium 对照。
- contact-equilibrium 不是使用裸 `qfrc_bias[joint]`：先用 footpad bottom corners
  求非负竖直接触力，使总重力与 COM 力矩平衡；再用接触点 Jacobian 得到
  `qfrc_contact`，由 `qfrc_bias - qfrc_contact` 求 `ctrl_eq`，并检查 root residual、
  ctrl range、force limit、接触和足底高度约束。
- 对 contact-equilibrium 候选施加 ±1°/±2° roll/pitch 与 ±1 cm root XY
  扰动，记录误差衰减、增长或跌倒。
- artifact 记录 base angular acceleration、COM/COP、接触力、joint error、
  actuator force 和饱和情况；得出决策后停止，不进入 Task061/Task062。

### R4a.1 — equilibrium 诊断修正（诊断与 artifact，不做行为 patch）

R4a 的解析接触力结果仍不足以作为 hold 候选，因为它没有把实际 MuJoCo soft-contact
forward dynamics 的初始 `qacc` 作为 gate。R4a.1 修正如下：

- 联合搜索 footpad bottom-corner contact forces 与 joint torque / ctrl range bounds；
  每只脚必须分别承重，不能只让总 COM 力矩平衡。
- 对候选 `qpos + ctrl + contact penetration` 调用实际 `mj_forward`，用
  `qacc`、`qfrc_constraint`、actual actuator force 与 saturation 重新评分和 gate。
- 只有 contact / joint / ctrl / penetration / 初始 `qacc` 全部达标的候选才能进入
  `contact_equilibrium_hold` 和扰动探针；infeasible 候选只记录 skipped，不混入
  hold/perturbation 统计。
- 扰动改为 root 速度扰动与单步 root 冲量扰动；summary 只汇总 feasible-equilibrium
  的扰动响应。
- `WholeBodyMuJoCoShard._is_fallen` 与相关诊断使用 `nominal_height * global_scale`
  的 scaled fall threshold。
- 新增专门测试覆盖 joint-bounded feasibility、feasible-only summary、扰动语义和
  scaled fall threshold。

### R4a.2 — joint-aware dynamic stance solve（诊断与 artifact，不做 controller）

独立复核证明 R4a.1 的 `majority_no_true_equilibrium` 是搜索空间结论：R4a.1 固定
R2 joint qpos，只枚举 penetration 与 base roll/pitch，不能推出 morphology /
inertia / contact 本身无 equilibrium。R4a.2 修正如下：

- 连续联合优化 base roll/pitch/yaw、contact penetration、joint qpos 与 ctrl；joint qpos
  只允许在 R2 stance 附近小幅移动，并保留实际 ctrl range。
- contact force 改为 SLSQP QP：直接求所有 support bottom-corner normal forces，并在
  目标中同时惩罚 force/moment residual 与 joint torque / ctrl bound 违规，停止组合枚举。
- 候选仍必须通过实际 `mj_forward` 的 root / joint qacc gate、actual foot contact、
  非足接触、actuator saturation、contact-force/root residual、joint force/ctrl bounds。
- 四足使用完全相同 solver 作为正对照；artifact 保留 4×2 四足全量统计，并要求每个
  range 至少有 feasible 四足正对照且 feasible hold 稳定。
- 对二足 4 seeds × 2 range 重新统计 feasible-only nominal hold 与速度/冲量扰动。
- 新增硬测试：biped seed0 必须找到已知正解；quadruped seed0 必须用同一 solver 通过；
  不再使用“若 feasible 才断言”的条件测试。

### R4b-1 — bounded feedback causality diagnosis（诊断，不改公共控制链）

R4a.2 证明多数二足存在真实 equilibrium，但 feasible-only hold / 扰动仍发散。R4b-1
只诊断 bounded lower-body feedback 是否是足够因果解释，不集成 controller：

- 只使用 R4a.2 artifact 中的 5 个 feasible 二足 equilibrium，固定 `qpos_eq + ctrl_eq`；
  非 feasible 二足不参与补救。
- 比较四种模式：`hold_baseline`、`attitude_only`（projected gravity + base angular
  velocity）、`com_cop_oracle`、`attitude_com_combined`。
- controller 输出仅为 private rollout 内的 lower-body position-target delta，不修改
  actuator `kp/kv`、reward、45D/193D schema、motor strength / latency / failure 链路；
  统一 gain grid，`max_delta <= 0.05 / 0.06 / 0.08 rad`，禁止逐 seed 手调。
- 速度/冲量扰动使用 paired nominal trajectory 的早期差分增长率，而不是只看最终是否跌倒。
- 增加 controller-on 1 秒、随后 off 的因果开关实验。
- 四足 feasible equilibrium 继续作为“不应被 destabilize”的正对照。

R4b-1 gate：5/5 二足 nominal 2 秒存活；50 个速度/冲量探针至少 45/50 存活；
actuator saturation / non-foot contact / unloaded-foot steps 均为 0；记录 clipping ratio 与
max delta；controller-off 实验可复现退化。只有 deployable 版本通过后，才允许设计
`ZeroActionHoldSolution`、升级 embodiment contract/hash。

### R4b-2 — bounded feedback authority / mapping diagnosis（诊断，不集成 controller）

R4b-1 证明当前统一 gain grid 没有通过生存 gate，但还不能区分“bounded target 空间
没有控制权”与“feedback 映射/权重没有用上控制权”。R4b-2 只做更窄的因果诊断：

- 仍只使用 R4a.2 的 5 个 feasible 二足 equilibrium，固定 `qpos_eq + ctrl_eq`；
  不改公共 env 控制链、actuator `kp/kv`、reward、45D/193D schema、motor process、
  latency 或 failure 语义。
- 对 roll / pitch 的 ±2° 姿态扰动、±0.10 rad/s 速度扰动，以及 ±1 cm COM/root
  offset，比较 baseline、当前 combined-high delta、反向 delta、以及同一 lower-body
  mapping 下的静态 `±0.02 / ±0.05 / ±0.08 rad` axis delta。
- 用实际 `mj_forward` 的 root `qacc` 计算 local restoring score，并记录 actuator force、
  saturation、ctrl bound clip、foot load、non-foot contact。
- 跑 2 秒 timeline：hold baseline、current combined low/high、inverted combined high，
  记录 tilt warning、foot unload 与 fall 的先后顺序，判断是否 contact loss 先于倾倒。
- artifact 必须包含 axis/kind breakdown，避免只用全局 median 掩盖 pitch/roll 差异。

R4b-2 只允许给出下一步诊断分支；不得把任何 R4b controller 接入公共环境。

### R4b-2i — independent equilibrium / authority audit（诊断，推翻错误因果链）

独立复核必须先验证 R4a.2 输入是否真是实际 MuJoCo equilibrium，再解释 feedback：

- 用 actual `mj_forward` 的完整 root / actuated-joint `qacc` 做 strict gate；解析 contact QP
  只能作为搜索先验，不能覆盖实际 soft-contact forward dynamics。
- 在 R4a.2 原姿态、joint adjustment、ctrl range 和 penetration 边界内做 diagnostic-only
  refinement，并分别运行 2 秒、10 秒 nominal hold 与原 10 个速度/冲量探针。
- feedback probe 必须先恢复 `ctrl_eq` 再计算 COP 和 delta；同时记录 equilibrium reference
  delta，区分 absolute controller bias 与 `delta(x)-delta(x_eq)` 小信号响应。
- 用 input finite difference 记录 hand mapping 的完整 root-qacc Jacobian 与逐 actuator
  contribution；不得用 `max_{delta in +/- set}` 的单轴最优值直接证明统一 polarity/weighting。
- root angle perturbation 与 MuJoCo free-joint tangent 对照；root x/y 全机平移不得再标为
  COM/support offset probe。

本审计仍不实现 controller，不改公共 env / actuator / reward / schema / motor process，也不进入
Task061 / Task062。

### R4a.3 — strict actual-equilibrium coverage（诊断 contract 修正）

R4b-2i 证明 R4a.2 的 `feasible` 标签只是 loose candidate label，不能作为 equilibrium
验收。R4a.3 将 equilibrium 诊断 contract 固化为：

- 解析 contact QP / R4a.2 artifact 只作为 candidate provenance，不再直接决定分支；
- 最终验收必须通过 strict actual `mj_forward` root / joint `qacc`、双脚实际载荷、
  non-foot contact、actuator saturation；
- strict initial equilibrium 之后还必须运行 fixed `qpos_eq + ctrl_eq` 的 nominal hold，
  只有 hold 也通过才算 `strict_contract_passed`；
- full coverage 先覆盖二足 4 seeds × 2 range fractions，不再只消费旧 R4a.2 的 5 个
  source-feasible record。

R4a.3 仍是诊断，不实现 controller，不改公共控制链、actuator `kp/kv`、reward、45D/193D
schema、motor strength / latency / failure 链路，也不进入 Task061 / Task062。

### R4a.3.1 — contact-preserving continuation / search adequacy diagnosis

R4a.3 的 3 个 failed endpoint 不能直接解释成 generator 物理不可行。先做同 topology
continuation 判断是否只是搜索落错 basin：

- seed0：`rf0 -> rf0.5`；
- seed3：`rf0 -> rf0.5`；
- seed1：`rf0.5 -> rf0`。

每条路径按 `0.05` range-fraction step 前进，失败时只做步长二分
`0.025/0.0125/0.00625`，并用上一 accepted strict `qpos/ctrl` 映射到同 topology
target physical 作为 warm-start。每个尝试记录 actual root / joint `qacc`、signed
COM-COP、每只脚 bottom height / contact count / normal load、active joint/ctrl
bounds、solver `nfev/optimality`，以及是否滑入 single-foot active set。

若 qacc-only continuation 失去双脚接触，则只允许增加 contact-preserving diagnostic
residual：actual root/joint `qacc`、左右脚 bottom height 到共同 penetration target
的误差、两脚最低实际载荷 deficit、与 warm-start 的小正则。最终验收仍沿用 R4a.3
strict gate；`solver.success=True` 不等价于通过。

失败分类固定为
`strict_double_support_equilibrium_found`、`single_support_equilibrium_only_found`、
`search_exhausted_without_certificate`、`kinematic_double_support_infeasible`、
`wrench_or_actuation_infeasible`。最后两类只有存在独立几何或静力学证据时才允许使用；
否则不得建议收紧 generator grammar。

### R4a.3.1a — true-continuation correctness diagnosis

R4a.3.1 的同 topology continuation 还需要排除一个诊断 bug：warm-start branch
可能被 target R2 stance 周围 `±0.08 rad` 的人工 bounds 重新锚定并剪断。新增独立
artifact，不覆盖 R4a.3.1 artifact：

- warm start 直接使用上一 accepted strict `qpos/ctrl`，target R2 stance 只用于记录
  与上一解的距离；
- joint bounds 改为 compiled physical joint limits；可选 trust region 只能以上一解
  为中心，默认关闭；
- 明确记录真实 joint-limit clip、ctrl-range clip、人工 trust-region clip，以及 target
  R2 stance 到上一解的距离；
- artifact 必须断言所有 continuation step 的人工 warm-start clip 为 0；
- 最终仍用 strict actual `qacc`、双脚实际载荷、0 non-foot contact、0 actuator
  saturation 和 2 秒 fixed `qpos_eq + ctrl_eq` hold 验收。

若恢复 `8/8`，说明旧 continuation/search bug 是主因，下一步转向显式 contact-wrench
equilibrium solver 并准备 `StanceSolutionV3(qpos_eq, ctrl_eq)`；若仍失败，停止继续堆
`least_squares` start / `max_nfev`，直接进入固定双脚 contact-mode 的
state-input-wrench 约束求解。此阶段仍不改公共 env、controller、generator、`kp/kv`，
也不进入 Task061/062。

### R4a.3.1b — fixed double-foot contact-mode state-input-wrench solve

R4a.3.1a 已排除 target R2 stance 人工剪断 branch，但仍未恢复 `8/8`。下一步不再
增加普通 least-squares starts 或 `max_nfev`，而是显式把双脚 contact mode 纳入变量和
约束：

- 对 3 个 failed endpoint 只做固定双脚模式诊断；
- state variables 为 root roll/pitch/yaw、penetration、actuated joint qpos；
- input variables 为 actuator position ctrl；
- wrench variables 为每个固定足底 contact point 的竖直 normal force；
- contact modes 来自确定性 start 的双脚最低角点模式，加一个 all-footpad-corners
  patch 模式；所有模式必须同时包含两只脚；
- residual 同时包含 root wrench balance、joint torque/input residual、selected contact
  heights、总法向力、per-foot load deficit、actuator ctrl/force bounds 与小正则；
- 最终验收仍必须通过 R4a.3 strict actual `mj_forward` qacc、双脚实际载荷、0 non-foot
  contact、0 actuator saturation 和 2 秒 fixed `qpos_eq + ctrl_eq` hold。

刚体 state-input-wrench solve 成功只表示解析固定接触模型内存在候选，不能绕过
MuJoCo soft-contact actual gate；search failure 也不能升级成物理不可行证书。

### R4a.3.1c — soft-contact force-closure / realization audit

R4a.3.1b 的 rigid contact-wrench candidates 不能直接解释为 MuJoCo soft-contact
本身的问题。先做一个力闭合与接触实现审计：

- 用 R4a.3 的 5 个 strict equilibrium 作为正对照，分别验证 hand actuator generalized
  force vs `data.qfrc_actuator`、actual contact force 经 Jacobian 映射 vs
  `qfrc_constraint`、以及完整动力学闭合
  `M*qacc = qfrc_actuator + qfrc_passive - qfrc_bias + qfrc_constraint`；
- 对 3 个 failed rigid candidates 记录 selected/unselected sole corners 高度、actual
  contact position / distance / normal / force / nearest corner、`J^T lambda_rigid`、
  `J^T f_actual`、`qfrc_constraint` 的逐 DOF 差值、analytic/actual 每脚载荷与 COP；
- 固定 candidate 的 joint qpos + ctrl，对 root penetration 做 `0–12 mm` sweep/bisection，
  记录总载荷是否穿过重量、双脚载荷是否同时满足、root/joint qacc 是否出现零点、
  active contact mode 是否保持一致。

决策门：

- 若 5 个 strict 正对照不能 5/5 闭合，则 R4a.3.1b 的 force/Jacobian/actuator mapping
  未校准，40 个 rigid candidates 暂停解释；
- 若正对照闭合但 failed active set 不一致，则修 fixed-contact active-set constraints；
- 若 active set 一致且 penetration 能恢复 strict equilibrium，则最终 solver 增加
  soft-contact realization stage；
- 若 active set / 动力学正确但 sweep 无零点，则进入 joint actual-contact refinement。

本阶段仍不增加 `max_nfev`、不准备 V3、不做 feedback、不改公共 env / controller /
generator / `kp/kv`，Task061/062 继续 blocked。

### R4a.3.1d — contact taxonomy + collision-free strict coverage correction

R4a.3/R4a.3.1c 的 contact gate 需要先把接触集合说清楚，再解释 failed candidates：

- contact taxonomy 固定拆成 `support_foot_floor_contacts`、
  `forbidden_nonfoot_floor_contacts`、`self_contacts`，并保留每个 geom-pair 明细；
- strict initial gate 与 2 秒 fixed hold 都要求双脚支撑、0 non-foot floor contact、
  0 self-contact；
- strict refinement 继续只做诊断，不改公共 env/controller/generator/`kp/kv`，但要允许
  upper-body / waist joint 在 compiled physical joint limits 内调整，并加入 self-collision
  clearance residual，避免旧 R2 stance 的 arm-trunk 穿插被误标为 strict equilibrium；
- R4a.3.1c 的 force closure 比较必须是同集合：foot hand reconstruction 只比较 filtered
  foot-floor EFC；full EFC 只比较 full `qfrc_constraint`。

本阶段必须重跑二足 4 seeds × 2 range fractions，不允许只重标旧 artifact。若 collision-free
coverage 仍未恢复 `8/8`，下一步仍停在 contact-mode / actual-contact refinement；不得准备
`StanceSolutionV3`、不得进入 feedback、不得改 generator，也不得进入 Task061/062。

### R4a.3.1e — flat double-foot active-set realization

R4a.3.1d 的 `4/8` 只能说明当前 refinement/search contract 覆盖不足，不能证明另外
`4/8` 物理不可行。下一步取消 selected/unselected corner 的离散歧义，先验证平脚全接触
名义姿态：

- 名义 static stance 目标强制两脚 footpad bottom faces 共面、近似水平，8 个 bottom
  corners 高度一致到同一 penetration target；
- 不要求 MuJoCo 恰好产生 8 个 contact point；最终只要求实际 contact taxonomy 里
  只有 expected foot-floor support contacts，且双脚分别承重；
- 禁止全部 self-contact 与 non-foot-floor contact；
- 联合优化 `qpos + ctrl + penetration`，最终验收只看 MuJoCo actual EFC、actual
  `qacc`、actual foot load 和 2 秒 fixed hold；
- rigid full-footpad wrench 只作为诊断，不得作为 strict acceptance；
- 所有关节以 compiled physical limits 为最终 fallback；可先跑局部 trust-region search，
  但失败后必须执行 full-limit fallback，避免 lower-body `±0.08 rad` 假阴性；
- 4 个 R4a.3.1d failed endpoint 全部跑，并至少带一个现有 collision-free strict sample
  走完全相同 solver path 作为正对照。

额外子探针：对 `biped:rf0:seed3` 固定 R4a.3.1d best `qpos`，只重新求 `ctrl`。若
input-only 恢复 strict，则属于 actuator equilibrium realization；否则继续 joint+ctrl+
penetration 联合 realization。

决策门：只有 full-patch 几何不可实现且有明确残差/证书时才讨论 generator grammar；
几何和 rigid wrench 可行但 actual dynamics 不可行时定位 compliant contact realization；
actual equilibrium 可行但 hold 失败时才回到闭环稳定性/feedback；collision-free strict 达到
`8/8` 时才设计 `StanceSolutionV3`。在此之前不得恢复 R4b 或 Task061/062。

### R4a.3.1f — lexicographic collision-free equilibrium realization

R4a.3.1e 的 `5/8` strict coverage 是可靠下界，但剩余 `3/8` 更可能是单阶段
residual formulation/search 问题，而不是 generator、feedback authority 或 MuJoCo contact
physics 已经被证明有问题。下一步把 flat/collision/contact/dynamics 拆成有序阶段：

- Kinematic phase：只解 `qpos`，把 flat double-foot patch 作为 hard acceptance
  constraint；使用所有允许 self-collision geom pair 的 `mj_geomDistance` continuous
  signed-distance clearance，而不是 self-contact count。保留 COM inside support、joint
  margin 和 foot-foot separation。
- Contact-entry phase：从 collision-free flat state 做小 penetration continuation，只要求
  进入预期 double-foot foot-floor contact mode；仍不硬编码 MuJoCo 必须产生恰好 8 个
  contact point。
- Dynamics phase：固定或紧约束 flat/clearance，再解 actual `qacc + ctrl`；每个 fixed
  `qpos` 先跑 bounded affine qacc-vs-ctrl linear subproblem 作为初始化和诊断，再做 actual
  MuJoCo dynamics refinement。
- 最终验收仍只看 actual EFC/contact taxonomy、strict root/joint `qacc`、双脚实际载荷、
  0 forbidden non-foot-floor contact、0 self-contact 和 2 秒 fixed hold。
- 若 strict coverage 恢复 `8/8`，只允许进入显式 contact-wrench equilibrium solver 与
  `StanceSolutionV3(qpos_eq, ctrl_eq)` 设计；仍不得直接集成 feedback/R4b，不得启动
  Task061/062。若仍失败且无独立几何或 wrench/actuation 证书，继续 solver/contact-mode
  diagnosis，不改 generator。

同时补强 R4a.3.1e 测试：正对照必须实际通过 strict+hold；每个 endpoint 若 local
attempts 全失败，必须执行 full-limit fallback。

## Log

- 2026-08-19：任务创建。诊断证据与复现命令见 task061 Log 及
  `task061-rtx-specialist-normal-walk/artifacts/stance_diagnosis_32x2_rf05.json`、
  `stance_diagnosis_32x2_rf00.json`、`stance_isolation_biped_seed557.json`、
  `stance_isolation_biped_simplest.json`。
- 已排除的候选（有数值证据，不作为本任务范围）：reset 决定性
  （`reset_max_qpos_delta=0`）、数值稳定性（`nan_seeds=0`）、关节限位
  （最小 margin 0.68 rad）、actuator 能力（0 次饱和、0 个超力矩上限）、
  mask/PPO/device（751 passed + CUDA smoke）。
- 已量化但延后的次要项：capsule `<inertial pos>` 落在近端关节而非
  capsule 中心（修正后 COM-脚偏移只从 0.178 m 到 0.161 m）、
  `_is_fallen` 阈值未乘 `global_scale`、GAE 把 timeout 当 terminal、
  reward 缺 base-height / foot-support / terminal-fall 项。
- 2026-08-19：先修正 R0 契约。新增 exact-physical `MorphologyInstanceKey` 与
  source-validated `StanceSolution`；明确 absolute qpos 不二次叠加 nominal offset。
  Checkpoint metadata 将 45D/193D tensor schema 与 embodiment/runtime 语义拆开，
  specialist manifest 从 topology-only hash 改为完整 physical instance hash。
- 2026-08-19：R0 verification：

  ```bash
  .venv/bin/python -m pytest tests/test_whole_body_extended.py \
      tests/test_whole_body_contract.py -q
  .venv/bin/python -m pytest -q
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/core/checkpoint.py \
      src/h200_locomotion_lab/robots/procedural_morphology.py \
      src/h200_locomotion_lab/robots/whole_body_stance.py \
      src/h200_locomotion_lab/robots/__init__.py \
      src/h200_locomotion_lab/tools/whole_body_ppo_smoke.py \
      tests/test_whole_body_extended.py
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_ppo_smoke \
      --family biped --num-envs 1 --updates 1 --rollout-steps 2 \
      --trial-seconds 0.1 --device cpu \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r0_checkpoint_contract_smoke.json
  ```

  定向 `23 passed`，全量 `752 passed`，本次文件 ruff clean，agent inspection
  成功。contract smoke 保持 `whole_body_v1_45` / 原 schema hash，同时写出独立的
  `procedural_whole_body_v1_capsule_nominal` embodiment hash 和 64 位 exact-instance
  manifest hash。实际加载 Task061 `artifacts/biped_smoke.pt` 后调用新 validator，
  因缺 `embodiment_contract_version/hash` 按预期拒绝。全仓 `ruff check .` 仍报告
  434 个本任务范围外的既有问题，未改动。
- 2026-08-19：在 run 隔离副本实现 R1 有面积的脚（
  `/home/admin1/workspace/run/locomotion_rl/task067-r1`）。`LinkBlueprint`
  新增 `foot` / `foot_size`；所有 terminal leg link 置为 foot，footpad 半长/半宽
  由整条腿长按 `0.16 / 0.09` 派生，不增加独立随机维度。`compile_mjcf`
  为 foot link 添加 `{link}_footpad` box geom，位置为 compiled local
  `-(length + radius)`，尺寸随 `global_scale * link_scale` 缩放；同 link 原
  capsule 的 `contype/conaffinity` 置 0。`WholeBodyMuJoCoShard._foot_geoms`
  改为只收集 footpad。R1 改变 contact geometry，因此 embodiment contract
  升为 `procedural_whole_body_v1_footpad_nominal`，hash 为
  `e01235aef2eed5a5ed06c32c1d2c3286dccebd41f35042ccd3c9738069cd3d46`；
  45D action / 193D observation schema 保持 `whole_body_v1_45` 不变。
- 2026-08-19：R1 还修正了 `tools/whole_body_stance_diagnosis.py` 的支撑点提取：
  box footpad 用底面角点计算 support hull，而不是继续用 geom center 点；否则
  R1 后二足 hull 仍会被诊断工具误判为退化线段。
- 2026-08-19：R1 verification：

  ```bash
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m pytest tests/test_whole_body_contract.py -q
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m pytest tests/test_whole_body_extended.py -q
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m ruff check \
      src/h200_locomotion_lab/robots/procedural_morphology.py \
      src/h200_locomotion_lab/envs/whole_body_mujoco.py \
      src/h200_locomotion_lab/tools/whole_body_stance_diagnosis.py \
      tests/test_whole_body_contract.py
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m pytest -q
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.whole_body_ppo_smoke \
      --family biped --num-envs 1 --updates 1 --rollout-steps 2 \
      --trial-seconds 0.1 --device cpu \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r1_footpad_contract_smoke.json
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.whole_body_stance_diagnosis \
      --families biped --seeds 4 --range-fraction 0.0 --horizon-steps 0 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r1_footpad_stance_diagnosis_biped4_rf00.json
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.inspect_agent
  ```

  定向 contract `15 passed`，extended `12 passed`；改动文件 ruff clean；全量
  `756 passed, 35 warnings`。R1 smoke artifact 写出新 embodiment version/hash，
  schema hash 保持 `6a3dfcc5f1ca4b27e2312e8ca402823c2740216a7c42eae1ca1fbe5aed278a58`。
  4-seed biped diagnosis 显示 `degenerate_support_all_feet=0`、
  `hull_area_median_all_feet=0.07772650562483298`，证明 footpad 面积已进入
  support hull；同时 `com_inside_support_all_feet=0`、`feet_near_floor_min=1`，
  符合预期地仍需 R2 nominal stance solver，完整 stance gate 未标 passed。
- 2026-08-19：继续在 run 隔离副本实现 R2 exact-physical static stance solver。
  `robots/whole_body_stance.py` 新增
  `solve_static_stance(model, data, blueprint, physical, ...)` 与
  `StaticStanceSolveError`；solver cache 使用 R0 的 exact
  `MorphologyInstanceKey` + stance solver contract hash。`WholeBodyMuJoCoShard`
  在 compile 后求解一次并校验 `StanceSolution`，所有 env reset 复用同一
  absolute `base_height` / joint qpos；`_set_targets` 的 zero-action midpoint
  直接使用 solution absolute qpos，禁止再次叠加 `nominal_offsets`。
  `ground_nominal_pose` 默认 margin 改为 `0.0`，并不再被 shard 训练 reset 路径调用。
- 2026-08-19：R2 还修正了 box geom bottom 诊断口径：
  `whole_body_stance_diagnosis._geom_bottom` 与 `ground_nominal_pose` 对 box 使用
  世界系底面角点最低值，而不是未旋转的 center-z minus size-z 近似。`whole_body_ppo_smoke`
  artifact 现在写出 `stance_solution` manifest、`stance_solution_hash` 和
  `stance_cache_key`。
- 2026-08-19：继续实现 R3 biped grammar 最小机械可行性约束。
  `MorphologyGeneratorConfig` 新增 `require_biped_ankle=True`、
  `mirror_biped_legs=True`。默认二足腿采样必含 `hip_pitch`、`hip_roll`、
  `knee_pitch`、`ankle_pitch`；左右腿只采一次离散结构并镜像复用，右腿 roll/yaw
  axis 符号镜像，左右连续 physical params 仍由各 link/joint 独立采样。四足 grammar
  不使用这两个 toggle。`structural_hash` 的 joint payload 现在包含 axis 向量，避免
  R3 的镜像轴符号在 split hash 中隐身。
- 2026-08-19：R2/R3 verification：

  ```bash
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m pytest tests/test_whole_body_contract.py -q
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m pytest tests/test_whole_body_extended.py tests/test_task061_rtx_specialist.py -q
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.whole_body_ppo_smoke \
      --family biped --num-envs 1 --updates 1 --rollout-steps 2 \
      --trial-seconds 0.1 --device cpu \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r3_static_stance_contract_smoke.json
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.whole_body_stance_diagnosis \
      --families biped quadruped --seeds 4 --range-fraction 0.0 \
      --horizon-steps 100 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r3_stance_diagnosis_4x2_rf00.json
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.whole_body_stance_diagnosis \
      --families biped quadruped --seeds 4 --range-fraction 0.5 \
      --horizon-steps 100 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r3_stance_diagnosis_4x2_rf05.json
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m ruff check \
      src/h200_locomotion_lab/robots/procedural_morphology.py \
      src/h200_locomotion_lab/robots/whole_body_stance.py \
      src/h200_locomotion_lab/envs/whole_body_mujoco.py \
      src/h200_locomotion_lab/robots/__init__.py \
      src/h200_locomotion_lab/tools/whole_body_ppo_smoke.py \
      src/h200_locomotion_lab/tools/whole_body_stance_diagnosis.py \
      tests/test_whole_body_contract.py
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m pytest -q
  PYTHONPATH=src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python \
      -m h200_locomotion_lab.tools.inspect_agent
  ```

  Contract/generator 定向 `18 passed`；extended + Task061 specialist device tests
  `14 passed`；全量 `759 passed, 35 warnings`；改动文件 ruff clean；agent inspection
  成功。最终 embodiment contract 为
  `procedural_whole_body_v1_footpad_static_stance`，hash
  `37f1e0bce3af26db1d7f5499f01bf28ced9faa4621670f5aac501f6d0f354579`，
  schema 仍为 `whole_body_v1_45`。smoke artifact 的 stance solution hash 为
  `e36d0d98b2a669682e8157931964f8c571e4f47cf2aa73573e63a14af0feb847`。
- 2026-08-19：R3 后 4x2 小型 stance diagnosis 显示 reset 几何目标基本达成：
  rf0.0 / rf0.5 的二足 `degenerate_support_all_feet=0`、
  `com_inside_support_all_feet=4/4`、`feet_near_floor_min=2`、
  `foot_height_spread_max=0.000469 / 0.003194`、
  `support_margin_median_all_feet=0.102954 / 0.103358`、`nan_seeds=0`、
  `seeds_with_actuator_over_force_limit=0`。四足 rf0.5 也通过几何小样本；
  四足 rf0.0 的 `foot_height_spread_max=0.008415` 在 4-seed 小样本中仍需跟踪。
  关键 blocker：二足 zero-action 2s 仍为 `zero_action_fall_ratio=1.0`
  （rf0.0 first fall step 44–68，rf0.5 first fall step 44–53），超过 gate
  `≤0.10`。按任务失败判定，完整 stance gate 未通过；不得进入 Task061 重跑或 Task062。
- 2026-08-19：按用户要求把 R1–R3 从 run 隔离副本
  `/home/admin1/workspace/run/locomotion_rl/task067-r1` 逐文件审查移植到主工作区，
  未整目录覆盖，避免破坏 Task068。移植范围只包含 Task067 目标文件：
  `robots/procedural_morphology.py`、`robots/whole_body_stance.py`、
  `envs/whole_body_mujoco.py`、`robots/__init__.py`、`tools/whole_body_ppo_smoke.py`、
  `tools/whole_body_stance_diagnosis.py`、`tests/test_whole_body_contract.py` 与本任务文档；
  同步 R1–R3 历史 smoke / stance diagnosis artifacts。移植后验证：

  ```bash
  PYTHONPATH=src .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/robots/procedural_morphology.py \
      src/h200_locomotion_lab/robots/whole_body_stance.py \
      src/h200_locomotion_lab/envs/whole_body_mujoco.py \
      src/h200_locomotion_lab/robots/__init__.py \
      src/h200_locomotion_lab/tools/whole_body_ppo_smoke.py \
      src/h200_locomotion_lab/tools/whole_body_stance_diagnosis.py \
      tests/test_whole_body_contract.py
  PYTHONPATH=src .venv/bin/python -m pytest tests/test_whole_body_contract.py -q
  ```

  改动文件 ruff clean；Task067 contract tests `18 passed`。重新生成
  `artifacts/r4a_r1_r3_migration_audit.json`，记录主树已具备 R1/R2/R3 marker。
- 2026-08-19：完成 R4a 动态平衡因果诊断。新增只读工具
  `tools/whole_body_dynamic_balance_diagnosis.py`，正式 artifact 为
  `artifacts/r4a_dynamic_balance_diagnosis_4x2.json`：

  ```bash
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis \
      --biped-seeds 0 1 2 3 --quadruped-seeds 0 1 2 3 \
      --range-fractions 0.0 0.5 --horizon-steps 100 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a_dynamic_balance_diagnosis_4x2.json
  PYTHONPATH=src .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py
  ```

  结果：二足 baseline 在 rf0.0/rf0.5 均 `fall_ratio=1.0`，first fall
  44–68 / 44–53；zero-gravity 与 root-locked 均 `fall_ratio=0.0`，确认
  gravity + free-base dynamics 是因果触发项。constant `qfrc_bias/Kp` preload
  仍 8/8 跌倒；dynamic `qfrc_bias/Kp` preload 仍 8/8 跌倒，且 first fall 可提前到
  32–54。接触一致静力求解的 root residual 与接触力 residual 均接近 0，但二足只有
  `2/8` 满足全部 contact / ctrl range / force / foot-height 约束；contact-equilibrium
  hold 仍 `7/8` 跌倒；扰动探针 `74/80` 跌倒、`5/80` 增长、`1/80` 衰减。四足
  baseline `8/8` 存活，contact-equilibrium `7/8` feasible。R4a decision：
  `contact_equilibrium_not_robust_and_unstable`。
- 2026-08-19：按独立复核意见完成 R4a.1 equilibrium 诊断修正。`tools/whole_body_dynamic_balance_diagnosis.py`
  schema 升为 `task067_r4a1_equilibrium_diagnosis_v2`：
  contact force solve 与 joint torque / ctrl bounds 联合评分；候选再经实际
  `mj_forward` 的 `qacc` / `qfrc_constraint` gate；只有初始 qacc 达标且每只脚分别承重的
  feasible equilibrium 才进入 hold/扰动。扰动从位姿扰动改为 root velocity 与单步 impulse。
  同时修正 `WholeBodyMuJoCoShard._is_fallen` 与 `whole_body_stance_isolation.py` 的
  scaled fall threshold。新增 `tests/test_task067_r4a1_equilibrium_diagnosis.py` 覆盖
  joint-bounded feasibility、完整 feasible candidate gate、feasible-only summary、扰动语义与
  scaled threshold。
  正式 artifact 为 `artifacts/r4a1_equilibrium_diagnosis_4x2.json`：

  ```bash
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis \
      --biped-seeds 0 1 2 3 --quadruped-seeds 0 1 2 3 \
      --range-fractions 0.0 0.5 --horizon-steps 100 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a1_equilibrium_diagnosis_4x2.json
  PYTHONPATH=src .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/envs/whole_body_mujoco.py \
      src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py \
      src/h200_locomotion_lab/tools/whole_body_stance_isolation.py \
      tests/test_task067_r4a1_equilibrium_diagnosis.py
  PYTHONPATH=src .venv/bin/python -m pytest tests/test_task067_r4a1_equilibrium_diagnosis.py -q
  ```

  结果：新增 R4a.1 tests `5 passed`；上述文件 ruff clean。正式诊断中二足
  baseline rf0.0/rf0.5 仍 `fall_ratio=1.0`，first fall 44–68 / 44–53；zero-gravity
  与 root-locked 均 `fall_ratio=0.0`。constant `qfrc_bias/Kp` preload 仍 8/8 跌倒；
  dynamic `qfrc_bias/Kp` preload 仍 8/8 跌倒。解析接触力 residual 接近 0，joint
  force/ctrl bounds 没有成为主要 blocker，但实际 `mj_forward` 初始 qacc 不达标：
  rf0.0 / rf0.5 的 `qacc_root_norm_min=2.599 / 2.538`，均高于 R4a.1 gate `1.0`，
  `qacc_joint_max_median=64.56 / 128.60`，高于 gate `10.0`；因此
  `feasible=0/4 + 0/4`，`contact_equilibrium_hold_feasible` 全部 skipped。四足 baseline
  仍 `8/8` 存活。R4a.1 decision：`majority_no_true_equilibrium`。

  收尾验证：

  ```bash
  PYTHONPATH=src .venv/bin/python -m pytest \
      tests/test_task067_r4a1_equilibrium_diagnosis.py \
      tests/test_whole_body_contract.py \
      tests/test_whole_body_extended.py \
      tests/test_task061_rtx_specialist.py -q
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
  ```

  结果：相关合同 / R4a.1 / blocked Task061 guard 测试 `37 passed`；agent inspection 通过。
- 2026-08-19：按独立复核意见完成 R4a.2 joint-aware dynamic stance solve。`tools/whole_body_dynamic_balance_diagnosis.py`
  schema 升为 `task067_r4a2_joint_aware_equilibrium_diagnosis_v3`：外层用
  SciPy least-squares 连续优化 base roll/pitch/yaw、penetration、joint qpos 与 ctrl；
  内层 contact force 使用 SLSQP QP，并联合考虑 joint torque / ctrl bounds。四足不再跳过
  equilibrium solve，而是使用完全相同 solver 作为正对照；feasible-only hold 与扰动统计
  只汇总通过完整 gate 的 equilibrium。新增 / 更新 R4a.2 专门测试，覆盖已知 biped seed0
  正解、quadruped seed0 正对照、完整 feasible candidate gate、feasible-only summary、
  速度/冲量扰动语义、scaled fall threshold 与决策正对照语义。

  正式 artifact 为 `artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json`：

  ```bash
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis \
      --biped-seeds 0 1 2 3 --quadruped-seeds 0 1 2 3 \
      --range-fractions 0.0 0.5 --horizon-steps 100 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json
  PYTHONPATH=src .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/envs/whole_body_mujoco.py \
      src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py \
      src/h200_locomotion_lab/tools/whole_body_stance_isolation.py \
      tests/test_task067_r4a1_equilibrium_diagnosis.py
  PYTHONPATH=src .venv/bin/python -m pytest \
      tests/test_task067_r4a1_equilibrium_diagnosis.py \
      tests/test_whole_body_contract.py \
      tests/test_whole_body_extended.py \
      tests/test_task061_rtx_specialist.py -q
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
  ```

  结果：上述文件 ruff clean；相关合同 / R4a.2 / blocked Task061 guard 测试 `39 passed`；
  agent inspection 通过。R4a.2 反转了 R4a.1 的全局无解结论：二足 rf0.0
  `3/4` feasible，rf0.5 `2/4` feasible，总计 `5/8`；`qacc_root_norm_median`
  为 `0.251 / 1.334`，`qacc_joint_max_median` 为 `0.667 / 6.471`，
  `max_joint_adjustment_median` 为 `0.0276 / 0.0562 rad`。所有 feasible 二足
  equilibrium 的 nominal hold 仍跌倒：rf0.0 `3/3`、rf0.5 `2/2`；速度/冲量扰动
  也全部跌倒：rf0.0 `30/30`、rf0.5 `20/20`。四足正对照使用同一 solver：
  rf0.0 `3/4` feasible、rf0.5 `1/4` feasible，且 feasible 四足 hold 均 `fall_ratio=0.0`。
  R4a.2 decision：`equilibrium_exists_but_perturbation_diverges`；下一步只允许
  Task067-R4b bounded base-attitude / COM feedback diagnosis，不进入 Task061/Task062。
- 2026-08-20：完成 R4b-1 bounded feedback causality diagnosis。新增独立诊断工具
  `tools/whole_body_bounded_feedback_diagnosis.py`，从 R4a.2 artifact 读取 feasible
  equilibrium，固定 `qpos_eq + ctrl_eq`，只在私有 MuJoCo rollout 内施加 bounded
  lower-body target delta。新增 `tests/test_task067_r4b1_bounded_feedback.py` 覆盖
  lower-body-only / bounded delta、paired nominal early-growth 语义和 R4b-1 gate 决策。

  正式 artifact 为 `artifacts/r4b1_bounded_feedback_diagnosis_5eq.json`：

  ```bash
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json \
      --horizon-steps 100 --off-step 50 \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4b1_bounded_feedback_diagnosis_5eq.json
  PYTHONPATH=src .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_bounded_feedback_diagnosis.py \
      tests/test_task067_r4b1_bounded_feedback.py
  PYTHONPATH=src .venv/bin/python -m pytest \
      tests/test_task067_r4b1_bounded_feedback.py \
      tests/test_task067_r4a1_equilibrium_diagnosis.py \
      tests/test_whole_body_contract.py \
      tests/test_whole_body_extended.py \
      tests/test_task061_rtx_specialist.py -q
  PYTHONPATH=src .venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
  ```

  结果：R4b-1 文件 ruff clean；相关合同 / R4a / R4b-1 / blocked Task061 guard 测试
  `42 passed`；agent inspection 通过。正式诊断中，四足正对照未被 destabilize：
  selected combined low gain 下 quadruped feasible equilibrium `4/4` 存活、0 saturation、
  0 non-foot contact、0 unloaded-foot steps。二足没有任何 controller 模式通过 nominal
  2 秒 gate：`hold_baseline`、`attitude_only`、`com_cop_oracle`、`attitude_com_combined`
  均为 `0/5` 存活，first fall 约 60–65 步。selected combined low gain 的扰动结果为
  `1/50` 存活、`49/50` 跌倒，0 saturation，但有 15 个 non-foot-contact steps 与
  133 个 unloaded-foot steps；COM/COP oracle 也仅 `1/50` 存活。controller-off 实验
  `5/5` 在 on phase 存活、off 后退化，但因为 controller-on nominal 自身未通过 2 秒 gate，
  该开关证据不足以授权集成。R4b-1 decision：`bounded_feedback_gate_failed`。
- 2026-08-20：完成 R4b-2 bounded feedback authority / mapping diagnosis。新增
  `tools/whole_body_feedback_authority_diagnosis.py` 与
  `tests/test_task067_r4b2_feedback_authority.py`，继续只使用 R4a.2 的 5 个 feasible 二足
  equilibrium；局部探针直接比较 actual `mj_forward` root `qacc` 的 restoring score，
  2 秒 timeline 记录 foot unload / tilt warning / fall 先后。

  正式 artifact 为 `artifacts/r4b2_feedback_authority_diagnosis_5eq.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_feedback_authority_diagnosis \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4b2_feedback_authority_diagnosis_5eq.json \
      --horizon-steps 100
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_feedback_authority_diagnosis.py \
      tests/test_task067_r4b2_feedback_authority.py \
      src/h200_locomotion_lab/tools/whole_body_bounded_feedback_diagnosis.py \
      tests/test_task067_r4b1_bounded_feedback.py
  .venv/bin/python -m pytest \
      tests/test_task067_r4b2_feedback_authority.py \
      tests/test_task067_r4b1_bounded_feedback.py \
      tests/test_task067_r4a1_equilibrium_diagnosis.py \
      tests/test_whole_body_contract.py \
      tests/test_whole_body_extended.py \
      tests/test_task061_rtx_specialist.py -q
  .venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
  ```

  结果：Ruff clean；相关 R4a/R4b/contract/Task061 guard 测试 `46 passed`；
  agent inspection 通过。authority probes 共 `60` 条，其中 angular probes `40` 条。
  `best_static_improves_angular=40/40`，`best_static_improvement_median=2.642`；
  当前 combined-high 仅 `23/40` 改善、`11/40` 成为最佳，global
  `current_improvement_median=0.147`。细分显示 roll angle 当前映射有效
  （`10/10` current best，median `2.371`），但 pitch angle 当前映射为负恢复
  （median `-2.918`），static pitch `±0.08` 或 inverted 才给出正恢复；roll/pitch
  rate 也主要由 static 最大 delta 胜出。timeline 仍 `0/5` nominal 存活：
  baseline first fall min `60`、current combined low `63`、current combined high `60`、
  inverted high `54`；actuator saturation 均为 `0`，baseline 的 contact loss
  没有先于 tilt warning。R4b-2 decision：
  `bounded_mapping_or_weighting_insufficient`；下一步仍只能继续 R4b deployable
  mapping/weighting 诊断，不允许 controller 集成、`ZeroActionHoldSolution`、contract/hash
  升级或 Task061/Task062。
- 2026-08-20：完成 R4b-2i 独立复核。新增只读诊断
  `tools/whole_body_equilibrium_audit.py`、硬回归
  `tests/test_task067_equilibrium_audit.py` 与 artifact
  `artifacts/r4b2_independent_equilibrium_audit_5eq.json`。未改公共 env/controller、actuator
  `kp/kv`、reward、45D/193D schema、motor strength/latency/failure，未进入 Task061/Task062。

  复现命令：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_equilibrium_audit
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_equilibrium_audit.py \
      tests/test_task067_equilibrium_audit.py
  .venv/bin/python -m pytest -q \
      tests/test_task067_r4a1_equilibrium_diagnosis.py \
      tests/test_task067_r4b1_bounded_feedback.py \
      tests/test_task067_r4b2_feedback_authority.py \
      tests/test_task067_equilibrium_audit.py
  ```

  验证结果：Ruff clean；Task067 相关测试 `17 passed`。独立 artifact 固化 source artifact
  SHA256、诊断及依赖源码 SHA256、每个 MJCF SHA256、morphology instance key、Python /
  NumPy / SciPy / MuJoCo 版本、solver 参数与 simulation timestep。

  R4a.2 的 5 个 source 状态 strict actual equilibrium 为 `0/5`，而不是 `5/5`：其 gate
  允许 `qacc_root_norm <= 1.0`、`qacc_joint_max <= 10.0`。实际 source root qacc norm 约
  `0.247–0.407`；5/5 的 signed `COM_x-COP_x` 为 `-4.0` 至 `-27.1 mm`，对应实际向后
  `qacc_x=-0.048` 至 `-0.310 m/s^2`。LIPM `g/z*(COM-COP)` 与实际 `qacc_xy` 方向和量级
  一致。解析 corner-force QP 声称静态平衡，但 actual MuJoCo soft-contact COP/wrench 不平衡，
  因此 `feasible` 是 false positive。

  在完全相同的原 R4a.2 bounds 内，只最小化 actual `mj_forward qacc`，5/5 均达到
  `root_qacc_norm < 8e-11`、`joint_qacc_max < 3e-10`、双脚有载、无 non-foot contact、
  无 saturation；随后 2 秒 nominal `5/5` 存活，10 秒 nominal 仍 `5/5` 存活。原 source
  nominal 为 `0/5`。所以“feasible equilibrium nominal hold 全部发散”不成立；真正的
  nominal fall 起因是 source state 的初始 actual contact/COP 漂移。strict equilibrium 的
  原 50 个 perturbation 中 `28/50` 存活，说明局部/有限扰动稳定性仍未过 gate，但这是与
  nominal equilibrium existence 分开的后续问题。

  R4b-2 工具还有直接影响结论的诊断缺陷：`_apply_probe_state()` 会 reset `ctrl=0`，
  `run_authority_probe()` 随即在恢复 `ctrl_eq` **之前**计算 COM/COP controller delta；x/y
  offset 同时平移 root、身体和双脚，在无限平面上保持 COM-COP 与 qacc 不变；static
  authority 对每个 probe 同时搜索 `+/-` delta 再取最大值，任何非零单轴 input derivative
  都几乎必然得到“改善”，因此 `40/40` 不能证明足够 authority 或正确统一 mapping。
  scalar desired-axis `qacc` 还忽略显著 off-axis / yaw coupling 与 contact active-set 改变。

  free root 的 qvel/qacc DOF `0:3` linear、`3:6` angular 对当前 generator 正确；roll/pitch
  quaternion 公式也正确。2° Euler perturb 与 `mj_integratePos` tangent 的姿态误差在本 5 个
  source 上最多 `3.89e-4 rad`，不是主因，但未来应直接用 tangent DOF。2° roll probe 已改变
  foot-load/contact mode，不能称为同一 contact mode 的局部线性化。

  在恢复 `ctrl_eq`、用 0.1° 小扰动并减去 equilibrium reference delta 后，attitude-only
  pitch `0/10` 改善、median restoring improvement `-0.00651`；roll 仅 `1/10`、median
  `-0.00229`。projected gravity 本身约为 `g_y=-roll, g_x=+pitch`，而当前 P command 与
  hand mapping 的 input derivative 同号，故 attitude P 对 roll 和 pitch 都是 anti-restoring；
  并非只有 pitch 有问题。COM/COP term 让 roll combined 看似有效（`9/10`），但 pitch
  combined 仍 `0/10`、median `-0.285`。fixed pitch mapping 的 pitch-qacc derivative 在
  5/5 都为正（`32.7–50.1` per target-rad）；roll mapping 仅 4/5 为正，seed 3 为 `-4.20`，
  且存在 roll-to-yaw、pitch-to-roll 等明显 coupling，证明统一 hip/knee/ankle hand weights
  不是 morphology-invariant allocation。

  timeline 的 20 ms sampled statistic 在其窄定义内可复现：baseline 5/5 的 15° tilt warning
  先于“整只脚法向载荷低于 5% body weight”。但它没有采样 t=0 / 2 ms substep、signed
  COM-COP、COP 边缘或 contact active set；source 在 t=0 已有 signed COP error，seed 0
  到明显 tilt 前该 error 和向后加速度持续增大且双脚仍有载。因此 timeline 不能排除、反而
  漏掉了 equilibrium/contact drift。四足确实经过同一 Python feedback/solver 调用路径，
  但其 baseline 已稳定、支撑多边形更宽，且 quadruped right-side axes 没有 biped 的镜像
  语义；`4/4 not destabilized` 只是弱 sanity control，不验证二足 mapping polarity。

  既有 R4b tests 在 artifact 缺失时会 skip；当前 checkout 有 artifact，所以不是“永远通过”，
  但测试只覆盖 restoring-score 代数、synthetic decision branch 与一个 roll probe，没有覆盖
  ctrl-eq-before-COP、pitch/P polarity、offset gauge invariance、contact-preserving tangent、
  source hash、multi-step causal response 或 paired controller-on/off。新增测试将 source
  artifact 设为必需，并固定了 loose feasible false positive、root translation invariance 与
  actual-qacc refinement 三个反证。
- 2026-08-20：完成 R4a.3 strict actual-equilibrium coverage。新增
  `tools/whole_body_strict_equilibrium_coverage.py` 与
  `tests/test_task067_strict_equilibrium_coverage.py`。新工具不再只读取 R4a.2 的
  `contact_equilibrium.status == feasible` 记录，而是对二足 4 seeds × 2 range fractions
  全量执行 strict actual refinement；R4a.2 的解析 contact QP 只保留为 candidate/provenance。
  最终 contract 需要 strict actual `mj_forward` qacc、实际双脚持续承重、0 non-foot contact、
  0 actuator saturation，以及 fixed `qpos_eq + ctrl_eq` 的 2 秒 nominal hold 全部通过。

  正式 artifact 为 `artifacts/r4a3_strict_equilibrium_coverage_4x2.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a3_strict_equilibrium_coverage_4x2.json \
      --families biped --horizon-steps 100 --max-nfev 1500
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_strict_equilibrium_coverage.py \
      tests/test_task067_strict_equilibrium_coverage.py
  .venv/bin/python -m pytest tests/test_task067_strict_equilibrium_coverage.py -q
  ```

  结果：Ruff clean；新增 strict coverage 测试 `4 passed`。正式 4×2 artifact decision 为
  `strict_equilibrium_coverage_incomplete`：二足 `5/8` 通过 strict actual equilibrium +
  nominal hold，accepted labels 为 `biped:rf0:seed0`、`biped:rf0:seed2`、
  `biped:rf0:seed3`、`biped:rf0.5:seed1`、`biped:rf0.5:seed2`；incomplete labels 为
  `biped:rf0:seed1`、`biped:rf0.5:seed0`、`biped:rf0.5:seed3`。旧 R4a.2 source-feasible
  `5/5` 都是 strict false positive（source strict actual equilibria `0/8`），且当前 strict
  refinement 没有把旧 source-infeasible 记录推进到 strict contract pass。因此尚未证明所有
  generator 二足都能在当前 bounds 内找到严格 equilibrium；不得以 R4a.2/R4b-1/R4b-2 的
  因果解释继续 controller 或训练分支。
- 2026-08-20：完成 R4a.3.1 contact-preserving continuation / search adequacy diagnosis。
  新增 `tools/whole_body_contact_preserving_continuation.py` 与
  `tests/test_task067_r4a31_contact_preserving_continuation.py`。工具只诊断 3 个 R4a.3
  failed endpoint：seed0 `rf0 -> rf0.5`、seed3 `rf0 -> rf0.5`、seed1 `rf0.5 -> rf0`；
  每一步用上一 accepted strict `qpos/ctrl` 映射到同 topology target physical，失败时
  只二分 range step，不增加普通 multistart。qacc-only 失败后记录 contact-preserving
  residual（actual qacc、共同 penetration target 的左右脚 bottom error、per-foot load
  deficit、warm-start 正则），最终仍用 R4a.3 strict gate 验收。

  正式 artifact 为 `artifacts/r4a31_contact_preserving_continuation_3fail.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_contact_preserving_continuation \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a3_strict_equilibrium_coverage_4x2.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31_contact_preserving_continuation_3fail.json \
      --family biped --range-step 0.05 --min-step 0.00625 \
      --horizon-steps 100 --max-nfev 1500
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_contact_preserving_continuation.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py -q
  ```

  结果：Ruff clean；R4a.3 + R4a.3.1 focused tests `8 passed`。正式 artifact decision 为
  `r4a31_single_support_or_search_failure`：continuation 没有恢复 `8/8`，combined strict
  contract 仍为 `5/8`，3 个 failed endpoint recovered `0/3`，且
  `kinematic_double_support_infeasible=0`、`wrench_or_actuation_infeasible=0`。endpoint
  分类为 `biped:rf0.5:seed0 = single_support_equilibrium_only_found`、
  `biped:rf0.5:seed3 = search_exhausted_without_certificate`、
  `biped:rf0:seed1 = search_exhausted_without_certificate`。

  关键数值：seed0 route 从 `rf0` 只能 strict 推进到 `rf0.1375`，再到 `rf0.14375`
  即使用 `0.00625` 二分和 contact-preserving residual 也未过 strict gate；其 direct
  endpoint strict-refinement 是精确单脚平衡，root `qacc_norm=1.14e-11`、
  joint `qacc_max=5.29e-11`、signed COM-COP 约 `[-4.75e-14, 3.72e-14]`，但左脚
  bottom `+0.0102 m`、`0` contact、`0 N`，右脚 bottom `-0.00278 m`、`2` contacts、
  `593.9 N`。seed3 route 从 `rf0` 连 `0.00625` 都不能 strict 推进；direct endpoint
  是单脚 active set 但 qacc 不 strict（root `0.215`、joint `0.546`），因此不是
  single-support equilibrium certificate。seed1 route 从 `rf0.5` 向 `rf0` 连 `0.00625`
  都不能 strict 推进；direct endpoint 双脚有接触和载荷但 qacc 不 strict（root `0.783`、
  joint `1.255`），只能归类为搜索耗尽无证书。

  结论：当前证据不能支持修改 generator grammar；尤其不能把 search exhaustion 写成
  physical infeasibility。下一步仍只能做 solver/contact-mode diagnosis，或若进入
  R4a.3.2，只能在已通过 strict contract 的 5 个（不是 8 个）equilibrium 上做私有诊断；
  Task061/062 继续 blocked。
- 2026-08-20：完成 R4a.3.1a true-continuation correctness diagnosis。新增
  `tools/whole_body_true_continuation_correctness.py` 与
  `tests/test_task067_r4a31a_true_continuation_correctness.py`。新诊断保留
  `artifacts/r4a31_contact_preserving_continuation_3fail.json`，单独写出
  `artifacts/r4a31a_true_continuation_correctness_3fail.json`；warm start 直接取上一
  accepted strict `qpos/ctrl`，joint bounds 只用 compiled physical joint limits（本次
  默认不启用 trust region），禁止 target R2 stance `±0.08 rad` re-anchor。

  正式命令：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_true_continuation_correctness \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a3_strict_equilibrium_coverage_4x2.json \
      --previous-r4a31-artifact .agent/task/task067-biped-stance-contract/artifacts/r4a31_contact_preserving_continuation_3fail.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31a_true_continuation_correctness_3fail.json \
      --family biped --range-step 0.05 --min-step 0.00625 \
      --horizon-steps 100 --max-nfev 1500
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_true_continuation_correctness.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py -q
  ```

  结果：Ruff clean；focused R4a.3/R4a.3.1/R4a.3.1a tests `12 passed`。
  新 artifact sha256 为
  `69543216ce85b73cd496322992c3b0f0ff32e8b5784bbc49fc6d7bd786808585`；旧
  R4a.3.1 artifact sha256 仍为
  `b9d5a91cd607bcd9590f1611bea2267299cad9806c97a95390cbaba34566544e`。

  新 artifact 的硬断言全部通过：
  `all_continuation_step_artificial_warm_start_clip_zero=true`、
  `artificial_warm_start_clip_violations=[]`、
  `solver_search_failure_not_promoted_to_physical_infeasible=true`。三条 route 的
  真实 joint-limit clip、ctrl-range clip、人工 trust-region clip 全部为 0；但 target
  R2 stance 到上一解的最大 joint distance 仍很大：seed0 route `0.145 rad`、
  seed3 route `0.225 rad`、seed1 route `0.399 rad`。其中硬回归 seed3
  `rf0 -> rf0.00625` 的人工 clip 为 `0.0`，不再出现 `0.145 rad` 级别的人工裁剪。

  coverage 未恢复：combined strict contract 仍为 `5/8`，failed endpoint recovered
  `0/3`。分类保持
  `biped:rf0.5:seed0 = single_support_equilibrium_only_found`、
  `biped:rf0.5:seed3 = search_exhausted_without_certificate`、
  `biped:rf0:seed1 = search_exhausted_without_certificate`；
  `kinematic_double_support_infeasible=0`、`wrench_or_actuation_infeasible=0`。seed0 true
  continuation 只能 strict 到 `rf0.0125` 后停止在 `rf0.01875`；seed3 从 `rf0` 连
  `0.00625` 都不能 strict 推进；seed1 从 `rf0.5` 连 `0.00625` 都不能 strict 推进。

  决策：没有证据支持 generator grammar 修正，也不再继续堆普通 least-squares / max-nfev；
  下一步应进入固定双脚 contact-mode 的 state-input-wrench 约束求解。公共 env、
  controller、generator、`kp/kv` 未修改；Task061/062 继续 blocked。
- 2026-08-20：完成 R4a.3.1b fixed double-foot contact-mode state-input-wrench solve。
  新增 `tools/whole_body_fixed_contact_mode_wrench_solve.py` 与
  `tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py`。新工具消费
  R4a.3 strict coverage 与 R4a.3.1a true-continuation artifact，只对 3 个 failed endpoint
  做固定双脚 contact mode 约束求解。每个候选显式包含 state、input 与 contact-wrench
  variables；contact modes 全部含双脚，且 search failure 不产生
  `kinematic_double_support_infeasible` 或 `wrench_or_actuation_infeasible` 证书。

  正式 artifact 为 `artifacts/r4a31b_fixed_contact_mode_wrench_solve_3fail.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_fixed_contact_mode_wrench_solve \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a3_strict_equilibrium_coverage_4x2.json \
      --continuation-json .agent/task/task067-biped-stance-contract/artifacts/r4a31a_true_continuation_correctness_3fail.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31b_fixed_contact_mode_wrench_solve_3fail.json \
      --family biped --horizon-steps 100 --max-nfev 350
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_fixed_contact_mode_wrench_solve.py \
      tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py \
      tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py -q
  ```

  结果：Ruff clean；focused R4a.3/R4a.3.1/R4a.3.1a/R4a.3.1b tests `17 passed`。
  新 artifact sha256 为
  `6467027a4c32739a09ae0275214a77afc65f87c6d865cd5f11a62618f4190a10`。
  artifact 断言：`all_modes_are_double_foot=true`、
  `search_failure_not_promoted_to_physical_infeasible=true`、
  `strict_acceptance_requires_actual_gate_and_2s_hold=true`。

  结果没有恢复 `8/8`：combined strict contract 仍为 `5/8`，endpoint recovered `0/3`，
  strict candidates `0`。但固定双脚刚体约束里出现 `40` 个 rigid-contact feasible
  candidates，说明 seed3/seed1 不能再解释成“解析双脚 wrench 不存在”；失败发生在
  刚体 wrench 候选进入 MuJoCo soft-contact actual gate 时。

  endpoint 结果：

  - `biped:rf0.5:seed0`：最终仍保留
    `single_support_equilibrium_only_found`，因为 direct endpoint 是精确单脚 actual
    equilibrium；固定双脚 rigid 候选 `12/16` 可行但 best actual gate 滑到 single-support，
    actual root `qacc_norm=9.85`、joint `qacc_max=957.6`。
  - `biped:rf0.5:seed3`：`fixed_contact_wrench_solution_found_actual_gate_failed`；
    rigid best root wrench `3.0e-09`、joint residual `3.1e-09`、height `1.9e-11`、load
    deficit `0`，但 actual gate root `qacc_norm=32.45`、joint `qacc_max=2537.5`。
  - `biped:rf0:seed1`：`fixed_contact_wrench_solution_found_actual_gate_failed`；
    rigid best root wrench `1.6e-09`、joint residual `1.6e-09`、height `1.5e-12`、load
    deficit `0`，但 actual gate root `qacc_norm=2.52`、joint `qacc_max=45.75`。

  决策：固定双脚刚体 contact-wrench 解不能直接作为 strict equilibrium 或
  `StanceSolutionV3`；没有物理不可行证书，也没有 generator grammar 修改依据。下一步只能
  继续 contact-mode/model diagnosis，重点解释 explicit wrench candidate 与 MuJoCo
  soft-contact actual qacc 的差异；公共 env、controller、generator、`kp/kv` 未修改；
  Task061/062 继续 blocked。
- 2026-08-20：完成 R4a.3.1c soft-contact force-closure / realization audit。新增
  `tools/whole_body_soft_contact_realization_audit.py` 与
  `tests/test_task067_r4a31c_soft_contact_realization_audit.py`。新工具先用 R4a.3 的
  5 个 strict equilibrium 校准解析方程与 MuJoCo：hand actuator qfrc vs
  `data.qfrc_actuator`、actual contact force/Jacobian vs `qfrc_constraint`、MuJoCo
  internal EFC `mj_mulJacTVec(data.efc_force)` vs `qfrc_constraint`，以及完整 dynamics
  equation closure。随后审计 R4a.3.1b 的 3 个 failed best candidates 的 sole-corner
  active set、actual contact realization、rigid/actual/constraint qfrc 差值、load/COP，
  并固定 joint qpos + ctrl 做 `0–12 mm` penetration sweep/bisection。

  正式 artifact 为 `artifacts/r4a31c_soft_contact_force_closure_realization_audit.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_soft_contact_realization_audit \
      --r4a3-json .agent/task/task067-biped-stance-contract/artifacts/r4a3_strict_equilibrium_coverage_4x2.json \
      --r4a31b-json .agent/task/task067-biped-stance-contract/artifacts/r4a31b_fixed_contact_mode_wrench_solve_3fail.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31c_soft_contact_force_closure_realization_audit.json \
      --family biped
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_soft_contact_realization_audit.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py \
      tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py -q
  ```

  结果：Ruff clean；focused R4a.3–R4a.3.1c tests `22 passed`。新 artifact sha256 为
  `086b777bfae33858db584acd209af4c599b14160ca30de57ae7916435cc6aefb`。
  artifact 断言：`strict_positive_control_count_is_5=true`、
  `internal_mujoco_efc_contact_closure_passes_all_positive_controls=true`、
  `full_dynamics_closure_passes_all_positive_controls=true`、
  `search_failure_not_promoted_to_physical_infeasible=true`。

  正对照校准结果：actuator qfrc `5/5` pass，完整 dynamics closure `5/5` pass（max
  residual `1.66e-13`），MuJoCo internal EFC contact closure `5/5` pass（max diff
  `0`）；但 hand actual-contact force/Jacobian reconstruction 只有 `2/5` pass，max qfrc
  diff `39.03`。因此 5 个 strict 正对照没有达到 5/5 full closure，R4a.3.1b 的
  `40` 个 rigid-contact feasible candidates 暂停作为解释证据；当前不能把原因归结为
  MuJoCo soft-contact 本身。

  failed candidate realization 结果：

  - `biped:rf0.5:seed0`：actual/analytic total load ratio `0.739`，左脚仍无实际 load；
    unselected corner 不更深，但 rigid vs actual/constraint qfrc root norm diff `360.27`。
  - `biped:rf0.5:seed3`：active set 不一致；selected min height `-1.91e-11 m`，
    unselected min height `-0.0026226 m`，未选角点深 `2.62 mm`；actual/analytic total
    load ratio `1.101`。
  - `biped:rf0:seed1`：actual/analytic total load ratio `0.835`；active set height
    没有 seed3 那种未选角点更深，但 rigid vs actual/constraint qfrc root norm diff
    `78.17`。

  penetration sweep：3/3 endpoint 的实际总载荷都穿过重量，3/3 有某些 penetration
  同时满足双脚载荷，但 strict actual equilibrium recovered `0/3`，active contact mode
  consistent `0/3`。因此不进入 V3、不做 feedback、不改 generator；下一步若继续，应先
  校准 explicit contact force-frame/Jacobian mapping against MuJoCo EFC，并修正
  fixed-contact active-set realization。
- 2026-08-20：完成 R4a.3.1d contact taxonomy + collision-free strict coverage correction。
  新增共享 contact taxonomy `tools/whole_body_contact_taxonomy.py`，并修正
  `tools/whole_body_strict_equilibrium_coverage.py`、
  `tools/whole_body_equilibrium_audit.py`、
  `tools/whole_body_soft_contact_realization_audit.py` 与对应测试。strict initial gate 与
  2 秒 hold 现在都要求：双脚支撑、0 non-foot floor contact、0 self-contact；contact
  artifact 明确记录 `support_foot_floor_contacts`、`forbidden_nonfoot_floor_contacts`、
  `self_contacts` 和每个 geom-pair 明细。strict refinement 对 upper-body / waist joint
  使用 compiled physical joint limits，并加入 self-contact clearance residual；lower-body
  仍保留 R4a.3 的 `0.08 rad` 诊断 trust region。

  正式 collision-free coverage artifact 为
  `artifacts/r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json \
      --families biped --horizon-steps 100 --max-nfev 1500
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_soft_contact_realization_audit \
      --r4a3-json .agent/task/task067-biped-stance-contract/artifacts/r4a3_strict_equilibrium_coverage_4x2.json \
      --r4a31b-json .agent/task/task067-biped-stance-contract/artifacts/r4a31b_fixed_contact_mode_wrench_solve_3fail.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31c_soft_contact_force_closure_realization_audit.json \
      --family biped
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_contact_taxonomy.py \
      src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py \
      src/h200_locomotion_lab/tools/whole_body_equilibrium_audit.py \
      src/h200_locomotion_lab/tools/whole_body_strict_equilibrium_coverage.py \
      src/h200_locomotion_lab/tools/whole_body_soft_contact_realization_audit.py \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py \
      tests/test_task067_r4a31d_contact_taxonomy_collision_free.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py \
      tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py \
      tests/test_task067_r4a31d_contact_taxonomy_collision_free.py -q
  ```

  结果：Ruff clean；R4a.3/R4a.3.1 focused regression `27 passed`。新 R4a.3.1d artifact
  sha256 为 `48a1237f1679063312e259006a42d2e73565667a903facf4026b20955174bd2f`，
  schema 为 `task067_r4a3_strict_equilibrium_coverage_v2_contact_taxonomy_collision_free`。
  collision-free strict coverage 为 `4/8`，accepted labels：
  `biped:rf0:seed0`、`biped:rf0:seed2`、`biped:rf0.5:seed1`、
  `biped:rf0.5:seed2`；incomplete labels：
  `biped:rf0:seed1`、`biped:rf0:seed3`、`biped:rf0.5:seed0`、
  `biped:rf0.5:seed3`。`strict_initial_self_collision_free=7/8`，
  `strict_nominal_hold_self_collision_free=4/4`（只统计实际执行 hold 的记录，denominator
  显式记录为 `4`），`source_feasible_false_positive=5`。

  关键修正：旧 `biped:rf0:seed3` 在 R4a.3 的 `5/8` strict set 中通过，但 collision-free
  correction 后不再通过；self-contact 可以清掉，但 best joint `qacc_max=0.00359` 仍超过
  strict 阈值。`biped:rf0.5:seed0` 的 best 仍不是双脚支撑，且存在 footpad-footpad
  self-contact。其余 4 个 accepted record 均为 strict initial self-contact `0`，2 秒 hold
  self-contact steps `0`。

  同步重跑的 R4a.3.1c same-set EFC artifact sha256 为
  `02b4b9a3b24ac969577b0bf328cb76ddd2d72312d267aaf298b7afffad80f356`，schema 为
  `task067_r4a31c_soft_contact_force_closure_realization_audit_v2_same_set_efc`。修正后：
  foot hand reconstruction vs filtered foot-floor EFC `5/5` pass，full EFC vs full
  `qfrc_constraint` `5/5` pass，完整 dynamics closure `5/5` pass，最大 hand/contact
  qfrc diff `2.27e-13`。因此旧的“force/Jacobian mapping bug”解释被撤回；R4a.3.1b
  rigid candidates 可解释，但当前 blocker 是 fixed-contact active set 不一致，仍是
  `biped:rf0.5:seed3` 的 unselected corner 更深问题。R4a.3.1b 后续应先修
  selected/unselected corner active-set constraints，但按本阶段决策门，应排在
  collision-free contract 修正之后。

  决策：collision-free strict coverage 未恢复 `8/8`，且没有
  `kinematic_double_support_infeasible` 或 `wrench_or_actuation_infeasible` 证书。因此不得改
  generator grammar、不得准备 `StanceSolutionV3`、不得进入 feedback / controller 集成；
  Task061/062 继续 blocked。下一步若继续，应在 collision-free contract 下修
  fixed-contact active set，并进入 fixed double-foot actual-contact refinement。
- 2026-08-20：完成 R4a.3.1e flat double-foot active-set realization。新增
  `tools/whole_body_flat_double_foot_realization.py` 与
  `tests/test_task067_r4a31e_flat_double_foot_realization.py`，并修正
  `tools/whole_body_strict_equilibrium_coverage.py` 的 hold self-collision 统计分母。1e
  取消 selected/unselected corner 离散 active-set 枚举，改为把两只 footpad 的 8 个 bottom
  corners 作为 flat double-foot patch 目标；最终 strict acceptance 不要求 MuJoCo 恰好生成
  8 个 contact，只要求 actual EFC/qacc、实际双脚载荷、0 non-foot-floor、0 self-contact
  与 2 秒 fixed hold。

  正式 artifact 为 `artifacts/r4a31e_flat_double_foot_active_set_realization.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_flat_double_foot_realization \
      --input-json .agent/task/task067-biped-stance-contract/artifacts/r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31e_flat_double_foot_active_set_realization.json \
      --family biped --positive-label biped:rf0:seed0 --input-only-label biped:rf0:seed3 \
      --local-max-nfev 450 --full-max-nfev 900 --horizon-steps 100
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_contact_taxonomy.py \
      src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py \
      src/h200_locomotion_lab/tools/whole_body_equilibrium_audit.py \
      src/h200_locomotion_lab/tools/whole_body_strict_equilibrium_coverage.py \
      src/h200_locomotion_lab/tools/whole_body_soft_contact_realization_audit.py \
      src/h200_locomotion_lab/tools/whole_body_flat_double_foot_realization.py \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py \
      tests/test_task067_r4a31d_contact_taxonomy_collision_free.py \
      tests/test_task067_r4a31e_flat_double_foot_realization.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py \
      tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py \
      tests/test_task067_r4a31d_contact_taxonomy_collision_free.py \
      tests/test_task067_r4a31e_flat_double_foot_realization.py -q
  ```

  结果：Ruff clean；focused R4a.3/R4a.3.1 regression `31 passed`。新 artifact sha256 为
  `09d55f196f7e14d1b20215e009680c1c07e66e669364f14468a0398b2e53c513`，schema 为
  `task067_r4a31e_flat_double_foot_active_set_realization_v1`。1e 跑了全部 4 个
  failed endpoints，并带 `biped:rf0:seed0` 作为 same-path 正对照；正对照通过。硬断言：
  `all_four_failed_endpoints_tested=true`、`positive_control_same_path_tested=true`、
  `final_acceptance_does_not_require_exactly_8_contacts=true`、
  `full_limit_fallback_after_local_failure_available=true`、
  `search_failure_not_promoted_to_physical_infeasible=true`。

  1e 从 4 个 failed endpoint 中恢复 `1/4`：`biped:rf0:seed3` 通过
  `collision_free_strict_double_support_equilibrium_found`，best 来自 local trust-region
  `r4a31d_best` start；actual root `qacc_norm=5.50e-10`、joint `qacc_max=1.02e-08`，
  双脚载荷约 left `270.68 N` / right `264.94 N`，0 non-foot-floor、0 self-contact，
  2 秒 hold 通过。其 flat patch 最大 height error / spread 约 `0.91 mm`，在 1e 的
  `1 mm` flat realization tolerance 内。

  子探针结论：`biped:rf0:seed3` 固定 qpos 只求 ctrl 不充分；input-only 后
  root `qacc_norm=1.75e-04`、joint `qacc_max=5.79e-04`，未过 strict。因此该恢复需要
  joint+ctrl+penetration 联合 realization，而不是单纯 actuator target 修正。

  剩余 3 个 endpoint 仍为 `flat_search_exhausted_without_certificate`：
  `biped:rf0:seed1` best 双脚支撑且无 forbidden contacts，但 joint `qacc_max=0.00715`、
  flat error `32.3 mm`；`biped:rf0.5:seed0` best 双脚支撑但仍有 `2` 个 self-contact，
  flat error `61.9 mm`；`biped:rf0.5:seed3` best 为 single-support 且有 `1` 个
  self-contact，flat error `88.8 mm`。三者 rigid full-footpad wrench diagnostic 也未给出
  可用 feasible certificate，但这不是几何或 wrench/actuation 物理不可行证书。

  决策：combined collision-free strict coverage 从 `4/8` 提升到 `5/8`，accepted labels
  为 `rf0:seed0`、`rf0:seed2`、`rf0:seed3`、`rf0.5:seed1`、`rf0.5:seed2`；仍未恢复
  `8/8`，且 `flat_patch_geometry_infeasible=0`、`wrench_or_actuation_infeasible=0`。
  因此不得改 generator grammar、不得准备 `StanceSolutionV3`、不得恢复 R4b/feedback、
  不得进入 Task061/062。下一步只能继续 fixed double-foot actual-contact refinement，
  重点处理剩余 3 个 endpoint 的 flat geometry/self-contact/contact realization。
- 2026-08-20：完成 R4a.3.1f lexicographic collision-free equilibrium realization。新增
  `tools/whole_body_lexicographic_collision_free_equilibrium.py` 与
  `tests/test_task067_r4a31f_lexicographic_collision_free_equilibrium.py`，并补强
  `tests/test_task067_r4a31e_flat_double_foot_realization.py` 的正对照/fallback 断言。1f
  将 1e 的单阶段 residual 拆成 kinematic qpos-only flat/continuous signed-distance
  clearance、small penetration contact-entry、bounded qacc-vs-ctrl linear subproblem 和
  actual MuJoCo dynamics refinement 四个诊断阶段；最终 strict acceptance 仍只由 actual
  EFC/contact taxonomy、strict `qacc`、双脚载荷、0 forbidden/self contact 与 2 秒 hold
  决定。

  正式 artifact 为
  `artifacts/r4a31f_lexicographic_collision_free_equilibrium_realization.json`：

  ```bash
  .venv/bin/python -m h200_locomotion_lab.tools.whole_body_lexicographic_collision_free_equilibrium \
      --r4a31d-json .agent/task/task067-biped-stance-contract/artifacts/r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json \
      --r4a31e-json .agent/task/task067-biped-stance-contract/artifacts/r4a31e_flat_double_foot_active_set_realization.json \
      --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31f_lexicographic_collision_free_equilibrium_realization.json \
      --family biped --positive-label biped:rf0:seed0 \
      --kinematic-max-nfev 900 --dynamics-max-nfev 900 --horizon-steps 100
  .venv/bin/python -m ruff check \
      src/h200_locomotion_lab/tools/whole_body_contact_taxonomy.py \
      src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py \
      src/h200_locomotion_lab/tools/whole_body_equilibrium_audit.py \
      src/h200_locomotion_lab/tools/whole_body_strict_equilibrium_coverage.py \
      src/h200_locomotion_lab/tools/whole_body_soft_contact_realization_audit.py \
      src/h200_locomotion_lab/tools/whole_body_flat_double_foot_realization.py \
      src/h200_locomotion_lab/tools/whole_body_lexicographic_collision_free_equilibrium.py \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py \
      tests/test_task067_r4a31d_contact_taxonomy_collision_free.py \
      tests/test_task067_r4a31e_flat_double_foot_realization.py \
      tests/test_task067_r4a31f_lexicographic_collision_free_equilibrium.py
  .venv/bin/python -m pytest \
      tests/test_task067_strict_equilibrium_coverage.py \
      tests/test_task067_r4a31_contact_preserving_continuation.py \
      tests/test_task067_r4a31a_true_continuation_correctness.py \
      tests/test_task067_r4a31b_fixed_contact_mode_wrench_solve.py \
      tests/test_task067_r4a31c_soft_contact_realization_audit.py \
      tests/test_task067_r4a31d_contact_taxonomy_collision_free.py \
      tests/test_task067_r4a31e_flat_double_foot_realization.py \
      tests/test_task067_r4a31f_lexicographic_collision_free_equilibrium.py -q
  ```

  结果：Ruff clean；focused R4a.3/R4a.3.1 regression `36 passed`。新 artifact sha256 为
  `0228a86ad30010a23ae770dc21ad693b7ece96fef801d3b6f561caf7f774f2e8`，schema 为
  `task067_r4a31f_lexicographic_collision_free_equilibrium_realization_v1`。1f 跑了 1e
  剩余 3 个 incomplete endpoints，并带 `biped:rf0:seed0` same-path 正对照。硬断言：
  `uses_lexicographic_phases=true`、
  `kinematic_phase_uses_continuous_signed_distance_not_contact_count=true`、
  `dynamics_phase_uses_no_integer_contact_count_residual=true`、
  `final_acceptance_does_not_require_exactly_8_contacts=true`、
  `search_failure_not_promoted_to_physical_infeasible=true`。

  1f 恢复全部剩余 3 个 endpoint，combined collision-free strict coverage 从 `5/8`
  提升到 `8/8`，`combined_incomplete_labels=[]`。恢复明细：
  `biped:rf0:seed1` best 为 tight kinematic tube、`r4a31d_best` start、penetration
  `0.25 mm`，actual root `qacc_norm=2.17e-08`、joint `qacc_max=5.15e-08`，双脚载荷
  left `223.71 N` / right `217.74 N`，flat spread `0.0817 mm`，minimum self-pair
  distance `0.131 m`，hold 通过；`biped:rf0.5:seed0` best penetration `0.75 mm`，
  root `7.29e-10`、joint `7.14e-09`，双脚载荷 left `162.20 N` / right `431.68 N`，
  flat spread `0.747 mm`，minimum self-pair distance `2.47 mm`，hold 通过；
  `biped:rf0.5:seed3` best penetration `0.50 mm`，root `7.69e-10`、joint `4.38e-09`，
  双脚载荷 left `324.23 N` / right `258.64 N`，flat spread `0.542 mm`，minimum
  self-pair distance `15.18 mm`，hold 通过。三个 best 均为 0 non-foot-floor contact、
  0 self-contact，且不要求 exact 8 contacts。

  结论：R4a.3.1e 剩余失败属于旧诊断搜索 formulation/active-basin 问题；没有几何或
  wrench/actuation 物理不可行证书，不能据此修改 generator grammar。1f 本身不准备
  `StanceSolutionV3`、不恢复 feedback/R4b、也不启动 Task061/062；下一步只允许设计显式
  contact-wrench equilibrium solver，并准备把 `qpos_eq + ctrl_eq` 纳入
  `StanceSolutionV3` 合同。

### R4a.3.1g — actual-dynamics feedforward stance solution V3

R4a.3.1f 的 `8/8` 只是存在性证明；根因复核表明公共 static stance contract 把
equilibrium state 与 equilibrium input 错当成同一个量。静态平衡需要独立
`qpos_eq` 与 `ctrl_eq`，position actuator 在 `qvel=0` 时由
`kp * (ctrl_eq - qpos_eq)` 提供关节平衡力矩。旧公共 reset/zero-action 路径把
`ctrl=qpos_stance`，导致 actuator force 为 0，足底接触不能同时闭合所有关节动力学行。

本 closed unit 将 1f 提炼成 artifact-free actual MuJoCo stance solver，并升级公共
stance solution contract：

- `solve_actual_dynamics_stance(...)` 只依赖 compiled MuJoCo model/data、blueprint 与
  exact physical params；不读取旧 R4a artifacts，不让 rigid contact-wrench 成为公共真值。
- V3 `StanceSolution` 保存 `root_pose_eq`、`joint_qpos_eq` 与独立
  `actuator_ctrl_eq`；root `x/y/yaw` gauge 固定为 0，roll/pitch 保留在 reset pose。
- biped joint 使用 compiled physical joint limits，并强制 endpoint `joint_margin >0.05`；
  ctrl 使用 actuator ctrlrange 并保留 `ctrl_margin >=0.01`。
- public `WholeBodyMuJoCoShard.reset()` 写入 `qpos_eq` 和 `ctrl_eq`；
  zero action 围绕 `actuator_ctrl_eq` 输出残差，不改 45D/193D、`kp/kv`、reward、
  motor process 或 controller。
- 审计工具 `whole_body_stance_solution_v3_audit.py` 直接构建 public shard，最终以
  actual EFC/qacc、双脚载荷、0 self/non-foot contact 与 2 秒 hold 验收；search/timeout
  failure 只写 `search_exhausted_without_certificate`，并显式
  `physical_infeasibility_claimed=false`。

#### Log

- Implemented:
  - `src/h200_locomotion_lab/robots/whole_body_actual_stance.py`
  - `src/h200_locomotion_lab/robots/whole_body_stance.py`
  - `src/h200_locomotion_lab/robots/procedural_morphology.py`
  - `src/h200_locomotion_lab/envs/whole_body_mujoco.py`
  - `src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py`
  - `src/h200_locomotion_lab/tools/whole_body_stance_solution_v3_audit.py`
  - `tests/test_whole_body_contract.py`
  - `tests/test_whole_body_extended.py`
  - `tests/test_task067_r4a31g_stance_solution_v3_audit.py`
- Contract versions:
  - `STANCE_SOLUTION_CONTRACT_VERSION=whole_body_static_stance_v3_actual_dynamics_feedforward`
  - `PROCEDURAL_EMBODIMENT_CONTRACT_VERSION=procedural_whole_body_v2_footpad_actual_stance_feedforward`
- Artifacts:
  - `.agent/task/task067-biped-stance-contract/artifacts/r4a31g_stance_solution_v3_actual_feedforward_audit_4seed.json`
  - `.agent/task/task067-biped-stance-contract/artifacts/r4a31g_stance_solution_v3_actual_feedforward_audit_biped9.json`
  - `.agent/task/task067-biped-stance-contract/artifacts/r4a31g_stance_solution_v3_actual_feedforward_audit.json`

Verification commands:

```bash
.venv/bin/python -m h200_locomotion_lab.tools.whole_body_stance_solution_v3_audit \
    --endpoint-seeds 4 \
    --matrix-seeds 4 \
    --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31g_stance_solution_v3_actual_feedforward_audit_4seed.json
.venv/bin/python -m h200_locomotion_lab.tools.whole_body_stance_solution_v3_audit \
    --endpoint-seeds 4 \
    --matrix-seeds 9 \
    --skip-quadruped-matrix \
    --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31g_stance_solution_v3_actual_feedforward_audit_biped9.json
.venv/bin/python -m h200_locomotion_lab.tools.whole_body_stance_solution_v3_audit \
    --endpoint-seeds 4 \
    --matrix-seeds 32 \
    --record-timeout-seconds 90 \
    --progress \
    --output-json .agent/task/task067-biped-stance-contract/artifacts/r4a31g_stance_solution_v3_actual_feedforward_audit.json
.venv/bin/python -m ruff check \
    src/h200_locomotion_lab/robots/whole_body_actual_stance.py \
    src/h200_locomotion_lab/robots/whole_body_stance.py \
    src/h200_locomotion_lab/robots/procedural_morphology.py \
    src/h200_locomotion_lab/envs/whole_body_mujoco.py \
    src/h200_locomotion_lab/tools/whole_body_dynamic_balance_diagnosis.py \
    src/h200_locomotion_lab/tools/whole_body_stance_solution_v3_audit.py \
    tests/test_whole_body_contract.py \
    tests/test_whole_body_extended.py \
    tests/test_task067_r4a31g_stance_solution_v3_audit.py
.venv/bin/python -m pytest \
    tests/test_whole_body_contract.py \
    tests/test_whole_body_extended.py \
    tests/test_task067_r4a31g_stance_solution_v3_audit.py -q
.venv/bin/python -m pytest tests/test_task067_r4a31*.py -q
```

Results:

- Ruff clean.
- `tests/test_whole_body_contract.py tests/test_whole_body_extended.py
  tests/test_task067_r4a31g_stance_solution_v3_audit.py`: `31 passed in 32.09s`.
- `tests/test_task067_r4a31*.py`: `32 passed in 158.07s`.
- 4x2 endpoint audit: biped strict V3 `8/8`, record build failures `0`,
  public zero-action falls `0`, min joint margin `0.06098`, min ctrl margin `0.01003`,
  max `|ctrl_eq-qpos_eq|=0.87889`, root x/y/yaw gauge fixed zero.
- biped9 audit: matrix biped `17/18` built/pass; only `biped:rf0:seed8` remained
  `search_exhausted_without_certificate`, with no physical infeasibility claim.
- full 32-seed matrix audit with 90s per-record timeout:
  - endpoint 4x2 still `8/8`;
  - biped matrix `49/64` built/pass, `15` search-exhausted records;
  - biped built records had public zero-action falls `0`, min joint margin
    `0.06000000018`, min ctrl margin `0.01003`;
  - failure labels:
    `biped:rf0:seed8`, `biped:rf0:seed9`, `biped:rf0:seed11`,
    `biped:rf0:seed21`, `biped:rf0:seed23`, `biped:rf0:seed30`,
    `biped:rf0.5:seed9`, `biped:rf0.5:seed11`, `biped:rf0.5:seed13`,
    `biped:rf0.5:seed14`, `biped:rf0.5:seed23`, `biped:rf0.5:seed24`,
    `biped:rf0.5:seed25`, `biped:rf0.5:seed26`, `biped:rf0.5:seed29`;
  - every failed biped record has `failure_classification=search_exhausted_without_certificate`
    and `physical_infeasibility_claimed=false`;
  - quadruped matrix built `64/64`, but public zero-action falls `3/64` under this audit,
    so the full public stance matrix is not passed.

Conclusion: the original 4x2 biped nominal stance blocker was the missing
equilibrium feedforward target, not feedback authority, generator grammar, or MuJoCo
soft-contact physics. However, the current V3 actual-dynamics solver is not yet a
production stance solver for the full 32-seed matrix: `49/64` biped strict coverage
is a solver/search coverage lower bound, not a physical infeasibility certificate.
Task061/062 and R4b remain blocked.

## Review

R0 contract gate（R1 前必须通过）：

- 相同 blueprint + 相同 physical 的 instance/cache key 完全一致；只改变 physical
  seed 时 key 不同，旧 `StanceSolution` 必须拒绝复用。
- `StanceSolution` qpos 明确为 absolute compiled coordinate；reset/action midpoint
  不得再次叠加 `nominal_offsets`。
- `WHOLE_BODY_SCHEMA_VERSION=whole_body_v1_45` 与现有 schema hash 保持不变。
- 新 checkpoint 必须携带并校验 embodiment contract version/hash 与 64 位
  manifest hash；缺字段或 expected hash 不匹配的旧 checkpoint 必须拒绝。

R0 状态：**passed**（只代表契约修正完成；R1–R3 与 stance gate 尚未执行）。

R1 状态：**passed**（contact geometry / foot classification 修正完成；R2/R3 与
完整 stance gate 仍未执行）。

R2 状态：**passed**（exact physical stance solution 与 reset/action midpoint 接入完成）。

R3 状态：**implemented, partial verification passed**（biped grammar 约束已落地；
小样本 reset 几何显著改善，但完整 stance gate 因二足 zero-action fall 仍 blocked）。

R4a 状态：**superseded by R4a.1**（R4a 找到 free-base/contact stability blocker，
但 equilibrium gate 还不够严格）。

R4a.1 状态：**superseded by R4a.2**。R4a.1 的 strict gate 有效，但搜索器固定
R2 joint qpos，只证明离散候选无解；`majority_no_true_equilibrium` 不再作为分支依据。

R4a.2 状态：**artifact reproduced, causal decision rejected by R4b-2i**。R1–R3 已逐文件
移植到主工作区；Task068 未被整目录覆盖。但 `equilibrium_exists_but_perturbation_diverges`
不能继续作为分支依据：5 个所谓 feasible source 均不满足 actual-dynamics equilibrium，
其 nominal fall 不能归因于 equilibrium 附近的 perturbation divergence。解析 contact QP
可作为 candidate generator，最终 equilibrium acceptance 必须改由严格 actual `mj_forward`
gate 与 nominal hold 验证。不得进入 Task061 / Task062。

R4b-1 状态：**experiment reproduced, gate failed, causal interpretation superseded**。其
`0/5` nominal 与 `1/50` selected perturbation 统计存在，但输入不是 strict equilibrium；
controller 还带有非零 equilibrium reference bias，controller-off 也没有 continuously-on
paired counterfactual。因此结果不能区分 mapping、authority 与 source drift，不能据此设计
`ZeroActionHoldSolution`、升级 contract/hash 或进入 Task061 / Task062。

R4b-2 状态：**artifact reproduced, decision not reliable / superseded by R4b-2i**。DOF、
restoring-score 代数与 quaternion 数学基本正确，但 source equilibrium false positive、
authority COP 在 `ctrl=0` 上计算、offset probe 是 gauge translation、`+/-` max 使 40/40
authority 证据近似必然成立，且 one-step scalar qacc 忽略 cross coupling/contact mode，故
`bounded_mapping_or_weighting_insufficient` 不能作为充分因果结论。

R4b-2i 状态：**independent diagnosis passed, implementation remains blocked**。当前最可能
原因按层级为：原 nominal fall 首因是 **equilibrium / actual-contact COP drift**；R4b hand
feedback 的 attitude P sign 和 morphology-dependent allocation 确有错误，但属于 source
equilibrium 修正后的二级 perturbation-stability 问题；现有 bounded target space 已有非零
authority，却没有证据证明其 authority 量值不足；R4b-2 诊断工具本身含足以污染 decision
的 bug/invalid probes。下一步先修诊断 equilibrium contract 并对 4x2 全集重跑 strict
actual equilibrium coverage；再在 strict equilibrium 上做 reference-centered multi-step
linearization/contact-Jacobian/actuator-direction 诊断。仍不得实现公共 R4b controller、设计
`ZeroActionHoldSolution`、升级 contract/hash 或进入 Task061 / Task062。

R4a.3 状态：**strict actual-equilibrium contract corrected / R4a.3.1f restores 8/8
collision-free coverage**。
诊断 contract 已修正为解析 contact QP 只生成候选，最终以 strict actual `mj_forward`
qacc、双脚实际载荷与 nominal hold 验收。原 R4a.3 4×2 coverage 为 `5/8`，但未禁止
self-contact；R4a.3.1d collision-free 修正后降为 `4/8`，R4a.3.1e flat realization
恢复到 `5/8`，R4a.3.1f lexicographic realization 最终恢复 full collision-free strict
coverage `8/8`。因此此前剩余失败是诊断搜索 formulation / basin 问题，不是 generator
物理不可行证书。下一步只允许设计显式 contact-wrench equilibrium solver，并准备
`StanceSolutionV3(qpos_eq, ctrl_eq)`；仍不得直接集成 feedback/R4b 或进入 Task061 /
Task062。

R4a.3.1 状态：**search adequacy diagnosis complete / no generator infeasibility
certificate**。同 topology continuation 没有恢复 `8/8`，3 个 failed endpoint recovered
`0/3`；`rf0.5:seed0` 被明确分类为
`single_support_equilibrium_only_found`，`rf0.5:seed3` 与 `rf0:seed1` 为
`search_exhausted_without_certificate`。没有独立 `kinematic_double_support_infeasible`
或 `wrench_or_actuation_infeasible` 证据，因此不得收紧 generator grammar；仍不得进入
Task061 / Task062。

R4a.3.1a 状态：**true-continuation correctness diagnosis complete / old stance-bound
clip excluded**。独立 artifact 保留旧 R4a.3.1 结果且证明所有 continuation step 的人工
warm-start clip 为 0；target R2 stance 与上一解确有大距离，但不再参与 branch
re-anchor。true continuation 仍未恢复 `8/8`，也没有物理不可行证书。因此下一步不是
继续增加 least-squares 搜索量，也不是修改 generator，而是固定双脚 contact-mode 的
state-input-wrench 约束求解；Task061 / Task062 继续 blocked。

R4a.3.1b 状态：**fixed contact-mode wrench solve complete / actual soft-contact gate
still fails**。固定双脚 state-input-wrench 约束在解析刚体层面为 seed3/seed1 找到可行
候选，但 3 个 failed endpoint 仍有 `0` 个通过 strict actual + 2 秒 hold，coverage 仍为
`5/8`。这排除了“只要写显式双脚 wrench 约束就能得到可用 strict equilibrium”的路径；
也没有 kinematic 或 wrench/actuation infeasibility certificate。因此不得准备
`StanceSolutionV3`、不得改 generator、不得进入 Task061/062；下一步若继续，只能诊断
explicit wrench candidate 与 MuJoCo soft-contact `qacc` 的模型差异。

R4a.3.1c 状态：**same-set force closure corrected / fixed-contact active-set blocker**。
R4a.3.1d 修正后，foot hand reconstruction 只与 filtered foot-floor EFC 比较，full EFC
只与 full `qfrc_constraint` 比较；5 个旧 strict 正对照 actuator qfrc、foot filtered EFC、
full EFC 与完整 dynamics equation 都 `5/5` 闭合。因此旧的 force/Jacobian mapping bug
结论被撤回，R4a.3.1b rigid candidates 不再因 apples-to-oranges EFC 比较而暂停解释。
failed audit 仍确认 seed3 fixed-contact active set 不真实：selected corner 在 `0` 附近，
但 unselected corner 深 `2.62 mm`。penetration sweep 没有恢复 strict equilibrium。
下一步只能修 fixed-contact selected/unselected corner active-set constraints，并在
collision-free contract 下做 actual-contact refinement；仍不得准备 `StanceSolutionV3`、
不得做 feedback、不得改 generator 或进入 Task061/062。

R4a.3.1d 状态：**collision-free strict coverage corrected / coverage remains incomplete**。
contact taxonomy 已拆分 `support_foot_floor_contacts`、`forbidden_nonfoot_floor_contacts`、
`self_contacts` 与 geom-pair 明细；strict initial gate 和 2 秒 hold 均要求 0 self-contact。
upper-body / waist joint refinement 使用 compiled physical joint limits，并加入
self-collision clearance residual。完整二足 4×2 重跑后 collision-free
`strict_contract_passed=4/8`，accepted 为 `rf0:seed0`、`rf0:seed2`、`rf0.5:seed1`、
`rf0.5:seed2`。旧 `rf0:seed3` 被移出 strict set；`rf0.5:seed0` 仍不是双脚支撑且有
footpad-footpad self-contact。hold self-collision 统计已修为只统计实际执行 hold 的记录
`4/4`，不再把 infeasible/dummy hold 算入分子。没有几何或 wrench/actuation 物理不可行
证书，因此不得改 generator grammar；Task061/062 继续 blocked。

R4a.3.1e 状态：**flat double-foot realization recovered 1 endpoint / superseded by
R4a.3.1f**。取消 selected/unselected corner 模式后，flat patch actual-contact solver
恢复 `biped:rf0:seed3`，combined collision-free strict coverage 从 `4/8` 提升到 `5/8`。
`rf0:seed3` 的 input-only probe 未恢复 strict，说明它需要 joint+ctrl+penetration 联合
realization，不是固定 qpos 的 actuator-only 问题。剩余
`rf0:seed1`、`rf0.5:seed0`、`rf0.5:seed3` 均为
`flat_search_exhausted_without_certificate`；没有 flat geometry 或 wrench/actuation
物理不可行证书。正对照 `rf0:seed0` 走同一路径通过。该结论被 R4a.3.1f 的
lexicographic formulation supersede，后者恢复剩余 3 个 endpoint。

R4a.3.1f 状态：**lexicographic collision-free realization complete / strict coverage
restored 8/8**。kinematic qpos-only phase 证明剩余 3 个 endpoint 均可达到 flat
double-foot、continuous signed-distance collision-free、COM inside support 的名义姿态；
contact-entry penetration continuation 均进入 expected double-foot foot-floor mode；dynamics
phase 通过 bounded qacc-vs-ctrl linear subproblem 初始化后恢复全部剩余 strict+hold。
combined accepted labels 覆盖二足 `4 seeds × 2 range_fraction` 全集，`combined_incomplete_labels=[]`。
没有任何 `kinematic_double_support_infeasible` 或 `wrench_or_actuation_infeasible`
certificate，因此不得改 generator grammar。1f 只是诊断 artifact，不改公共 env/controller/
generator/`kp/kv`，也不准备 V3。该结论已被 R4a.3.1g 的 actual-dynamics feedforward
公共 contract 集成 supersede；不再把显式 rigid contact-wrench solver 设为权威层。

R4a.3.1g 状态：**endpoint V3 feedforward fixed / full matrix solver coverage
incomplete**。公共 `StanceSolution` 已升级为独立 `qpos_eq + actuator_ctrl_eq`，reset
使用 `qpos_eq`，zero action 围绕 `ctrl_eq` 输出残差。原二足 `4 seeds × 2 range_fraction`
endpoint 通过 strict actual qacc、双脚载荷、0 self/non-foot contact 和 2 秒 hold：
`8/8`，endpoint record build failures `0`，public zero-action falls `0`，min joint margin
`0.06098`，min ctrl margin `0.01003`，root x/y/yaw gauge 为 0。完整 32-seed matrix
尚未通过：biped `49/64` built/pass，15 个失败全部分类为
`search_exhausted_without_certificate` 且 `physical_infeasibility_claimed=false`；
quadruped built `64/64`，但 public zero-action falls `3/64`。因此根因“缺失 equilibrium
feedforward target”对原 4×2 成立，但当前 actual-dynamics stance solver 还不是生产
coverage solver；不得改 generator、不得恢复 R4b/feedback、不得进入 Task061/062。

R4a.2/R4b historical replay compatibility（2026-08-21）：Task069 全量回归暴露
7 个旧 Task067 诊断测试失败，根因不是几何漂移，而是 R4a.2 historical artifact 仍携带
`procedural_whole_body_v1_footpad_static_stance` contract key，公共 runtime 已升级为
`procedural_whole_body_v2_footpad_actual_stance_feedforward`。已在 R4b/strict coverage
共用 `_build_shard` 边界加显式 replay binding：先重建当前 runtime shard，再要求 artifact
的 `blueprint_hash` 与 `physical_hash` 逐字段匹配；source contract 只允许当前 v2 或精确
白名单旧 v1 hash `37f1e0bce3af26db1d7f5499f01bf28ced9faa4621670f5aac501f6d0f354579`。
未知 contract、错 hash、blueprint/physical 篡改均 fail closed。新生成的 R4b/strict
payload 额外区分 `source_replay_contracts` 与 `runtime_contract`，避免把旧 v1 source
artifact 静默标成 v2 通过证据。未修改旧 artifact、public generator default、stance
solver 或 controller；一个旧 R4a.2 seed0 known-positive 测试已改为验证 historical
artifact record 本身，而不是要求当前 v2 solver 复现旧 v1 search 起点。验证结果：
focused 原失败组 `27 passed`，`tests/test_task067*.py` 为 `59 passed`，相关 Ruff clean，
全量 `.venv/bin/python -m pytest -q` 为 `825 passed, 35 warnings`。

Stance gate 状态：**blocked pending full-matrix actual-dynamics solver coverage**。
公共二足 endpoint zero-action hold 已由 V3 feedforward 修复，但完整 32-seed stance matrix
仍未通过；当前不得进入 Task061 pilot、Task062 shared MLP、Task063 或 Task064。

Stance gate（32 个 train-like 二足 seed + 32 个四足 seed，
`range_fraction=0.0` 和 `0.5` 各跑一次，用
`tools/whole_body_stance_diagnosis.py` 出 JSON）：

| 指标 | 二足 pass | 四足 pass |
| --- | --- | --- |
| `degenerate_support_all_feet` | `0/32` | `0/32` |
| `hull_area_median_all_feet` | `≥0.02 m²` | `≥0.10 m²` |
| `com_inside_support_all_feet` | `32/32` | `32/32` |
| `support_margin_median_all_feet` | `≥ +0.02 m` | `≥ +0.05 m` |
| reset `feet_near_floor` | 每个 seed `= 腿数` | 每个 seed `= 腿数` |
| reset `foot_height_spread` | `≤0.005 m` | `≤0.005 m` |
| zero-action 2s `zero_action_fall_ratio` | `≤0.10` | `≤0.05` |
| `nan_seeds` | `0` | `0` |
| `actuators_over_force_limit` | `0` | `0` |
| `min_joint_limit_margin_min` | `>0.05 rad` | `>0.05 rad` |

失败判定：任一行不达标就停在本任务继续诊断，不允许改 reward、不允许
提高 PD、不允许进入 Task061 重跑或 Task062。

允许进入 Task062 的条件（必须全部满足，按顺序）：

1. 本任务 stance gate 通过；
2. Task060 usability gate 用新 generator 重跑 2000/2000 通过，且
   新增记录 `min_height ≥ 0.7 * stance_height` 的比例 ≥0.90；
3. Task061 的 100×10s specialist quality gate 在二足和四足各自通过
   （zero-fall ≥0.95、normalized velocity error ≤0.25、
   non-foot contact ≤0.05、roll/pitch p95 ≤0.45 rad）。

在 1–3 全部通过之前，Task062 shared MLP、Task063 hidden motor process、
Task064 GRU/TXL 一律不启动。
