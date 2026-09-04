# Task069 — LocoFormer 论文可证形态包络修正

状态：**in_progress**。

进入条件：Task051 已提供可复现的 primitive-link 两足/四足生成器；Task067 的
stance 审计只对当前 `procedural_whole_body_v2_footpad_actual_stance_feedforward`
契约有效。

退出条件：新增一个版本化、论文可证的 LocoFormer-style morphology profile，覆盖
两足、四足、轮式两足、轮式四足四个家族；旧 v2 profile 的 XML、manifest、hash、
cache/checkpoint 语义不被静默改变；完成确定性、编译、有限步物理、slot/mask、视觉
gallery 和回归验证。

本任务唯一允许的 claim：

> 本仓库实现了与 LocoFormer 论文公开描述一致的、可验证的四家族程序化形态包络。

不得声称“官方生成器复现”“官方代码移植”或“像素级/参数级 parity”。截至任务创建时，
官方项目页只公开论文和视频，没有公开形态生成器源码。

## Route

### 1. 权威证据与 reference

只把作者论文和项目页当作 LocoFormer 事实来源：

- 官方项目页：<https://generalist-locomotion.github.io/>
- arXiv v1：<https://arxiv.org/html/2509.23745>
- Appendix A.1 / Figure 6：四类训练形态分别为 quadruped、wheeled quadruped、
  biped、wheeled biped；训练形态是程序生成体，不使用在售机器人的精确参数。
- Figure 7：G1、H1、GR1、TRON1、Berkeley Humanoid、A1、Spot、ANYmal C、
  TRON1-W、Go2-W 是 unseen evaluation embodiments，不是生成器模板。

本地已保存并验证的官方 Figure 6 reference：

- `../task067-biped-stance-contract/artifacts/independent_r4a31g_review/`
  `locoformer_official_fig6a_quadruped.png`
  - 1565×508 RGBA
  - SHA-256 `ded3e8bc9130b7918b6336698e3db1ad7db1290794045d5619f338eed5ce5d31`
- `../task067-biped-stance-contract/artifacts/independent_r4a31g_review/`
  `locoformer_official_fig6c_biped.png`
  - 1565×508 RGBA
  - SHA-256 `9714c051ac26320c126a517561794c83aae59dde392a21c5f19463108bc62605`

`lucidrains/locoformer` README 自称 WIP，且未由官方项目页链接；可用于一般 TXL 实现
比较，但不得作为本任务的 morphology ground truth，也不得下载或 vendor。

### 2. 已确认的本地偏差

| 维度 | 论文公开包络 | 当前本地实现 | 本任务要求 |
| --- | --- | --- | --- |
| family | 两足、四足及各自轮式版 | 只有 `biped` / `quadruped` | 补齐四个显式 family，manifest 不得混类 |
| wheel | 轮式两足、轮式四足是训练类别 | 没有实际 wheel topology | 生成真实 wheel joint、rolling geom、actuator/slot/mask |
| joint grammar | joint 集合与顺序可变化 | 有变化，但 biped 默认强制 ankle、左右腿镜像 | 新 profile 必须显式支持有/无 ankle 与可配置镜像策略；旧 profile 保持原样 |
| morphology identity | 训练体是程序生成体 | primitive-link 方向正确 | 不得把 G1/Go2 等 named robot 参数硬编码进训练 grammar |
| visual evidence | Figure 6 展示四家族的群体多样性 | 仅有零散 Task067 stance 截图 | 每个 family 输出群体 montage 和近景结构图 |
| claim boundary | 论文核心还包括 TXL、长上下文和大规模 RL | Task051/067 只覆盖生成与站姿 | 本任务只关闭 morphology envelope，不声称 policy/训练复现 |

当前 footpad、actual-stance feedforward、cache identity 是本仓库为了可运行性增加的契约，
不是论文公开的 LocoFormer 要求。新 profile 可以复用这些基础设施，但不得把它们写成
“官方设计”。

### 3. 允许修改范围

- `src/h200_locomotion_lab/robots/procedural_morphology.py`
- `src/h200_locomotion_lab/robots/whole_body_slots.py`（仅当 wheel slot/mask 确有缺口）
- `src/h200_locomotion_lab/robots/__init__.py`
- 新增一个 task-scoped manifest/gallery/verification tool；不得复制出第二套生产 generator
- 直接相关的 `tests/test_whole_body_*.py` 或新建 `tests/test_task069_*.py`
- `.agent/doc/locoformer.md`
- 本 task 的 `task.md` 与 `artifacts/`

超出以上范围必须先在 Log 说明理由并拆成新的 closed unit。尤其不得顺手修改 policy、
reward、PPO/TXL、bounded feedback、Task061/062 训练链或 Task067 stance solver。

### 4. 禁止项

- 不下载 checkpoint、机器人资产、数据集、上游仓库或 simulator asset。
- 不启动 H200；所有实现与 smoke 默认走本地 RTX 5060 Ti + MuJoCo/headless。
- 不以更多随机 restart、放宽 NaN/碰撞阈值或跳过失败记录来制造通过。
- 不覆盖、重置或回滚现有 dirty worktree。
- 不在原 v2 contract 名称下改变确定性输出。
- 不把旧 Task067 的 stance artifact 自动解释为新 profile 的通过证据。
- 不要求与 Figure 6 颜色、相机、像素完全一致；验收对象是结构包络和物理契约。

## Closed units

### R0 — 论文 claim 与 legacy baseline 冻结

#### Route

1. 将论文中直接可证、推断、未知三类信息写入
   `artifacts/r0_source_contract.json`，每一项带来源段落或 figure。
2. 在任何生产修改前，对 v2 profile 的固定 seed 集合保存 blueprint manifest、compiled
   XML SHA-256、contract version/hash、instance/cache key，写入
   `artifacts/r0_legacy_v2_baseline.json`。
3. 至少覆盖 `biped/quadruped × seeds 0..31 × range_fraction 0/0.5`；失败也必须保留
   expected denominator，不能只记录 built records。

#### Log

- 2026-08-20：在生产 generator 修改前生成 `artifacts/r0_source_contract.json` 与
  `artifacts/r0_legacy_v2_baseline.json`。legacy baseline 为
  `biped/quadruped × seeds 0..31 × range_fraction 0/0.5 = 128` 条，`128/128`
  build/compile 通过；Python 3.11.16、MuJoCo 3.5.0，硬件假设 RTX 5060 Ti-first、
  本地 MuJoCo/headless、H200 disabled。baseline 保存了 blueprint/physical/instance/XML
  SHA-256、contract version/hash 和 capture 时 dirty-file 清单。

#### Review

- 未生成 source contract 或 legacy baseline 时，禁止进入 R1。
- 任何无法由论文证实的数值范围必须标为本仓库设计选择，不得伪装成官方参数。

### R1 — 版本化 grammar 与旧契约隔离

#### Route

1. 新增明确命名的新 profile/contract version；不得覆盖
   `procedural_whole_body_v2_footpad_actual_stance_feedforward`。
2. 旧 profile 对 R0 全部样本必须 byte/hash stable。
3. 新 profile 至少可构造确定性 witnesses：
   - biped 有 ankle / 无 ankle；
   - biped mirrored / 显式非 mirrored leg grammar；
   - biped 有 arms / 无 arms；
   - quadruped 至少两种不同 joint-axis/order signature；
   - trunk aspect ratio、limb length、mass/COM 等连续参数仍与 topology identity 分离。
4. profile、family、wheel topology 必须进入完整 manifest 和 instance/cache identity。

#### Log

- 2026-08-20：新增显式 `locoformer_paper_faithful_morphology_v1` profile 与
  `procedural_locoformer_paper_faithful_v1` embodiment contract；保留旧
  `MorphologyGenerator()` 为 v2 默认。新 profile manifest/structural hash 明确包含
  profile、family、wheel topology；固定拓扑下 trunk/limb 几何与物理 hash 独立变化。
  biped 的 ankle/arms/mirror toggles 和 quadruped axis/order variation 均有测试 witness。
  论文只证明四类 family/程序生成训练体；数值 ranges 明确标为 local design choice。

#### Review

- 旧 profile 任一 baseline hash 漂移即 fail closed。
- 新旧 contract 共享同一 cache key、checkpoint identity 或结构 hash 时 fail closed。
- 只更换渲染外观、没有结构 witnesses 时不得通过。

### R2 — 轮式两足与轮式四足

#### Route

1. 实现 `wheeled_biped` 与 `wheeled_quadruped` 的显式 family 表示，或语义完全等价且
   manifest 中不可混淆的表示。
2. wheeled biped 每条承重腿有一个 terminal wheel；wheeled quadruped 每条腿有一个
   terminal wheel。wheel 必须是有转轴、接触 geom、actuator 和 semantic slot 的动力学
   部件，不能只是视觉 cylinder。
3. unified action/observation slots 对缺失/存在 wheel 的 selector 和 mask 必须可逆、无重复，
   不得改变 legacy fixed-slot 解释。
4. wheel 半径、宽度、轴向、range/continuous joint、friction 与 actuator limits 必须写入
   manifest；连续物理 randomization 仍需确定性。

#### Log

- 2026-08-20：`wheeled_biped` 每条记录 2 个 terminal wheel，
  `wheeled_quadruped` 每条记录 4 个 terminal wheel；wheel 均为 continuous hinge、
  lateral axis `(0,1,0)`、cylinder contact geom、motor actuator，并在 manifest 写出
  radius/width/axis/range/friction/effort limit。R3 matrix 的四个 family 均记录
  `32/32` compile、slot round-trip 和 wheel XML checks；非轮式 family 无 active wheel slot。

#### Review

- wheel joint/geom/actuator/slot 任一缺失，或 rolling axis 与结构明显不一致时 fail。
- 非轮式 family 意外获得 active wheel slot 时 fail。
- 不能为了 wheel smoke 修改 reward、controller 或 stance solver。

### R3 — 全矩阵、物理 smoke 与视觉 gallery

#### Route

对四个 family 各取 seeds `0..31`，以 expected denominator `32` 独立报告：

1. 两次独立生成的 manifest/XML/hash 完全一致。
2. `32/32` MuJoCo compile；所有 mass/inertia/limit/actuator 参数有限且物理符号有效。
3. reset 后至少 100 个 2 ms step 无 NaN/Inf、无 solver fatal、状态保持有限；本项不把
   “不会跌倒”当作 morphology pass，但必须单列 fall/contact/penetration 数据，不能隐藏。
4. 每个 family 输出：
   - 32-sample montage；
   - 至少 4 个近景 oblique/side/front render；
   - active joint/slot 标注或配套 JSON；
   - wheel family 的轮轴与接触近景。
5. `artifacts/r3_morphology_matrix.json` 必须报告完整 denominator、build/compile/smoke
   failures、joint count、axis/order signatures、arm/ankle/mirror witnesses、wheel count、trunk/
   limb ratio和所有 artifact SHA-256。
6. 生成 PNG 后，执行 agent 必须使用能实际解码并显示本地图片的 image viewer，逐张打开
   四套 montage 和全部最终入选近景；不能只检查文件存在、尺寸、SHA 或让测试读取像素。
   每个 family 都要在 Log 记录亲眼观察到的：躯干与肢体连接、腿数和前后/左右布局、
   足端或轮端位置、轮轴方向、离地/穿地、断链/悬浮 geom、自碰撞式重叠、异常比例、
   reset 姿态和相机裁切。发现问题时必须写出 `family/seed/view`，修正后重新截图并复看；
   不得从 montage 中移除异常 seed 来制造正常外观。

#### Log

- 2026-08-20：`artifacts/r3_morphology_matrix.json` 报告四 family 各自 expected
  denominator 32，合计 128；built/compiled/deterministic/finite-100-step-smoke/
  slot-mask/wheel checks 全部 `32/32`。MuJoCo Renderer 生成四套 32-sample montage、
  每 family 4 张近景和 active-slot JSON。首次 quadruped 近景有上缘裁切，已将相机距离
  从 2.0 调到 2.5，重渲染并复看。
- 2026-08-21：根据独立 reviewer 复核修复 R3 false-green：paper profile 的 arm attachment
  改为按 trunk 实际半宽、arm radius、physical link-scale 上界和 clearance 计算；reset
  诊断现在显式区分 world body、生成 body 质量、qpos/qvel/qacc/ctrl/actuator force、
  warning、floor contact、self contact、terminal floor distance 和 rollout penetration。
  四 family 全部为 `finite_physics=32/32`、`finite_smoke=32/32`、`reset_self_collision_free=32/32`
  和 `reset_terminal_floor_clear=32/32`；reset 的离地 terminal 不再被伪写成 near-floor，已在
  `r3_manual_visual_observations.json` 披露。确定性检查改为独立重采 physical 并比较
  blueprint/physical/XML 全字段 hash。四套 montage 和 16 张近景再次用本地 image viewer
  打开复看，未见 arm/trunk 自碰撞、断链或裁切；rollout 接触/穿透作为单独数据保留。

#### Review

- 任一 family `built < 32`、`compiled < 32`、出现 NaN/Inf/solver fatal，R3 不通过。
- montage 只展示成功样本但 JSON 隐去失败 denominator，视为 false green。
- 视觉接近 Figure 6 只能作 sanity check，不能替代结构化验证。
- 只有 PNG 文件、自动图像统计或 renderer 成功记录，但没有执行 agent 的逐图视觉观察
  Log，R3 不通过。viewer 无法解码、画面全黑/透明、机器人过小不可辨或关键结构被裁切，
  均须重新渲染，不能按“截图已生成”通过。

### R4 — 迁移、回归与最终 claim gate

#### Route

1. legacy checkpoint/cache 对 legacy profile 保持可读；新 profile 必须要求新 contract
   identity，错误版本 fail closed。
2. Task067 既有 artifact 明确标为 v2-only；新 profile 如需 stance/training readiness，另开
   task 重新验证，不在本任务借用旧通过结论。
3. 运行直接测试、全量相关测试、Ruff 和 agent inspection；写
   `artifacts/r4_final_verification.json`。其中必须包含 `visual_inspection` 数组：每张最终
   图片的相对路径、SHA-256、viewer 解码结果、执行 agent 的结构观察、发现的问题和
   修正后复看结论。
4. 更新 `.agent/doc/locoformer.md`：区分 official paper facts、paper-faithful local profile、
   legacy v2 profile、仍未完成的 TXL/scale/real-robot claims。

#### Log

- 2026-08-20：`artifacts/r4_final_verification.json` 已生成，包含 R0/R3 machine-readable
  gate、20 条 `visual_inspection`、production pre/post SHA、cache/checkpoint fail-closed
  probe 和命令证据。相关 pytest `38 passed in 57.02s`，Ruff clean，agent inspection
  返回 0；legacy verify `128/128`。新旧 contract/cache key distinct，旧 checkpoint
  对新 contract 拒绝，新 contract 正常接受。
- 2026-08-20：另运行全量 `.venv/bin/python -m pytest -q`，结果为
  `7 failed, 813 passed, 35 warnings`。7 个失败全部属于既有 Task067 equilibrium 测试：
  它们读取的 artifact 仍标记 `procedural_whole_body_v1_footpad_static_stance`，而当前冻结
  legacy runtime 是 `procedural_whole_body_v2_footpad_actual_stance_feedforward`；物理与
  blueprint hash 一致，仅 contract version/hash 不一致。Task069 未修改 Task067 artifact、
  stance solver 或 controller chain，故该结果记录为范围外 baseline mismatch，不伪装成全量
  通过，也不在本任务内修复。剩余风险是尚未有独立只读 reviewer 的批准，因此本 task 保持
  in_progress，不写 passed。
- 2026-08-21：按用户要求修复上条全量 pytest 失败，不再把它保留为范围外失败。
  根因是 Task067 R4a.2/R4b/strict coverage 诊断测试把旧 artifact 的
  `procedural_whole_body_v1_footpad_static_stance` instance key 与当前 v2 runtime key
  做全字段相等比较，且一个 known-positive 测试仍把当前 v2 seed0 当作旧 R4a.2 source。
  修复为显式 historical artifact replay binding：blueprint/physical hash 必须逐字段匹配，
  source contract 只允许当前 v2 或精确白名单旧 v1 hash；未知 contract、错 hash、
  blueprint/physical 篡改均 fail closed。输出 payload 现在同时写
  `source_replay_contracts` 与 `runtime_contract`，不把旧 v1 artifact 静默解释成 v2
  通过证据，也不修改旧 artifact、generator default、stance solver 或 controller。
  验证结果：原失败 focused group `27 passed`；Task067 相关 `59 passed`；Task069
  指定 pytest `38 passed`；相关 Ruff clean；`inspect_agent` 返回 0；全量
  `.venv/bin/python -m pytest -q` 为 `825 passed, 35 warnings`。本 task 仍保持
  `execution_verified_pending_independent_readonly_review`，不写 passed。
- 2026-08-21：Task069 新增的两个 gate 测试加入后，指定 pytest 为 `40 passed`；Ruff
  和 inspect 仍通过。独立 reviewer 的 split-run full regression 汇总为
  `342 + 367 + 118 = 827 passed, 0 failed, 35 warnings`，证据见
  `artifacts/r4_full_pytest_split_summary.json`。R4 已按当前 verification-tool/source
  hash、R3 matrix、20 条显式 viewer observation 和当前 regression 状态重建；仍需新的
  独立只读 reviewer 复查本次修复后的 artifact，故不写 passed。
- 2026-08-21：针对独立 reviewer 的 REQUEST_CHANGES 完成证据闭环修复：用当前
  `source-contract` writer 重建 `artifacts/r0_source_contract.json`，其中
  `procedural_training_bodies` 已固定为 `author_paper:§2.1 Task Generation`；R4
  记录 R0 artifact/source SHA，并校验 artifact 的 verification-tool SHA 与当前 writer
  一致。`verify_legacy_baseline()` 现 fail-closed 比较 `compiled/status/error` 及
  `contract_version/contract_hash`，复核结果仍为 `128/128`。R3 matrix 重新生成后，
  rollout 接触事件逐条包含 `step/kind/geom1/geom2/distance`；逐图 observation manifest
  仍覆盖 4 套 montage 与 16 张近景，重新打开后 `20/20` 通过。四家族所有 required gate
  仍为 `32/32`，指定 pytest `40 passed`、Ruff clean、inspect exit 0；verifier 实现覆盖
  五个执行/contract 字段；状态继续保持
  `execution_verified_pending_independent_readonly_review`，等待新的独立只读复核。

#### Review

- 需要独立只读 review 检查 source contract、legacy hash stability、四家族矩阵、gallery、
  cache/checkpoint fail-closed 和 claim wording。
- 任一 gate 未满足时状态保持 `in_progress` 或 `blocked`，不得写 `passed`。

## Verification commands

执行 agent 可增加更窄测试，但不得删减最终这些验证：

```bash
.venv/bin/python -m pytest -q \
  tests/test_whole_body_contract.py \
  tests/test_whole_body_extended.py \
  tests/test_whole_body_usability_gate.py \
  tests/test_task069_*.py

.venv/bin/python -m ruff check \
  src/h200_locomotion_lab/robots \
  tests/test_whole_body_contract.py \
  tests/test_whole_body_extended.py \
  tests/test_whole_body_usability_gate.py \
  tests/test_task069_*.py

.venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
```

若 shell 无法匹配 `tests/test_task069_*.py`，说明任务测试尚未落地，最终验证不得通过。

## Log

- 2026-08-20：任务由用户要求创建，目的是让后续 agent 修正当前生成器与 LocoFormer
  官方论文形态包络的已知偏差。
- 创建时确认：官方论文/项目页公开四家族与 unseen evaluation robots，但未公开 exact
  generator code；因此本任务采用“paper-faithful envelope”而非“exact reproduction”措辞。
- 创建时确认：当前 `MorphologyFamily` 只有 `biped/quadruped`；默认
  `require_biped_ankle=True`、`mirror_biped_legs=True`，没有 wheel family。当前 Task067
  independent review 还显示 quadruped stance 走 legacy fallback；该 stance 问题不属于
  Task069 的实现范围，也不能被 morphology gallery 掩盖。
- 硬件假设：RTX 5060 Ti-first、本地 MuJoCo/headless；不启用 H200。

## Review

状态：**execution_verified_pending_independent_readonly_review**。

执行证据：

- `artifacts/r0_source_contract.json`
- `artifacts/r0_legacy_v2_baseline.json`
- `artifacts/r3_morphology_matrix.json`
- `artifacts/r4_final_verification.json`

执行 agent 已完成 R0–R4 的实现、回归和逐图复看；仍需独立只读 reviewer 复查 source
contract、legacy hash stability、四家族矩阵、gallery 以及 claim wording 后，才能将
任务状态改为 passed。无论 reviewer 结果如何，最终 claim 仍不包含官方源码复现、policy/
TXL 训练、sim2real 或 100k-robot scale。

最终通过必须同时满足：

1. R0–R4 各自的 Route / Log / Review 完整且有 machine-readable evidence；
2. 四家族结构、wheel dynamics、slot/mask、determinism、compile、finite smoke 全部通过；
3. v2 legacy baseline 全部 hash stable，新 contract/cache/checkpoint identity fail closed；
4. 四套群体 montage 与近景图可直接打开，且与 JSON/manifest 一一对应；执行 agent 已
   用 image viewer 逐图查看并留下结构观察，独立 reviewer 也至少复看四套 montage、每个
   family 一个近景及全部被标记异常后重渲染的图片；
5. 指定 pytest、Ruff、agent inspection 返回 0；
6. 独立 reviewer 明确批准“paper-faithful morphology envelope”这一窄 claim；
7. Review 明确保留：这不是官方源码复现，不证明 LocoFormer policy、长上下文训练、
   sim2real 或 100k-robot scale。
