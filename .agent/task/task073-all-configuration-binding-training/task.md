# Task073 — Remaining 16 configurations: physical binding, nominal walking, and randomization

状态：**planned / blocked_by_task072**。

## 进入条件

Task072 的 exact-bound G1 与 Go2 必须在同一冻结 lineage 上通过 numerical evaluation、视频和
verifier。Task072 未通过时，本任务只允许维护 contract/registry，不得启动其余构型训练，也不得
用旧 Go2 artifact 绕过入口 gate。

## 目标与固定分母

在 Task070 冻结的 18-case denominator 中，保留 Task072 已证明的 G1 与 Go2，并把物理绑定与训练
依次扩展到其余 16 个构型：

| 分组 | exact case ids | 数量 |
| --- | --- | ---: |
| Tier A remaining centers | `engineai_pm01`、`spot_base`、`deeprobotics_lite3` | 3 |
| Tier B wheel compositions | `unitree_g1_wheeled`、`engineai_pm01_wheeled`、`spot_base_wheeled`、`unitree_go2_wheeled`、`deeprobotics_lite3_wheeled` | 5 |
| Tier C humanoid candidates | `agibot_x1_serial`、`agibot_x2_ultra`、`engineai_t800`、`engineai_t800pro`、`limx_hu_d04`、`booster_t1_23`、`booster_t1_29`、`robotera_star1` | 8 |

每个 case 独立执行：`结构/actuator accounting -> 质量/COM/惯量/摩擦/电机 tuple 绑定 -> reset/contact/
action gate -> fixed-command nominal training -> numerical/video/verifier pass -> domain randomization`。
前一 gate 不通过时不得进入后一 gate，case 仍留在固定分母中并记录失败原因。

Task072 冻结的 G1/Go2 nominal descriptor、physics/motor binding、action/reward config 与 pass evidence
在 Task073 中是**只读基线**。002–006 的“绑定/扩展其余 case”只指 remaining 16；Task073 不得原地
改绑 G1/Go2。若共享实现或配置改动会改变 G1/Go2 nominal 行为，则 Task072 lineage 失效，必须在新
freeze 上重跑 G1 与 Go2，之后才能继续引用它们进入 18-case matrix。

## Subtasks

1. `001-freeze-denominator-and-execution-order.md`：冻结 18-case registry、Task072 baseline 与执行顺序。
2. `002-physical-binding-contract.md`：统一质量/COM/惯量、摩擦、transmission 与 coherent motor tuple
   的逐 case fail-closed 绑定规范。
3. `003-tier-a-remaining-centers.md`：绑定并 nominal train PM01、Spot、Lite3。
4. `004-tier-b-wheel-compositions.md`：绑定并 nominal train 五个 wheel composition。
5. `005-tier-c-variable-dof-and-provenance.md`：完成八个人形 candidate 的 variable-DoF adapter、
   transmission 与物理 provenance gate。
6. `006-tier-c-nominal-training.md`：对通过 005 的八个 candidate 逐 case 训练 nominal walking。
7. `007-post-nominal-domain-randomization.md`：只在对应 case nominal pass 后，分阶段加入物理与电机
   randomization 并重训/复验。
8. `008-final-matrix-and-review.md`：汇总 18-case exact denominator、证据、失败与最终 claim。

## Code implementation

### Workspace and owners

Task072 pass 后，从其 freeze commit 新建独立 worktree
`/home/admin1/workspace/run/locomotion_rl/task073-1`、branch
`codex/task073-all-configuration-training`；不得在 Task072 frozen worktree 上继续改。拟新增文件均在
本 task 中由 subtask 首次创建。新 worktree 建立后，先用 `apply_patch` 将
`/home/admin1/workspace/proj/locomotion_rl/.agent/task/task073-all-configuration-binding-training/` 的九份
Markdown 权威契约逐字同步到 branch 同路径并记录 SHA；不得在执行时另写一套 task contract：

| owner | responsibility |
| --- | --- |
| `.agent/task/task073-all-configuration-binding-training/task073_case_registry.py` | 18-case registry、输入 SHA、状态机 |
| `.agent/task/task073-all-configuration-binding-training/task073_physical_source_allowlist_v1.json` | remaining 16 的逐 field 本地 source locator、role 与 unknown；只允许显式文件，不做全盘搜索 |
| `.agent/task/task073-all-configuration-binding-training/task073_pipeline.py` | `bind`、`smoke`、`train`、`render`、`eval`、`randomize`、`verify` CLI 编排 |
| `src/h200_locomotion_lab/robots/physical_binding.py` | immutable physical binding schema、XML overlay、validator；不改 Task070 frozen XML |
| `src/h200_locomotion_lab/robots/topology_local_adapter.py` | Tier C topology-local action/observation schema |
| `src/h200_locomotion_lab/envs/whole_body_mujoco.py` | 复用 Task072 的 `model_xml/model_xml_sha256/stance_solution` 外部 XML 入口，并增加 optional actuation mode/topology-local adapter；45-slot 默认行为不变 |
| `src/h200_locomotion_lab/training/whole_body_ppo.py` | configurable obs/action dim；默认仍为 193/45 |
| `tests/test_task073_*.py` | registry、binding、wheel、variable schema、nominal、randomization、final verifier |

CLI 固定以 task-local 脚本路径执行，不使用省略号式模块命令。统一 artifact root 为
`.agent/task/task073-all-configuration-binding-training/artifacts/v1/`，case 目录固定为
`<case_id>/{binding,nominal,randomization,verify}/`。Task072 freeze 未通过时，CLI 只允许
`registry build/validate`；其他 subcommand 必须返回非零。

### Source registry root

结构输入固定来自 Task070 attempt010：
`.agent/task/task070-archetype-constrained-standable-morphology/artifacts/
preview_task070_v2_descriptor_driven_attempt010/`。registry 按上表 exact case id 绑定对应
`<case_id>_seed000` 目录；每个目录必须恰有一个 `*_anonymous_preview_manifest.json`，loader 只按该
manifest 的 `paths.xml`/`paths.descriptor` 取文件并复算 `xml_sha256`/`descriptor_sha256`，不得靠 glob
任选 XML，也不得从 attempt005/006/009 混取。G1/Go2 nominal evidence 只从 Task072 freeze manifest
导入并保持只读。

### Quantitative source allowlist

`task073_physical_source_allowlist_v1.json` 每个 remaining case、每个 field group 固定保存
`{group,path,locator,value,raw_sha256,role,units,quantitative_allowed,note}`；group 只能是
`link_inertial|contact|transmission|motor|controller|wheel_engineering`。文件存在但没有某字段时，该
group 的 `path/locator/value` 为 null 并写 blocked reason，禁止跨型号回填。允许扫描的根只有：

| case | local source roots |
| --- | --- |
| `engineai_pm01`, `engineai_t800`, `engineai_t800pro` | `.external/task070_reference_sources/engineai_robotics_native_sdk/assets/`；PM01 另审计 `.external/task070_reference_sources/engineai_amp/serial_pm01.urdf` |
| `spot_base` | `.external/task070_reference_sources/boston_dynamics_spot_sdk/spot_base_model.urdf` |
| `deeprobotics_lite3` | `.external/task070_reference_sources/deep_robotics_model/{Lite3.urdf,Lite3.xml}` |
| `agibot_x1_serial` | `.external/task070_reference_sources/agibot_x1_infer/src/module/{sim_module/model/mjcf/robot/xyber_x1/xyber_x1_serial.xml,dcu_driver_module/cfg/dcu_x1.yaml}` |
| `agibot_x2_ultra` | `.external/task070_reference_sources/agibot_x2_urdf/X2_URDF-v1.4.0/{X2-Ultra.urdf,X2-Ultra.xml}` |
| `limx_hu_d04` | `.external/task070_reference_sources/{limx_humanoid_description,limx_humanoid_rl_deploy_python/controllers/HU_D04_01}/` |
| `booster_t1_23`, `booster_t1_29` | `.external/task070_reference_sources/booster_assets/robots/T1/` |
| `robotera_star1` | `.external/task070_reference_sources/robotera_models/star1/urdf/l3_with_hand_fixedpin_xml.urdf` |
| five wheel cases | 对应 center 的 rows，加 Task070 manifest `actuation_stack` 与 `src/h200_locomotion_lab/robots/archetype_morphology.py::_compose_terminal_wheels`；role 必须是 `local_engineering_module`，不得标 named parity |

allowlist 自身和每个 raw source 的 SHA 都进入 registry；路径不在上表、SHA 漂移或 locator 解析失败即
阻断对应 case。此 task 不通过网络补 source；缺 quantitative motor/transmission 字段的 candidate
预置 unknown/blocked，留在 18-case denominator。

## Route

Task073 不采用“先把 16 个都塞进一个 sampler 再看平均值”的路线。每个 case 必须先有 source-
audited nominal center，并通过自己的 locomotion gate，之后才可以进入该 case 的 randomized
distribution。Biped、quadruped 和 wheel 使用 family-appropriate reward/contact/action semantics；
wheel 必须是 continuous velocity/torque-compatible control 与 active balance，不能复用普通 position
foot target。

物理配置只能来自已清理的 source/config 或明确标注的 anonymous engineering prior。未知字段保持
unknown 并阻断 quantitative training；不得把相似型号数值、统一 URDF placeholder 或独立乱采的
torque/speed/gain 拼成“真实 config”。Tier C 不能静默截断、合并或冒充 frozen 45-slot；必须有显式
versioned variable-DoF adapter 与 mask/schema。

Task073 新增 adapter/config 必须以 versioned extension 实现，并验证 Task072 G1/Go2 nominal base 的
identity 不变。007 对 G1/Go2 的 randomization 只能作为只读 nominal base 上的新 overlay；它不得回写
或替换 Task072 binding，并必须同时重跑 nominal non-regression 对照。

## 里程碑与 claim 边界

- `nominal_18_complete`：Task072 的 G1/Go2 加 Task073 的其余 16 个 case 均通过各自 fixed-command
  nominal walking gate；这不等于 domain-randomized robustness。
- `task073_passed`：18/18 均完成声明的 post-nominal domain-randomized training/evaluation，所有
  artifact 与 freeze lineage 可验证，且独立 review 无重大 finding。

本任务最多证明指定 anonymous asset 在指定 MuJoCo、reward、command 与 randomization contract 下
可训练并通过 gate。不得 claim named-robot parameter parity、真实电机系统辨识、sim2real、硬件部署、
LocoFormer 官方 reproduction 或 runtime motor-fault adaptation。

## Log

- 2026-08-27：任务及八个 subtask contract 建立；Task072 尚未通过，因此没有 Task073 case 被
  执行或标记 train-ready。

## Review

状态：**not_started**。

必须保留 18-case 固定分母和逐 case provenance。Task073 未满足进入条件前不得执行训练；未达到
`nominal_18_complete` 不得声称所有构型会走；未完成 18/18 post-nominal randomization 不得标记
Task073 passed。
