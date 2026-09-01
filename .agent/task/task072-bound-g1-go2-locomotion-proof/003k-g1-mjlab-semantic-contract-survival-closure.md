# 003k — G1 MJLab task-owned semantic contract and survival closure

状态：**ready_for_authorized_gpu / not_trained / not_passed**。

003k 是 003j pilot failure 之后的唯一活动修复任务。它不把失败归因于“训练轮数不够”，也不允许
继续在 `Unitree-G1-Flat` 上逐个删除已知 parent 项。目标是一次性关闭 Task072 的完整训练语义：
Task072 必须拥有 observations、actions、commands、events、rewards、terminations、curriculum、metrics、
task sensors 和 PPO 配置的显式 runtime 合同；parent 只能提供 MJLab 构造基础设施，不能继续静默决定
任务行为。

本文件只授权实现、CPU 验证和 clean source commit。**不授权 GPU capacity、one-update、pilot、eval、
proof、video 或 freeze。** CPU gate 通过后必须停止并报告，GPU 需用户另行明确授权。

## 1. Frozen diagnosis

### 1.1 003j 不是单纯“训练不够”

003j 已从随机初始化完成 `4096 x 24 x 21 = 2,064,384` transitions。训练合同、optimizer、finite、
runtime reward table 和 forward pilot checks 均通过，但存活出现确定性回退：

| checkpoint | median first fall | common-prefix mean vx | median +x |
| --- | ---: | ---: | ---: |
| `model_0` | `2.44 s` | `-0.3204 m/s` | `-0.6655 m` |
| `model_7` | `2.68 s` | `0.2865 m/s` | `0.7245 m` |
| `model_14` | `1.42 s` | `0.4085 m/s` | `0.5495 m` |
| `model_20` | `1.14 s` | `0.4628 m/s` | `0.5006 m` |

四个 checkpoint 的 256-env、20 s fixed-command eval 均为 `zero_fall_ratio=0`。`model_20` 的左右脚
contact ratio 约为 `0.985/0.990`，alternating touchdown transitions 为 `0`。策略学会了向前倒，
没有学会交替行走。

003j TensorBoard 还给出同一方向的训练期证据：

| update | mean episode reward | mean episode length |
| --- | ---: | ---: |
| `7` | `-2.756772` | `66.58` steps |
| `14` | `-2.211146` | `64.83` steps |
| `20` | `-1.607051` | `54.82` steps |

updates `7..20` 的 reward/episode-length Pearson correlation 为 `-0.957527`；updates `14..20` 为
`-0.984532`。reward 变好而 episode 变短，不是 walking improvement，而是 termination exploit。

### 1.2 Parent phase 与 reward phase 不一致

当前 source 中：

- reward 的 `REWARD_V3_PHASE.period` 为 `0.8 s`；
- reward `phase_gait`、`out_of_phase_double_support` 和 `clearance` 都消费该 `0.8 s` 周期；
- parent `src.tasks.velocity.mdp.observations.phase` 的 actor/critic config 仍保留
  `params={"period": 0.6, "command_name": "twist"}`；
- 当前 `build_task_cfg()` 替换了 `env_cfg.rewards`，却没有替换或校验 actor/critic 的 phase term。

因此 policy 看到的是 `0.6 s` clock，reward 评价的是 `0.8 s` clock。它们每 `2.4 s` 才重新对齐，
且在一个短 episode 内持续漂移。仅检查 reward 表无法发现这个跨 manager 错误。

### 1.3 v3 缺少足够的终止代价

003i 明确排除了 parent `is_terminated`，003j 的实际 23-term reward table 也没有 terminal term。
`nonfoot_contact` 不能替代 terminal penalty：003j 的该项 TensorBoard count 为 `21`，min/max/last
全部为 `0`；`fell_over` 会在 torso 达到 `70°` bad-orientation 时触发，可能早于任何 non-foot body
接触地面。

003j 最差的已观测 rolling per-step reward 近似为：

```text
r_worst = -4.436747 / 94.63 = -0.04689
```

对 constant per-step reward `r`、discount `gamma` 和 terminal cost `P`，把终止延后一步的收益差为：

```text
G(T+1) - G(T) = gamma^T * (r + (1 - gamma) * P)
```

当前 `gamma=0.99`，要让“多活一步”优于立即终止，需
`P > -r_worst / (1-gamma) = 4.689`。加 `25%` 固定安全边际后为 `5.861`，取可审计的
post-dt terminal cost `P=6.0`。control `dt=0.02`，因此 v4 使用 raw `-1`、positive weight `300.0`，
实际 terminal-step contribution 为 `-1 * 300 * 0.02 = -6.0`。

这不是恢复 parent 的不透明 `is_terminated=-200`。它是只针对 Task072 唯一 non-timeout
`fell_over` 的显式、dt-aware、数据推导的有界成本，且必须分别验证 terminated 与 timeout。

### 1.4 Action distribution 是次级证据，不是本轮调参变量

003j 的 Gaussian std 从 `1.0` 增至约 `1.0478`，raw scalar clip fraction 约从 `0.3196` 增至
`0.356`。这些值继续进入 per-update telemetry，但 003k 不改变 action mapping、clip、network、
entropy、learning rate 或 PPO 超参数。先关闭已证实的 phase/termination/parent semantic 漏洞，避免把
多个变量混成一次试验。

## 2. Authority, source baseline and ownership

执行 frame 固定为：

```text
/home/admin1/workspace/run/locomotion_rl/task071-1
```

文档建立时 baseline HEAD 为
`374389b623a20794b85d5d9c0adcdc8d329cb837`；003j implementation commits
`52d6924` 与 `f799115` 继续作为代码历史。执行者开始前必须只读记录 HEAD、branch 与
`git status --short`，保留既有 untracked `artifacts/`，不得删除、覆盖或提交 binary artifacts。

003k 的实现 ownership 仅限：

- `.agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py`；
- `tests/test_task072_locomotion_proof.py`；
- 本文件和 Task072 `task.md` 的 Route / Log / Review 状态更新。

只有现有结构确实无法完成 runtime semantic extraction 时，才允许在同一 task 目录增加一个小型
JSON-native helper；不得新建第二套 runner、第二套 reward evaluator 或通用配置框架。不得修改
external checkout、MJLab package、asset XML、contact profile、stance、shared PPO library 或其他 task。

实现完成后以一个 coherent source commit 交付。不得 amend 或重写 003j 历史 commits。

## 3. Rejected inputs and new lineage

以下全部只保留为 `rejected_diagnostic`，不得 resume、warm-start、复制 optimizer state、重命名为 v4
或作为 pass evidence：

- 003h 的 checkpoint、pilot、eval；
- 003i `model_0.pt` 及 one-update artifact；
- 003j `model_0/7/14/20.pt`、capacity、one-update、pilot、eval 和 pilot gate；
- 所有 003b–003g 旧 checkpoint。

003k 新常量必须为：

```text
TASK072_ACTIVE_SUBTASK = "003k"
LINEAGE_ID = "mjlab_g1_7capsule_task_v4_semantic_closed"
REWARD_CONTRACT_VERSION = "task072_mjlab_biped_phase_contact_survival_v4"
TASK072_GAIT_PERIOD_S = 0.8
DEFAULT_OUTPUT_ROOT =
  ROOT / "artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v4_semantic_closed"
```

asset/profile/stance 仍绑定已验证的 `mjlab_g1_7capsule_task_v2` 和 single-ground runtime XML；只创建
新的 task/runtime/reward lineage，不修改物理资产。新 root 不得覆盖或符号链接到 v3 artifacts。
当前 `.agent/task/.../artifacts/` 与 repo-root `artifacts/` 同时存在历史 v3 文件；003k 明确只写上面的
repo-root `ROOT/artifacts/...v4...`。task-dir artifact tree 只读，不再产生第三份或双写。

## 4. Parent trust boundary: replace complete semantic tables

`load_env_cfg("Unitree-G1-Flat")` 可继续用于取得 MJLab dataclass 类型、flat-plane scene scaffold、sim
和 viewer 基础设施；`load_rl_cfg()`/`load_runner_cls()` 可继续取得 RSL-RL 类型和 runner class。
但 `build_task_cfg()` 不得再通过“先继承全部，再 pop 几个已知名字”定义 Task072。

在 register 前，以下表必须由 Task072 literal builder **整表替换**：

| semantic owner | v4 要求 |
| --- | --- |
| `scene.entities["robot"]` | 使用现有 v2 runtime XML、stance init、collision 与 actuator builder 完整替换 |
| `scene.sensors` | 只含本文件第 6 节的两个 task sensors |
| `observations` | actor/critic 都由第 5 节的 ordered literal table 重建 |
| `actions` | 只含现有 signed `joint_pos` action contract |
| `commands` | 只含第 7 节 fixed `twist` |
| `events` | 只含第 7 节两个 deterministic reset events |
| `rewards` | 只含第 9 节 ordered 24-term v4 table |
| `terminations` | 只含 `time_out` 与 `fell_over` |
| `curriculum` | exact empty dict `{}` |
| `metrics` | 只含显式 `mean_action_acc` |
| agent/PPO | 每个第 8 节字段显式赋值或 fail-closed assert |

禁止以 `pop("push_robot")`、`pop("foot_friction")` 等作为 completeness proof。整表替换后，parent
未来新增 term 会自然消失。任何 manager 出现未列出的 active name、callable、weight、params 或顺序都必须
在 CPU verifier 失败。

允许由 parent 提供但必须进入 resolved semantic payload 的 infrastructure 仅有：

- sim/MuJoCo solver fields 与 `decimation`；
- flat plane terrain scaffold、scene extent 与 viewer；
- MJLab config dataclass/manager/runner classes。

其中 sim 必须保持 `timestep=0.005`、`decimation=4`、`step_dt=0.02`、iterations `10`、
ls_iterations `20`、ccd_iterations `50`、contact_sensor_maxmatch `64`、effective `nconmax>=128`、
`njmax>=600`。resolved values 必须写入 payload；不满足时不得自行修补成另一套 physics。

## 5. Exact observation contract

所有 term 的 `noise=None`、term `history_length=0`、`scale=None`，除表内 params 外无隐藏 params。
actor/critic group 均为 `concatenate_terms=True`、`enable_corruption=False`、`history_length=1`。critic
必须用独立 term config 构造，不能通过共享可变 actor term object 留下别名。

### 5.1 Actor ordered table

| order | name | exact callable `module.qualname` | exact params |
| ---: | --- | --- | --- |
| 0 | `base_ang_vel` | `mjlab.envs.mdp.observations.builtin_sensor` | `{"sensor_name":"robot/imu_ang_vel"}` |
| 1 | `projected_gravity` | `mjlab.envs.mdp.observations.projected_gravity` | `{}` |
| 2 | `command` | `mjlab.envs.mdp.observations.generated_commands` | `{"command_name":"twist"}` |
| 3 | `phase` | `src.tasks.velocity.mdp.observations.phase` | `{"period":0.8,"command_name":"twist"}` |
| 4 | `joint_pos` | `mjlab.envs.mdp.observations.joint_pos_rel` | `{}` |
| 5 | `joint_vel` | `mjlab.envs.mdp.observations.joint_vel_rel` | `{}` |
| 6 | `actions` | `mjlab.envs.mdp.observations.last_action` | `{}` |

### 5.2 Critic ordered table

Critic order `0..6` 与 actor 完全相同，随后精确追加：

| order | name | exact callable `module.qualname` | exact params |
| ---: | --- | --- | --- |
| 7 | `base_lin_vel` | `mjlab.envs.mdp.observations.builtin_sensor` | `{"sensor_name":"robot/imu_lin_vel"}` |
| 8 | `foot_height` | `src.tasks.velocity.mdp.observations.foot_height` | `asset_cfg=SceneEntityCfg("robot", site_names=FOOT_SITES)` |
| 9 | `foot_air_time` | `src.tasks.velocity.mdp.observations.foot_air_time` | `{"sensor_name":"feet_ground_contact"}` |
| 10 | `foot_contact` | `src.tasks.velocity.mdp.observations.foot_contact` | `{"sensor_name":"feet_ground_contact"}` |
| 11 | `foot_contact_forces` | `src.tasks.velocity.mdp.observations.foot_contact_forces` | `{"sensor_name":"feet_ground_contact"}` |

Resolved `asset_cfg` 必须 canonicalize 为 entity name、site names 和实际 site IDs；不能以 Python object
repr 进入 JSON。

### 5.3 Cross-manager phase invariant

`TASK072_GAIT_PERIOD_S=0.8` 是唯一 gait period source。以下五处必须引用同一常量或在 builder 中
fail-closed 比较，禁止分别写两个 magic number：

- actor `phase.params.period`；
- critic `phase.params.period`；
- reward `phase_gait.params.period`；
- reward `out_of_phase_double_support.params.period`；
- reward `clearance.params.period`。

reward offsets 固定 `[0.0, 0.5]`，stance fraction 固定 `0.55`。observation phase 使用同一
`episode_length_buf * step_dt` global clock，moving command 下 reset phase 为 sin/cos `[0,1]`；第一个
control step为 `0.02/0.8=0.025` cycle。stand command 的 phase-zeroing 行为保留，但 Task072 fixed
command `vx=0.5` 不触发该 mask。

CPU verifier 必须从 actual registered env 的 actor/critic ObservationManager active table读取 params，
再与 actual RewardManager 三个 phase terms 做跨表断言。检查版本字符串或只检查 builder constant 不算通过。

## 6. Exact task sensor contract

`scene.sensors` 只允许以下两个 config；parent `terrain_scan` 和 `self_collision` 都不得存在。XML 内置
robot IMU 等 sensor 属于 asset entity，不计入这个 task sensor tuple。

### 6.1 `feet_ground_contact`

- class: `mjlab.sensor.ContactSensorCfg`；
- primary: `ContactMatch(mode="subtree", entity="robot",
  pattern="^(anon_limb0_ankle_roll_link|anon_limb1_ankle_roll_link)$")`；
- secondary: `ContactMatch(mode="body", pattern="terrain")`；
- fields: `("found", "force")`；
- reduce: `"netforce"`；
- `num_slots=1`、`track_air_time=True`、`history_length=0`、`secondary_policy="first"`；
- runtime 必须 resolve 为左右各 7 个 foot capsules against the single scene terrain。

### 6.2 `nonfoot_ground_contact`

- primary: `ContactMatch(mode="body", entity="robot", pattern=<anchored exact alternation>)`；
- alternation 必须来自 runtime XML 的全部 named bodies 减去 exact `FOOT_BODIES`，保存 resolved body
  names/IDs；禁止手写一个可能漏 body 的子集；
- secondary: `ContactMatch(mode="body", pattern="terrain")`；
- fields: `("found",)`；
- reduce: `"none"`；
- `num_slots=1`、`track_air_time=False`、`history_length=0`、`secondary_policy="first"`。

`self_collision` 当前没有 v4 reward/termination/observation consumer，因此删除，不能继续作为隐藏
parent sensor。003k 不新增 self-collision reward；若以后需要，必须单独 versioned task。

## 7. Commands, events, terminations, metrics and episode

### 7.1 Fixed command

`commands` 只含 `twist`，exact resolved config 为：

```json
{
  "entity_name": "robot",
  "heading_command": false,
  "heading_control_stiffness": 0.5,
  "rel_standing_envs": 0.0,
  "rel_heading_envs": 0.0,
  "init_velocity_prob": 0.0,
  "resampling_time_range": [1000000000.0, 1000000000.0],
  "ranges": {
    "lin_vel_x": [0.5, 0.5],
    "lin_vel_y": [0.0, 0.0],
    "ang_vel_z": [0.0, 0.0],
    "heading": null
  },
  "debug_vis": false
}
```

train 和 eval 都必须 fixed-command；eval 在每次 auto-reset 后继续执行现有 `_force_fixed_command` 并
断言实际 command tensor 全为 `[0.5,0,0]`，不能只依靠超长 resampling interval。

### 7.2 Events

`events` 只允许以下 ordered names：

1. `reset_base`: `mjlab.envs.mdp.events.reset_root_state_uniform`，mode `reset`，pose x/y/z/yaw
   全为 `[0,0]`，`velocity_range={}`；
2. `reset_robot_joints`: `mjlab.envs.mdp.events.reset_joints_by_offset`，mode `reset`，asset
   `robot`, joints `(".*",)`，position/velocity range 都为 `[0,0]`。

不得存在 startup/interval DR、push、friction、encoder bias、COM、terrain randomization 或隐式
event。`curriculum` 必须 exact `{}`。

### 7.3 Terminations

`terminations` 只允许：

| order | name | callable | `time_out` | params |
| ---: | --- | --- | ---: | --- |
| 0 | `time_out` | `mjlab.envs.mdp.terminations.time_out` | `true` | `{}` |
| 1 | `fell_over` | `mjlab.envs.mdp.terminations.bad_orientation` | `false` | `{"limit_angle":1.2217304763960306}` |

`1.2217304763960306 rad` 即 `70°`。003k 不改 termination angle，也不增加 contact termination。
`episode_length_s=10000.0` 保持 train/eval 相同；formal eval 的 20 s horizon 由 evaluator 控制，不能通过
改 env time-out 假装存活。

### 7.4 Metrics

`metrics` 只含 `mean_action_acc`，callable 为 `mjlab.envs.mdp.metrics.mean_action_acc`，params `{}`。
它仅用于 telemetry，不进入 reward 或 pass gate。

## 8. Exact action and PPO contract

### 8.1 Action

现有 `task072_mjlab_signed_headroom_v1` 保持不变：

- exact 29/29 semantic-to-anonymous joint order；
- stance actuator equilibrium 为 offset；
- existing per-joint negative/positive motor-headroom scales；
- policy raw action 经 wrapper clip 到 `[-1,1]` 后进入 signed action term；
- `clip_actions=1.0`；
- one-update/pilot 对 raw clip 只做完整、finite、integer-backed、可重算 telemetry，不恢复
  `.10/.50/.25` 错误 absolute gates。

semantic payload 必须保存实际 target order、offset、negative/positive scale、policy domain、29/29
mapping 和既有 action contract payload SHA。003k 不重新推导或调整 action scale。

### 8.2 Actor, critic and PPO

`load_rl_cfg()` 后必须把下列字段逐一显式赋值或逐一 assert；不能只保存 parent version：

| component | exact v4 value |
| --- | --- |
| actor hidden dims / activation | `[512,256,128]` / `elu` |
| actor obs normalization | `true` |
| actor distribution | `GaussianDistribution`, `init_std=1.0`, `std_type="scalar"` |
| critic hidden dims / activation | `[512,256,128]` / `elu` |
| critic obs normalization | `true` |
| value loss coefficient / clipped value | `1.0` / `true` |
| PPO clip / entropy | `0.2` / `0.01` |
| learning epochs / mini-batches | `5` / `4` |
| learning rate / schedule | `0.001` / `adaptive` |
| gamma / lambda / desired KL | `0.99` / `0.95` / `0.01` |
| max grad norm | `1.0` |
| optimizer runtime class | `Adam`，记录 actual fully-qualified class |
| rollout steps | exact `24` |
| resume / upload | `false` / `false` |
| logger | `tensorboard` |

`seed`、`num_envs`、`max_iterations`、`save_interval` 和 output task ID 是 stage controls，按第 11 节
允许变化。其他 agent/PPO 字段不允许在 capacity、one-update、pilot 或 eval 间变化。

## 9. Reward v4: retain gait and add bounded fall cost

### 9.1 Ordered active table

v4 保留 v3 的全部 23 个实际 callable、formula、params 和 positive weight，只在末尾追加一个 term。
禁止借本轮改变 centered tracking、pose、gait、clearance 或 regularization weights。

| order | term | exact task-local callable | weight |
| ---: | --- | --- | ---: |
| 0 | `track_xy_centered` | `task072_reward_track_xy_centered` | `2.0` |
| 1 | `track_yaw` | `task072_reward_track_yaw` | `0.50` |
| 2 | `upright` | `task072_reward_upright` | `0.25` |
| 3 | `tilt` | `task072_reward_tilt` | `5.0` |
| 4 | `height` | `task072_reward_height` | `0.25` |
| 5 | `stand_support` | `task072_reward_stand_support` | `0.30` |
| 6 | `phase_gait` | `task072_reward_phase_gait` | `0.50` |
| 7 | `out_of_phase_double_support` | `task072_reward_out_of_phase_double_support` | `0.35` |
| 8 | `clearance` | `task072_reward_clearance` | `0.50` |
| 9 | `touchdown_airtime` | `task072_reward_touchdown_airtime` | `0.10` |
| 10 | `soft_landing` | `task072_reward_soft_landing` | `0.10` |
| 11 | `foot_slip` | `task072_reward_foot_slip` | `0.20` |
| 12 | `nonfoot_contact` | `task072_reward_nonfoot_contact` | `0.20` |
| 13 | `pose_hip` | `task072_reward_pose_hip` | `0.20` |
| 14 | `pose_knee` | `task072_reward_pose_knee` | `0.30` |
| 15 | `pose_ankle` | `task072_reward_pose_ankle` | `0.20` |
| 16 | `pose_waist` | `task072_reward_pose_waist` | `0.10` |
| 17 | `pose_arm_wrist` | `task072_reward_pose_arm_wrist` | `0.05` |
| 18 | `joint_velocity` | `task072_reward_joint_velocity` | `0.02` |
| 19 | `joint_limit` | `task072_reward_joint_limit` | `0.05` |
| 20 | `action_magnitude` | `task072_reward_action_magnitude` | `0.01` |
| 21 | `action_rate` | `task072_reward_action_rate` | `0.01` |
| 22 | `base_angvel_xy` | `task072_reward_base_angvel_xy` | `0.02` |
| 23 | `fall_terminated` | `task072_reward_fall_terminated` | `300.0` |

前 23 项的 canonical params 继续由当前 v3 independent expected table 验证，包括 exact torso/body/site/
joint IDs、stance SHA、29-joint groups、phase、contact sensors 和 action name。所有 callable ID 必须是
实际 `__module__ + __qualname__`；不能从 active table 自己生成 expected table 后与自己比较。

### 9.2 Exact terminal callable

`task072_reward_fall_terminated(env)` 无 params，返回 shape `[num_envs]` float tensor：

```python
-env.termination_manager.terminated.to(dtype=torch.float32)
```

`terminated` 必须是 MJLab 已排除 timeout 的 non-timeout aggregate。由于第 7.3 节只允许一个
non-timeout term，它与 actual `fell_over` 等价。若 runtime API 的 `terminated` 包含 timeout，禁止猜测
兼容；CPU gate 直接失败并在本 task Review 记录 blocker。

RewardManager 仍统一执行 control-dt scaling；不得在 callable 内再乘 `dt`。breakdown 必须同时给出
raw、`raw * weight` pre-dt、`raw * weight * step_dt` contribution 和 live total，避免混用。

### 9.3 Required probes

扩展现有 reward fixture adapter，不新建第二套 evaluator。除 v3 三个 nonterminal oracle 外，加入：

| fixture | `terminated` | `time_out` | fall raw | pre-dt | dt contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | `0` | `0` | `0` | `0` | `0` |
| fell-over only | `1` | `0` | `-1` | `-300` | `-6` |
| timeout only | `0` | `1` | `0` | `0` | `0` |

`terminated && time_out` overlap 为 invalid fixture，必须 fail closed。现有 nonterminal v3 oracle 保持
`static_both=-0.5542411176571156`、`ideal_phase_matched=1.7`、
`persistent_left_only=-0.26012009890715004`、ideal-static margin
`2.2542411176571155`；新增 terminal term在这些 fixtures 均为零，不能改变它们。

CPU actual-runtime verifier 必须读取 actual RewardManager 的 ordered 24 terms、callables、weights、params
和 live dt relation；fixture 只负责受控 terminal flags/formula，不得伪造 actual manager 配置。

## 10. Canonical full semantic payload

用一个 helper 替换当前只记录 command/event names/reward names 的浅 payload：

```python
task072_runtime_semantic_payload(
    env_cfg,
    agent_cfg,
    registration,
    *,
    render_mode: str | None,
) -> dict[str, JSONNative]
```

返回值本身不含自引用 SHA。调用方对
`json.dumps(payload, sort_keys=True, separators=(",", ":"))` 取 SHA-256，并把 payload path、raw SHA、
payload SHA 写入 verifier/manifests。helper 只能从传入的 actual resolved cfg/registration 抽取，禁止重算
另一套“理想 config”。

payload 至少完整包含：

- schema、active subtask、lineage、source commit、external MJLab commit、runtime XML/profile/stance SHA；
- sim、decimation、step dt、episode、terrain、single-ground audit 和 viewer；
- robot init/collision/actuation resolution；
- 两个 sensor 的完整 config 与 resolved bodies/sites/geoms/IDs；
- actor/critic observation group 和 ordered term name/callable/params/noise/scale/history；
- action ordered term、29 target names、offset、signed scales、domain 和 payload SHA；
- command full config；
- events、rewards、terminations、curriculum、metrics 的 ordered active tables；
- actor、critic、distribution、PPO、optimizer、runner fields；
- registration/task ID、stage controls 和 fixed-command assertion。

Python callables canonicalize 为 `module.qualname`；tuples、SceneEntityCfg、dataclass、Path、numpy/torch scalar
都先转为稳定 JSON-native 值。不得保存 object repr、内存地址或 unordered set。

从 config 构造 actual 1-env `ManagerBasedRlEnv` 后，再生成一份 runtime-manager table。config table 与
actual manager table必须 name/order/callable/weight/params 等价；runtime-resolved IDs 可作为额外字段，不能
反过来掩盖 config mismatch。

Capacity、one-update、pilot、每个 eval 与 aggregate gate 都必须引用同一 semantic payload SHA。允许的
stage diff 只限：

```text
env.scene.num_envs
env.seed
env.render_mode
agent.seed
agent.max_iterations
agent.save_interval
registration.task_id
registration.num_envs
registration.max_iterations
registration.transitions_per_update
```

capacity 的 `num_envs` 可为 `2048/4096/6144`；train seed 固定 `720301`，eval seed 固定 `720400`。
`episode_length_s` 不再列入 train/eval allowlist，必须始终 `10000.0`。若确有 MJLab registration-only
字段无法一致，先把实际 diff 写入 Review 并停止，不得执行者自行扩大 allowlist。

## 11. Implementation route

### A. Close builders

1. 将 current constants/version/output root 切换到 003k/v4；保留 v3 artifacts 只读。
2. 把 observations、sensors、actions、commands、events、rewards、terminations、curriculum、metrics
   改为 task-owned whole-table builders。
3. 显式赋值/assert 第 8 节 agent/PPO fields。
4. 将五个 phase consumer 绑定同一 `TASK072_GAIT_PERIOD_S`。
5. 加入第 24 个 `fall_terminated`，保持前 23 项完全不变。

### B. Close evidence

1. 扩展现有 active-table/canonical payload helper，覆盖所有 manager 与 agent semantic fields。
2. 扩展现有 reward fixture probe，验证 terminal/timeout/dt；不得复制 reward formulas。
3. CPU actual-runtime verifier 验证 config table、actual manager table、phase cross-invariant、24-term
   breakdown、fixed command、single-ground、finite obs/reward 和 exact termination separation。
4. train wrapper 继续每 update 记录 raw action clip；另记录 policy std、mean episode reward、mean episode
   length、completed `fell_over` count 和 completed timeout count。

### C. Close evaluator and pilot gate

现有 eval loop 顺序保持唯一：policy obs -> policy action -> `RslRlVecEnvWrapper.step(action)` -> 立即 clone
`outer.reset_terminated` 与 `outer.reset_time_outs` -> 再读取 robot/contact state。继续验证
`done == terminated | time_out`、两者不 overlap，并只累计每个 env 首次 reset 前的 common prefix。

在现有 per-env trace 上增加，不建第二个 rollout：

- left/right touchdown：`contact[t-1]==false && contact[t]==true`；
- simultaneous touchdown 单独计数，不写入 alternating side sequence；
- alternating sequence 只收录“该 step 恰有一侧 touchdown”的 chronological side labels；相邻 labels
  不同计一次 alternation；
- left/right single support：每 step exact `left && !right` / `right && !left`；
- 所有 gait 统计均只使用该 env 的 common prefix；reset 后数据不得拼接进原 episode。

003k aggregate pilot gate 保留 003j 全部 finite、schema、forward 和 survival gates，并增加：

| gate | exact requirement |
| --- | --- |
| `model20` survival | median first fall `>=2.5 s` |
| learned survival | `model20 >= model0 + 0.5 s` |
| mid-run non-regression | `model14 >= model7 - 0.25 s` |
| late non-regression A | `model20 >= model7 - 0.10 s` |
| late non-regression B | `model20 >= model14 - 0.25 s` |
| forward velocity | `model20` common-prefix mean vx `>=0.05 m/s` |
| forward displacement | `model20` median +x `>=0.10 m` |
| left/right touchdowns | each side: fraction of env prefixes with `>=1` touchdown `>=0.95` |
| left/right single support | each side: fraction of env prefixes with `>=1` step `>=0.95` |
| alternation | median per-env alternating touchdown transitions `>=2` |

训练期 anti-exploit telemetry 还必须计算：

```text
reward_improvement_7_to_20 = mean_reward[20] - mean_reward[7]
length_ratio_20_to_7 = mean_episode_length[20] / mean_episode_length[7]
```

若 `reward_improvement_7_to_20 >= 0.25` 且 `length_ratio_20_to_7 < 0.90`，pilot gate 失败并标记
`reward_up_survival_down`。updates `7..20` reward/length Pearson correlation 记录为 evidence；不单独以 noisy
correlation 设 gate。上述 eval survival non-regression gates 始终 blocking，因此 003j 的 model7→20
明显回退不能再次晋级。

## 12. Minimal tests and CPU validation

本任务禁止用大量 fake tests 替代 runtime 验证。测试预算固定如下：

- 修改现有 reward active-table test：term count `23 -> 24`，检查 exact new callable/weight/order 和三个
  terminal fixtures；
- 修改现有 canonical config/eval/pilot tests 以消费 full semantic payload 和新增 gait/survival fields；
- **最多新增一个** focused test，名称固定为
  `test_task072_v4_cross_manager_contract_closes_phase_and_fall`。它调用 actual task builder，断言
  actor/critic/reward period 一致、exact termination table、terminal dt contribution 和 semantic payload；
- 不新增 broad fake-env suite、不复制 24 个 reward formula、不写 version-string-only test、不运行全量
  repository pytest。

CPU validation 命令范围固定为：

```bash
/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python -m py_compile \
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py
PYTHONPATH=$PWD/src /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python -m pytest -q \
  tests/test_task072_locomotion_proof.py \
  -k 'reward or v4_cross_manager or common_prefix or pilot_gate or semantic_contract'
git diff --check
```

执行顺序固定：先在 dirty implementation 上跑上述快速检查；通过后只提交 runner、focused test 和冻结的
contract docs，得到 clean source commit；再从该 clean HEAD 重跑同一检查和下面的 formal CPU verifier。
semantic payload 的 `source_commit` 必须等于这个 clean HEAD，不能指向修改前 baseline。ignored/untracked
artifacts 不进入 commit。formal verifier 生成后，可用单独 result-doc commit 更新 Log/Review；verifier
记录的是执行时 `contract_doc_input_raw_sha256`，result-doc 更新不反向改写已执行合同。

然后运行扩展后的现有 CPU reward/eval contract verifier，输出唯一新 artifact：

```text
003k_v4_semantic_reward_eval_contract_verifier.json
```

verifier 必须 `passed=true`，记录 source/runner/test/task doc/raw+payload SHA、actual active tables、phase
cross-invariant、terminal fixtures、sim/single-ground/fixed-command evidence。CPU gate 未过就停止；不得为了
测试通过放宽 table、allowlist、terminal cost 或 gait gates。

CPU clean commit 后更新本文件状态为
`ready_for_authorized_gpu / not_trained / not_passed`，列出 commit 与 SHA，然后停止。

## 13. Separately authorized GPU route

只有用户在 CPU clean commit 后明确授权，才按以下唯一顺序运行；所有 CUDA 进程必须处于
`/home/admin1/workspace/run/.gpu.lock` ancestor 下：

1. v4 fresh capacity smoke：`2048x24`、required `4096x24`、optional `6144x24`；不得复用 v3 capacity；
2. exact fresh one-update：`4096x24x1`，seed `720301`；
3. one-update 全合同 passed 后，fresh pilot：`4096x24x21`，seed `720301`，保存
   `model_0/7/14/20`；
4. training acceptance passed 后，对四个 checkpoint 分别运行 256-env、20 s、seed `720400`
   fixed-command eval；
5. 运行 aggregate pilot gate，并更新 Log/Review。

每一阶段都从随机初始化或该 stage 自身 manifest 明确声明的 checkpoint 开始；pilot 不消费 one-update
checkpoint。stage failure 立即停止，保留 artifact，不继续下一个阶段。

即使 003k pilot 全部通过，也只能标记
`pilot_passed / trained / proof_not_run` 并新建 `003l` versioned proof task。不得在 003k 直接启动
650-update proof、video、004 freeze、Task073 或 Task074。

## 14. Acceptance checklist

### Implementation/CPU acceptance

- [x] complete task-owned manager tables replace partial parent cleanup；
- [x] actor、critic、三个 reward phase consumers 使用 exact `0.8 s` shared clock；
- [x] v3 全部 gait terms 保留，active reward exact 24 terms；
- [x] `fall_terminated` 只惩罚 non-timeout termination，dt contribution exact `-6.0`；
- [x] actual observation/reward/termination manager tables 与 independent expected contract 一致；
- [x] full JSON-native semantic payload/SHA 覆盖全部 task behavior；
- [x] train/eval semantic diff 只含 frozen allowlist；
- [x] existing evaluator 保持 terminated/timeout/common-prefix 正确性并增加 per-env gait metrics；
- [x] action/PPO/asset/stance/physics 未被顺带调参；
- [x] focused tests、CPU verifier、`py_compile`、`git diff --check` passed；
- [x] clean source commit 已记录，GPU 未启动。

### Later GPU acceptance

- [ ] fresh v4 capacity 和 one-update passed；
- [ ] fresh 21-update pilot training contract passed；
- [ ] model `0/7/14/20` exact eval set 完整且 semantic payload SHA 一致；
- [ ] no `reward_up_survival_down`；
- [ ] survival、forward、touchdown、双侧 single-support、alternation gates 全部 passed；
- [ ] 未消费任何 v3 checkpoint；
- [ ] pass 后只新建 003l，未越级 proof/freeze/Task073/Task074。

## Log

- 2026-09-01：根据 003j fresh pilot/eval 与 current source audit 建立 003k。冻结根因是 parent
  observation phase `0.6 s` 与 reward phase `0.8 s` 分叉，以及无 terminal cost 导致的
  reward-up/survival-down exploit。本次只建立详细实现合同，未修改 source、未运行 CPU verifier 或
  GPU。
- 2026-09-01：实现 bounded CPU slice：切换 003k/v4 lineage/output root，统一 reward/phase period，
  保留 v3 reward 并追加 `fall_terminated` 的 raw `-1`、weight `300.0`，扩展 fixture oracle 覆盖
  normal/fell-over/timeout。未运行 GPU、training、eval、proof、video 或 freeze。
- 2026-09-01：完成 003k CPU closure source commit
  `d169a90ac59b34a8282bca51108c199e622531ab`。从 clean source HEAD 重跑：
  `py_compile` passed；targeted pytest
  `tests/test_task072_locomotion_proof.py -k 'reward or v4_cross_manager or common_prefix or pilot_gate or semantic_contract'`
  为 `7 passed, 59 deselected`；`git diff --check` passed。随后运行 CPU-only
  `verify-reward-eval-contract`，产物
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v4_semantic_closed/003k_v4_semantic_reward_eval_contract_verifier.json`
  passed，raw SHA-256
  `fa28bab4c8800d49c9e91ead8f38fa39092187f6b993dd2a301190aac37ffb7f`，其
  `source_commit` 为同一 clean source commit。未启动 GPU、training、eval、proof、video 或 freeze。

## Review

状态：**ready_for_authorized_gpu / not_trained / not_passed**。

003k implementation/CPU gate 已关闭：Task072 现在拥有 task-owned observations/actions/commands/events/
rewards/terminations/curriculum/metrics/sensors/PPO runtime contract，actor/critic/reward phase 统一为
`TASK072_GAIT_PERIOD_S=0.8`，RewardManager actual table 为 ordered 24 terms，`fall_terminated` 对
non-timeout termination 的 dt contribution 为 `-6.0`，train/eval semantic diff 仅含冻结 allowlist。

003k 未训练且仍 not_passed；GPU capacity、one-update、pilot、eval、proof、video、freeze 均未授权也未运行。
下一步只能在用户另行明确授权后，从 fresh v4 capacity 开始。
