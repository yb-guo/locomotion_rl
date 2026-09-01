# 003i — G1 MJLab reward-v3、eval 语义与 survival repair

状态：**one_update_failed / smoke_trained / not_passed**。

003i 是 003h 后验审计后的唯一活动 subtask。它只关闭三个已经定位的合同错误：

1. `task072_mjlab_biped_phase_contact_v3` 只被写成 version 字符串，真实
   `RewardManager` 仍保留 parent `foot_gait`；
2. 20 s eval 把 `reset_terminated` 与 `reset_time_outs` 合并为 fall，并在无 survivor 时把
   vx/x 写成 0 或 sentinel；
3. 训练只暴露瞬时 clip fraction，没有 update 级聚合，也没有阻止 model7 到 model20 的
   survival 回退。

本 subtask 必须先实现、测试并 runtime 验证，再另行请求 GPU。它只重跑 fresh-init pilot；即使
pilot continuation gate 通过，也不等于 walking proof，不能直接进入 004。通过后必须另建后续
versioned proof subtask，并再次取得正式训练授权。

## Route

`003h rejected_diagnostic -> 003i CPU implementation/tests -> user-authorized capacity/one-update ->
fresh pilot -> pilot continuation gate -> future versioned proof subtask -> 004`。

003i pending、CPU gate 失败、未获 GPU 授权、one-update gate 失败或 pilot gate 失败时，
004、Task073、Task074 全部保持 blocked。2026-09-01 GPU 授权已执行到 one-update；one-update
clip gate 失败后已停止，未运行 fresh pilot、eval、pilot gate、proof、video 或 freeze。

## Frozen rejected evidence

下列文件原样保留以便审计，但绝不消费为初始化、warm-start、optimizer state、proof、video 或
freeze：

| artifact | SHA-256 | 状态/原因码 |
|---|---|---|
| 003h `model_20.pt` | `08a700768ce8310fe20dcb87e96653bd450ada22475fabbf9e8765533b4a17b9` | `rejected_diagnostic`; `reward_contract_not_instantiated`, `eval_timeout_conflation` |
| 003h pilot manifest | `ef75777584f7a3a3b8a622ef120d7ec677c38477cdb5dcb8a893fe10f90c7231` | 同上 |
| 003h fixed eval | `59f614ea2e0e4c5414f543bc6696cb582ad075338620cd6183a199accfa47fb0` | 同上 |

后验只读诊断用于解释 gate，不是新 pass evidence：003h TensorBoard 仍出现
`Episode_Reward/foot_gait`；真实 manager 有 inherited `foot_gait weight=.5`。旧 deterministic
replay 的 first-fall/common-prefix 依次为 model0 `2.24 s/-0.728549 m`、model7
`2.62 s/-0.554933 m`、model14 `1.36 s/+0.393896 m`、model20
`1.02 s/+0.340923 m`；model20 相对 model7 明显退化。旧 20 s JSON 的 vx/x=0 是 evaluator
丢弃全部 fallen env 后的假零，不能解释为“完全没有前进”。

## Scope 与代码所有权

默认只允许修改：

- `.agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py`；
- `tests/test_task072_locomotion_proof.py`；
- 本 subtask 与 Task072 状态文档。

禁止修改 `.external/unitree_rl_mjlab`、asset XML、contact profile、stance、signed-headroom action
数学、PPO 超参数、observation、termination limit、fixed command、DR/curriculum 或 proof gate。
需要超出上述范围时停止并新建有独立 delta/验收的 subtask，不能在 003i 顺手改变科学变量。

## Exact implementation contract

### 1. Reward-v3 信号与采样时刻

reward contract id 固定为 `task072_mjlab_biped_phase_contact_v3`。旧 003h 同名字符串没有
canonical payload/manager table，已被拒绝；只有本节完整实现、payload SHA 和 runtime table
同时闭合才算 v3。

所有物理量在真实 `ManagerBasedRlEnv.step()` 的 RewardManager 时刻读取：完成 decimation、
`episode_length_buf += 1`、termination compute 之后、reset 之前。不得从 reset 后 observation
回推 reward。control `dt=0.02 s`，RewardManager 对 `raw * weight` 只乘一次 `dt`；term function
不得预乘 dt。bonus raw 为非负，penalty raw 为非正，表内 weight 全为正。

固定定义：

- `M = I(sqrt(vx_cmd^2+vy_cmd^2)+abs(wz_cmd) > 0.1)`；本任务固定命令使 `M=1`；
- `k = env.episode_length_buf` 在 reward compute 时的逐环境值；reset 后为 0，首个 action 后为 1；
- `phase = ((k * 0.02) mod 0.8) / 0.8`；`leg_phase_i=(phase+offset_i) mod 1`；
- offsets 固定 left/right `[0.0, 0.5]`，`desired_contact_i = I(leg_phase_i < 0.55)`；
- `contact_i = I(current_contact_time_i > 0)`，来自 `feet_ground_contact` 对左右 logical-foot
  body subtree 与唯一 `terrain` 的真实 contact；每脚 7 capsules 只能 OR 聚合为一个 logical value；
- `touchdown_i = feet_ground_contact.compute_first_contact(0.02)_i`，touchdown air-time 使用
  `last_air_time_i`；
- foot height/velocity 使用已绑定 left/right foot site，height 是相对 terrain z 的米值；
- `a_raw` 是 policy 输出、`Task072ClipLoggingVecEnvWrapper` clip 前的 29 维 normalized action；
  wrapper 继续使用 `agent_cfg.clip_actions=1.0`，`a=clip(a_raw,-1,1)` 是送入 action manager、供
  reward 使用的 applied normalized action，`a_prev` 每环境 reset 为 0；
- `q_ref` 是当前 bound stance `stance_solution.joint_qpos`，不得改用 XML zero/default pose。

phase 只能使用 episode-local `episode_length_buf`；禁止 global runner/update step。termination 或
timeout reset 后下一 episode 必须重新从 `k=0` 开始。

### 2. Exact active reward table

实现时必须新建一个显式完整 dictionary 并整体赋给 `env_cfg.rewards`；禁止继承 parent 后
`pop("feet_gait")`、alias、局部删除或只改 `REWARD_CONTRACT_VERSION`。exact active name set
共有 **23** 项：

| term | exact top-level callable qualname | raw function | weight |
|---|---|---|---:|
| `track_xy_centered` | `task072_reward_track_xy_centered` | `exp(-((vx-.5)^2+vy^2+vz^2)/.25)-1`，base/body-frame m/s | `2.0` |
| `track_yaw` | `task072_reward_track_yaw` | `exp(-wz^2/.25)`，base/body-frame rad/s | `.50` |
| `upright` | `task072_reward_upright` | `clip(-g_z,0,1)` | `.25` |
| `tilt` | `task072_reward_tilt` | `-(g_x^2+g_y^2)` | `5.0` |
| `height` | `task072_reward_height` | `-((root_z-z_stance)/.10)^2` | `.25` |
| `stand_support` | `task072_reward_stand_support` | `(1-M)*I(contact_left or contact_right)` | `.30` |
| `phase_gait` | `task072_reward_phase_gait` | `M*mean_i(I(contact_i == desired_contact_i))` | `.50` |
| `out_of_phase_double_support` | `task072_reward_out_of_phase_double_support` | `-M*I(contact_left and contact_right and not(desired_left and desired_right))` | `.35` |
| `clearance` | `task072_reward_clearance` | 对 `desired_contact_i=false and contact_i=false` 的脚取 `exp(-((height_i-.10)/.05)^2)` 均值；集合为空为 0；整体乘 `M` | `.50` |
| `touchdown_airtime` | `task072_reward_touchdown_airtime` | `mean_i(touchdown_i*clip(last_air_time_i/.5,0,1))` | `.10` |
| `soft_landing` | `task072_reward_soft_landing` | `mean_i(touchdown_i*exp(-(foot_vz_i/.5)^2))` | `.10` |
| `foot_slip` | `task072_reward_foot_slip` | `-mean_i(contact_i*(foot_vx_i^2+foot_vy_i^2))` | `.20` |
| `nonfoot_contact` | `task072_reward_nonfoot_contact` | `-mean_j(I(nonfoot_body_j contacts terrain))`；j 是全部 robot body 去掉两个 logical-foot body 后的 resolved set | `.20` |
| `pose_hip` | `task072_reward_pose_hip` | hip 6 joints 的 `-mean((q-q_ref)^2)` | `.20` |
| `pose_knee` | `task072_reward_pose_knee` | knee 2 joints 的 `-mean((q-q_ref)^2)` | `.30` |
| `pose_ankle` | `task072_reward_pose_ankle` | ankle 4 joints 的 `-mean((q-q_ref)^2)` | `.20` |
| `pose_waist` | `task072_reward_pose_waist` | waist 3 joints 的 `-mean((q-q_ref)^2)` | `.10` |
| `pose_arm_wrist` | `task072_reward_pose_arm_wrist` | left/right arm、elbow、wrist 14 joints 的 `-mean((q-q_ref)^2)` | `.05` |
| `joint_velocity` | `task072_reward_joint_velocity` | 29 active joints 的 `-mean(qd^2)`，qd rad/s | `.02` |
| `joint_limit` | `task072_reward_joint_limit` | 29 joints 的 `-mean(v^2)`；`v=max((q-hi90)/(hi-lo),0)+max((lo90-q)/(hi-lo),0)`，`lo90/hi90=center +/- .9*half_range` | `.05` |
| `action_magnitude` | `task072_reward_action_magnitude` | `-mean(a^2)` | `.01` |
| `action_rate` | `task072_reward_action_rate` | `-mean((a-a_prev)^2)` | `.01` |
| `base_angvel_xy` | `task072_reward_base_angvel_xy` | `-(wx^2+wy^2)`，base/body-frame rad/s | `.02` |

pose groups 必须由 29-row semantic mapping 显式解析并断言互斥、并集正好 29：
`limb[01]_hip_*` 6、`limb[01]_knee_pitch` 2、`limb[01]_ankle_*` 4、`waist_*` 3、
`left_arm_* + right_arm_*` 14。不得用 anonymous joint 索引位置猜分组。

`nonfoot_contact` 必须增加受控 `nonfoot_ground_contact` sensor：primary 是 runtime resolved robot
bodies 排除两个 logical-foot bodies，secondary 是唯一 `terrain`，raw 是 primary-body contact
boolean 的均值负值。不得把 self-collision、foot-terrain 或字符串模糊匹配混入。

表中 23 个 callable 必须是 runner 模块顶层函数；禁止 nested `<locals>` callable 和
`.external...mdp.rewards` callable。每项 `RewardTermCfg.params` 的 canonical keys 也固定：

- velocity/orientation/height/base-angvel：`asset_name="robot"`、resolved torso body name/id；
  tracking 项另含 `command_name="twist"` 和表内 denominator，height 另含从 bound stance 读取的
  `stance_height` 与 stance payload SHA；
- stand/phase/double-support：`command_name="twist"`、`sensor_name="feet_ground_contact"`、
  `command_threshold=.1`；phase/double-support 另含 `period=.8`、`offsets=[0,.5]`、
  `stance_fraction=.55`；
- clearance/airtime/landing/slip：`sensor_name="feet_ground_contact"`、resolved left/right site
  names/ids；分别另含表内 `.10/.05`、`.5`、`.5` 数值；
- nonfoot：`sensor_name="nonfoot_ground_contact"`、resolved nonfoot body names/ids、
  `terrain_name="terrain"`；
- pose：`asset_name="robot"`、该组 ordered semantic names、resolved anonymous names/ids、
  ordered `q_ref` 和 stance payload SHA；
- joint velocity/limit：ordered 29 semantic names、resolved names/ids；limit 另含
  `soft_fraction=.9` 和 runtime lower/upper arrays；
- action magnitude/rate：`action_name="joint_pos"`；rate 另含 reset previous action=0；
- 所有浮点值 canonicalize 为 JSON number，tuple canonicalize 为 JSON list；禁止保存 Python object
  repr、未解析 regex 或仅保存 version。23 个 `RewardTermCfg.params` 本身只允许 JSON-native
  scalar/list/dict；不得放 `SceneEntityCfg` 等需要第二次 runtime resolve 的对象。

`tests/test_task072_locomotion_proof.py` 必须维护一份独立 literal
`EXPECTED_TASK072_REWARD_V3_ACTIVE_TABLE`，逐项写死 name、qualname、weight 和上述 expected params；
它不得调用 runner 的 builder/canonicalizer 来生成 expected value。manager test 将真实 runtime
table 与该 literal 比较，避免“actual 与由 actual 自己生成的 expected 永远相等”。

### 3. Reward oracle 与真实 runtime gates

pure oracle 与 runtime fixture 都穷举 `k=1..40`（50 Hz 的完整 0.8 s phase cycle）。除下列差异外，
fixture 固定 upright、root z=`z_stance`、q=`q_ref`、qd=0、a/a_prev=0、w=0、无 nonfoot contact、
无 touchdown：

| fixture | velocity/contact/foot height | weighted pre-dt mean gate |
|---|---|---:|
| `static_both` | vx/vy/vz=0；双脚 contact；height=0 | `<= -0.25` |
| `ideal_phase_matched` | vx=.5, vy/vz=0；contact 精确等于 desired；所有 desired swing 脚 height=.10 | `>= 1.00` |
| `persistent_left_only` | vx/vy/vz=0；左脚恒 contact、右脚恒 noncontact；两脚 height=0 | `<= 0` |

另断言 `ideal_phase_matched - static_both >= 1.25`。按本表、40 个离散 phase 和 touchdown=0，
oracle 期望约为 static `-0.554241`、ideal `1.700000`、persistent-left `-0.260120`、margin
`2.254241`；误差超 `1e-6` 即拒绝，不能只检查宽松不等式。

受控 probe 不允许实现者自行向真实 MuJoCo state/sensor buffer 打补丁。唯一 harness 分两层：

1. 先实例化真实 1-env train `ManagerBasedRlEnv`，从它的实际 `RewardManager` 按 active order 提取
   已 resolve 的 23 个 `RewardTermCfg`；这一步同时完成 manager-level table 检查；
2. `Task072RewardFixtureAdapter` 是 tests 中的 tensor-backed 最小 env，固定实现这些 callable 唯一
   允许读取的接口：`num_envs/device/step_dt/episode_length_buf/max_episode_length_s`、
   `command_manager.get_command("twist")`、`scene["robot"].data` 的 root pose/body velocity/
   projected gravity/joint/site tensors、`scene["feet_ground_contact"]` 的 contact/air-time data 与
   `compute_first_contact(dt)`、`scene["nonfoot_ground_contact"].data.found`、以及
   `action_manager` 的 applied/current/previous action；读取未列接口必须使测试失败；
3. 将第 1 步提取的 exact resolved cfg 原样交给官方 `mjlab.managers.RewardManager`，env 参数只换为
   第 2 步 adapter。以 `scale_by_dt=False` 验证 raw*weight，以 `scale_by_dt=True, dt=.02` 验证只乘
   一次 dt。fixture 逐字段装载上表三种状态和 k=1..40；不得重新构造另一套 reward dict；
4. pure oracle 与上述官方 RewardManager 的逐项、weighted pre-dt、dt contribution 和 total abs
   diff 全部 `<=1e-6`，且 finite。

这里的 `RewardManager(cfg, env, *, scale_by_dt=...)`、`active_terms`、`get_term_cfg(name)`、
`compute(dt)` 和 `get_active_iterable_terms(env_idx)` 都是当前 pinned MJLab 的既有 API，不是 003i
要新增的 manager API。CPU verifier 先用 `inspect.signature`/`hasattr` 断言这些接口存在；缺失即因
runtime drift fail closed。唯一允许新增的抽取 helper 是
`task072_reward_breakdown_from_manager(manager, env, dt)`：它严格按 `manager.active_terms`，通过
`manager.get_term_cfg(name)` 调用 actual `cfg.func(env, **cfg.params)`，返回 ordered rows
`{name, raw, weight, weighted_pre_dt, dt_contribution}` 与两个 total。它不得读取 expected table、
不得重写公式。断言 `scale_by_dt=False` 的 `compute(dt)` 等于 pre-dt total，`scale_by_dt=True` 的
`compute(dt)` 等于 dt total，而 `get_active_iterable_terms()` 始终等于 ordered weighted-pre-dt rows。

真实 1-env MuJoCo CPU rollout不做不可控的 ideal-contact 注入，只验证 live boundary：实际 manager
输出 reward 等于既有 `get_active_iterable_terms()` 之和乘 `.02`、实际 active table/SHA 正确、首步 phase
使用 `k=1`、reset 后 `episode_length_buf=0`、连续 120 个 zero-action control steps内无
terminated/timeout 且全部 component finite。zero-action rollout 只是 runtime sanity，不是 locomotion pass。

### 4. Manager-level verification（禁止字符串代替 runtime）

测试必须实例化真实 train `ManagerBasedRlEnv`/`RewardManager`，同时重建 eval config。至少断言：

1. active term names 的 set、顺序和 count=23 与上表 exact 相等；
2. 每项实际 callable 的 source file、`__module__`、`__qualname__` 是受控 v3 实现；
3. 每项实际 weight、resolved params、period `.8`、offsets `[0,.5]`、stance fraction `.55`、
   command threshold `.1` 完全相等；
4. active table 中没有 `foot_gait`、`feet_gait`、`is_terminated`，没有 alias、重复名或 parent
   callable；
5. train/eval 实例的 canonical active table 相等。只检查 version 字符串、source grep、dict
   中出现某个字面量，均不算通过。

canonical reward payload 固定包含 schema/id、23-term ordered table、function source SHA、module/
qualname、weight、JSON-normalized resolved params、signal sampling、phase 与 dt semantics。runtime spec、
capacity、one-update、pilot、eval manifest 都保存其 canonical JSON、payload SHA 和 actual manager
active-table SHA；train/eval 重建不等即 fail closed。

### 5. Eval cause separation 与 common-prefix 算法

外部 eval horizon 固定 `H=20/.02=1000` control steps；env `episode_length_s` 保持与 training 相同的
`10000 s`，不得设成 20 s。fixed eval 使用 deterministic inference policy、256 env、seed
`720400`、command `[.5,0,0]`。

eval 不得绕开 policy-compatible wrapper。唯一构造为
`outer=ManagerBasedRlEnv(...)`、
`base_env=RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)`、
`env=Task072ClipLoggingVecEnvWrapper(base_env, semantic_joint_names, rollout_steps=1000)`，并断言
`agent_cfg.clip_actions==1.0`。外层 logging wrapper 在 clip 前观察 policy raw action，然后调用
`base_env.step()`；actor/critic TensorDict、base-wrapper clipping、action manager 和
`_force_fixed_command` 语义与 training 相同。

`env.reset()` 后立即 `_force_fixed_command(env)`、通过 runtime binding 中
`actor.term_slices["command"]`（当前应为 `[6,9]`，不得另行硬猜）断言实际 twist 与 actor command
slice 都是 `[.5,0,0]`，再保存 `start_x` 并令所有 env `active=true`。对 1-based step j 精确执行：

```text
active_before = active
action_raw = deterministic_policy(obs)
obs, reward, done, extras = env.step(action_raw)  # wrapper 记录 raw，再按 1.0 clip
terminated = outer.reset_terminated.clone()
time_out = outer.reset_time_outs.clone()
assert done.bool() == (terminated | time_out)
_force_fixed_command(env)
assert actual command 和 obs command slice 仍为 [.5,0,0]
new_terminated = active_before & terminated
new_time_out = active_before & time_out
assert no env is both new_terminated and new_time_out
active_after = active_before & ~terminated & ~time_out
只用 active_after 的 reset 后返回状态中的 post-step sample 更新 vx、last_x 和 contact/event
new_terminated 的 first_fall_seconds = j * 0.02
active = active_after
```

两类 outer flags 必须在 `env.step()` 返回后、任何下一步调用或 robot/sensor metric 读取前立即
`.clone()`；禁止从 wrapper 合并后的 `done` 或 `extras["time_outs"]` 反推。若当前 MJLab 版本没有
这两个 exact outer attributes，CPU verifier 必须 fail closed，而不是切换到另一套 loop。

MJLab 在 step 返回前已 auto-reset，故 terminated/time-out step 的 robot state 是新 episode state，
必须通过 `active_after` 排除；不得用 `active_before` 累积该 sample。首次 termination/time-out 后，
该 env 的后续 auto-reset episode全部忽略。touchdown、single-support、alternation、contact fraction
同样只在 active_after 上累积。

输出 schema 至少包含：

- `reset_terminated={count, ratio, env_ids}` 与
  `reset_time_outs={count, ratio, env_ids}`，二者分开；
- `zero_fall_ratio=1-reset_terminated.ratio`；20 s 内任何 timeout 都令 evaluator `passed=false`；
- `first_fall_seconds={min,p10,median,p90,max}`；未 terminated survivor 的值固定为 20 s；
- `common_prefix.active_env_steps`、`valid_env_count`、按有效 env-step 加权的 `mean_vx`、按 env
  的 `mean_x_displacement`/`median_x_displacement`；每 env x 是最后一个有效 nonterminal x 减
  `start_x`；
- `common_prefix` contact/touchdown/single-support/alternation；
- `survivor_full_horizon={survivor_count, mean_vx, mean_x_displacement, ...}`。survivor=0 时指标写
  JSON `null` 并使相应 gate fail，禁止写 0、`1e9` 或用 reset 后轨迹补数。

pilot continuation 只消费 `common_prefix` 与 first-fall；正式 walking proof 仍消费 zero-fall 和
survivor full-horizon 的既有 walking gate。两类字段不能互相冒充。

deterministic fake-env unit tests 必须覆盖：timeout-only 不算 fall且 evaluator fail；termination-only
first-fall；同 batch 混合 cause；所有 env early fall 仍保留非零 common-prefix vx/x；termination
step 返回的假 reset 位移/contact 不污染；无 survivor 输出 `null` 而非 0/sentinel；cause overlap
fail closed。

### 6. Training-period action clip aggregation

统计边界固定在 policy 与原 `RslRlVecEnvWrapper` 之间。新增模块顶层 composition wrapper
`Task072ClipLoggingVecEnvWrapper(base_env, semantic_joint_names, rollout_steps)`：它把 raw action 交给
整数 accumulator 后原样调用 `base_env.step(raw_action)`；原 base wrapper 继续按
`agent_cfg.clip_actions=1.0` clip，再交给 `Task072SignedJointPositionAction`。禁止把统计放进 action
term，因为那里看到的已经是 wrapper-clipped action，会伪造为 0 clip fraction。

logging wrapper 必须透明转发 runner 需要的 reset/step/attribute/unwrapped/observation 接口；除增加
统计外，给同一 action 时的 obs/reward/done/extras 和 applied joint target 必须与原 base wrapper
bitwise/equal-tolerance 一致。每次 `step` 在 base clip 前统计 `abs(a_raw)>1.0`（严格大于，不是
`>=`）。wrapper 内部 `steps_in_update` 每到 exact `rollout_steps=24` 时，在同一个 `step()` 尾部
snapshot 一条 immutable update record、追加到 `completed_update_records`、再把全部计数原子清零；
update index 从 0 单调递增。禁止依赖 logger callback 或复制/覆盖 `MjlabOnPolicyRunner.learn()`。
现有 action term 的瞬时 `task072_clip_fraction` 不得进入正式 artifact。

one-update 与 pilot 的唯一挂载点都在创建 runner 之前：

```text
outer = ManagerBasedRlEnv(...)
base_env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
env = Task072ClipLoggingVecEnvWrapper(base_env, semantic_joint_names, agent_cfg.num_steps_per_env)
runner = runner_cls(env, ...)
runner.learn(updates)
records = env.drain_task072_clip_update_records()
assert len(records) == updates and env.steps_in_update == 0
```

`drain_task072_clip_update_records()` 只在 `runner.learn()` 返回后调用一次，返回并清空 completed
records；runner 持有并实际 step 的必须就是该 `env`，不得把 logging wrapper 放在 runner 外面。
one-update 要得到 1 条 record，21-update pilot 要得到 ordered 21 条；任一数量或 update index
不符即训练 manifest fail。eval 使用同一 wrapper 保持 raw-action/clip 边界，但 `rollout_steps=1000`
且其 record 不进入训练 progression。

每 update 固定输出：

- `scalar_clip_fraction = clipped_scalars / (N*T*29)`；
- `env_step_any_clip_fraction = env_steps_with_any_clip / (N*T)`；
- 29 个按 semantic joint 命名的 `per_joint_clip_fraction[j] = clipped_j/(N*T)`；
- `max_abs_raw_action`；
- 原始 numerator、denominator、`num_envs=N`、`rollout_steps=T`、`joint_count=29`、update index。

同一数值写入 TensorBoard tags `Diagnostics/action_clip/{scalar_fraction,env_step_any_fraction,
max_abs_raw}` 与 `Diagnostics/action_clip/joint/<semantic_joint>`、stdout 单行 JSON、
`progression.json` 每-update record；manifest 保存完整 progression SHA 和最后 7 updates 的 pooled
summary。训练函数在 drain 后把 21 条 record 按其 update index 写入同一 run 的 TensorBoard；
具体用 `torch.utils.tensorboard.SummaryWriter(log_dir=runner.logger.log_dir)` 新 writer 写完即 close，
不得依赖已经 stop 的 runner writer。last 7 固定指 updates 14..20，pool numerator/denominator 后再除，
禁止简单挑最小 update。
one-update deterministic action tensor必须可逐项精确复算，并比较 logging wrapper 与原 base
wrapper 的 applied action/target 等价；NaN、缺字段、denominator 不等 `N*T*29` 或 update reset
失败都 fail closed。

pilot clip gate：updates 14..20 pooled `scalar_clip_fraction <=0.10`、pooled
`env_step_any_clip_fraction <=0.50`、任一 joint pooled fraction `<=0.25`。`max_abs_raw_action` 记录但
本 pilot 不设额外阈值；如果它 non-finite 则失败。

## CPU acceptance matrix

在任何 CUDA 命令前，顺序固定为：

1. unit tests：23-term pure oracle、完整周期 probes、pose 29/29 partition、action clip exact counts；
2. manager tests：真实 train/eval manager active table/function/weight/params/phase 与 canonical SHA；
3. evaluator fake-env tests：cause separation、common-prefix、auto-reset exclusion、null semantics；
4. runtime CPU verifier：1-env manager component sum、phase reset、120-step zero-action sanity；
5. negative tests：错 key、缺 term、parent callable、weight/phase drift、train/eval reward SHA drift、
   forged checkpoint/manifest、cause overlap、缺 clip metric都必须在 env/GPU work 前拒绝；
6. `py_compile`、以下两份 scoped pytest 全量、`git diff --check` 全部返回 0：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK072=.agent/task/task072-bound-g1-go2-locomotion-proof
env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$TASK_PY" -m py_compile \
  "$TASK072/task072_mjlab_contact_runner.py" tests/test_task072_locomotion_proof.py
env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q \
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py
git diff --check
```

runner 必须新增 CPU-only `verify-reward-eval-contract` command，并生成
`003i_reward_eval_contract_verifier.json`；`passed=true` 只有上述 reward、manager、eval schema、clip
schema、zero-action sanity 全部通过时才允许。随后提交受控 source/tests/docs，记录 selected source
commit、runner/test/reward payload/verifier SHA；未提交的 source 或任一 mismatch 不得申请 GPU。

## GPU execution contract（需另行用户授权）

artifact root 固定为现有 single-ground lineage 下的新 003i 前缀，不覆盖 003h：

```text
artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v3_single_ground/
  003i_reward_eval_contract_verifier.json
  003i_capacity_smoke_2048_4096_6144.json
  003i_one_update_4096x24_seed720301/
  003i_pilot_4096x24x21_seed720301/
  003i_eval_pilot_model_{0,7,14,20}_fixed_vx0p5_seed720400.json
  003i_pilot_gate.json
```

所有 CUDA processes 必须是 `/home/admin1/workspace/run/.gpu.lock` holder 的后代并在 artifact 记录
`gpu_lock.held_by_ancestor=true`。因 runner/config/reward/eval schema SHA 已改变，003h capacity 和
one-update 均不可复用。授权后顺序固定：

1. capacity candidates `2048/4096/6144`、rollout 24；至少 `4096 x 24=98,304` 必须通过；
2. 只消费上述 003i capacity artifact 的 exact `4096x24x1` one-update，从随机初始化，验证 optimizer
   step、parameter delta、23 reward terms 和 clip aggregates；
3. 新 source commit、随机初始化 seed `720301` 跑 `4096x24x21=2,064,384` transitions，
   `resume=false`，save/eval `model_0/model_7/model_14/model_20`；
4. 四个 checkpoint 都必须由同一 pilot manifest 列出 path/SHA/update/transitions，并用 fixed
   seed `720400`、256 env、20 s、vx=.5 的新 evaluator 独立 reload；
5. 生成 aggregate `003i_pilot_gate.json`。任何一步失败立即停止，不运行 proof、video 或 freeze。

weights、logs、checkpoints、GPU monitor 等 binary/large artifacts 不提交 Git；只在本 subtask Log
记录路径、SHA、数值 verdict 和 source commit。

### Pilot continuation gate

四个 eval 与训练 progression 全部 finite，且 timeout count 都为 0。不得挑 model7 或 best checkpoint
绕过 final regression。exact gates：

- model20 median first-fall `>=2.5 s`；
- model20 median first-fall `>= model0 + 0.5 s`；
- model14 median first-fall `>= model7 - 0.25 s`；
- model20 median first-fall `>= model7 - 0.10 s` 且 `>= model14 - 0.25 s`；
- model20 common-prefix mean vx `>=0.05 m/s`；
- model20 common-prefix median +x displacement `>=0.10 m`；
- updates 14..20 的三个 clip gates全部通过。

失败状态固定为 `pilot_failed / trained / not_passed`，保留新 run 为 diagnostic 并停止。通过状态只能是
`ready_for_separately_authorized_proof / pilot_passed / not_passed`；它不解锁 004，也不能把 pilot
checkpoint freeze。后续 full proof 仍必须从 003i pilot 是否允许 continuation 的结论出发，按另一个
versioned subtask 和既有正式 walking gate 执行。

## Log

- 2026-09-01：完成 bounded 003i runner/test repair。Source commit
  `407f053ab2376e482738179f8f3ac53ad26424e8`；runner SHA
  `f12ae2ade55815094fb9a29b046d87892909a22d641d7d482e82e464c8818233`；test SHA
  `dde574f527fc31980b65ed922bcd648c1a9ce481693e49d77b55dbfc87a7be6a`；actual
  23-term manager active-table SHA `323feac6197abc6d706205f39d5f332b834e87332d453919ccfb1998f5eea7e2`；
  reward payload SHA `ef8fe4b5c8f3edcbd293cc170b03ae99665e79f04d289105f57f04e19d617051`。
  CPU verifier artifact
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v3_single_ground/003i_reward_eval_contract_verifier.json`
  SHA `28f5a660e74599472035dee7849c84d7771122185b7869a301d0a4261f5ceae3` passed；checks
  `reward_manager_api/train_active_table/eval_active_table/train_eval_reward_match/reward_payload/registration_active_sha/canonical_config/fixture_oracle/eval_schema/clip_schema/zero_action_sanity`
  全部 true。新增 `pilot-gate` JSON reducer，强制 model0/7/14/20 continuation gate 和 updates
  14..20 action-clip pooled thresholds；training/eval manifest 对 clip schema、integer counters、分数、
  阈值、capacity consumption 和 progression SHA fail closed；active-table validator 绑定 q_ref/source
  hash。验证：`py_compile` passed；scoped pytest
  `tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py` returned
  `80 passed, 35 warnings`；`git diff --check` passed。未运行 CUDA、capacity、one-update、pilot、
  proof、video 或 freeze。
- 2026-09-01：整理脏文件后提交 `3fad223dc37f7d00f89582c50cda2fed5f2fcfa2`，并在该 clean HEAD
  重新确认 003i CPU verifier passed。runner SHA
  `f12ae2ade55815094fb9a29b046d87892909a22d641d7d482e82e464c8818233`、test SHA
  `dde574f527fc31980b65ed922bcd648c1a9ce481693e49d77b55dbfc87a7be6a`、reward payload SHA
  `ef8fe4b5c8f3edcbd293cc170b03ae99665e79f04d289105f57f04e19d617051`、actual manager active-table SHA
  `323feac6197abc6d706205f39d5f332b834e87332d453919ccfb1998f5eea7e2` 均与 CPU verifier 一致；
  refreshed CPU verifier artifact SHA
  `e3058b1385ee3d6136b541a4efa1e433b76f2072caf2ec0cd3400e4765f95f80`。
  003i capacity smoke 在 `/home/admin1/workspace/run/.gpu.lock` ancestor 下通过，artifact SHA
  `6cbb7fb4e42908cfd304dec8f7f7b462dc0423e54e670c052d835d2f0853e462`；
  candidates `2048/4096/6144` 全部 finite/passed，required selected `4096 x 24 = 98,304`，
  `gpu_lock.held_by_ancestor=true`。随后运行 exact `4096x24x1` one-update seed `720301`，
  生成 `model_0.pt` SHA
  `3a810edb8e2c5f7d8f9f1825309cd14da202f396e7f70b9c62ba76ca5e315e30`、manifest SHA
  `b8920fc985b90bec63b7864765ba5a47652b57bb49026067cf390b37e776c33c`、progression SHA
  `3a6d4dc656eac3e3d126e729dc14056bf2e85f05f0f801d2591589d62baaa598`、policy ONNX SHA
  `3a260a462922140f9876e378d2270d64307a1ec97345647173e6d2d565ab0478`；capacity consumption checks
  全部 true，`training_execution_complete=true`，但 one-update `passed=false`，原因是 action clip
  gate failed：scalar clip fraction `0.3195660470545977 > 0.10`、env-step-any clip fraction
  `0.9999898274739584 > 0.50`、max per-joint clip fraction
  `0.3235677083333333 > 0.25`（`left_arm_shoulder_pitch`），max abs raw action
  `5.651471138000488`。按 stop rule 未运行 `4096x24x21` fresh pilot、model `0/7/14/20`
  eval 或 `003i_pilot_gate.json`；003i 状态为
  `one_update_failed / smoke_trained / not_passed`。

## Review

- [x] exact 23-term v3 已整体替换 parent reward；gait 保留且 centered velocity 使 static full reward
  probe 非正，不存在 parent/alias term。
- [x] 真实 train/eval ManagerBasedRlEnv/RewardManager active table、callable、weight、params、phase、
  dt 与 canonical SHA 全部闭合，而非只检查版本字符串。
- [x] eval 分开统计 terminated/timeout，common-prefix、median first-fall 与 auto-reset exclusion 的
  fake/runtime tests 通过。
- [x] action clip 每 update 的整数计数、四类输出、last-7 pooled gate 均可复算。
- [x] CPU matrix 与 source commit 已通过；若未取得单独 GPU 授权，状态停在
  `awaiting_user_gpu_authorization / not_trained / not_passed`。
- [x] GPU capacity passed 后 exact one-update 因 action clip scalar/env-step/per-joint gates 失败而停止；
  failure artifact 已保留，未继续 pilot/eval/gate。
- [ ] fresh-init pilot 通过 continuation gate 后也只允许新建 proof subtask；否则停止
  004/Task073/Task074。
