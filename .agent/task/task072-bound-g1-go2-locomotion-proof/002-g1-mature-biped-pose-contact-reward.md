# 002 — G1 mature biped pose/contact reward

状态：**passed**。Owner：reward/environment owner；review owner：Task072 verifier owner。

## Route

1. 保留 fixed-command linear/yaw tracking、upright 与 base-height 目标，但把 reward 分解为可单项
   记录的 task、pose、contact/gait 和 regularization components。
2. 增加围绕 knee-bent nominal stance 的 per-semantic-group pose deviation；至少区分 hip、knee、
   ankle、waist、arm/wrist，避免 29 DoF 使用一个无语义的统一姿态权重。
3. 增加最小成熟 biped contact shaping：左右脚 load-bearing contact、swing/air-time、foot
   clearance、landing、slip 和 non-foot contact。所有接触项必须读取实际 MuJoCo contact，不得用
   action 或关节角伪造接触状态。
4. 增加 joint velocity、joint-limit、action magnitude/rate、base angular velocity 等有限
   regularization，并在 artifact 中保存每项 reward 的量级，防止某一项主导总回报。
5. 删除 `-200` 巨大跌倒惩罚。fall/upright violation 继续终止 episode，但 terminal reward 不得
   在正常单步 reward 尺度上形成数量级主导；最终选择和 ablation 必须记录。
6. 只给 exact-bound biped 配置启用该 reward；Go2 继续使用 quadruped-specific contact/reward
   semantics。共享环境若需新增观测或 metric，只做最小扩展并保持 legacy 默认不变。

本地 `/home/admin1/workspace/proj/unitree_rl/doc` 与课程压缩包只作为 reward/action 组成和训练预算
参考；许可证尚未清理，不复制其代码或 checkpoint，也不把其 29DoF/23DoF observation contract
冒充当前 anonymous G1 contract。

## Log

- 旧 Task072 reward 主要包含 tracking、heading、upright、tilt、height、non-foot contact 与
  action regularization，并使用 `-200` fall penalty；缺少完整 per-joint pose 与足端 gait/contact
  shaping。
- 2026-08-27：尚未实施；不得记录为 mature reward 已绑定。
- 2026-08-27：实施 `Task072BipedReward` 与 `Task072QuadrupedReward`。G1 reward 记录 20 个
  raw/weighted components，使用 MuJoCo footpad-floor contact 生成 foot contact/height/speed/air-time/
  touchdown metrics；旧 biped `-200` 和 quadruped `-50` fall reward penalty 已移除，fall 仅作为
  termination。`reward-contract` 生成 G1/Go2 reward_config；targeted pytest
  `biped_reward or foot_contact or reward_components or no_fall_penalty` 为 1 passed。
- 2026-08-27：在 nominal_v3 实施 `task072_biped_phase_contact_v2`：moving branch command 固定
  `vx=0.5`，period `0.8 s`，left/right offsets `0.0/0.5`，stance threshold `0.55`，G1 shard
  使用 `upright_threshold=cos(0.8)`；moving 时 static both-contact 的 phase subtotal 周期均值测试
  `<=0`，alternating contact subtotal 为正且更高。命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py reward-contract
  --case unitree_g1 --case unitree_go2 --output-root
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3`。G1 reward_config raw SHA
  `ed515be4a236ec1e5cc5e7d1214fc8d9d28a41b64845bdde3613e51b0d2e656f`；Go2 reward_config raw SHA
  `79c8823d70d0ebb61cbbb1902a24e2d0bd3dd7963b7cb514a051ab61b7fb5d89`；fall_penalty 均为 `null`。
  两份指定测试为 `28 passed`。

## Code implementation

### Environment metrics

只在 `WholeBodyMuJoCoShard.step()` 增加 reward 所需的 post-step/pre-reset metrics，均按 unified 45-slot
或左右 load-bearing limb 顺序输出：`joint_position`、`joint_velocity`、`foot_contact`、
`foot_height`、`foot_planar_speed`、`foot_vertical_speed`、`foot_air_time`、`touchdown`。foot geom 只从
`LinkBlueprint.foot` 对应 `<link>_footpad` 与 floor contact 解析，不使用字符串猜测 action 或 qpos；
reset 时 air-time 清零。现有 metric key/value 不改。

### Historical rejected v1 formula

在 task-local CLI 新增 frozen `Task072BipedRewardConfig`，并把现有
`Task072LocomotionReward` wrapper 拆成 `Task072BipedReward` 与 `Task072QuadrupedReward`。不要把
biped reward 写进 shared `_reward()`：`Task072BipedReward.step(action)` 必须先调用
`self.shard.step(action)`，只读取该返回值的 post-step/pre-reset metrics，再以
`dataclasses.replace(result, reward=total, metrics=merged_metrics)` 替换 reward；reset、observation、
termination 和 motor process 仍由原 shard 负责。`merged_metrics` 保留所有旧 key，并新增
`reward_components`。nominal pose 直接取绑定的 `stance_solution.joint_qpos`。以下是本仓库
`task072_biped_pose_contact_v1` 初始设计，不声称来自 Unitree/LocoFormer 官方：

| component | 单环境 raw value | weight |
| --- | --- | ---: |
| `track_xy` | `exp(-||v_xy-[0.5,0]||² / 0.25)` | `2.0` |
| `track_yaw` | `exp(-(wz-0)² / 0.25)` | `0.5` |
| `upright` | `clip(-gravity_z, 0, 1)` | `0.5` |
| `height` | `-((z-z_stance)/0.10)²` | `0.25` |
| `pose_{hip,knee,ankle,waist,arm_wrist}` | negative mean squared `(q-q_nominal)` for that semantic group | `0.20,0.30,0.20,0.10,0.05` |
| `support` | `1` when either foot contacts floor | `0.30` |
| `swing_clearance` | mean over non-contact feet of `exp(-((height-0.08)/0.04)²)` | `0.10` |
| `touchdown_air_time` | mean `touchdown * clip(air_time/0.5,0,1)` | `0.10` |
| `soft_landing` | mean `touchdown * exp(-(vertical_speed/0.5)²)` | `0.10` |
| `foot_slip` | negative mean `contact * ||foot_planar_speed||²` | `0.20` |
| `nonfoot_contact` | negative boolean non-foot/floor contact | `0.20` |
| `joint_velocity` | negative active-slot mean `qvel²` | `0.02` |
| `joint_limit` | negative active-slot mean squared normalized soft-limit violation outside inner 90% range | `0.05` |
| `action_magnitude` | negative active-slot mean `action²` | `0.01` |
| `action_rate` | negative active-slot mean `(action-previous_action)²` | `0.01` |
| `base_angvel_xy` | negative `wx²+wy²` | `0.02` |

总 reward 是表中 weighted values 的直接和；所有 raw/weighted component 都写入
`result.metrics['reward_components']`。`fall` 只保留在 `trial_done`，reward 中不得出现 terminal fall
项。任何权重调整必须产生新 `reward_version` 与 config SHA，不能原地改 v1。

### Frozen quadruped branch

同时新增 frozen `Task072QuadrupedRewardConfig` 和 `Task072QuadrupedReward`，供 001 的 Go2 smoke、
005 正式训练以及 Task073 Spot/Lite3 复用。`task072_quadruped_fixed_command_v1` 精确保留恢复基线的
非 terminal quadruped shaping：

`4*exp(-4*||v_xy-command_xy||^2) + exp(-4*(wz-command_yaw)^2)
+ 1.5*exp(-2*heading_error^2) + clip(-gravity_z,0,1)
- 2*||gravity_xy||^2 - nonfoot_contact
- 0.05*mean_active((action-previous_action)^2)`。

它不包含 biped height/pose/air-time 项，也不包含旧 `-50*fall`；fall 同样只终止 episode。raw 与
weighted component key 固定为 `track_xy, track_yaw, heading, upright, tilt, nonfoot_contact,
action_rate`。若后续要加入 quadruped gait shaping，必须新建 reward version，并因 freeze identity
变化回到 003 重跑 G1，不能在 005 临时修改。

验证与产物：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q \
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py \
  -k 'biped_reward or foot_contact or reward_components or no_fall_penalty'
```

测试覆盖两个 reward 的 component sum、左右 foot contact/touchdown/air-time、reset、finite、semantic
group coverage，以及 biped `-200`/quadruped `-50` terminal penalty 均不存在。CLI 在不训练 Go2 的
前提下生成并 smoke 两个 frozen config：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK072=.agent/task/task072-bound-g1-go2-locomotion-proof
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK072/task072_locomotion_proof.py" reward-contract \
  --case unitree_g1 --case unitree_go2 --output-root "$TASK072/artifacts/nominal_v3"
```

输出分别为 `unitree_g1/reward_config.json` 与 `unitree_go2/reward_config.json`，各自含 version、全部
weights、semantic/contact mapping、config SHA 和 source SHA。缺 component、NaN、Go2 config 未生成
或 shared legacy reward 漂移时停止 003。

Frozen quadruped branch 的 heading 定义固定为：从 post-step、pre-reset 的 canonical-root world
quaternion `(wxyz)` 提取 yaw；reference 是该 env 每次 reset 后保存的 canonical-root yaw；
`error=atan2(sin(yaw-reference), cos(yaw-reference))`。`trial_done` 处理完成后才更新新 trial 的
reference，禁止用 reset 前旧值或跨 env reference。

### R1 v2 formula

状态：**reopened / not_passed**；v1 rejected，不原地修改。新增 `task072_biped_phase_contact_v2`：
真实 MuJoCo contact，period=`0.8 s`，left/right offset `[0.0,0.5]`，stance threshold=`0.55`。定义
`global_phase=((trial_step/control_hz) % period)/period`、
`leg_phase_i=(global_phase+offset_i)%1`、`desired_contact_i=(leg_phase_i<threshold)`。
当 `||command||>0.1` 时，phase contact match 取代 moving 时 any-foot support，并显式惩罚 phase 外
双支撑；当 `||command||<=0.1` 时关闭 gait/clearance 项并只启用站立 support。本任务命令固定为
`0.5 m/s`，因此正式训练始终走 moving 分支。
Task072 bad-orientation 限值=`0.8 rad`：当前 MuJoCo shard 使用 projected-gravity 判定，因此 builder
必须传 `upright_threshold=cos(0.8)`，不得误把 `0.8` 直接填入该字段。保留既有 canonical stance
height termination；fall 仅 termination，无巨大 penalty。

| component | raw formula | initial weight |
|---|---|---:|
| track_xy | `exp(-(||v_xy_body-[.5,0]||²+vz_body²)/.25)` | 2.0 |
| track_yaw | `exp(-wz²/.25)` | .5 |
| upright | `clip(-gravity_z,0,1)` | .25 |
| tilt | `-||gravity_xy||²` | 5.0 |
| height | `-((z-z_stance)/.10)²` | .25 |
| stand_support | `I[||command||<=.1] * I[any foot contact]` | .30 |
| phase_gait | `I[||command||>.1] * mean_i(I[contact_i==desired_contact_i])` | .50 |
| out_of_phase_double_support | `-I[||command||>.1] * I[both_contact and not desired_both]` | .35 |
| clearance | desired-swing/non-contact feet mean `exp(-((height-.10)/.05)²)` | .50 |
| touchdown_airtime | touchdown mean `clip(air_time/.5,0,1)` | .10 |
| soft_landing | touchdown mean `exp(-(vertical_speed/.5)²)` | .10 |
| foot_slip | `-mean(contact*planar_speed²)` | .20 |
| nonfoot_contact | negative non-foot/floor contact fraction | .20 |
| pose_hip | group negative MSE around `q_stance` | .20 |
| pose_knee | group negative MSE around `q_stance` | .30 |
| pose_ankle | group negative MSE around `q_stance` | .20 |
| pose_waist | group negative MSE around `q_stance` | .10 |
| pose_arm_wrist | group negative MSE around `q_stance` | .05 |
| joint_velocity | negative active-slot mean `qvel²` | .02 |
| joint_limit | negative normalized inner-90% violation MSE | .05 |
| action_magnitude | negative active-slot mean `action²` | .01 |
| action_rate | negative active-slot mean `(action-previous_action)²` | .01 |
| base_angvel_xy | `-(wx²+wy²)` | .02 |

Record every raw/weighted component and contact state (left/right single, both, none), airtime,
touchdown, slip. Unit test constructs same speed/pose with phase-matched alternating contact and static
both-foot contact and asserts alternating reward > static。测试必须以 `control_hz=50` 穷举一个完整
`0.8 s` 周期的 40 个 phase sample，并断言 moving 分支中 static-both 的
`phase_gait + out_of_phase_double_support + stand_support` 加权周期均值 `<=0`；当前 threshold/weight
下该值为 `0.275-0.315+0=-0.040`。同时断言 phase-matched alternating 的同一 subtotal 严格为正且
更高，防止“静态双支撑仍有净 contact bonus”。Any weight change creates a new reward version and config
SHA。`clearance` 只统计 desired swing 且实际未接触的脚；分母为零时 raw value 为零，不得奖励双脚
静止。

课程参考已逐页核对：

- `/home/admin1/workspace/proj/unitree_rl/doc/实践2：设计感知与动作空间，实现宇树G1粗糙地形行走策略7.12版.pdf`
  页 6–8：relative observation、joint-position action、air-time/clearance 和 termination；
- `/home/admin1/workspace/proj/unitree_rl/doc/实践4：蹲姿行走策略，G1速度+骨盆高度MDP设计.pdf`
  页 6–8、21–23：Task/Style/Reg/Penalty、body-frame tracking 和真实 contact；
- `/home/admin1/workspace/proj/unitree_rl/doc/实践3：人形机器人动作空间 HoST Sim2Sim 部署.pdf`
  页 4–6、10、12–14：仅用于区分 get-up current-q incremental 与 walking stance residual。

本地 G1 `feet_gait`/config 只提供可审计的设计参考，不复制其代码、checkpoint 或整套权重。Task072
保持 `q_stance` residual，不采用 HoST current-q increment，不采用课程中的 command curriculum。

## Review

通过条件：每个 reward component 可独立复算且 finite；pose/contact 信号与 G1 semantic slot 和
实际 foot contact 一致；`-200` 已移除；zero-action、untrained 和短 PPO rollout 未出现 NaN、奖励
爆炸或明显 reward hacking。仅改权重而仍无 pose/contact 结构不通过本 subtask。

历史 v1 曾通过当时 unit contract。G1 reward_config SHA
`fe9b6992cfcda4b2af69723e7147836d152bd2544085c4ff267b910f8b388ee9`；Go2 reward_config SHA
`0edf81478735de189bf4a1d87ce8304b5c1fd2488b8662310e9eaa2b0a2e44f9`。

R1 状态：**passed**。v2 component sum/finite、actual MuJoCo foot contact、alternating>static phase
test、pose semantic group coverage、legacy default 回归与 no-fall-penalty 均通过。后续 G1 pilot 已暴露
训练质量失败，但不是 002 reward contract 缺失被包装为通过；003 继续保持 failed/not_passed。
