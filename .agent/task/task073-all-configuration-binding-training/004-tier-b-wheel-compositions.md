# 004 — Tier B wheel compositions

## Route

1. 对 G1、PM01、Spot、Go2、Lite3 的已绑定 non-wheel center 组合本地 terminal wheel module，保持
   原 motor accounting，再显式增加 2 个 biped wheel 或 4 个 quadruped wheel motors。
2. 独立绑定 wheel mass/COM/inertia、radius/width、contact friction、axis/frame、continuous
   transmission 与 torque/velocity-compatible motor tuple；不得复用 position-joint action semantics。
3. 先通过 compile/reset、clearance/contact、paired wheel response、canonical root 与 active-balance
   gate，再在固定 `0.5 m/s`、无 randomization、无 curriculum 下训练 nominal wheel locomotion。
4. 每个 case 独立生成 checkpoint progression、paired baselines、20 s evaluation、8 s 视频和
   verifier；视频必须能审计 wheel rotation、ground contact、body balance 和 forward progress。

## Log

- Task070 wheel cases只有 actuator smoke，不证明 active balance 或 wheel locomotion。
- 2026-08-27：五个 wheel composition 的 nominal training 尚未执行。

## Code implementation

在 `physical_binding.py` 增加 `WheelMotorBinding`，并在 Task073 derived XML overlay 中把 wheel slot 的
position actuator 替换为 MuJoCo velocity actuator；non-wheel actuator 不变。每个 wheel record 固定包含
`slot, radius, width, local_axis, parent_frame, effort_limit, velocity_limit, kv, source_role,
source_sha256`。若 source 只有 local engineering module，必须明确该 role，不能标 vendor parity。

`WholeBodyMuJoCoShardConfig` 新增 optional `actuation_mode_by_slot`；默认全 position。Task073 新增
`ContinuousWheelAdapter.target(action, binding)`，计算
`omega_target = clip(action,-1,1) * velocity_limit`，velocity actuator `forcerange` 固定为
`[-effort_limit,+effort_limit]`。`_set_targets()` 按 slot mode 分支；不得对 wheel 计算 position midpoint。
同时扩展 Task072 已有的 external-model validator：未声明 mode 时继续逐 actuator 验证原 position
semantics；声明为 wheel velocity 时验证 joint transmission、gear、velocity gain/bias、ctrlrange 和
forcerange，并拒绝 position/velocity mode 与 XML 不一致。wheel 的 stance control 固定为零角速度，
non-wheel stance control 仍来自 `stance_solution.actuator_ctrl`；stance artifact 与 derived wheel XML
绑定同一 SHA。

五个 exact case id 为 `unitree_g1_wheeled`、`engineai_pm01_wheeled`、`spot_base_wheeled`、
`unitree_go2_wheeled`、`deeprobotics_lite3_wheeled`。pipeline 先验证 2/4 wheel axis/sign、连续旋转、
饱和、ground contact 和 zero-command hold，再用 root velocity tracking + upright/tilt + wheel slip/
effort regularization 训练 active balance；锁 wheel 或把 body 滑动当位移必须失败。

新增 frozen `Task073WheelRewardConfig`，version 为 `task073_wheel_velocity_v1`。单环境 raw/weight 固定为：

| component | raw value | weight |
| --- | --- | ---: |
| `track_xy` | `exp(-||v_xy-[0.5,0]||^2/0.25)` | `2.0` |
| `track_yaw` | `exp(-wz^2/0.25)` | `0.5` |
| `upright` | `clip(-gravity_z,0,1)` | `0.5` |
| `height` | `-((z-z_stance)/0.10)^2` | `0.25` |
| `wheel_support` | mean wheel/floor contact | `0.20` |
| `rolling_error` | `-mean(contact*(v_roll-radius*omega)^2)` | `0.20` |
| `lateral_slip` | `-mean(contact*v_lateral^2)` | `0.20` |
| `nonwheel_pose` | negative mean squared non-wheel `(q-q_nominal)` | `0.10` |
| `wheel_effort` | negative mean squared normalized wheel force | `0.01` |
| `action_rate` | negative active mean `(action-previous_action)^2` | `0.01` |
| `nonload_contact` | negative non-wheel/non-foot floor-contact boolean | `0.20` |
| `base_angvel_xy` | negative `wx^2+wy^2` | `0.02` |

`v_roll`/`v_lateral` 必须从 wheel local axis、canonical ground normal 和 MuJoCo contact-point velocity
构造；不能用 root velocity 代替。总 reward 是 weighted sum，fall 只 termination、无 terminal penalty。
每个 raw/weighted component、wheel contact、omega 和 slip 写 metrics；config 与 mapping SHA 写
`nominal/reward_config.json`。测试用解析场景覆盖正确 rolling、侧滑、锁 wheel、悬空 wheel、axis sign
和 component sum。

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q tests/test_task073_wheel_adapter.py
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" nominal --case unitree_g1_wheeled
```

其余四个 case 使用相同命令替换 `--case`；每个输出独立 binding/nominal/video/verifier。默认 position
path、Task070 frozen XML 与 Task072 G1/Go2 replay 必须回归不变。

## Review

通过条件：5/5 wheel cases 分别通过 physics/action/active-balance、numerical、video 和 verifier gate。
静态 stance、锁 wheel、脚端滑行或由 non-wheel center pass 推断 wheel pass 均不接受。
