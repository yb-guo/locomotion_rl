# 003 — G1 fixed-command nominal training

状态：**failed / not_passed**。`nominal_v3` G1 pilot 已失败；不得启动 proof、004 或 005。

## Route

1. 锁定 exact-bound anonymous G1、flat terrain 和命令 `vx=0.5 m/s, vy=0, yaw=0`。
2. 显式关闭质量、COM、惯量、摩擦、电机、delay、push、sensor、terrain 和 command
   randomization；关闭 standing-to-walk、速度范围或 reward curriculum。
3. 使用 001 的逐 slot action scale 与 002 的 biped reward，从随机初始化 PPO 训练。先做 bounded
   progression 检查；只有 survival/tracking 曲线有改善且无 NaN/OOM 才扩大到足以证明行走的预算。
   训练预算不足不能被报告为资产不可训练。
4. 保存定期 checkpoint、训练曲线、seed、transition count、wall time、RTX 5060 Ti VRAM、完整
   config 与 source SHA。正式 checkpoint 必须从磁盘重新加载后评估。
5. 对 selected checkpoint、untrained policy 和 zero action 使用相同 reset seeds/command 运行
   paired evaluation；生成 20 s numerical report 与 8 s/400-frame headless MuJoCo 视频。
6. 由 agent 打开视频检查 upright alternating gait、forward progress、足端接触和无拖地/漂移；
   视觉失败即保留失败，不允许只凭 aggregate metrics 通过。

## Log

- 旧 G1 runs 的 256k–1.024M transitions 均快速跌倒；官方 G1 Task048 成功基线约为 63.9M
  transitions。该差异仅用于预算判断，不允许复用 Task048 checkpoint。
- 2026-08-27：新 action/reward contract 下的 nominal G1 training 尚未执行。
- 2026-08-27：完成 G1 `smoke` stage：4 envs、32 rollout steps、2 updates、256 transitions，
  从随机初始化保存 `initial.pt`，最终 `run_manifest.json` 写出；pytest 两份指定测试文件全量为
  25 passed。
- 2026-08-27：完成 G1 `pilot` stage：32 envs、64 rollout steps、1000 updates、2,048,000
  transitions，从随机初始化训练并保存 checkpoint progression。20 s paired diagnostic eval：
  selected zero-fall ratio `1.0`，planar velocity error `0.4972`，yaw error `0.2210`，gravity XY
  `0.5282`，forward displacement `0.2094 m`；untrained planar error `0.4550`，zero-action planar
  error `0.4611`。pilot 不满足 velocity-improvement 晋级条件，也不满足 Task072 planar/gravity/video
  gate，因此未启动 proof。
- 2026-08-27：生成 pilot diagnostic `walk.mp4`、`walk.json`、`eval.json`、`case_verifier.json`、
  `pilot_gate.json` 与 `agent_visual_observation.json`。视频 sidecar `render_passed=false`，
  8 s forward displacement `0.1762 m`；人工视觉检查为 not passed。
- 2026-08-27：nominal_v3 G1 smoke completed。命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py train --case
  unitree_g1 --stage smoke --run-dir
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/unitree_g1/smoke`。结果
  2 updates、256 transitions、`fall_count=0` both updates。Current-source smoke artifact SHA：
  `run_manifest.json` `6c71b2e21fc79ba1317598cc61dc14ac47492f6f4051a75a5c27156dc9daa612`，
  `progression.json` `9277e6bf3508ca4c2af0a5ef83664a312fc90aaf0b22b79c20079622f60f0fb3`。
  Config records `learning_rate=0.0001`、`entropy_coef=0.01`、`target_kl=0.01`、all randomization false、
  motor_fault false、observation transform `task072_relative_scaled_observation_v1`。两份指定测试为
  `28 passed`。
- 2026-08-27：nominal_v3 G1 pilot training completed。命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py train --case
  unitree_g1 --stage pilot --run-dir
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/unitree_g1/pilot`。
  Progression: 1000 updates、2,048,000 transitions；printed checkpoints update 200/400/600/800/1000
  reward_mean `1.0493/1.1464/1.1994/1.0428/1.2227`，fall_count `22/22/23/18/21`。Artifact SHA:
  `run_manifest.json` `f85a98575463151247d9753f9c31d5e2958d9b98868085bf81a78509e991b109`；
  `progression.json` `9850923d34721506478a75f76dbf20288c482fb4b0f589b0d4f80992e85d3f10`；
  `final.pt` `73e1657e604fa8c0bef1350d33e4c442872d8fbcbb643f6d68a3cb5bc43c24b3`。
- 2026-08-27：nominal_v3 G1 pilot video/render failed。命令：
  `env MUJOCO_GL=egl PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py render --case
  unitree_g1 --run-manifest
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/unitree_g1/pilot/run_manifest.json
  --output .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/unitree_g1/pilot/walk.mp4
  --diagnostic-output
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/unitree_g1/pilot/diagnostic_rollout.json`。
  `walk.mp4` SHA `7b7f8a56dc688f1af10e92f427a9b65db6364a0eaf05b3a24f3ff8c8e6e82990`；
  `walk.json` SHA `2f7324b25848e0984b7c549aae3d37fdb7519af7311ffd76f0c580aaca91bc5e`；
  `diagnostic_rollout.json` SHA `4bc0a82294dc1f1fe6a0dce314db4aa4d2271737fc6c6e4fe696baa876500809`。
  Sidecar `render_passed=false`，fall_count `3`，done_count `3`，8 s forward displacement `0.1341 m`。
  Diagnostic summary: left/right noninitial touchdowns `0/0`，alternating transitions `0`，
  left/right single support `0.000/0.015`，both-contact fraction `0.985`，no-contact `0.000`，
  target/actual clamp fraction `0`，mean projected-gravity XY `0.1088`。
- 2026-08-27：agent opened `walk_contact_sheet.png` generated from the pilot video. Visual check failed:
  no stable forward walking, no left/right alternating swing-touchdown gait, mostly persistent double support/
  dragging, and visible late-trial collapse. `agent_visual_observation.json` SHA
  `8cddacb41cf4eed830e623071845a90a20adce987b9b7c25fe4a9e989cbeee6d`；contact sheet SHA
  `81dd05d6641e21b685b770a253b85cbcfe427f7508fab84621b64f46a6d66927`。
- 2026-08-27：nominal_v3 G1 paired eval and verifier failed closed。Eval command produced
  `eval.json` SHA `5a87ee29d7ecde030510794c039b7287b529d3f6830d2de0bd96ff262e5cd80f` and
  `eval_trace.json` SHA `4eaae902c11746e606c39fe31b4fe40f7d50f8a226c7e209b80601857109ac7d`。
  Selected policy: zero_fall_ratio `0.0`，planar error `0.3775`，yaw error `0.1108`，gravity XY
  `0.1143`，forward displacement mean `0.5862 m`，nonfall displacement `null`，first-trial length
  `120` steps for all 20 envs。Update0 baseline planar error `0.4158`，zero-action baseline planar error
  `0.4227`；common-prefix margins update0/zero `0.0557/0.0917` pass，full-horizon displacement margins
  update0/zero `0.1911/0.1925 m` fail the `2.0 m` comparison gate。Eval failure reasons:
  `zero_fall_ratio_below_threshold, planar_velocity_error_above_threshold,
  nonfall_forward_displacement_not_finite, zero_action_forward_displacement_margin_below_2m,
  update0_forward_displacement_margin_below_2m, video_verified_false`。
  `case_verifier.json` SHA `cc8cb77d7e33ae33f857f68f47796732335d568ec638adbc028aeb6c919cc586`，
  `pilot_gate.json` SHA `fa43ce3485427afa33c1c9dfe2de1afa06c4f8126a652856485310502ebb34d8`。
  Pilot gate failed reasons: `touchdowns, alternating, single_support, contact_fractions, kl,
  survival_improved, diagnostic_motion`。PPO attempted-minibatch KL mean/p95/max
  `0.00744/0.01534/0.03858`，clip fraction mean/p95 `0.1056/0.2461`，early-stop fraction `0.886`。
  Per contract, proof/004/005 were not started。
- 2026-08-27：final focused validation command
  `env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src"
  /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python -m pytest -q
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py` returned `28 passed`。
  A GPU-visible rerun immediately after pilot hit CUDA OOM in torch Adam health check, so the final
  assertion validation used CPU-visible execution.

## Code implementation

### Stage contract

在 task-local CLI 新增 `Task072TrainStageConfig` 与唯一入口 `train --stage`。三个 stage 都从随机权重
开始，不接受 `--resume`/外部 checkpoint，command ranges 三轴均退化为 `(0.5,0.5)`、`(0,0)`、
`(0,0)`，并传 `MotorProcessConfig(no_event_probability=1.0)`。manifest 必须断言 mass/COM/friction/
motor/terrain/sensor/push randomization 与 curriculum 全为 false。

| stage | envs | rollout steps | updates | transitions | train seed | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke` | 4 | 32 | 2 | 256 | 72072 | finite forward/backward、parameter delta、无 OOM |
| `pilot` | 32 | 64 | 1000 | 2,048,000 | 72072 | 判断 action/reward 是否产生生存与 tracking 改善 |
| `proof` | 32 | 64 | 31,200 max | 63,897,600 max | 72072 | 与已知 official-G1 success budget 同量级；每 200 updates checkpoint |

每个 stage 在第一次 optimizer update 前都保存随机初始化的 `initial.pt`；proof 的 `initial.pt` 是正式
untrained baseline，checkpoint manifest 绑定 seed、policy config 与 source SHA。pilot 晋级条件固定为：
最后 25% rollout 的 median first-fall time 至少比 update-0 baseline 高 `1.0 s`
且不低于 `3.0 s`，planar velocity error 至少改善 `0.05 m/s`，所有 loss/gradient/observation/reward
finite。不满足就停止并记录 action/contact/reward diagnostics，不启动 proof。proof 每 1000 updates
对当前 checkpoint 跑 20 s/20-env gate；首次全 gate 通过即可停止，否则到 max transitions 后失败退出。

正式 eval 使用 seed `172072` 创建 20 个确定性环境（第 `i` 个环境 seed 固定为
`172072 + 1009*i`），render seed 固定 `272072`。`eval` 必须在三个全新 shard 上依次运行
`selected`、`untrained`、`zero_action`，每组都从完全相同的 20 个 reset state 开始：selected 从磁盘
加载 proof checkpoint 并取 deterministic policy mean；untrained 从磁盘加载 proof `initial.pt` 并取
deterministic mean；zero_action 每步写全零 action。三组 initial `qpos/qvel/command` 的逐 env SHA 必须
相等，否则 pairing fail closed。

`eval.json` schema 固定为
`{schema_version,case_id,lineage,horizon_s,num_envs,eval_seed,pairing,
policies:{selected,untrained,zero_action},comparisons,gate}`。每个 policy record 含 kind、checkpoint SHA
（zero 为 null）、20 个 episode rows、aggregate survival/fall/velocity/yaw/gravity/displacement metrics；
`pairing` 含 env seeds 和 initial-state SHA；同目录 `eval_trace.json` 保存下面 R1 定义的逐步原始序列，
`eval.json` 必须绑定其相对路径、raw SHA 和 canonical payload SHA。除主数值 gate 外，selected
full-horizon mean forward
displacement 必须分别比两个 baseline 至少多 `2.0 m`；planar velocity learning margin 使用下面 R1
定义的逐环境 common-alive-prefix error，必须分别至少低 `0.05 m/s`。否则证明
“训练产生行走”的 comparison gate 为 false。case verifier 必须自己重新构造三组 shard、重放磁盘
checkpoint 并复算，不能相信 report 内的 gate 布尔值。

正式 proof 的实现顺序必须是 train -> render -> eval（eval 绑定 video sidecar）-> case verify。以下
proof 命令只能在本文件末尾的 R1 pilot block 全部通过后执行，不得整段无条件连跑：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK072=.agent/task/task072-bound-g1-go2-locomotion-proof
CLI="$TASK072/task072_locomotion_proof.py"
ROOT_OUT="$TASK072/artifacts/nominal_v3/unitree_g1"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" train --case unitree_g1 --stage proof --run-dir "$ROOT_OUT/proof"
env MUJOCO_GL=egl PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" render --case unitree_g1 \
  --run-manifest "$ROOT_OUT/proof/run_manifest.json" --output "$ROOT_OUT/walk.mp4"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" eval --case unitree_g1 \
  --run-manifest "$ROOT_OUT/proof/run_manifest.json" --video-sidecar "$ROOT_OUT/walk.json" \
  --output "$ROOT_OUT/eval.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" verify-case --case unitree_g1 \
  --report "$ROOT_OUT/eval.json" --output "$ROOT_OUT/case_verifier.json"
```

新增 budget/config/seed/no-randomization/progression/early-stop tests。agent 必须打开 `walk.mp4`，并写
`agent_visual_observation.json`，字段固定为 `agent_visual_check_passed, upright_alternating_gait,
forward_progress, foot_contact_plausible, dragging_or_skating, observation`；只有数值与视觉都通过才进入
004。

## Review

通过条件：G1 同时满足 Task072 主任务的 20 s 数值 gate、positive displacement、checkpoint
progression、paired baselines、8 s 视频和 agent visual check。任何 randomization/curriculum 非空、
使用旧 checkpoint、视频缺失或 verifier 只读取 report 而不重放 checkpoint，均判定失败。

当前 nominal_v3 状态：**failed / not_passed**。失败发生在 G1 pilot 的 video、20 s paired eval 和
pilot verifier gate，故按路线停止，未执行 proof、004 或 005。

历史 nominal_v2 状态：**failed / not_passed**。失败发生在 pilot 晋级 gate，故按路线停止，未执行 proof、004 或
005。

## R1 repair contract

状态：**failed / not_passed**；nominal_v2 rejected，nominal_v3 repair 已实施但 G1 pilot 未通过。Owner：training/eval owner。
在 `WholeBodyMuJoCoShardConfig` 新增三个 optional/default-compatible 字段：
`observation_joint_reference_by_slot: Mapping[str,float] | None = None`、
`observation_joint_velocity_scale: float = 1.0`、
`observation_base_angular_velocity_scale: float = 1.0`。`_observation()` 在调用 whole-body encoder 前
应用变换；Task072 G1 传入完整 `stance_solution.joint_qpos` mapping、`0.05`、`0.2`，得到同维
`q-q_stance`、`qvel*0.05`、`base_angvel*0.2`。base linear velocity 保持当前 schema。默认
`None/1/1` 必须数值等价于 legacy；不加 noise、history、phase observation 或新网络。manifest 记录
version `task072_relative_scaled_observation_v1`、mapping/config SHA。API test 断言 slot coverage、维度、
数值变换与 legacy identity。

在 `WholeBodyPPOConfig` 新增 `target_kl: float | None = None`。新增 frozen
`PPOMinibatchDiagnostics(epoch, index, approx_kl, clip_fraction, applied)`；`PPODiagnostics` 新增
`early_stopped, minibatches_attempted, minibatches_completed, epochs_completed, learning_rate,
minibatches: tuple[PPOMinibatchDiagnostics,...]`。对每个 minibatch，在 backward/optimizer step 前固定
计算 `log_ratio=new_log_prob-old_log_prob`、`ratio=exp(log_ratio)`、
`approx_kl=mean((ratio-1)-log_ratio)`、
`clip_fraction=mean(abs(ratio-1)>clip)`。每次尝试都记录一行；已经执行至少一个 minibatch 且当前
`approx_kl > 1.5*target_kl` 时，该行 `applied=false`，不 backward/step，并停止其余 epoch。其余已
step 行为 `applied=true`。`target_kl=None` 时 optimizer 次数、顺序和数值行为保持原样，只增加
telemetry。R1 target=`.01`、initial lr=`1e-4`、entropy=`.01`；其余变化必须逐项 ablation 且各版
记录 config SHA。

`progression.json` 每个 update row 必须原样保存上述完整 minibatch records，以及 reward components、
contact 四状态/airtime/touchdown/slip、45D active action 分布、逐 slot target
would-clamp/actual-clamp、value/grad。pilot KL/clip sample set 是 1000 个 update row 中全部 attempted
minibatch records 展平后的 finite 值：mean 为算术平均，p95 固定用 nearest-rank
`sorted(values)[ceil(.95*N)-1]`，max 为最大值；early-stop fraction 是
`count(update.early_stopped)/1000`。verifier 必须从这些 raw records 重算 gate，不得读取预先写好的
aggregate 布尔值。缺 row、空 sample、`applied` count 与 `minibatches_completed` 不等或 aggregate
不一致均 fail closed。

Pilot 固定 2,048,000 transitions，fail-closed 条件为：8 s 每脚至少 4 次非初始 touchdown，压缩连续
contact 后的 left/right alternating touchdown transitions 至少 6 次；left/right single-support
fraction 均 `>=.05`、both-contact fraction `<=.70`、no-contact fraction `<=.20`；正常 target
would-clamp 与 actual-clamp fraction 均 `<=1e-6`，且任一 active slot 都不得高于该阈值；
attempted-minibatch approximate KL mean `<=.015`、p95 `<=.03`、max `<=.05`，attempted-minibatch
clip fraction mean `<=.20`、p95
`<=.35`，KL early-stop update fraction `<=.50`；selected 相对每个 baseline planar
velocity error 至少改善 .05 m/s；median first-fall 比 update-0 至少改善 1.0 s 且≥3.0 s；所有
loss/gradient/observation/reward finite，8 s forward displacement `>=.50 m`、mean projected-gravity
XY `<=.45`。agent 必须实际打开 8 s 视频，确认 torso/pelvis 向前移动、左右腿交替、存在真实 swing
和 touchdown、无持续拖滑；视觉失败等同 pilot 失败。任一失败停止，不跑 proof。

8 s `diagnostic_rollout.json` 固定使用 selected deterministic policy、render seed `272072` 和 400 个
control step；每步保存 `step, trial_done, foot_contact[2], touchdown[2], foot_air_time[2],
foot_height[2], foot_planar_speed[2], canonical_root_world_x, projected_gravity_xy[2],
active_action, unclamped_target_by_slot, ctrl_target_by_slot, target_would_clamp_by_slot,
actual_clamp_by_slot`。非初始 touchdown 是同一 trial 内 previous contact=false、current=true 且
`step>0` 的 rising edge；reset 边界不计。只取“恰有一只脚 touchdown”的事件并按 step 形成 L/R
序列，collapse 相邻同标签后，`alternating_transitions=max(len(sequence)-1,0)`；同时 touchdown 不计入
交替数。contact fraction 分母固定 400；global clamp 分母为 `400*active_slot_count`，per-slot 分母
固定 400。report aggregate 必须由这些 rows 复算，视频 sidecar 绑定同一 rollout seed、checkpoint
SHA 和 diagnostic raw/payload SHA。

Eval pairing 仅在 selected/untrained/zero_action 三组每个 env 的 initial qpos、qvel、command SHA
全部相等时为 true，否则 false 且 verifier fail。initial-state SHA 固定为现有 `payload_sha256()` 对
`{"qpos": qpos.tolist(), "qvel": qvel.tolist(), "command": command.tolist()}` 的 canonical JSON hash，
三组必须先从磁盘各建全新 shard，再逐 env 比较 hash。

`eval_trace.json` 固定保存三个 policy、20 env、1000 control step 的 first-trial raw arrays：每个
policy/env record 含 `seed, initial_state_sha256, alive_before[H], trial_done[H],
local_linear_velocity_xyz[H,3], local_yaw_velocity[H], projected_gravity_xy[H,2],
canonical_root_world_x[H]`。env 首次 terminal 后不 reset；余下 `alive_before=false`，terminal root 值
保持到 H，仅用于 fixed-horizon displacement，不能进入 alive-prefix velocity。令 `n_policy,e` 为
该 record 从 step 0 起 `alive_before=true` 的连续 sample 数，范围 `[1,H]`；terminal transition 因
step 前仍 alive 被包含，无 terminal 时 `n=H`。对 selected 与每个 baseline、每个 env 定义
`k_e=min(n_selected,e,n_baseline,e)`，逐步 planar error 为
`sqrt((vx-0.5)^2+vy^2)`，comparison 为
`mean_e(mean(error_baseline[e,0:k_e])-mean(error_selected[e,0:k_e]))`，必须 `>=0.05 m/s`。
survival 和 terminal-or-H root-x displacement 另由完整 first trial 报告，不使用 common prefix。
`eval.json` 由 trace 派生；case verifier 一方面从 trace raw rows复算全部数字，另一方面新建 shards
重放磁盘 checkpoint，要求 pairing、gate 和 aggregate 在 `abs_tol=1e-6` 内一致。v3 产物不得覆盖 v2。

R1 pilot 命令顺序固定为 train -> render -> eval -> verify-case；不能只训练后凭 progression 晋级：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK072=.agent/task/task072-bound-g1-go2-locomotion-proof
CLI="$TASK072/task072_locomotion_proof.py"
ROOT_OUT="$TASK072/artifacts/nominal_v3/unitree_g1"
PILOT="$TASK072/artifacts/nominal_v3/unitree_g1/pilot"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" train --case unitree_g1 --stage smoke \
  --run-dir "$ROOT_OUT/smoke"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" train --case unitree_g1 --stage pilot --run-dir "$PILOT"
env MUJOCO_GL=egl PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" render --case unitree_g1 \
  --run-manifest "$PILOT/run_manifest.json" --output "$PILOT/walk.mp4" \
  --diagnostic-output "$PILOT/diagnostic_rollout.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" eval --case unitree_g1 \
  --run-manifest "$PILOT/run_manifest.json" --video-sidecar "$PILOT/walk.json" \
  --output "$PILOT/eval.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" verify-case --case unitree_g1 \
  --report "$PILOT/eval.json" --diagnostic "$PILOT/diagnostic_rollout.json" \
  --pilot-gate-output "$PILOT/pilot_gate.json" --output "$PILOT/case_verifier.json"
```

## R2 single-variable experiment contract

状态：**planned / not_run**。本节取代 R1 的后续训练路线，但不改写 R1/nominal_v3 的失败证据。
Owner：training/eval owner。R2 artifact root 固定为
`artifacts/nominal_v4/unitree_g1/`；不得覆盖 `nominal_v2` 或 `nominal_v3`。

### Causality and lineage

`nominal_v3/pilot` 冻结为 `E0` 失败基线：193D actor observation 无 phase、每步 raw reward
直接进入 GAE/return、rollout batch 2048、minibatch 256、4 epochs、constant LR `1e-4`、hard
KL stop `target_kl=.01`。其 first-100-update mean value loss/grad norm 分别为
`154.5699113/50.9869969`，KL early-stop update fraction 为 `0.886`；8 s both-contact
fraction `0.985`，左右 noninitial touchdowns `0/0`。

| variant | parent config | 唯一允许的科学 delta | 预算 |
| --- | --- | --- | ---: |
| `E1_phase` | E0 | shared policy observation 追加 phase sin/cos | 2,048,000 |
| `E2_reward_dt` | E1 | 最终 scalar reward 乘 control `dt` | 2,048,000 |
| `E3a_adaptive_kl` | E2 | hard stop 改为 adaptive-LR optimizer strategy | 2,048,000 |
| `E3b_large_minibatch` | E2 | 仅在 E3a 失败时使用 large-minibatch strategy | 2,048,000 |
| `E4a_roll_authority` | selected E3 | 只验证/必要时改变四个 hip/ankle-roll residual bounds | diagnostic + optional 2,048,000 |
| `E4b_contact_geometry` | selected E3 | 只验证/必要时改变 foot collision geometry | diagnostic + optional 2,048,000 |
| `progression_20m` | selected E4 integration | 配置冻结；只增加预算 | 20,000,768 |

“依次”表示后一项以已接受的前一配置为 parent，不表示从前一项 checkpoint 续训。E1/E2/E3 和任何
E4 training ablation 都必须用 train seed `72072` 从随机初始化开始；禁止 `--resume`、外部
checkpoint 和 Task048 checkpoint。除 E1 必然改变 actor/value first-layer input shape 外，相邻实验的
初始化 tensor SHA 必须相同；E1 使用相同 seed/initializer，并把预期 shape/SHA 差异限制到
actor/value first layers。
每个 manifest 必须写
`variant_id,parent_variant,parent_config_sha256,delta_allowlist,source_sha256,initial_checkpoint_sha256`，
verifier 对 canonical config diff 做 allowlist 检查。共同不变量仍为 fixed command
`(0.5,0,0)`、flat terrain、相同 reset/eval/render seeds、无 randomization/curriculum、相同网络
hidden layers、GAE/PPO loss 系数和 exact-bound asset；未声明的差异一律 fail closed。

每个 2.048M training ablation 都生成完整 progression、20 s paired eval、8 s contact diagnostic、
视频和 agent visual observation。此阶段的用途是归因，不要求在 2.048M 直接通过 full locomotion
gate。只有下述对应 mechanistic gate 通过，才可进入下一项。

### E1 — Task048-style phase observation

只在 Task072 G1 shared policy observation 尾部追加
`[sin(2*pi*phase), cos(2*pi*phase)]`，observation schema 从 193D 变为 195D。phase 使用当前
Task072 reward 已有的 per-trial clock：
`phase = ((trial_step * control_dt) % gait_period_s) / gait_period_s`；`gait_period_s` 保持
当前 `0.8 s`，不能顺手改为 Task048 的 `0.6 s`，也不能改变 reward phase、reward weights、
action、PPO 或预算。observation 与 reward 必须调用同一个 phase helper、使用同一个
pre/post-step timing，不能各自维护 counter。这里复用的是 Task048 的表示形式，不是其 checkpoint。

测试必须断言 reset 和 quarter-cycle 分别为 `(0,1),(1,0),(0,-1),(-1,0)`，phase 有界且 finite，
原 193 个 observation 元素逐值不变，manifest 记录 clock source/period/schema。E1 correctness
通过后保留 phase 进入 E2；2.048M 的 survival、contact、touchdown、tracking 差异只作为 effect
report，不因尚未形成行走而删除 phase。

### E2 — reward/return scale

只做一次 scaling：先按 E1 原公式求出 raw reward components 和 `raw_total_reward`，再令
`ppo_reward = raw_total_reward * control_dt`，其中 `control_dt=0.02`。GAE 的 gamma/lambda、
advantage normalization、value coefficient、global grad clip `1.0` 和所有 component weight
保持不变；不得同时做 return normalization、reward clipping 或逐 component 重调。progression
同时保存 raw/scaled reward、return/GAE target mean/std/p95、value prediction mean/std，以及
clip 前 grad norm。

E2 scale gate 使用前 100 updates，从 raw rows 重算，必须同时满足：

- 所有 reward/return/value/loss/gradient finite；
- mean value loss 至少比 E1 parent 降低 `10x`，且绝对值 `<=15.4569912`；
- mean pre-clip grad norm 至少比 E1 parent 降低 `5x`，且绝对值 `<=10.1973994`；
- scaled/raw total reward ratio 在每个非零 sample 上等于 `0.02`（`abs_tol=1e-7`），零 sample
  两者同时为零。

未通过 scale gate 就停止，不进入 E3；不能用“gradient 已被 clip 到 1”替代 clip 前 grad norm 证据。

### E3 — remove hard KL stop

主分支 `E3a_adaptive_kl` 禁用 hard KL early stop，保持 batch 2048、minibatch 256、4 epochs、
initial LR `1e-4`，使用 Task048/RSL-style adaptive schedule，`desired_kl=.01`：用同一批
Gaussian policy 的 mean/std 计算 scheduler KL；KL `>.02` 时 LR 除 `1.5`，KL 在
`(0,.005)` 时 LR 乘 `1.5`，LR clamp 到 `[1e-5,1e-2]`。每次 scheduler decision、LR
before/after、scheduler KL 和现有 sampled approximate KL 都写入 raw progression。
E3a 的 semantic single delta 是 optimizer strategy replacement，config diff allowlist 仅含
`hard_kl_stop,schedule,desired_kl`。

E3a gate 要求每个 update 都完成全部 4 epochs/32 minibatches、hard-stop fraction 精确为零、LR
始终在 bounds 内，并继续满足 attempted-minibatch approximate KL mean/p95/max
`<=.015/.03/.05`、clip fraction mean/p95 `<=.20/.35`。不满足时 E3a rejected，不得直接混加
batch change。

只有 E3a rejected 才允许从 E2 另起 sibling `E3b_large_minibatch`：禁用 hard stop，constant
LR 仍为 `1e-4`，只把 minibatch `256 -> 1024`；不启用 adaptive LR。E3b 使用同一 KL/clip
gate 且每 update 完成 4 epochs/8 minibatches。E3a/E3b 只能择一进入 E4；两者一起使用属于新的
integration experiment，不能记为这里的单变量结果。
E3b 的 semantic delta 同样是 optimizer strategy replacement，config diff allowlist 仅含
`hard_kl_stop,minibatch_size`。

### E4 — roll authority and contact geometry

先做两个互不依赖的 no-update diagnostic，二者都从 selected E3 config/asset 出发：

- `E4a_roll_authority`：对左右 `hip_roll`、`ankle_roll` 逐 slot 做
  `u in {-1,-.5,0,.5,1}` sweep，其他 action 为零；记录 stance-relative requested/unclamped/actual
  target、0.5 s realized excursion、steady-state error、peak actuator force、effort utilization、
  joint-limit/collision-limit 和 target/actual clamp。当前有效 residual bounds
  hip-roll 约 `0.2094/0.2063 rad`、ankle-roll `0.23562 rad`；Task048 scales
  `0.350661/0.438577 rad` 仅作诊断参考，不能直接当成结论。若每个方向均可达、median steady-state
  error `<=20%` commanded excursion、无 clamp/limit violation，则保留 001 contract。否则候选
  variant 只允许改变这四个 residual bounds，必须重跑 001 behavior gate 后再做独立 2.048M training。
- `E4b_contact_geometry`：冻结 policy/action/reward/friction/mass/inertia/joints/actuators，枚举 foot
  geom 并做 heel/toe/inner-edge/outer-edge controlled touchdown；记录 first-contact height、contact
  count/point、normal-force centroid、support polygon、penetration 和 tangential slip。当前 bound
  asset 的每脚单个 wide box 与 Task048 每脚七个 capsule 只是假设差异。若要改 geometry，只能建立
  foot-collision-only asset variant，摩擦仍保持当前值，不能同时调 friction。

E4a 与 E4b 必须是 siblings；如果两项改变都各自有正向证据，再新增 `E4c_integration` 做 smoke
和 2.048M confirmation，不能把 combined run 当作单变量证据。任何 E4b geometry 变化都会改变
exact-bound XML/source SHA，必须回 Task071 重新生成/验证 parent artifact 并更新 Task072 binding；
未完成该重绑定时，E4b 只能是诊断结果，不能进入正式 progression。

### 10M/20M progression and full proof

选定配置冻结后，从随机初始化启动一条 `progression_20m` lineage。32 envs x 64 steps 下，
update `4883` 为 `10,000,384` transitions，update `9766` 为 `20,000,768` transitions；
两点都保存可重放 checkpoint，并用相同 seeds 跑 paired eval、contact diagnostic 和视频。

这里判断的是是否进入 Task048-like 学习区间，不是提前要求 full walk：Task048 的约 9.8M
`model_100` 仍只有 `0.011 m/s`，约 19.7M 的 `model_200` 仍接近零速，但约 14.7M 时 mean
episode length 已从约 64 steps 上升到 300–390 steps。因此 progression gate 固定为：

- 10M/20M 全程 finite、无 OOM，且持续满足 E2 value/grad scale gate 和 selected E3 optimizer gate；
- 20M 最后 10% rollout mean episode length `>=300` control steps，20 s paired eval 的 median
  first-fall `>=6.0 s`；
- 20M median first-fall 相对本 lineage update-0 和 2.048M checkpoint 都至少增加 `2.0 s`；
- 从 10M 到 20M，下列五项至少两项达到改善阈值：median first-fall `+1.0 s`、mean episode
  length `+50 steps`、common-prefix planar error `-0.025 m/s`、both-contact fraction
  `-0.05`、8 s 双脚合计 noninitial touchdowns `+4` 且 alternating transitions `+2`；
- action clamp、collision/contact integrity、lineage/config diff verifier 全部通过。

20M 允许仍未达到速度或 alternating-gait full gate；它只授权扩大预算。progression gate 失败时
不得启动 full proof，也不得据此宣称 asset 不可训练。

progression 通过后，同一条从随机初始化开始的 lineage 才进入 full-proof phase，最大总预算
`63,897,600` transitions（update `31200`）。20M continuation 必须恢复 policy、critic、
optimizer、adaptive-LR state、observation normalizer、MuJoCo qpos/qvel/ctrl、last action、
reward/contact/airtime state、所有 RNG 和 env/trial counters；新增
uninterrupted-vs-resume equivalence test，下一 update 的 tensors/metrics SHA 不一致即 fail closed。
这项 same-lineage continuation 是 R2 selected-run 的特例，不允许用于 E1–E4 消融。full proof
仍执行原 20 s/20-env numerical gate、paired baselines、8 s video、agent visual check 和独立
checkpoint replay verifier；通过后才能进入 004 freeze 和 005 Go2 rerun。
