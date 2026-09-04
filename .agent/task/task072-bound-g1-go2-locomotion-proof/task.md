# Task072 — Bound G1/Go2 nominal locomotion proof

状态：**in_progress / not_passed**。

## 目标与边界

在 Task071 已绑定的 exact anonymous Unitree G1 与 Go2 上，用随机初始化 PPO 证明固定命令
`vx=0.5 m/s, vy=0, yaw=0` 的 nominal forward locomotion。先修正 G1 的动作尺度与 biped
reward；G1 通过后冻结全部代码、配置和资产 lineage，再在完全相同的 source commit 上重跑
Go2、视频和 verifier。

本任务不启动 Task073，不扩展到其余 16 个构型，不启用质量、COM、摩擦、电机、push、sensor
noise 或 terrain randomization，也不使用 command curriculum。Task048 checkpoint 与本地课程资料
只能作为设计/预算参考，不得作为 Task072 初始化权重或 pass evidence。

### R1 diagnosis and reopen

`nominal_v2` 已拒绝。8 s deterministic `final.pt` 重放为 both-foot contact `1.0`，左右脚均无
air-time/gait；总 reward 约 `1.832`。target clamp 为 overall `4.28%`、left/right ankle roll
`71%/21%`、waist pitch `32.25%`。pilot 的 mean/max approximate KL 为 `0.249/8.351`，mean/max
clip fraction 为 `0.614/0.938`。eval 又把 selected 的 `1000` steps 与只存活 `147/160` steps 的
untrained/zero baselines 直接比较，并无条件写 `paired_baselines_verified=true`。这些证据分别重开
001、002、003；旧 v2 unit/smoke 结果只保留为历史记录，不再是当前 pass evidence。

## Subtasks

1. `001-g1-motor-tuple-slot-action-scale.md`
   - 从同一 transmission-resolved coherent motor tuple 与 stance-relative 正/负 joint headroom 推导
     每个 G1 slot 的非对称绝对关节位置残差尺度；
   - 禁止继续使用统一 joint-range fraction，也禁止仅靠运行时 hard clamp 消化过大的 residual；
   - 在冻结前用同一 adapter 加载 Go2 tuple 做 config/no-update smoke，避免冻结后才补代码。
2. `002-g1-mature-biped-pose-contact-reward.md`
   - 换成可分解审计、command-gated 的 biped phase/pose/contact reward；
   - 明确奖励左右交替并惩罚 phase 外双支撑，拒绝“长期双脚接地也拿高分”；
   - 删除当前主导训练的 `-200` 巨大跌倒惩罚，fall 仍作为 termination。
3. `003-g1-fixed-command-nominal-training.md`
   - 固定 `0.5 m/s`，关闭全部 domain randomization 和 curriculum；
   - 加入同维 observation transform、target-KL PPO guard、完整 contact/action diagnostics 和真正
     fail-closed 的 paired eval；
   - 从随机初始化 PPO 先重跑 2.048M pilot；pilot 数值与视频均过才允许 proof。
4. `004-freeze-g1-passing-lineage.md`
   - 只有 G1 全 gate 通过后，冻结实现 commit、配置、descriptor、asset、checkpoint 与 verifier
     SHA；
   - 冻结后不允许为 Go2 改代码。
5. `005-go2-same-lineage-rerun.md`
   - 在冻结的同一代码 lineage 上，从随机初始化重跑 exact-bound Go2；
   - 重新产出 checkpoint progression、paired baselines、20 s eval、8 s 视频和 verifier 结果。

## Code implementation

### Canonical workspace and recovery baseline

- 不在大量脏改动的主工作区实现。接续 worktree 固定为
  `/home/admin1/workspace/run/locomotion_rl/task071-1`，branch 固定为
  `codex/task072-bound-walk-proof`；开始时只读记录 `git rev-parse HEAD`、
  `git branch --show-current` 和 `git status --short`，不得覆盖其中现有 Task071/072 改动。
- 本任务六份 Markdown 的权威契约位于
  `/home/admin1/workspace/proj/locomotion_rl/.agent/task/task072-bound-g1-go2-locomotion-proof/`。执行者在
  第一处代码修改前，用 `apply_patch` 将这六份文件逐字同步到 recovery worktree 的同路径并比较
  SHA；不得把 recovery worktree 的旧 task 文档反向覆盖本契约。这样 004 才能把实际执行契约纳入
  freeze commit。
- 该 worktree 中现有 task-local CLI
  `.agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py` 是恢复基线；
  001–005 在这个文件上做最小修改。当前主工作区缺少该 CLI，不得从主工作区的测试失败反向创建
  第二套实现。
- 依赖解释器固定使用
  `/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python`，在 worktree 根运行时显式传入
  `PYTHONPATH=$PWD/src`；不要使用该 worktree 当前缺 pytest 的 Python 3.14 `.venv`。

### Behavior owners

| owner | 实现责任 |
| --- | --- |
| `src/h200_locomotion_lab/envs/whole_body_mujoco.py` | `WholeBodyMuJoCoShardConfig`、`WholeBodyMuJoCoShard._set_targets` 和 post-step joint/contact metrics；默认 45-slot/scalar 路径必须不变 |
| `.agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py` | parent artifact loader、motor tuple resolver、reward、stage config、train/eval/render/freeze/verify CLI |
| `src/h200_locomotion_lab/algorithms/ppo.py` | optional target-KL early stop 与 PPO diagnostics；`target_kl=None` 必须保持原更新行为 |
| `src/h200_locomotion_lab/training/whole_body_ppo.py` | Task072 PPO config 接线和 rollout reward/contact/action diagnostics；默认 trainer 行为不变 |
| `tests/test_task072_locomotion_proof.py` | task-local action/reward/budget/lineage/CLI contract |
| `tests/test_whole_body_extended.py` | shared environment 的 scalar legacy 与新增 optional path 回归 |

### Frozen inputs and output root

CLI 的 `_load_parent_artifacts()` 必须从
`.agent/task/task071-multimorphology-training-readiness/artifacts/` fail-closed 加载：

- `official_sim_physics_overlay_v1.json`；
- `r1_g1_go2_bound_official_sim_physics_overlay_v1.json`；
- `r2_env_contract_smoke.json`；
- `r3_ppo_update_smoke.json`。

每份 parent 的 raw SHA-256、canonical JSON payload SHA、declared case denominator 和内部引用 XML SHA
都进入 Task072 static lineage；文件缺失、内部 SHA 不闭合或不含 `unitree_g1`/`unitree_go2` 时立即
`ValueError`。

Task071 overlay 不是 Task070 morphology/motor metadata 的替代品。resolver 还必须直接读取以下 frozen
Task070 attempt010 输入；JSON payload SHA 定义为对完整 parsed object 执行
`json.dumps(payload, sort_keys=True, separators=(",", ":"))` 后取 SHA-256。Task071 对应 record 的
`frozen_input` path/raw SHA 必须与表中值逐项相等：

| case / input | attempt010 相对路径 | raw SHA-256 | canonical payload SHA-256 |
| --- | --- | --- | --- |
| G1 descriptor | `unitree_g1_seed000/unitree_g1_29dof_structural_descriptor.json` | `6464ad8af464956ca8c722a95fddd94b7183c0cdd153134b0cbda12f6199662e` | `cd16bbb3bea241eaec802dbcd7ad4b25550d90246d2816e7c7d23c8f2b453855` |
| G1 manifest | `unitree_g1_seed000/unitree_g1_29dof_anonymous_preview_manifest.json` | `fcb581ac1feb5454bebf7251098548f10648f9f478160adfee0fa764b3405967` | `7d1641f79f1ae72cbddc0e355a4af64d4154da8b5996f8ec3c744d49f0a07f99` |
| G1 XML | `unitree_g1_seed000/unitree_g1_29dof_anonymous_preview.xml` | `35f6e56eb17b018fa1288db6f74eb8c42fc6616c599008c5050a6af8805120f1` | n/a |
| Go2 descriptor | `unitree_go2_seed000/unitree_go2_12dof_structural_descriptor.json` | `795fd0549643cf96ca83385d0c67ba7fb68485b074c16f610a4c197179e82bac` | `09ed1b69922019213d21f6ee8144e64aca688cbee4968952ab58d16e2e016fd1` |
| Go2 manifest | `unitree_go2_seed000/unitree_go2_12dof_anonymous_preview_manifest.json` | `a7afd7b32706c27d276c1b71dc527d05ac3c3fede16edde32b6152633169f398` | `c5f1c10b165fd399026f854154f4a012841c75fd90a55aae79a0d002988d35c0` |
| Go2 XML | `unitree_go2_seed000/unitree_go2_12dof_anonymous_preview.xml` | `296ad8fb2ae42f1bb1e437c5e722914794676c7c6f0da51b9a60c674d85ebfa9` | n/a |

attempt010 root 固定为
`/home/admin1/workspace/proj/locomotion_rl/.agent/task/task070-archetype-constrained-standable-morphology/artifacts/preview_task070_v2_descriptor_driven_attempt010/`；
recovery worktree 不复制第二份 artifact，直接读取该 canonical absolute root。
static lineage 必须保存这六个 path、raw SHA、四个 JSON payload SHA，以及 manifest 内
`blueprint_manifest.profile_metadata.motor_configuration`、`actuation_stack.coherent_motor_config` 的
canonical payload SHA；缺一项或交叉引用不闭合即失败。

统一 artifact root 为
`.agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v3/`，不得覆盖 nominal_v2。布局只有
以下一种解释：

```text
nominal_v3/
  action_contract.json
  unitree_{g1,go2}/
    reward_config.json
    smoke/{initial.pt,run_manifest.json,progression.json,checkpoints/...}
    pilot/{initial.pt,run_manifest.json,progression.json,checkpoints/...,diagnostic_rollout.json,
           eval_trace.json,eval.json,pilot_gate.json,walk.mp4,walk.json,case_verifier.json,
           agent_visual_observation.json}
    proof/{initial.pt,run_manifest.json,progression.json,checkpoints/...}
    eval_trace.json
    eval.json
    walk.mp4
    walk.json
    case_verifier.json
    agent_visual_observation.json
  freeze/task072_freeze_manifest.json
  task072_verifier.json
```

stage 目录拥有该次训练的 update-0、progression、checkpoints 和 manifest；selected checkpoint 的实际
相对路径只能由同目录 `run_manifest.json` 声明。case 根目录只放 proof checkpoint 的正式
render/eval/verify 输出，pilot diagnostic 只放 `pilot/`，两者不得互相引用。根目录
`action_contract.json` 是同时含 G1/Go2 的唯一 combined payload；case/stage manifest 只保存对应 JSON
pointer 和整份 action contract raw/payload SHA。每个 manifest/report 都保存其消费文件的 artifact-root
相对路径与 raw SHA；JSON 另存 canonical payload SHA。不得复制第二份 contract，也不得用隐含
“latest checkpoint”。

## Route

严格按 `001 -> 002 -> 003 -> 004 -> 005` 执行。任何阶段失败都保留完整失败 artifact，并停止向
后续 gate 晋级；不得用缩短 horizon、改 command、启用 curriculum 或引入随机化来掩盖 nominal
失败。允许在 G1 pass 前做有记录的 reward/action ablation，但最终 G1 与 Go2 证据必须绑定同一个
冻结 commit；冻结后若改动任何训练、环境、动作、奖励、评估或 verifier 代码，则 lineage 失效，
G1 和 Go2 都必须重跑。Go2 的 tuple mapping、action adapter 配置和 no-update target smoke 必须在
G1 freeze 前准备并验证；这不算提前启动 Go2 训练。

Task072 的每个正式 case 都必须包含：

- exact Task071-bound asset/descriptor/physics/motor provenance 与 SHA；
- 随机初始化 checkpoint progression；
- paired zero-action 与 untrained-policy baseline；
- 20 秒、至少 20 episode 的 deterministic evaluation；
- 8 秒、400 frame 的 headless MuJoCo 视频；
- 从 checkpoint 重新加载并独立 rollout 的 fail-closed verifier；
- finite observation/action/reward/loss、positive forward displacement 和无 lineage mismatch。

数值 gate 固定为：zero-fall ratio `>= 0.95`、mean planar velocity error `<= 0.35 m/s`、mean
yaw velocity error `<= 0.35 rad/s`、mean projected-gravity XY norm `<= 0.35`。缺少视频、checkpoint、
baseline、progression、SHA 或任一指标失败，均视为该 case 未通过。

## Log

- 2026-08-27：旧 action adapter 使用
  `stance_midpoint + scalar_action_scale * action * half_joint_range`。在 G1 `action_scale=0.35`
  下，相对 `0.25 * effort_limit / kp` 的逐 slot motor-aware 尺度约为 `0.22x–9.01x`；ankle roll
  欠驱动而 wrist 可被过度命令。该统一 range fraction 不是 coherent motor config。
- 2026-08-27：历史 Go2 run 曾通过 20 s 行走 gate，但其 CLI/source SHA 与当前实现不同，只作为
  behavioral evidence，不计入 Task072 最终 pass。
- 2026-08-27：历史 G1 direct/curriculum/low-learning-rate runs 均在约 `1.56–2.14 s` 跌倒；因此
  Task072 当前未通过。简单 standing-to-walk curriculum 不再作为修复路线。
- 2026-08-27：本轮仅建立任务与子任务契约，未执行新的训练、视频或 verifier。
- 2026-08-27：在 recovery worktree 完成 001/002 实现与验证：G1 29-slot、Go2 12-slot
  motor-tuple-derived action amplitude 生成并通过 smoke；G1 biped reward 与 Go2 frozen quadruped
  reward configs 生成；两份指定测试文件全量为 25 passed。
- 2026-08-27：G1 `smoke` stage passed；G1 `pilot` stage 完成 2,048,000 transitions 但 pilot gate
  failed。20 s selected eval 为 zero-fall ratio `1.0`、planar error `0.4972`、yaw error `0.2210`、
  gravity XY `0.5282`、forward displacement `0.2094 m`；untrained/zero baselines planar error 分别
  `0.4550`/`0.4611`，训练没有达到 `0.05 m/s` learning margin。8 s pilot diagnostic video
  `render_passed=false`，forward displacement `0.1762 m`；人工视觉检查 not passed。按 003 contract
  未启动 proof，未执行 freeze 或 Go2 rerun。
- 2026-08-27：权威 Task072 六份 Markdown 已从主工作区同步到 recovery worktree；同步后 SHA 分别为
  `task.md=1dc8dcacc1e746ff0f29f17104570ef361b2312344cf88d968e3db4e5f3c7172`，
  `001=2c7967c3b31b888a91ac4d2fe1f0e4d470bf67500a71be354b176ad9e557ba85`，
  `002=8bc18c11b82ecd638e0bf03bbc0a803ed7f342e44743aace73a0f0c25eb3fb4e`，
  `003=38dbe850eb717f6b5cb8febb9749208fbce331e169ab2f7c2fc762d839c51b75`，
  `004=ed162820fc4a6f65bf6099f7d5f699aae47b02cab3f127e99608b65d5e63bb9f`，
  `005=f3f37b0e9012f991078310b32d8fd791a516d43819bc468990c51bb9b345883d`。
- 2026-08-27：nominal_v3 001/002 implemented and verified in recovery worktree
  `/home/admin1/workspace/run/locomotion_rl/task071-1` on branch `codex/task072-bound-walk-proof`。
  `action_contract.json` SHA `ea50671d5614f20887ad13dda96d72425bedb0ddeda9f3561c8e0233094370f9`，
  G1/Go2 reward_config SHA `ed515be4a236ec1e5cc5e7d1214fc8d9d28a41b64845bdde3613e51b0d2e656f` /
  `79c8823d70d0ebb61cbbb1902a24e2d0bd3dd7963b7cb514a051ab61b7fb5d89`；指定两份测试 `28 passed`。
- 2026-08-27：nominal_v3 G1 smoke passed, then G1 pilot completed 2.048M transitions but failed。
  Smoke artifacts: `run_manifest.json` SHA `6c71b2e21fc79ba1317598cc61dc14ac47492f6f4051a75a5c27156dc9daa612`，
  `progression.json` SHA `9277e6bf3508ca4c2af0a5ef83664a312fc90aaf0b22b79c20079622f60f0fb3`。
  Pilot artifacts: `run_manifest.json` SHA `f85a98575463151247d9753f9c31d5e2958d9b98868085bf81a78509e991b109`，
  `progression.json` SHA `9850923d34721506478a75f76dbf20288c482fb4b0f589b0d4f80992e85d3f10`，
  `final.pt` SHA `73e1657e604fa8c0bef1350d33e4c442872d8fbcbb643f6d68a3cb5bc43c24b3`，
  `walk.mp4` SHA `7b7f8a56dc688f1af10e92f427a9b65db6364a0eaf05b3a24f3ff8c8e6e82990`，
  `walk.json` SHA `2f7324b25848e0984b7c549aae3d37fdb7519af7311ffd76f0c580aaca91bc5e`，
  `diagnostic_rollout.json` SHA `4bc0a82294dc1f1fe6a0dce314db4aa4d2271737fc6c6e4fe696baa876500809`，
  `eval.json` SHA `5a87ee29d7ecde030510794c039b7287b529d3f6830d2de0bd96ff262e5cd80f`，
  `eval_trace.json` SHA `4eaae902c11746e606c39fe31b4fe40f7d50f8a226c7e209b80601857109ac7d`，
  `pilot_gate.json` SHA `fa43ce3485427afa33c1c9dfe2de1afa06c4f8126a652856485310502ebb34d8`，
  `case_verifier.json` SHA `cc8cb77d7e33ae33f857f68f47796732335d568ec638adbc028aeb6c919cc586`，
  `agent_visual_observation.json` SHA `8cddacb41cf4eed830e623071845a90a20adce987b9b7c25fe4a9e989cbeee6d`。
  20 s paired eval selected zero_fall_ratio `0.0`，planar error `0.3775`，yaw error `0.1108`，
  gravity XY `0.1143`，nonfall displacement `null`；common-prefix planar margins update0/zero-action
  `0.0557/0.0917` pass，but full-horizon displacement margins `0.1911/0.1925 m` fail the `2.0 m`
  comparison gate。video `render_passed=false` with fall_count `3`, done_count `3`, 8 s forward
  displacement `0.1341 m`。Agent visual check failed: no stable forward walking, no alternating
  swing/touchdown, mostly persistent double support/dragging, late collapse。
  Pilot gate failed reasons: `touchdowns, alternating, single_support, contact_fractions, kl,
  survival_improved, diagnostic_motion`。Per route, proof/freeze/Go2 rerun were not started.
- 2026-08-27：final focused validation command
  `env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src"
  /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python -m pytest -q
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py` returned `28 passed`。
  A prior immediate GPU-visible rerun after pilot hit CUDA OOM inside torch Adam health check; CPU rerun
  was used to verify code assertions after freeing the test from GPU state.
- 2026-08-28：确认 `nominal_v3` 是冻结的失败基线，不覆盖、不追加训练。R2 改为依次完成 phase
  observation、reward/return scaling、KL optimizer、roll authority/contact geometry 和 10M/20M
  progression；各消融必须从随机初始化独立训练并只允许一个声明过的 delta。

## Review

状态：**not_passed**。

Task072 只有在 exact-bound G1 和 Go2 都在同一冻结 source commit/config contract 上满足全部数值、
视频、baseline、checkpoint 与 verifier gate 后才能标记 passed。Task071 的 one-update PPO smoke、旧
lineage Go2 成功、短时 actuator response 或视觉可用都不能替代本任务的 locomotion proof。

历史 nominal_v2 Review 曾记录 001/002 unit contract passed、003 failed at G1 pilot；R1 的重放诊断曾
重开 001/002。当前 nominal_v3 执行结果为 **001 passed、002 passed、003 failed at G1 pilot、
004/005 not_run**；没有 clean git freeze，也没有 Go2 same-lineage rerun。因 G1 pilot diagnostic、视觉、
20 s eval 和 verifier gate 均失败，已按 route 停止。R2 只重新打开 003 的训练设计，不改变
nominal_v3 的失败结论；Task072 继续保持 **not_passed**。

## R2 single-variable recovery route

R2 的权威细节在 `003-g1-fixed-command-nominal-training.md`。执行顺序固定为：

1. `E1 phase`：只在当前 actor/value 共用的 policy input 追加与 Task048 同表示形式的 phase
   `sin/cos` observation；
2. `E2 reward_dt`：只将每步总 reward 乘 control `dt`，让 return/value target 回到合理量级；
3. `E3 optimizer`：主分支只用 adaptive LR 替代 hard KL stop；大 minibatch 只能作为同父配置的
   sibling fallback，不能与 adaptive LR 合并后声称单变量；
4. `E4a roll authority` 与 `E4b contact geometry`：分别验证 ankle/hip-roll 控制权和足底 collision
   geometry。任何 asset geometry 变化都必须回到 Task071 重做 exact-bound lineage；
5. 选定配置先从随机初始化跑一条 20M progression，在 10M、20M 固定 checkpoint 做 paired eval、
   contact diagnostic 和视频。只有进入预注册的 Task048-like 学习区间，才让同一条 lineage 继续
   进入 full-proof phase，最大总预算仍为 63,897,600 transitions。

R2 所有 run 继续锁定 G1 flat fixed command、seed family、无 randomization/curriculum、Task048
checkpoint 禁用。004/005 仍由 G1 full proof gate 阻塞。
