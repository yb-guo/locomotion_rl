# Task072 — Bound G1/Go2 nominal locomotion proof

状态：**in_progress / not_passed**。

## 目标与边界

在 Task071 已绑定的 exact anonymous Unitree G1 与 Go2 上，用随机初始化 PPO 证明固定命令
`vx=0.5 m/s, vy=0, yaw=0` 的 nominal forward locomotion。先修正 G1 的动作尺度与 biped
reward；G1 通过后冻结全部代码、配置和资产 lineage，再在完全相同的 source commit 上重跑
Go2、视频和 verifier。

Task073 在 003c G1 walking gate 与 004 freeze 完成前保持 blocked。本任务不扩展到其余 16 个构型，
不启用质量、COM、摩擦、电机、push、sensor
noise 或 terrain randomization，也不使用 command curriculum。Task048 checkpoint 与本地课程资料
只能作为设计/预算参考，不得作为 Task072 初始化权重或 pass evidence。

003b/003c 是在旧 Task071-bound v1 失败证据之外新增的 versioned G1 walking lane：003b 只替换
terminal-contact profile，003c 使用被绑定的 MJLab training contract，必须把 G1 实际训练到 walking
gate。它们不得修改或补判原始
exact-bound/custom-PPO claim；上述 fixed-command/no-curriculum 约束继续适用于原 Task072 route，003c
训练按 MJLab contract 执行并以固定 `vx=0.5` eval 判定。

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
3a. `003a-e3a-mjlab-kl-correctness-repair.md`
   - 保留 rejected E3a artifacts，修复 joint-KL reduction、per-minibatch LR scheduling 与
     pre-tanh likelihood identity；
   - 先通过 no-update correctness gate，未经用户明确授权不得运行 repaired smoke 或 2.048M；
   - repaired E3a 通过前继续阻塞 E3b/E4、progression、freeze 与 Go2 rerun。
3b. `003b-g1-mjlab-terminal-contact-alignment.md`
   - 从当前 Task071-bound G1 派生新的 `mjlab_g1_7capsule_v1` contact variant；每只 logical foot
     使用 7 个 MJLab capsule，删除旧大 box，不改变 inertial、joint 或 actuation；
   - environment 与 stance 按 logical foot 聚合多 geom contact，并重新生成 stance 与 bound SHA；
   - 本 subtask 只允许 asset/no-update 验收，禁止训练。
3c. `003c-g1-mjlab-terminal-contact-training.md`
   - 旧 `proof_1024x24_seed720301` 保持完整证据，但重新定性为
     `rejected_runtime_binding_mismatch`；
   - 不再解释为“正确 runtime binding 下的 G1 无法行走”。
3d. `003d-g1-mjlab-runtime-binding-repair.md`
   - 新建 `mjlab_g1_7capsule_task_v2` asset/profile/stance lineage；
   - 对齐 Unitree-G1-Flat runtime contact material，显式绑定 stance default qpos、root pose、
     action offset 和 29/29 semantic joint mapping；
   - 后验 compiled audit 发现 asset `floor` 与 scene `terrain` 重合，故本 route rejected。
3e. `003e-g1-mjlab-repaired-walking-proof.md`
   - 历史上完成 `2048/4096/6144` capacity smoke 和
     `4096 x 24 x 650 = 63,897,600` transitions 随机初始化训练；
   - fixed-command walking gate 未通过，且后验确认训练 runtime 为 double-ground；全部证据 rejected，
     不得解释为正确 single-ground binding 下的失败。
3f. `003f-g1-mjlab-single-ground-runtime-repair.md`
   - 保持 v2 contact asset/profile/stance immutable，新增
     `mjlab_g1_7capsule_task_v3_single_ground` runtime lineage；
   - 运行时移除 asset `floor`，编译后只允许 scene `terrain`，并 fail closed 审计 14 个 foot capsules；
   - 只运行 CPU/no-update verifier，正式 v3 capacity/training 需另行授权。
4. `004-freeze-g1-passing-lineage.md`
   - 只有 G1 全 gate 通过后，冻结实现 commit、配置、descriptor、asset、checkpoint 与 verifier
     SHA；
   - 冻结后不允许为 Go2 改代码。
5. `005-go2-same-lineage-rerun.md`
   - 旧 standalone Go2 rerun route 保留为历史合同，不再位于当前 immediate chain；
   - Go2 和其他构型统一在 Task073 资产迁移后由 Task074 Tier A 从随机初始化训练测试。

## Code implementation

### 003a verifier repair status

Task-local repaired verifier `task072_e3a_repair_verifier.py` 已绑定现有 no-update、smoke、旧 rejected
E3a artifacts、全部 current source hashes、run/static/checkpoint lineage 和 update0 random-init，并重生成
schema-3 R4 gate（正式 SHA
`27c75fe817d4a8e8f556c37208f248c5e503f4109ae0d2840b3d11652928e693`；verifier source SHA
`f14c173f878006234e09911949ff826b4f6aac3df6d64bb54b1a0b20c9c0192f`）。此前
`7722371e68684f521d61beebc681104345e5e20bb01dc1c841a18c303e221c5e`、
`f34069875981c704e617ed5e0c246e72d7a1e8209850373bf2fffa9af80059dc`、
`39ff24bd9b34c8e3d6be56050855c394b017b693d974aceef7aee1c3d9404fbd`、
`110713b5bd5f5b7686480f197bd94b49c6f49ee7ae8f4b8c1ec1a4158df52551`、
`605787f787143281e3c3e8a3448923f8ce4a0dbbb0fc195316346bad6e858b00`、
`e718630badafa2b502dda1e5f3f518c54b3af2371e9456f772aa20b8efcfb182`、
`83f95ccdc5aaaafcbcf50f64d8ccd4c1cfe3ca110b8e3f63fbc97d123aa59a2a` 均仅作 rejected/superseded audit history。
R5 已完成 2.048M，但 exact-init binding 与原 optimizer thresholds 均失败；Task072 仍 not_passed。

### Canonical workspace and recovery baseline

- 不在大量脏改动的主工作区实现。接续 worktree 固定为
  `/home/admin1/workspace/run/locomotion_rl/task071-1`，branch 固定为
  `codex/task072-bound-walk-proof`；开始时只读记录 `git rev-parse HEAD`、
  `git branch --show-current` 和 `git status --short`，不得覆盖其中现有 Task071/072 改动。
- 本任务 Markdown 的权威契约位于
  `/home/admin1/workspace/proj/locomotion_rl/.agent/task/task072-bound-g1-go2-locomotion-proof/`。执行者在
  第一处代码修改前，用 `apply_patch` 将全部当前 subtask contract 逐字同步到 recovery worktree 的同路径并比较
  SHA；不得把 recovery worktree 的旧 task 文档反向覆盖本契约。这样 004 才能把实际执行契约纳入
  freeze commit。
- 该 worktree 中现有 task-local CLI
  `.agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py` 是恢复基线；
  当前 subtasks 在这个文件上做最小修改。当前主工作区缺少该 CLI，不得从主工作区的测试失败反向创建
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

原始路线为 `001 -> 002 -> 003 -> 004 -> 005`；其失败证据保持不变。当前有效链路严格为
`003b v1 geometry pass -> 003c rejected runtime binding mismatch -> 003d rejected double-ground ->
003e rejected contaminated training -> 003f single-ground runtime repair -> separately authorized v3
walking run -> 004 freeze -> Task073 asset migration -> Task074 18-case training`。
003f no-update verifier 未通过不得运行新的 capacity smoke；新的 v3 walking run 未通过不得 freeze 或
启动 Task073。
004 必须绑定新的 contact/stance/asset lineage，不能
复用旧 `official_sim_physics_overlay_v1` stance 或把旧失败 artifact 补判通过。任何阶段失败都保留完整失败 artifact，并停止向
后续 gate 晋级；不得用缩短 horizon、改 command、启用 curriculum 或引入随机化来掩盖 nominal
失败。Task073 的通用重构会建立新 asset lineage，Task074 必须重新训练 G1 和 Go2；Task072 003c
checkpoint 不得被 Task074 复用。

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
- 2026-08-28：R2 `E1_phase` completed in recovery worktree from random initialization, seed `72072`,
  2,048,000 transitions, artifact root `artifacts/nominal_v4/unitree_g1/E1_phase/`. Focused tests returned
  `32 passed`. E1 manifest records parent `E0`, parent config SHA
  `2de70da74e8f65b979836e59379aa3b54ea85487f142eff7634e11509b5794e3`, observation dim `195`, phase
  schema `193->195`, and scientific config diff limited to
  `configuration.actor_observation_dim` plus `configuration.phase_observation`. `e1_gate.json` SHA
  `965fe72a07ebd908223f3a5f62b824946a43b0763582d15588cba5a61d9dfc9a` passed the E1 mechanistic gate.
  Effect artifacts still show no walking: 8 s both-contact fraction `1.0`, left/right noninitial touchdowns
  `0/0`, alternating transitions `0`, fall_count `5`, forward displacement `0.1805 m`; 20 s selected
  zero-fall ratio `0.0`, planar error `0.3174`, yaw error `0.2648`, gravity XY `0.2121`; agent visual
  check failed due persistent double-support dragging. Per R2 this does not reject phase; proceed to
  `E2_reward_dt`.
- 2026-08-28：After reboot, preserved the interrupted `E2_reward_dt` partial run
  (`initial.pt`, `update000200.pt`) as
  `artifacts/nominal_v4/unitree_g1/E2_reward_dt_interrupted_reboot_20260828T040811Z/`; no resume was
  used. Added task-local `verify-e2-scale` after reviewer feedback and preserved source-drift/interrupted
  E2 attempts as separate `E2_reward_dt_*` directories. Re-ran focused tests, returning `36 passed`, then
  completed final verifier-source `E2_reward_dt` from random initialization with seed `72072` and
  2,048,000 transitions under `artifacts/nominal_v4/unitree_g1/E2_reward_dt/`. `e2_gate.json` SHA
  `fe41b57a91b715254f09625b73339e6403d02c1caa696ce6b73c0a8571b2910e` passed the scale gate:
  parent `E1_phase` config SHA `0632a74728d6b33a26747dfdb7e47f4531df3a10c82f234587bb2adc72b6c226`,
  scientific config diff only `configuration.reward_scale`, first-100 value loss/grad norm
  `0.02188625/5.26026373` vs E1 parent `137.5181938/40.9377101`, and reward scale max sample error
  `1.862645149230957e-09` with zero mismatch count `0`. Effect artifacts still do not prove walking:
  render `render_passed=false`, 8 s both-contact fraction `1.0`, no noninitial touchdowns or alternating
  transitions, 20 s selected zero-fall ratio `0.0`, yaw error `0.4468`, and visual double-support dragging.
  Per R2 this authorizes `E3a_adaptive_kl`, not a locomotion proof.
- 2026-08-28：R2 `E3a_adaptive_kl` implemented and run from random initialization, seed `72072`,
  2,048,000 transitions, artifact root `artifacts/nominal_v4/unitree_g1/E3a_adaptive_kl/`. After binding
  `src/h200_locomotion_lab/algorithms/ppo.py` in static lineage, the pre-lineage-fix run was preserved as
  `E3a_adaptive_kl_provenance_incomplete_20260828T063858Z/` and the provenance-complete run was used.
  Focused tests returned `41 passed`. `e3a_gate.json` SHA
  `3cb5594110432cf6bc4ab10bd4d74acec470468e8fe2cb16d05177ceb1706d1e` rejected E3a: all updates
  completed 4 epochs/32 minibatches and hard-stop fraction was `0.0`, but attempted-minibatch KL
  mean/p95/max `4.539716/3.420841/28128.240234` exceeded the `<=0.015/0.03/0.05` gate, and clip
  fraction mean/p95 `0.684878/0.828125` exceeded `<=0.20/0.35`. Effect artifacts also failed
  locomotion: render `render_passed=false`, 20 s selected zero-fall ratio `0.0`, planar error `0.3694`,
  yaw error `1.0213`, and visual check showed rocking/leaning without reliable alternating gait. Per the
  gate-fail stop rule, no E4 or E3b run was started.
- 2026-08-28：只读复核确认 rejected E3a 并不等价于本机 MJLab 1.2.0 / RSL-RL 5.0.1 semantics：
  scheduler KL 对 29 个 active action dimensions 取了均值而 policy log-prob 求和，LR 又只在完整
  32-minibatch update 后调整；高 `log_std` 下 squashed-action `atanh` round-trip 还使同策略首
  minibatch 出现非零 KL/clip。新增 `003a-e3a-mjlab-kl-correctness-repair.md`，先做 no-update
  correctness repair；本轮未改实现、未跑训练，E3b/E4 继续 blocked。
- 2026-08-28：执行 `003a-e3a-mjlab-kl-correctness-repair.md` 的 R0-R3 no-update correctness
  repair。新增 raw-action likelihood replay、old Gaussian mean/std rollout fields、per-minibatch
  joint-KL adaptive scheduler 和 `verify-e3a-kl-repair` fail-closed verifier；static lineage 进一步
  绑定 `src/h200_locomotion_lab/masked_distribution.py`。未运行 repaired smoke、2.048M training、
  E3b、E4、20M progression、freeze 或 Go2 rerun。
- 2026-08-28：`E3a_mjlab_kl_repair/no_update_correctness_gate.json` SHA
  `d2e3650be8e3b112688898bdb8441fb761c14171455a574c1d186adcfbbb80c3` passed R0-R3：
  旧 E2/E3a artifact SHA 均保持不变，rejected E3a 仍为 rejected；本机参考版本记录为
  `mjlab==1.2.0`、`rsl-rl-lib==5.0.1`；regular/saturated raw likelihood identity error 均 `0.0`，
  same-policy sampled approx KL `0.0`、clip fraction `0.0`；29-active-dim joint KL fixture
  `.004/.01/.021` 对应 increase/hold/decrease；真实 `ppo_update` 首 minibatch telemetry 记录
  identity error、sampled approx KL、clip fraction、scheduler KL 和 no-update parameter delta 均为
  `0.0`；raw evaluator/mask mismatch 现在 fail closed；原 E3a optimizer gate 阈值未改。验证：
  `py_compile` passed、`git diff --check` passed、focused pytest returned `46 passed`。
- 2026-08-28：按用户授权只执行 003a R4 repaired smoke，未运行 R5/2.048M。`E3a_mjlab_kl_repair/smoke/`
  完成 4 envs、32 steps、2 updates、256 transitions，CUDA smoke 正常返回且无 OOM，两个 update
  `fall_count=0.0`。Artifact SHA：`run_manifest.json`
  `04d1aba1a459832f704bb3724ec04cc70cf026ed6fef93543cc8ab57a628db40`，`progression.json`
  `28bfe92d62275985623945f56a13e9d91009d91fbfff95fd0d11635a5e500651`，`initial.pt`
  `c657a6a7a8bf31c7b798390ee811a953c31f4b78b7adc1b2bcd6c480a9a0da39`，`final.pt`
  `0c851a233f99c930aa78ec290c9a786336502f6f160c7707efd939a0f2273688`。新增
  最初 `r4_repaired_smoke_gate.json` SHA
  `7722371e68684f521d61beebc681104345e5e20bb01dc1c841a18c303e221c5e` 后续因 preservation=false
  仍 pass 的 fail-open 问题被拒绝；smoke 原始 artifacts 未变。正式独立 verifier 从原始证据重算 8 个
  minibatch、两个 update 的 first-MB identity、243,803 个 policy 参数差分、全部 checkpoint/source
  lineage，并生成 schema-3 gate SHA
  `27c75fe817d4a8e8f556c37208f248c5e503f4109ae0d2840b3d11652928e693`。no-update gate 与旧 rejected
  E3a artifact SHA 保持不变。
- 2026-08-30：按用户授权完成 repaired E3a R5：RTX 5060 Ti、seed `72072`、32 envs、64 steps、
  1000 updates、2,048,000 transitions，训练正常返回且无 OOM。`run_manifest/progression/initial/final`
  SHA 分别为 `4b8bc6ce658fe28829e4b0b71386cf8c058e7fa0952b3ac21e4c1a886a98a85e`、
  `cb008bf389f46cf0d8008ac1c8c8d88507d5da8425236c0bb028097566f4069b`、
  `25ebde4805f7874f4dc99dd494bf03f3c8085e6367d4469056414b9b871d6802`、
  `9208a97d7191cec1425504194db776c85c3082902c25efeefacaa5d45d2f1398`。Optimizer gate SHA
  `65697821095c9f7274d85603b470b074e2ae3393e6b3e58277da215e385294ca` fail closed：update0 不等于
  fixed R4/E2 initialization；只读 metrics 另有 KL mean/max 与 clip mean/p95 超阈值。未启动
  render/eval、E3b 或 E4。
- 2026-08-30：只读资产审计纠正此前错误判断：Task072 实际 bound G1 的逐 body mass 与 Task071
  选定 Unitree source 一致；异常的 `32.08 kg` 分布属于未用于训练的 Task070 raw preview。进一步确认
  当前 bound asset 为每脚一个 box，而本地 MJLab 示例为每脚 7 个 capsule。按用户要求新增 003b
  alignment-only 与 003c separately-authorized training contract；本轮未改实现、未生成资产、未训练。
- 2026-08-30：用户进一步纠正执行口径：003c 必须把 G1 实际训练到 walking gate，不能只做
  trainability admission；训练预算统一按累计 environment transitions 计算，不用 update/iteration
  count 比较不同并行度。后续任务重排为 Task073 资产迁移、Task074 18-case 正式训练。
- 2026-08-30：完成 003b `mjlab_g1_7capsule_v1` contact alignment，未覆盖旧 Task071/Task072
  artifacts。新 XML SHA `bd06eff122d35044018f3867a9d227346af4df847a8c56ce1df3f4cd074faf36`；
  contact profile SHA `304e464577636d45322e98547db6ef8557585c2bd1c3d254ee898e440b41156d`、payload SHA
  `2523a11840ae28cc1d2402c02d341a4965819925814ed986315e1661e351d857`；stance SHA
  `3671c9335e58591a5d9252c9aa38d02689579c222ada81432851b9d327030e19`、payload SHA
  `7218694acd65774e253a6d4d6900ed302fd6a735817a8fec41b26acb678f9e2b`；compiled invariant diff SHA
  `11f2911f3eab726bb877f00397ca64924a47635af1ad0b464250c99c2febe129` passed；compiled contact geometry SHA
  `0ec19a7335fab7ed528761ec407e52c6525c7f0cbc73c5c608fc2fbf5cbc3c17` passed；no-update smoke SHA
  `670b6a4ceccbee2a20d2c44c7d1314962e511f4d89b0b306bc97f550f6611f0c` with `optimizer_step_calls=0`。
  Independent verifier SHA `c47304991b595e4f5ef746a6901ccfc365f8318dc97600e8aa72c2cfd5c39cfe` passed。
- 2026-08-30：003c RTX capacity smoke passed for `256/512/1024` envs，selected `1024` envs、
  `24` rollout steps/env，`24576` transitions/update；capacity artifact
  `artifacts/mjlab_contact_training/g1/r1_capacity_smoke.json` passed，记录
  `h200_used=false`、`external_downloads_performed=false`、`task048_checkpoint_used=false`。正式训练
  run dir `artifacts/mjlab_contact_training/g1/proof_1024x24_seed720301/`，seed `720301`，从随机初始化
  跑 `2600` updates，累计 `63,897,600` transitions；training manifest SHA
  `96781895798e0c8c1a2e2ae89368ac8c10a9be427106df192ef914fa8584b6de`。
- 2026-08-30：003c fixed-command independent reload evals all failed walking gate. Observed transitions:
  `model_407.pt` `10,027,008` transitions, checkpoint SHA
  `7ac16a655fad723a5f460412095e179a84133ad1f12a2dc5d68c6a700e913eac`, eval SHA
  `060971ad0edb83c2354b610ab4cb86f5f601a26068547a3fed575fda78f8f264`, mean vx
  `0.0001045351`, x `0.0020907023`, planar error `0.5133395195`; `model_814.pt` `20,029,440`,
  checkpoint SHA `bbb9b2d488f5276b76c2e9c4cda8f256355409a3243d110a96fb85ce9db5fdd6`, eval SHA
  `863a133f66bba034c9ca2688dbfb20de0d91d81e2354c11e16e7a91b27cda879`, mean vx `0.0000345447`,
  x `0.0006908933`, planar error `0.5425215960`; `model_1628.pt` `40,034,304`, checkpoint SHA
  `70942b02d8591ef59db4f7d538b94c4dc2090d1bd47454e58db51c78b59f51b0`, eval SHA
  `b94391ee470eda5c1b612912a7d61689034711e4a2aa826717893513cf80137b`, mean vx `0.0005213788`,
  x `0.0104275765`, planar error `0.5364387035`; `model_2599.pt` `63,897,600`, checkpoint SHA
  `5e6f614114a1e4d1e3e78f2b0d63824664254e132d7c3d8f55ddfb2a74699ff4`, eval SHA
  `05ca1bcbd2b9b113374d69936aa3f94fd95d176b7ca1f763060f180bc6a684f9`, mean vx `0.0001334263`,
  x `0.0026685263`, planar error `0.5329391360`。All four evals had zero-fall `1.0` and passed yaw,
  gravity, touchdown, single-support and alternating-contact checks, but failed forward velocity,
  +x displacement and planar tracking. No 8 s passing video was generated; 004/Task073/Task074 were not
  started.
- 2026-08-30：003c TensorBoard training scalars at target checkpoints were recorded from
  `events.out.tfevents.1788088651.admin1-B860M-K.1483996.0`: step `407` mean reward `-6.19351`,
  train xy/yaw errors `0.343544/0.53608`, fell_over `3.5`; step `814` mean reward `-6.81898`,
  xy/yaw `0.79855/1.20915`, fell_over `1.79167`; step `1628` mean reward `-5.06848`,
  xy/yaw `0.671423/1.03302`, fell_over `1.25`; step `2599` mean reward `-1.66894`,
  xy/yaw `0.904601/1.07719`, fell_over `1.25`。
- 2026-08-31：旧 003c `proof_1024x24_seed720301` 重新定性为
  `rejected_runtime_binding_mismatch`：003b stance 未进入 MJLab init/default pose，29 个 runtime
  `default_joint_pos` 全为零，runtime root height `0.8` vs stance `0.8534139176251306`，zero-action
  target 未使用 `actuator_ctrl_eq`，`1024 x 24` cadence 与官方 `4096 x 24` PPO batch 不等价，且 effective
  contact material 未对齐 Unitree-G1-Flat runtime task。旧 run 数值证据保留但不再支撑正确配置下的失败结论。
- 2026-08-31：完成 003d `mjlab_g1_7capsule_task_v2` runtime binding repair。v2 XML SHA
  `c41bfe757fbeb51f094a08457258d17004989948be6eb1fac5bbf3eafa644f93`；contact profile SHA
  `de1fcd515052afe488f3b769fad3649eeb3d509b8ac4ab6f4558cb45137d4f21`、payload SHA
  `dc4e9d9ac898ad0f9e837f5e89125e0f4c6e547ffa9e2824a10abfa145a69f46`；runtime material SHA
  `6179293ff3186429a5e4a21727ba55a2989801c3538bc7021203d3f9dc037804`；recomputed stance SHA
  `b2cf38c891bdc5e6b7bf5c4eaaed3cb42fee5b45e40ac773bc4b339dd261aac4`、payload SHA
  `85d29988805b4ef82d15bf575280b8d20ff304cfb14a3317cf277ee3783cc492`、internal solution SHA
  `cc522b67380713954480c3e9781be01fc6ad96445fb133d410f213f551f5ce9a`。No-update verifier
  `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2/runtime_binding_verifier.json` passed，SHA
  `26b93587a11cbc12c0de32e8ea0bef23020351e47f367901ce83afe5b13ff36b`：
  29/29 explicit mapping, no unmapped/duplicate joints, default qpos max error `1.4767210787525187e-08`,
  action offset and zero-action target max error `2.2015378742246128e-08`, root pose max error
  `4.416348237798657e-08`, joint-pos-rel reset error `0.0`, qpos/action-target min delta
  `3.810971428740453e-06`, actor/critic observation layouts `(98)/(113)` exact, compiled material
  full vector/contype/conaffinity checks passed, `optimizer_step_calls=0`, `parameter_delta_max_abs=0.0`。
- 2026-08-31：003e capacity smoke attempted with `flock -n /home/admin1/workspace/run/.gpu.lock`
  and candidates `2048/4096/6144`, but the lock was held by another docker compose task
  (`2056863/2057703/2057729`). No GPU capacity artifact, optimizer smoke, formal training, reward B,
  checkpoint eval, passing video or 004 freeze was produced.
- 2026-08-31：003e runner gates tightened after review：CUDA capacity requires inherited
  `/home/admin1/workspace/run/.gpu.lock`; capacity artifact must select passed `4096 x 24 = 98,304`
  transitions/update evidence; `one-update-train` must consume that artifact and use exactly `4096 x 24`。
  Negative CLI probes for missing 4096 and missing inherited lock both failed before env creation.
- 2026-08-31：003e 在 inherited GPU lock 下重新运行并通过 official-equivalent capacity gate：
  `2048/4096/6144` env candidates 均 passed，selected `4096 x 24 = 98,304` transitions/update；
  capacity artifact SHA `836a125164d185cd8a78226218f8e592dbfc5342a88864968b664d93ade65c37`。
  Capacity-consuming one-update smoke passed，SHA
  `a423f2ccf2cbbd59c9c893b63e74bfb0ebaa4f7263c650f048828f465e4c1321`。
- 2026-08-31：003e repaired baseline 使用 `mjlab_g1_7capsule_task_v2` 从随机初始化完成
  `4096 x 24 x 650 = 63,897,600` transitions，seed `720301`，training manifest SHA
  `88de879629cb83d7cba8bad34c9d919c16d44a6d0d0721636106d26060f0110b`，wall time
  `1965.1720781326294 s`。Checkpoint SHA：`model_100.pt`
  `bf0a3c99d96fd07bca82f8fb9b5e476e8c8a5061b760f34f5a5014530dfc996d`，`model_200.pt`
  `9e0dd9ba6700c6915d6162a7af06bc7997d67992519e3b9839a8c5ec79072512`，`model_400.pt`
  `7a5487f2e6204ec641281a702a2b6f3813ae1f4f639f0fe8e70757c8ed7d92ba`，final `model_649.pt`
  `f3828137d2f8056824fe2aab41dbe09c454169c78d202ceb238b68ad553bd18f`；`h200_used=false`、
  `external_downloads_performed=false`、`task048_checkpoint_used=false`。
- 2026-08-31：003e fixed-command independent reload evals at `model_100/model_200/model_400/model_649`
  all failed walking gate。All four had zero-fall `1.0` and finite obs/reward. `model_100`/`model_200`
  still lacked right touchdown, left single support and alternation; `model_400`/`model_649` passed
  touchdown/single-support/alternation but failed forward velocity, +x displacement and planar tracking.
  Final `model_649` eval SHA `02458b2803e709244a205cb1faa63e01ce3dfca911c473e73b5a0758a7d98f57`：
  mean vx `-0.0006511346`，x `-0.0130226929`，planar error `0.5035098791`，yaw `0.0302346162`，
  gravity XY `0.0306854099`，touchdowns `left=531/right=6`，single support `left=6/right=646`，
  alternation `6`。No 8 s passing video was generated; 004 freeze、Task073、Task074 remain blocked.
- 2026-08-31：后验 exact compiled-model audit 发现 003d/003e runtime 同时包含 z=0 的
  `robot/floor` 与 `terrain`。冻结 stance 上共有 56 个 foot-plane contact records，28 个对
  `robot/floor`、28 个对 `terrain`；但 `feet_ground_contact.secondary` 只匹配 `terrain`。因此 003d
  incomplete verifier 的 pass 被撤销，003e capacity/training/checkpoints/evals 全部按
  `rejected_runtime_binding_mismatch_double_ground` 保留；training manifest 的 `passed=true` 只表示
  run/checkpoint 产出完成。它们不能证明正确 binding 下无法行走，也不能 freeze 或初始化 v3。
- 2026-08-31：完成 003f `mjlab_g1_7capsule_task_v3_single_ground` runtime repair。Frozen v2 asset
  未修改，SHA 仍为 `c41bfe757fbeb51f094a08457258d17004989948be6eb1fac5bbf3eafa644f93`；runtime
  spec SHA `9216ef486aa9b535412c65b198e5a168d76d595763e88cf900057a65aa33874a`。CPU no-update
  verifier passed，artifact SHA `8ced901ec7b4eb5b69d29e2098d2411e0db84f27c33529e7061d9df0cd467dc1`，
  runner SHA `742ece071c4c77e652a3d09971f2a78b0a53ccd7d99ce6eef1f7fca4e3f2a33b`：compiled
  planes exactly `[terrain]`，14/14 foot geoms only contact terrain，`foot_ground_pairs=28`，
  `hidden_plane_pairs=0`，全部 binding checks true，optimizer steps/parameter delta 均为 0。聚焦 pytest
  `6 passed`；未运行 GPU capacity、optimizer update、训练、eval 或 video。

## Cleanup record

- 2026-08-31：按用户请求以 `gio trash` 清理两个已确认无效目录：`artifacts/mjlab_contact_training/g1`（10 `.pt`，62,819,574 file bytes，约 60M）与 `artifacts/mjlab_runtime_binding/g1/mjlab_g1_7capsule_task_v2`（9 `.pt`，51,918,322 file bytes，约 50M）。两路径已不存在；contact v2 与 runtime v3 保留。默认输出根改为 v3；runner SHA `30d18f0c07d105d5f65de65010df49fef88bf304d8d221f1d425ce4b1c6f2a5e`，v3 verifier SHA `377daa19e8d84950208f8b0b6f820ffd5360a9e00e10ef1d752c9d48299f27a1`，verifier passed。
- 2026-08-31：reviewer fix 后最终 runner SHA `b323fce3d90889f2836512be6888021f343a00c3e06354b39c4a64242708a57c`，v3 verifier SHA `551d0743a029ca033b02d049a88cb48f5c36cd01d00bb58a3d737f4bffdf420f`；registration 的 `lineage_id`、`run_name`、`experiment_name` 均为 `mjlab_g1_7capsule_task_v3_single_ground`，verifier passed。

## Review

状态：**not_passed**。

Task072 只有在 exact-bound G1 和 Go2 都在同一冻结 source commit/config contract 上满足全部数值、
视频、baseline、checkpoint 与 verifier gate 后才能标记 passed。Task071 的 one-update PPO smoke、旧
lineage Go2 成功、短时 actuator response 或视觉可用都不能替代本任务的 locomotion proof。

历史 nominal_v2 Review 曾记录 001/002 unit contract passed、003 failed at G1 pilot；R1 的重放诊断曾
重开 001/002。当前 nominal_v3 执行结果为 **001 passed、002 passed、003 failed at G1 pilot、
004/005 not_run**；没有 clean git freeze，也没有 Go2 same-lineage rerun。因 G1 pilot diagnostic、视觉、
20 s eval 和 verifier gate 均失败，已按 route 停止。R2 只重新打开 003 的训练设计，不改变
nominal_v3 的失败结论；`E1_phase` 机械 gate 已通过，`E2_reward_dt` scale gate 已通过并授权进入
`E3a_adaptive_kl`，但 `E3a_adaptive_kl` optimizer gate 已失败/rejected。按本次 gate-fail stop
规则，未启动 E3b 或 E4；`003a` 的 R0-R3 no-update correctness repair 与 R4 repaired smoke formal
gate 已通过，R5 2.048M 也已执行，但 exact-init binding 和原 optimizer gate 均失败，故 rejected。
003b contact alignment 已通过；003c 已按 transition budget 从随机初始化跑满
`63,897,600` transitions，但随后因 runtime binding mismatch 被拒绝；003d 的 joint/pose/action/material
局部 repair 后验发现 double-ground，故其 incomplete no-update pass 已撤销；003e 虽用
`4096 x 24 x 650` 跑满 `63,897,600` transitions，但同样受 double-ground 污染，已 rejected，不能再称为
正确 v2 binding 下的 walking failure。003f single-ground runtime no-update verifier 已通过，但未启动新的
v3 capacity/training/eval/video。G1/Go2 full proof 仍未通过，Task072 继续保持 **not_passed**。

## R2 single-variable recovery route

R2 的权威细节在 `003-g1-fixed-command-nominal-training.md`。执行顺序固定为：

1. `E1 phase`：只在当前 actor/value 共用的 policy input 追加与 Task048 同表示形式的 phase
   `sin/cos` observation；
2. `E2 reward_dt`：只将每步总 reward 乘 control `dt`，让 return/value target 回到合理量级；
3. `E3 optimizer`：主分支只用 adaptive LR 替代 hard KL stop；大 minibatch 只能作为同父配置的
   sibling fallback，不能与 adaptive LR 合并后声称单变量；
4. `E4a roll authority` 与 `E4b contact geometry`：分别验证 ankle/hip-roll 控制权和足底 collision
   geometry。E4b 现在由 003b/003c 的新 versioned contact asset route 承接；禁止原地修改 Task071
   overlay v1，必须重做 bound XML、stance、lineage 和 verifier；
5. 选定配置从随机初始化按累计 transitions 跑 progression；checkpoint 至少覆盖 10M、20M、40M，
   最大总预算为 `63,897,600` transitions。所需 updates 必须由实际并行环境数和 rollout length 反推；
   只有完整 walking gate 通过才允许提前停止。

R2 所有 run 继续锁定 G1 flat fixed command、seed family、无 randomization/curriculum、Task048
checkpoint 禁用。004/005 仍由 G1 full proof gate 阻塞。
