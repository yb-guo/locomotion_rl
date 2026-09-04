# 003 — Tier A remaining center morphologies

## Route

按 PM01、Spot、Lite3 顺序逐 case 执行 002 的物理绑定与 runtime gate。随后使用 Task072 冻结的同一
training/eval/verifier framework，在 flat terrain、`vx=0.5 m/s, vy=0, yaw=0`、无 randomization、无
curriculum 下从随机初始化 PPO 训练：PM01 使用 biped pose/contact semantics；Spot 与 Lite3 使用
quadruped semantics。

每个 case 独立保存 checkpoint progression、paired zero/untrained baselines、20 s evaluation、8 s
视频、agent visual check 与 verifier。一个 case 失败不允许由另外两个 case 或 tier 平均值抵消。

## Log

- G1/Go2 属于 Task072；本 subtask 只处理 Tier A remaining 3。
- 2026-08-27：PM01、Spot、Lite3 nominal walking 尚未执行。

## Code implementation

`task073_pipeline.py` 新增 `train_stage(case_id, stage)` 与 `run_nominal(case_id)`：前者只执行一个
`smoke|pilot|proof` stage，后者按 gate 串联三个 stage，再执行 render/eval/verify。它们读取 002 的
bound XML/binding verifier，并从 Task072 freeze manifest 读取而不是重新解释以下 contract：motor-aware action formula、fixed
`(0.5,0,0)`、stage budgets/seeds、20 s numerical gate、8 s video 和 checkpoint verifier。

- `engineai_pm01` 使用序列化的 Task072 biped reward schema，并以其自己的 stance nominal pose 和
  23 active slots重建 semantic groups。resolver 只读 structural descriptor：`module=left_leg|right_leg`
  再按 semantic slot 中唯一的 `hip|knee|ankle` 分到对应组，`module=waist` 进入 waist，
  `module=left_arm|right_arm` 进入 arm_wrist；23/23 必须恰好覆盖一次，否则 fail closed；
- `spot_base`、`deeprobotics_lite3` 使用 freeze 中 Go2 quadruped reward config，但各自读取自己的
  12-slot tuple、stance/contact geoms。load-bearing contact 只由 blueprint `foot=true` 与 binding
  `contact_role=foot` 的交集解析；不得按 link name 猜测；
- 三个 case 禁止共享 checkpoint、optimizer、normalizer 或 action scale。

每个 case 严格执行 `bind -> smoke -> pilot -> proof -> render -> eval -> verify`：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
CLI="$TASK073/task073_pipeline.py"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" nominal --case engineai_pm01
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" nominal --case spot_base
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" nominal --case deeprobotics_lite3
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q tests/test_task073_pipeline.py -k tier_a
```

CLI 内部使用 Task072 同名 artifact layout：`nominal/{run_manifest.json,progression.json,final.pt,
eval.json,walk.mp4,walk.json,agent_visual_observation.json,case_verifier.json}`。只有各自 case verifier
通过才 transition；一个失败不阻止记录其他 case evidence，但 Tier A 3/3 gate 保持 false。

## Review

通过条件：3/3 分别通过与 Task072 同等级的 numerical/video/verifier gate，且 source/physics/motor
provenance 可审计。通过本 subtask 只建立 Tier A nominal 5/5，不代表 wheel、candidate 或 domain
randomization 已通过。
