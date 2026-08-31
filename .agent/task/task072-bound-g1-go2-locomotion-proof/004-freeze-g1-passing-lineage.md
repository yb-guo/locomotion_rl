# 004 — Freeze the G1-passing lineage

状态：**not_run / blocked_by_003h_authorized_pilot_and_walking_gate**。旧 `nominal_v3` 与 003g 均已失败；
只有 003h repair commit 之后的新 G1 MJLab passing lineage 通过全部适用 gate，且本 freeze allowlist
覆盖 MJLab runner、action/reward contract、runtime spec、external pin、config、eval、render、verifier
和测试后才执行。

## Route

1. 只有 003h 后续授权训练的数值、交替 gait、视频与独立 verifier walking gate 全部通过后才执行冻结；smoke、
   finite loss 或仅能站立不得建立 passing lineage。
2. 将动作 adapter、reward、环境、trainer、evaluation、render、verifier、测试和任务 contract 固定在
   一个明确 git commit；记录 branch、commit SHA、dirty-state audit 和执行环境。
3. 冻结 001 已验证的 G1/Go2 action-adapter 配置与 tuple mapping；记录 G1/Go2 descriptor、MJCF、
   physics binding、motor tuple、训练 config，以及已产生的 checkpoint、report、video 与 verifier
   SHA-256 和生成命令。Go2 尚未训练的 artifact 明确记为 pending，不伪造 hash。
4. verifier 必须重新加载 checkpoint 并重跑 evaluation，校验 source/config/asset/checkpoint SHA；
   只读取已有 JSON 或接受 lineage mismatch 必须 fail closed。
5. 冻结后不允许为了 Go2 修改环境、训练、reward、action、evaluation、render 或 verifier 源码。
   若任何源文件或 contract 改动，废弃本 freeze，重新执行 G1 003 和本 subtask。

本轮新的主要输出是供 Task073 使用的 frozen G1 reference/contact/stance/training lineage。下方旧
Go2-paired freeze 细节保留作历史审计；Task073 通用资产重构后，Task074 必须重新训练 G1/Go2，不能
复用 003c checkpoint。

## Log

- 历史 Go2 pass 的 CLI SHA 与当前 source 不同，证明仅有 artifact SHA 而没有 source freeze 不足以
  建立 paired G1/Go2 claim。
- 2026-08-27：G1 尚未通过，因此本 lineage 尚未冻结。
- 2026-08-27：nominal_v3 G1 pilot failed（video/eval/verifier gate failed），因此未执行 freeze，
  未提交 Task072 freeze commit，`task072_freeze_manifest.json` 未生成。
- 2026-08-30：003b contact alignment passed，但 003c 在 `63,897,600` transitions 后 fixed-command
  walking gate 仍失败：final `model_2599.pt` 独立 20 s/256-env eval `passed=false`，mean vx
  `0.0001334263 m/s`，mean +x displacement `0.0026685263 m`，planar tracking error `0.5329391360`。
  未生成 passing video，未运行 freeze command，未创建 `task072_freeze_manifest.json`，未授权 Go2、
  Task073 或 Task074。

## Code implementation

在 CLI 的 `_static_lineage()` 基础上新增 `freeze_command()`；它只在 G1 case verifier 与 agent visual
observation 均为 true、Go2 action smoke 为 true 后运行。先提交全部受控实现/测试/config/task contract，
再运行 freeze；Go2 正式训练前不得再修改这些文件。最终 task Log 更新必须等 Go2 完成后另作 docs-only
commit，不能冒充训练 source commit。

freeze manifest schema 固定为
`{schema_version, git:{commit,branch,controlled_dirty:false}, sources:[{path,sha256}], parent_artifacts,
assets, action_contracts, reward_configs, g1_checkpoint, g1_report, g1_video, go2_action_smoke, commands}`；
Go2 尚未生成的 checkpoint/report/video 只写 `status: pending`，不能写假 SHA。`sources` 至少覆盖 CLI、
environment、trainer、policy、tests、`pyproject.toml`、`uv.lock`。

`sources` 必须精确等于以下 controlled source allowlist，并记录 commit-tree blob 的 SHA（不是 working
tree 猜测值）：

```text
src/h200_locomotion_lab/envs/whole_body_mujoco.py
src/h200_locomotion_lab/algorithms/ppo.py
src/h200_locomotion_lab/training/whole_body_ppo.py
src/h200_locomotion_lab/policies/whole_body_mlp.py
src/h200_locomotion_lab/masked_distribution.py
src/h200_locomotion_lab/robots/whole_body_adapter.py
.agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py
.agent/task/task072-bound-g1-go2-locomotion-proof/task.md
.agent/task/task072-bound-g1-go2-locomotion-proof/001-g1-motor-tuple-slot-action-scale.md
.agent/task/task072-bound-g1-go2-locomotion-proof/002-g1-mature-biped-pose-contact-reward.md
.agent/task/task072-bound-g1-go2-locomotion-proof/003-g1-fixed-command-nominal-training.md
.agent/task/task072-bound-g1-go2-locomotion-proof/003a-e3a-mjlab-kl-correctness-repair.md
.agent/task/task072-bound-g1-go2-locomotion-proof/003b-g1-mjlab-terminal-contact-alignment.md
.agent/task/task072-bound-g1-go2-locomotion-proof/003c-g1-mjlab-terminal-contact-training.md
.agent/task/task072-bound-g1-go2-locomotion-proof/004-freeze-g1-passing-lineage.md
.agent/task/task072-bound-g1-go2-locomotion-proof/005-go2-same-lineage-rerun.md
tests/test_task072_locomotion_proof.py
tests/test_whole_body_extended.py
pyproject.toml
uv.lock
```

Task072 相对已单独提交的 Task071 parent baseline 的 staged diff 必须是该 allowlist 的子集；每个实际
Task072 修改都必须在 staged diff 中，allowlist 外 staged path 必须失败。allowlist 中即使本轮未改的
policy/adapter/dependency lock 仍必须进入 manifest `sources`，使执行依赖闭合。

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK072=.agent/task/task072-bound-g1-go2-locomotion-proof
CLI="$TASK072/task072_locomotion_proof.py"
git status --short
git add src/h200_locomotion_lab/envs/whole_body_mujoco.py \
  src/h200_locomotion_lab/algorithms/ppo.py \
  src/h200_locomotion_lab/training/whole_body_ppo.py \
  src/h200_locomotion_lab/policies/whole_body_mlp.py \
  src/h200_locomotion_lab/masked_distribution.py \
  src/h200_locomotion_lab/robots/whole_body_adapter.py \
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py \
  .agent/task/task072-bound-g1-go2-locomotion-proof/task.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/001-g1-motor-tuple-slot-action-scale.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/002-g1-mature-biped-pose-contact-reward.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/003-g1-fixed-command-nominal-training.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/003a-e3a-mjlab-kl-correctness-repair.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/003b-g1-mjlab-terminal-contact-alignment.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/003c-g1-mjlab-terminal-contact-training.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/004-freeze-g1-passing-lineage.md \
  .agent/task/task072-bound-g1-go2-locomotion-proof/005-go2-same-lineage-rerun.md \
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py \
  pyproject.toml uv.lock
git diff --cached --name-only
git commit -m "task072: bind motor-aware nominal locomotion"
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q \
  tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" freeze \
  --g1-report "$TASK072/artifacts/nominal_v3/unitree_g1/eval.json" \
  --g1-visual "$TASK072/artifacts/nominal_v3/unitree_g1/agent_visual_observation.json" \
  --action-contract "$TASK072/artifacts/nominal_v3/action_contract.json" \
  --output "$TASK072/artifacts/nominal_v3/freeze/task072_freeze_manifest.json"
```

实现必须拒绝受控 source 与 commit tree 不一致、parent/config/checkpoint SHA 漂移和缺失 G1 gate。
若恢复 worktree 中尚有 Task071 的受控 source 改动，执行者必须先按 Task071 的实际文件 allowlist
单独提交 parent baseline；不得用 `git add -A`、不得把无关脏文件混入 Task072 freeze commit。
冻结后若任何受控 source 改动，004 失效并回到 003 重跑。

## Review

通过条件：G1 pass evidence 与唯一 clean source commit/config/asset identity 一致，Go2 tuple/config
已在该 commit 通过 pre-freeze smoke；verifier 能拒绝任一被篡改或错 lineage 的 source、config、
descriptor、checkpoint 或 report。没有 commit freeze 时不得进入 005 的正式 Go2 run。

当前状态：**not_run / blocked_by_003h_authorized_pilot_and_walking_gate**；003h 只完成 CPU contract
closure，尚未运行新 authorized pilot、numeric eval、video 或 independent verifier，004 正确阻塞。
