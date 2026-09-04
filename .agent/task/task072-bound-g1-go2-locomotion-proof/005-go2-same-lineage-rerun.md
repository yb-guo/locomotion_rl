# 005 — Go2 rerun on the frozen G1 lineage

状态：**not_run**。只有 004 的 `nominal_v3` freeze 成立后才执行。

## Route

1. checkout/verify 004 冻结的同一 source commit；运行前确认所有受控源文件与 config 无 diff。
2. 使用 exact Task071-bound anonymous Go2 与其自己的 transmission-resolved motor tuple；逐 slot
   动作尺度沿用 001 的同一推导代码，但使用 Go2 tuple，不能借用 G1 数值。
3. Go2 使用同一 commit 内的 quadruped-specific pose/contact reward 配置，不把 biped foot/pose
   semantics 生搬到四足。命令仍固定为 `vx=0.5 m/s, vy=0, yaw=0`，关闭全部 randomization 与
   curriculum，并从随机初始化 PPO 重训。
4. 重新生成 checkpoint progression、paired zero/untrained baselines、20 s evaluation、8 s/400
   frame 视频和 verifier report；历史不同 SHA 的 Go2 artifact 不得复用为正式 evidence。
5. 运行统一 verifier，一次性校验 G1 与 Go2 均来自 004 freeze 且均满足主任务 gate。

## Log

- 历史 Go2 曾达到 20/20 20 s、zero fall、planar error 约 `0.0903`、yaw error 约 `0.1019` 和
  forward displacement 约 `10.91 m`，但 source lineage 不同，必须重跑。
- 2026-08-27：同 lineage Go2 正式 run 尚未执行。
- 2026-08-27：nominal_v3 G1 pilot failed before proof/freeze；因此未 checkout/verify freeze，
  未启动 Go2 smoke/pilot/proof/eval/render/verifier。

## Code implementation

本 subtask 不新增代码；只运行 004 commit 中已经通过 pre-freeze smoke 的 shared adapter/CLI。CLI
`train --case unitree_go2` 必须读取 Go2 12-slot tuple 和既有 quadruped reward 分支，不能复用 G1
29-slot数值或 biped pose/contact config。Go2 使用与 003 相同 stage budget、train/eval/render seeds，
仍从随机权重开始且 randomization/curriculum/fault 全关闭。

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK072=.agent/task/task072-bound-g1-go2-locomotion-proof
CLI="$TASK072/task072_locomotion_proof.py"
ROOT_OUT="$TASK072/artifacts/nominal_v3/unitree_go2"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" verify-freeze \
  --manifest "$TASK072/artifacts/nominal_v3/freeze/task072_freeze_manifest.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" train --case unitree_go2 --stage smoke --run-dir "$ROOT_OUT/smoke"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" train --case unitree_go2 --stage pilot --run-dir "$ROOT_OUT/pilot"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" train --case unitree_go2 --stage proof --run-dir "$ROOT_OUT/proof"
env MUJOCO_GL=egl PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" render --case unitree_go2 \
  --run-manifest "$ROOT_OUT/proof/run_manifest.json" --output "$ROOT_OUT/walk.mp4"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" eval --case unitree_go2 \
  --run-manifest "$ROOT_OUT/proof/run_manifest.json" --video-sidecar "$ROOT_OUT/walk.json" \
  --output "$ROOT_OUT/eval.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$CLI" verify \
  --freeze-manifest "$TASK072/artifacts/nominal_v3/freeze/task072_freeze_manifest.json" \
  --g1-report "$TASK072/artifacts/nominal_v3/unitree_g1/eval.json" \
  --go2-report "$ROOT_OUT/eval.json" \
  --output "$TASK072/artifacts/nominal_v3/task072_verifier.json"
```

Go2 视频也必须由 agent 打开并生成同 schema `agent_visual_observation.json`；最终 `verify` 同时重放
两份 checkpoint/baseline/eval，并要求 source commit 等于 freeze commit。若为 Go2 修改代码，立即
废弃本 run 与 004 freeze，回到 003 重跑 G1。

## Review

通过条件：Go2 独立满足 Task072 全 gate，且 G1/Go2 的 source commit、trainer、environment、
evaluation、render 和 verifier lineage 完全一致。若为修 Go2 改过代码，则 004 freeze 失效，必须
回到 003 重跑 G1；不得仅重跑 Go2 后宣称 Task072 passed。

当前状态：**not_run**；因 004 未冻结，同 lineage Go2 rerun 正确阻塞。
