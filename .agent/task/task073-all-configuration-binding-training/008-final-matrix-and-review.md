# 008 — Final 18-case matrix and review

## Route

1. 汇总固定 18-case matrix：source/asset/descriptor、actuator/transmission、mass/COM/inertia、friction、
   motor tuple、adapter、runtime gate、nominal training、video/verifier 与 randomization 状态。
2. 每项记录 exact command、hardware、source commit、config/checkpoint/artifact SHA、seed、transition
   count、failure reason 和 state-machine transition；不得只记录聚合成功率。
3. 复跑统一 verifier，拒绝 stale report、checkpoint/config/source mismatch、缺视频与被缩小的
   denominator。
4. 执行独立只读 review，检查 Task070/071/072 claim 未被覆盖、family action/reward 语义未混用、
   nominal-before-randomization 顺序与 unknown provenance fail-closed。

## Log

- 2026-08-27：最终 matrix 尚未生成；Task073 保持 blocked by Task072。

## Code implementation

`task073_pipeline.py` 新增 `build_matrix(registry)` 与 `verify_matrix(path)`。每行固定包含
`case_id, tier, family, state, source_shas, binding_sha, schema_hash, nominal_config_sha,
checkpoint_sha, eval_sha, video_sha, visual_observation_sha, randomization_overlay_shas,
randomized_checkpoint_sha, randomized_eval_sha, commands, hardware, seeds, transition_count,
failure_reason`。任何缺失字段用 null，并使对应 gate false；不能省略失败 row。

`verify_matrix` 先验证 exact 18 ids/denominator，再逐 case 调用与训练时不同进程的 checkpoint replay
verifier，核对 report/video/config/source SHA 与 state transition。输出：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" matrix \
  --registry "$TASK073/artifacts/v1/registry.json" \
  --output "$TASK073/artifacts/v1/final/task073_matrix.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" verify-matrix \
  --matrix "$TASK073/artifacts/v1/final/task073_matrix.json" \
  --output "$TASK073/artifacts/v1/final/task073_final_verifier.json"
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q tests/test_task073_final.py
```

只有 18/18 nominal row verifier pass 才写 `nominal_18_complete=true`；只有 18/18 最终 cumulative
randomization verifier pass 且独立 review clean 才写 `task073_passed=true`。verifier 本身不得修改
registry/matrix 或降低 gate。

## Review

`nominal_18_complete` 只有 18/18 nominal case 全通过时为 true；`task073_passed` 只有 18/18 均完成
post-nominal randomization 且最终 reviewer 无重大 finding 时为 true。任何失败或 unknown 必须原样
保留，不能通过改 denominator、降低 gate 或扩大 claim 消除。
