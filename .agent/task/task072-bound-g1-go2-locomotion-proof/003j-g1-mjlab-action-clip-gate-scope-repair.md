# 003j — G1 MJLab action-clip gate scope repair

状态：**gpu_pending / not_trained / not_passed**。

003j supersedes 003i only for the gate-scope mismatch. The 003i one-update run,
manifest, model and diagnostics remain rejected evidence and must not be
resumed, relabeled or consumed.

## Route

`003i rejected one-update diagnostic -> 003j CPU implementation/tests/verifier ->
clean source commit -> authorized 003j capacity -> fresh exact one-update ->
fresh pilot/evals -> pilot gate -> separately authorized proof`。

No 003i checkpoint, optimizer state, pilot or artifact is an input. The source
baseline is `59e45c7d5bb707af8d8803a2950316e701afba9b` in recovery frame
`/home/admin1/workspace/run/locomotion_rl/task071-1`.

## Log

- 2026-09-01：003j opened after the frozen 003i diagnosis. The root cause is
  stage-scope leakage: one-update `passed` depended on pilot-only absolute clip
  thresholds (`.10/.50/.25`).
- 2026-09-01：CPU implementation and regression coverage are in progress.
- 2026-09-01：CPU `py_compile`、聚焦 clip/one-update/pilot tests、`git diff --check` 与
  `003j_reward_eval_contract_verifier.json` 均通过；verifier 已绑定 source commit
  `bc10d887c2b426806b515da92d768838e7b5a7ea`。
- 2026-09-01：003j GPU capacity 未启动；`flock -n /home/admin1/workspace/run/.gpu.lock`
  返回 busy，holder 为其他 EmbodiedGen GPU 任务（PID 49088/49099）。按 stop rule 标记
  `gpu_pending`，不等待、不轮询，不运行 capacity、one-update、pilot 或 eval。

## Review

The clip boundary is evidence-only in one-update and pilot: records and pooled
summaries must be finite, exact, integer-backed, joint-complete and arithmetically
recomputable. No absolute clip threshold is an acceptance check. Survival
non-regression and forward-motion pilot gates remain blocking.

One-update acceptance includes exact shape, capacity consumption, checkpoint
production, transitions, clip-record validity, optimizer-step count, positive
finite actor+critic parameter delta, finite real loss dictionaries, exact runtime
reward terms and canonical reward-table SHA match, enabled RSL-RL NaN checking,
and finite exact-rollout evidence. Policy distribution lineage is recorded from
the actual runtime and is not retuned by this subtask.

CPU validation is limited to runner `py_compile`, focused existing Task072
clip/one-update/pilot tests, `git diff --check`, and the existing CPU runtime
verifier, producing a new `003j_reward_eval_contract_verifier.json`. If those
gates pass, GPU scope is limited to the separately authorized 003j capacity
smoke, fresh `4096x24x1` one-update seed `720301`, fresh `4096x24x21` pilot,
four fixed-command evals with seed `720400`, and the aggregate pilot gate.
Stop after the pilot result; do not start proof, video, freeze, Task073 or
Task074.
