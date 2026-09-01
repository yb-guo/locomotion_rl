# 003j — G1 MJLab action-clip gate scope repair

状态：**pilot_failed / trained / not_passed**。

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
- 2026-09-01：CPU implementation and regression coverage completed on source commit
  `f799115a5575d5c492c834467ec4ad35ad816750`; runner SHA
  `4b0612ab88cd1c60701ac354a46b5573196ebcf988bf4c3511ce2fcc16a9c4e8`。The final
  manifest validator compares serialized acceptance-check key sets, so canonical JSON
  key ordering cannot reject an otherwise valid runtime manifest.
- 2026-09-01：CPU `py_compile`、聚焦 clip/one-update/pilot tests、`git diff --check` 与
  `003j_r2_reward_eval_contract_verifier.json` 均通过；verifier artifact SHA
  `6ddffb76ec98ee08b2a27358813da4989a1d7242254eabac160dcf09a9a321ff`。
- 2026-09-01：003j GPU capacity 未启动；`flock -n /home/admin1/workspace/run/.gpu.lock`
  返回 busy，holder 为其他 EmbodiedGen GPU 任务（PID 49088/49099）。按 stop rule 标记
  `gpu_pending`，不等待、不轮询，不运行 capacity、one-update、pilot 或 eval。该条为
  后续 r2 GPU 授权前的历史记录。
- 2026-09-01：按授权在 inherited `/home/admin1/workspace/run/.gpu.lock` 下完成 r2
  capacity smoke；`2048/4096/6144` 均 passed，selected `4096 x 24 = 98,304`
  transitions/update，`gpu_lock.held_by_ancestor=true`。Artifact
  `003j_r2_capacity_smoke_2048_4096_6144.json` SHA
  `21890c28dcaf5dc22d78abc1e765fbacf426fc9e51ea546345ff78fd9323fb0b`。
- 2026-09-01：fresh exact `4096 x 24 x 1` one-update seed `720301` passed from random
  initialization。实际 optimizer steps `20/20`；actor+critic parameter delta
  `max_abs=0.004554471932351589`、changed `438331`，loss dictionaries finite；实际
  runtime reward active term count `23`、table SHA
  `323feac6197abc6d706205f39d5f332b834e87332d453919ccfb1998f5eea7e2`，
  check-for-NaN enabled，observed rollout `24`，all finite。Clip 只保留 schema/count/
  integer/fraction/finite/pooled arithmetic evidence；此 run scalar clip fraction
  `0.3195793765714799`、env-step-any `0.9999898274739584`、max raw `5.651525974273682`
  均不再被 one-update threshold 拒绝。Manifest SHA
  `5d4ce6e8d278600aaf3de54b8a9c62336de8fe9d266428358c993e5e0b45486d`，model SHA
  `c24ff439e6950842c5a6cc20c3647534ea1f542147d7e0bd4e0766ad9f867871`。
- 2026-09-01：fresh `4096 x 24 x 21` pilot seed `720301` completed `2,064,384`
  transitions and passed the training acceptance contract (`420/420` actual optimizer
  steps, finite positive parameter delta, finite losses, exact reward/NaN/rollout
  evidence)。Pooled updates `14..20` clip summary remained finite and arithmetically
  valid (scalar fraction `0.351400340914922`, env-step-any `0.9999956403459821`, max raw
  `5.885787487030029`); no clip threshold was applied. Training manifest SHA
  `d2dd617050b5ecfdf11a6456769ed50ad0de5e978abf10dac5ec89bb959041b0`；checkpoint
  SHAs: model0 `80357096fa668eb0659e4e5b229302bc83e9a541ff6fd871a9f3b346135a8fd6`,
  model7 `0d87c3f180ed16f510ecdf782274ec75a1b22c2e6526ef9b616c1597ba29c95e`,
  model14 `c9451fb351ea426a71ab14d8a66974ca19c61916410fb283990d22f324c5f970`,
  model20 `84cd5dae651821b00d67525deda164453d031b791d859b1ba8666195af3a7c22`。
- 2026-09-01：manifest-bound fixed-command evals `model_0/7/14/20` ran with seed
  `720400` and finite observations/rewards, zero timeouts, but all four policies fell
  before the 20 s horizon (`zero_fall_ratio=0`), so no individual eval passed. Median
  first-fall / common-prefix mean vx / median x were respectively `2.44/-0.3204/-0.6655`,
  `2.68/0.2865/0.7245`, `1.42/0.4085/0.5495`, and `1.14/0.4628/0.5006` for models
  `0/7/14/20`; eval clip evidence was recorded without thresholds.
- 2026-09-01：aggregate `003j_r2_pilot_gate.json` ran under the same lock and failed
  closed。Training/eval-set/schema/clip/reward checks all passed, forward checks passed,
  but continuation failed at model20 `>=2.5 s`, model20-vs-model0 `+0.5 s`, model14-vs-
  model7 `-0.25 s`, model20-vs-model7 `-0.10 s`, and model20-vs-model14 `-0.25 s`。
  Gate SHA `c460a0d5095b88943c0550ec6bae6c35c373e718407e4d379029e8d057fce43f`。
  Per stop rule, no proof, video, freeze, Task073 or Task074 was started。

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
verifier, producing a new `003j_r2_reward_eval_contract_verifier.json`. If those
gates pass, GPU scope is limited to the separately authorized 003j capacity
smoke, fresh `4096x24x1` one-update seed `720301`, fresh `4096x24x21` pilot,
four fixed-command evals with seed `720400`, and the aggregate pilot gate.
Stop after the pilot result; do not start proof, video, freeze, Task073 or
Task074. The r2 pilot gate failed on the required survival non-regression
comparisons; the subtask remains `pilot_failed / trained / not_passed`.
