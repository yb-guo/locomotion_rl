# 003a — E3a MJLab/RSL KL correctness repair

状态：**failed / R5_completed_rejected / optimizer_gate_failed / stopped_before_render_eval**。

Owner：PPO optimizer、policy likelihood 与 Task072 verifier owner。

本 subtask 修复已拒绝 `E3a_adaptive_kl` 暴露的 optimizer/likelihood correctness 问题。现有
`artifacts/nominal_v4/unitree_g1/E3a_adaptive_kl/` 保持 immutable rejected diagnostic evidence，
不得覆盖、续训或补判通过。修复后的候选 variant 预留为 `E3a_mjlab_kl_repair`，parent 仍为已通过
scale gate 的 `E2_reward_dt`；只有用户另行授权后，才允许从 seed `72072` 随机初始化运行。

本 task 不启动 E3b/E4，不改变 reward、observation、action residual bounds、asset、command、batch、
minibatch、epochs、entropy coefficient 或训练预算，不切换成 Task048 checkpoint，也不使用 H200、外部
下载、randomization 或 curriculum。为保持 action contract，环境仍接收现有 tanh-squashed action；
不得借“对齐 MJLab”顺手改成 unsquashed environment action。

## Route

1. **R0 — contract and lineage reconciliation**
   - 在任何代码修改前，保留 rejected E3a 的 manifest/progression/gate SHA；
   - fail-closed 绑定 `algorithms/ppo.py`、`masked_distribution.py`、`whole_body_mlp.py`、trainer、
     task-local CLI 与 verifier；
   - 记录本机只读参考版本 `mjlab==1.2.0`、`rsl-rl-lib==5.0.1` 及所比对文件 SHA；参考包不成为
     runtime dependency。
2. **R1 — likelihood identity without training**
   - rollout 保存 old Gaussian mean/std 与 pre-tanh raw action；环境 action 仍为 `tanh(raw_action)`；
   - old/new log-prob 必须对同一个 raw sample 求值，不能从 clamp 后的 squashed action 反推 raw；
   - 在参数未更新时，普通样本和超出 `atanh(1-1e-6)` 的饱和样本都必须满足
     `max_abs_log_prob_delta <= 1e-5`、`abs(approx_kl) <= 1e-6`、`clip_fraction == 0`。
3. **R2 — MJLab/RSL scheduler semantics**
   - analytic `KL(old||new)` 先对每个 sample 的 active action dimensions 求和，再对 minibatch 求均值；
   - 每个 minibatch 在 optimizer step 前，用 rollout 保存的 old distribution params 与当前 params
     做 scheduler decision；
   - 保持 `desired_kl=.01`、`KL>.02` 时 LR `/1.5`、`0<KL<.005` 时 LR `*1.5`、LR clamp
     `[1e-5,1e-2]`；inactive slots 不得进入 KL；
   - raw telemetry 至少保存 update/epoch/minibatch、joint scheduler KL、LR before/after、decision、
     sampled approximate KL、clip fraction 与 same-policy identity error。
4. **R3 — no-update correctness gate**
   - synthetic 29-active-dimension fixture 必须证明 joint KL 等于 per-dimension KL 之和，而不是均值；
   - 构造 joint KL `.004/.01/.021`，分别验证 increase/hold/decrease；
   - 验证 first minibatch 在任何 optimizer step 前 likelihood identity 成立；
   - canonical config diff 只能是 E2 到 adaptive optimizer strategy 的既有 allowlist；不得移动原
     E3a `approx_kl <= .015/.03/.05` 与 clip `<= .20/.35` gate。
5. **R4 — authorized smoke only**
   - 仅在 R0–R3 全部有验证证据且用户明确授权后，运行最短现有 smoke；
   - smoke 必须 finite、完整记录每-minibatch scheduler，并证明首个未更新 minibatch
     `approx_kl≈0`、`clip_fraction=0`；失败即停止，不进入 2.048M。
6. **R5 — authorized repaired E3a rerun**
   - 仅在 smoke 通过且用户再次授权后，从 E2 config、seed `72072` 随机初始化，在新的
     `E3a_mjlab_kl_repair/` artifact root 完整运行 2,048,000 transitions；禁止 resume 旧 E3a；
   - 继续使用原 optimizer KL/clip gate、20 s paired eval、8 s diagnostic、视频和 verifier；
   - repaired gate 通过前，E3b、E4、20M progression、004 freeze 与 005 Go2 rerun全部保持 blocked。

## Log

- 2026-08-28：只读检查 `E2_reward_dt` 与 provenance-complete `E3a_adaptive_kl` artifacts，并对照本机
  `mjlab==1.2.0` / `rsl-rl-lib==5.0.1`。Task072 scheduler 对 batch×29 active action elements 取均值，
  而 sampled log-prob 对 active action dimensions 求和；E3a scheduler KL mean/p95/max
  `0.008998/0.012922/0.018500` 换成 joint 口径约为 `0.260956/0.374747/0.536491`。
- 2026-08-28：E3a 在同一 update 尚未执行 optimizer step 的 `epoch=0,index=0` 已出现非零乃至巨大
  approximate KL；update 1000 为 `0.890499`、clip fraction `0.269531`。代码路径显示 rollout old
  log-prob 使用原始 Gaussian sample，但训练只保存 squashed action，evaluate 再从 clamp 后 action
  `atanh` 重建 raw；final active mean `log_std=0.9125`、mean std `2.4964` 时该 round-trip 不再一致。
- 2026-08-28：本轮只建立 003a contract；未修改实现、未运行测试或训练、未启动 E3b/E4，未使用
  Task048 checkpoint、外部下载或 H200。
- 2026-08-28：在 recovery worktree
  `/home/admin1/workspace/run/locomotion_rl/task071-1` 完成 R0-R3 correctness repair；未运行
  smoke 或 2.048M 训练。实现改动限定在 raw likelihood / adaptive scheduler / verifier 接线：
  rollout 保存 pre-tanh raw action、old Gaussian mean/std，环境 action 继续为 `tanh(raw_action)`；
  PPO 对 raw sample 重算 old/new log-prob，adaptive scheduler 改为每 minibatch optimizer step 前
  决策，analytic KL 为 active dimensions 求和后 minibatch mean；raw telemetry 记录 scheduler KL、
  LR before/after、decision、sampled approx KL、clip fraction 与 same-policy identity error。
- 2026-08-28：新增并运行 no-update verifier：
  `verify-e3a-kl-repair --output
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v4/unitree_g1/E3a_mjlab_kl_repair/no_update_correctness_gate.json`。
  输出 SHA `d2e3650be8e3b112688898bdb8441fb761c14171455a574c1d186adcfbbb80c3`，`r0_r3_passed=true`，
  `training_started=false`，`smoke_started=false`。该 gate 重新绑定旧 E2/E3a SHA：E2 manifest
  `25ef89af5c7460701a8b8f5f2b48de876dbcf34b7405e54dac7471e261b35597`、E2 gate
  `fe41b57a91b715254f09625b73339e6403d02c1caa696ce6b73c0a8571b2910e`、rejected E3a manifest
  `3974dbadfe7d96b234878cc14fe6edfb58def64ea2a154747688bda5f118e33a`、progression
  `45e9a52ee07a6166fe1ffe2cd1af43a44a4f9bda7fdb3e4f07261782db080d27`、gate
  `3cb5594110432cf6bc4ab10bd4d74acec470468e8fe2cb16d05177ceb1706d1e`，并确认 rejected E3a
  仍为 rejected。
- 2026-08-28：R0 记录本机只读参考 `mjlab==1.2.0`、`rsl-rl-lib==5.0.1`，source SHA 分别包含
  RSL PPO `a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c`、RSL distribution
  `4631eae1939dcd6b065d79c3c0128a88ba2c6126d08f46944c8795682a208a1b`、MJLab init
  `866b4b673f8bf2a779fd18c97d875833d55520c7e99157edf3b5a759fd8632cc`；参考包未成为 runtime
  dependency。
- 2026-08-28：R1-R3 no-update gate metrics：regular/saturated raw likelihood identity error 均
  `0.0`，same-policy sampled approx KL `0.0`，clip fraction `0.0`；旧 atanh-from-squashed 路径对
  saturated raw sample 的 drift 为 `2610.556640625`。29-active-dim joint KL fixtures 为
  `.004 -> 0.0039998293`、`.01 -> 0.0099995732`、`.021 -> 0.0209999681`，scheduler decision
  分别为 increase/hold/decrease。canonical config diff 仍限于 E2 到 adaptive optimizer strategy
  allowlist，原 E3a approx-KL/clip gate 阈值保持 `0.015/0.03/0.05` 与 `0.20/0.35`。
- 2026-08-28：根据 reviewer P1/P2 修复后重跑 gate：`adaptive_kl=True` 时缺少 `raw_actions`
  或 old distribution params 现在 fail closed，不再回退到 update-end scheduler；no-update verifier
  通过真实 `ppo_update` 首 minibatch telemetry 验证 pre-step identity，记录
  `first_minibatch_identity_error=0.0`、`first_minibatch_sampled_approx_kl=0.0`、
  `first_minibatch_clip_fraction=0.0`、`first_minibatch_scheduler_kl=0.0`、
  `no_update_parameter_delta_max=0.0`。第二轮 reviewer 继续发现 raw evaluator `TypeError` fallback
  仍可能回到 squashed likelihood；已改为 raw replay evaluator/mask support failure 直接 fail closed，
  并新增回归测试。
- 2026-08-28：验证命令：
  `env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src"
  /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python -m py_compile
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py
  src/h200_locomotion_lab/masked_distribution.py
  src/h200_locomotion_lab/policies/whole_body_mlp.py
  src/h200_locomotion_lab/algorithms/ppo.py
  src/h200_locomotion_lab/training/whole_body_ppo.py
  tests/test_task072_locomotion_proof.py` passed；`git diff --check` passed；focused pytest
  `tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py` returned `46 passed`。
  `E3a_mjlab_kl_repair/` 目录只包含 `no_update_correctness_gate.json`，没有 checkpoint、
  progression、render/eval 或 smoke artifact。
- 2026-08-28：按用户明确授权只运行 R4 repaired smoke，未运行 R5/2.048M。命令：
  `env PYTHONPATH="$PWD/src" /home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
  .agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py train --case
  unitree_g1 --stage smoke --variant E3a_mjlab_kl_repair --device cuda:0 --run-dir
  .agent/task/task072-bound-g1-go2-locomotion-proof/artifacts/nominal_v4/unitree_g1/E3a_mjlab_kl_repair/smoke`。
  训练正常返回：2 updates、4 envs、32 rollout steps、256 transitions、两次 update 的
  `fall_count=0.0`，reward mean `0.02099193088361062/0.021519083820749074`。
- 2026-08-28：最初手工汇总的 R4 gate SHA
  `7722371e68684f521d61beebc681104345e5e20bb01dc1c841a18c303e221c5e` 后续审核发现 preservation
  字段为 false 却仍 pass，已明确 rejected；该 gate 不能作为通过证据。Smoke artifacts 本身保持不变并
  写入独立 `smoke/`：
  `run_manifest.json` SHA `04d1aba1a459832f704bb3724ec04cc70cf026ed6fef93543cc8ab57a628db40`，
  `progression.json` SHA `28bfe92d62275985623945f56a13e9d91009d91fbfff95fd0d11635a5e500651`，
  `initial.pt` SHA `c657a6a7a8bf31c7b798390ee811a953c31f4b78b7adc1b2bcd6c480a9a0da39`，
  `final.pt` SHA `0c851a233f99c930aa78ec290c9a786336502f6f160c7707efd939a0f2273688`。
  后续正式 verifier 从这些原始 artifact 重算所有 JSON/checkpoint/source/lineage；8 个 minibatch 均有
  scheduler telemetry，两个 update 的第一个未更新 minibatch
  `approx_kl=0.0`、`clip_fraction=0.0`、`scheduler_kl=0.0`、`same_policy_identity_error=0.0`；
  参数更新 finite 且非零，`delta_max_abs=0.0006282080430537462`、`delta_l2=0.19559913081391125`。
  no-update gate SHA 仍为
  `d2e3650be8e3b112688898bdb8441fb761c14171455a574c1d186adcfbbb80c3`，旧 rejected E3a artifacts
  保持原 SHA。
- 2026-08-30：用户明确授权 R5 后，在 RTX 5060 Ti 上以 GPU lock、`uv.lock`、Python 3.11 和
  locked/offline `uv run --isolated` 从 seed `72072` 随机初始化完成 `32 envs × 64 steps × 1000
  updates = 2,048,000` transitions；未使用 Task048 checkpoint、外部下载或 H200。训练正常返回且无
  OOM，update `200/400/600/800/1000` 的 `fall_count` 为 `17/21/4/3/5`。`run_manifest.json` SHA
  `4b8bc6ce658fe28829e4b0b71386cf8c058e7fa0952b3ac21e4c1a886a98a85e`，`progression.json` SHA
  `cb008bf389f46cf0d8008ac1c8c8d88507d5da8425236c0bb028097566f4069b`，`initial.pt` SHA
  `25ebde4805f7874f4dc99dd494bf03f3c8085e6367d4469056414b9b871d6802`，`final.pt` SHA
  `9208a97d7191cec1425504194db776c85c3082902c25efeefacaa5d45d2f1398`。
- 2026-08-30：真实 R5 暴露两项失败。其一，sampled approximate-KL 的理论零点存在 float32
  cancellation：983 个 raw 值位于 `[-5.0059e-9,-2.3283e-10]`，row aggregate 的最大重算差为
  `4.5984e-9`，而 clip aggregate 差为 `0.0`。Verifier 只把 per-minibatch lower bound 和 approx
  aggregate tolerance 绑定为 `1e-8`，clip 仍为 `1e-9`，所有原 upper thresholds 未改；新增边界测试后
  locked/offline focused pytest 为 `58 passed, 1 deselected`。被 deselect 的既有 no-update reference
  test 需要未进入 `uv.lock` 的 `rsl_rl/mjlab`，正式 no-update artifact SHA 未改。
- 2026-08-30：修正数值误杀后，官方 optimizer gate 仍 fail closed：R5 update0 policy 与 R4/E2 fixed
  random initialization 不完全相等，243,803 个参数中 96,351 个不同，最大差 `1.9372e-7`。R5 locked
  runtime 为 MuJoCo `3.12.0`，既有 R4/E2 manifest 记录 `3.5.0`；该 runtime/init drift 不允许补判。
  即使只读越过此项重算，KL mean/p95/max 为 `0.0150795/0.0274182/0.0554737`，clip mean/p95 为
  `0.220988/0.3828125`，原五项阈值中四项失败。`e3a_gate.json` SHA
  `65697821095c9f7274d85603b470b074e2ae3393e6b3e58277da215e385294ca`；按 gate-fail stop，未运行
  render、20 s eval、8 s diagnostic、E3b 或 E4。

## Review

2026-08-28：新增独立 task-local `task072_e3a_repair_verifier.py`，不改已被 smoke manifest 绑定的训练
CLI/source。Verifier 以显式 smoke/optimizer 子命令、固定 artifact/source/current-file SHA、严格 JSON、
完整 run/static/checkpoint lineage、每-minibatch scheduler/first-MB 和 update0 random-init 检查修复旧 R4
自证问题。旧无效 R4 gate SHA `7722371e68684f521d61beebc681104345e5e20bb01dc1c841a18c303e221c5e`
及未满足最终合同的中间 gate SHA
`f34069875981c704e617ed5e0c246e72d7a1e8209850373bf2fffa9af80059dc`、
`39ff24bd9b34c8e3d6be56050855c394b017b693d974aceef7aee1c3d9404fbd`、
`110713b5bd5f5b7686480f197bd94b49c6f49ee7ae8f4b8c1ec1a4158df52551`、
`605787f787143281e3c3e8a3448923f8ce4a0dbbb0fc195316346bad6e858b00`、
`e718630badafa2b502dda1e5f3f518c54b3af2371e9456f772aa20b8efcfb182`、
`83f95ccdc5aaaafcbcf50f64d8ccd4c1cfe3ca110b8e3f63fbc97d123aa59a2a` 均只保留为 rejected/superseded
audit history；使用现有 smoke 重生成官方 gate。
正式 gate schema 为 3，SHA
`27c75fe817d4a8e8f556c37208f248c5e503f4109ae0d2840b3d11652928e693`；verifier source SHA
`f14c173f878006234e09911949ff826b4f6aac3df6d64bb54b1a0b20c9c0192f`。Gate 重算并绑定 243,803 个
policy 参数，`delta_max_abs=0.0006282080430537462`、`delta_l2=0.19559913081391125`，并证明 update0
与 E2 同 seed 随机初始化完全一致。真实 R5 的 1000 updates/32,000 minibatches 已由 optimizer
verifier 检查，但 exact update0 binding 与原 optimizer 阈值均失败。

状态：**R4_formal_gate_passed / R5_completed_rejected / optimizer_gate_failed**。

R5 artifacts 保留为 rejected evidence，不能成为 R2 selected E3 候选。Render/eval、E3b/E4、20M
progression、freeze 与 Go2 rerun 继续 blocked；Task072 保持 not_passed。
