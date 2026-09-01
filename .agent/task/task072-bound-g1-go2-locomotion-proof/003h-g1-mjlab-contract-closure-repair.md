# 003h — G1 MJLab contract-closure repair

状态：**rejected_diagnostic / trained / not_passed**。

003h 是 003g 失败后的 versioned repair。003g 仅保留为 single-ground failed walking evidence；其
training contract 继承了 `Unitree-G1-Flat` 的 command randomization、domain randomization、curriculum、
parent reward 和对称 action scale，因此不得作为下一次训练的父合同。

## Route

历史 route 为 `003f single-ground runtime -> 003g failed evidence -> 003h contract closure ->
user-authorized capacity/one-update/pilot`。该 route 已执行，但因 reward/eval 合同错误关闭；当前
successor 固定为 `003h rejected_diagnostic -> 003i reward/eval/survival repair`。

003h 当时唯一训练路线是 MJLab。旧 custom-PPO 路线和 E3a artifacts 只作为 historical/rejected evidence；
历史 SHA drift 必须被 verifier 拒绝，不得重写旧 artifact 使其通过。

## Log

- 2026-08-31：修复 `WholeBodyMuJoCoShard` logical foot reference site 行为。未显式配置 site 时使用已存在
  geom fallback；显式配置了不存在的 site 仍 fail closed。
- 2026-08-31：在 Task072 本地 MJLab runner 中新增
  `task072_mjlab_signed_headroom_v1` action contract。contract 从 v2 XML actuator range、stance action
  offset 和 motor tuple formula 生成 29/29 rows：motor delta、5% safety margin、negative/positive
  headroom、signed negative/positive amplitude。policy 输出域固定为 clipped `[-1, 1]`，position target
  按 raw action 符号选择 amplitude；zero action 等于 stance action offset。
- 2026-08-31：训练与 eval 统一消费 canonical fixed-command config：`[0.5, 0, 0]`、standing probability
  0、command resampling effectively disabled、push/friction/encoder/COM randomization disabled、actor
  corruption disabled、curriculum empty、random init PPO/no resume。eval 只允许 env count、seed、horizon、
  render mode、task id 和由 env count 派生的 transition count differ。
- 2026-08-31：文档曾声称移除 inherited `is_terminated` 和 `feet_gait` terms；后验确认 runner 实际
  只尝试删除错误键 `feet_gait`，parent 实际 key 为 `foot_gait`，且没有完整替换 RewardManager。
  因而 `task072_mjlab_biped_phase_contact_v3` 仅是标签，不是已实例化的 reward-v3 合同；触发
  `reward_contract_not_instantiated` rejection，转入 003i。
- 2026-08-31：runner 不再回退到 `/home/admin1/workspace/proj` 的可变 external symlink；缺少 frame-local
  `.external/unitree_rl_mjlab` 时 fail closed。manifest 记录 external path、expected commit、actual commit
  和 tracked-clean 状态。
- 2026-08-31：`evaluate` 必须显式传入训练 manifest，checkpoint path 与 SHA 必须由 manifest 声明；
  任意 path/hash mismatch 均拒绝。新增 MJLab `render`、`verify-reload`、`freeze` fail-closed commands；
  在 numeric eval、video、independent verifier 未通过前不会生成 passing freeze。
- 2026-08-31：CPU scoped tests：`tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py`
  为 `75 passed`。未运行 GPU capacity、one-update、pilot、proof training、passing video 或 freeze。
- 2026-09-01：收到用户授权后，在 repair commit
  `49a15b816b03378dc7a48e51726e0051a0de8a6b` 下执行 003h GPU route。Capacity smoke passed，
  artifact SHA `214524c37b9bb2c079292dc189b96b0555e08137c70236d1b775c476a81bcd5e`，selected
  `4096 x 24 = 98,304` transitions/update，highest passed `6144` envs，记录
  `gpu_lock.held_by_ancestor=true`。Capacity-consuming one-update smoke passed，manifest SHA
  `68806a6f706983f149a41d8fab0eb47487d6d5afdff698f37fe7c11f710aa21d`。随后执行 21-update pilot
  `4096 x 24 x 21 = 2,064,384` transitions，随机初始化 seed `720301`，manifest SHA
  `ef75777584f7a3a3b8a622ef120d7ec677c38477cdb5dcb8a893fe10f90c7231`，最终 checkpoint
  `model_20.pt` SHA `08a700768ce8310fe20dcb87e96653bd450ada22475fabbf9e8765533b4a17b9`。
  Manifest-bound 20 s fixed-command eval on `model_20.pt` failed walking gate，eval SHA
  `59f614ea2e0e4c5414f543bc6696cb582ad075338620cd6183a199accfa47fb0`；后验确认 eval 将最后一步
  `reset_time_outs` 与 `reset_terminated` 合并为 fall，并在无 survivor 时写入 0/sentinel，故按
  `eval_timeout_conflation` rejection 保留。model_20、pilot manifest、eval artifact 均为
  `rejected_diagnostic`，禁止 resume/warm-start/proof/freeze。未运行 proof training、passing video 或 freeze。

## Review

状态：**rejected_diagnostic / trained / not_passed**。003h 的训练可执行性证据保留，但 reward contract
未实例化、eval timeout 语义错误，且未记录可审计 clip/survival non-regression；不得复用 003g/003h
checkpoint，必须执行 003i fresh-init route，不得生成 passing video 或 freeze。
