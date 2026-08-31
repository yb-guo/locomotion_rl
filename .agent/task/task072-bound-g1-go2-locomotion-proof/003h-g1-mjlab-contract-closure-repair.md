# 003h — G1 MJLab contract-closure repair

状态：**ready_for_authorized_pilot / not_trained / not_passed**。

003h 是 003g 失败后的 versioned repair。003g 仅保留为 single-ground failed walking evidence；其
training contract 继承了 `Unitree-G1-Flat` 的 command randomization、domain randomization、curriculum、
parent reward 和对称 action scale，因此不得作为下一次训练的父合同。

## Route

003f single-ground runtime -> 003g failed evidence -> 003h contract closure -> user-authorized
capacity/one-update/pilot only after this source commit is selected。

003h 唯一活动训练路线是 MJLab。旧 custom-PPO 路线和 E3a artifacts 只作为 historical/rejected evidence；
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
- 2026-08-31：移除 active MJLab reward 中 inherited `is_terminated` 和 `feet_gait` terms；003h reward
  lineage 记录为 `task072_mjlab_biped_phase_contact_v3`。fall 仍作为 termination，但不再是 reward
  penalty。
- 2026-08-31：runner 不再回退到 `/home/admin1/workspace/proj` 的可变 external symlink；缺少 frame-local
  `.external/unitree_rl_mjlab` 时 fail closed。manifest 记录 external path、expected commit、actual commit
  和 tracked-clean 状态。
- 2026-08-31：`evaluate` 必须显式传入训练 manifest，checkpoint path 与 SHA 必须由 manifest 声明；
  任意 path/hash mismatch 均拒绝。新增 MJLab `render`、`verify-reload`、`freeze` fail-closed commands；
  在 numeric eval、video、independent verifier 未通过前不会生成 passing freeze。
- 2026-08-31：CPU scoped tests：`tests/test_task072_locomotion_proof.py tests/test_whole_body_extended.py`
  为 `75 passed`。未运行 GPU capacity、one-update、pilot、proof training、passing video 或 freeze。

## Review

状态：**ready_for_authorized_pilot / not_trained / not_passed**。003h 已关闭 training/eval/action/reward/
runtime/provenance 的主要合同分叉，并以 CPU tests 证明 fail-closed 行为。下一步训练 agent 只有在用户
授权 GPU capacity 与 pilot 后，才能从本 repair commit 重新开始；不得复用 003g checkpoint。
