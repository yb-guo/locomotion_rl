# 003g — G1 MJLab single-ground walking proof

状态：**in_progress / not_passed**。

在 003f 已通过的 `mjlab_g1_7capsule_task_v3_single_ground` runtime lineage 上，使用 RTX 5060 Ti
从随机初始化运行 v3 single-ground G1 正式训练。训练沿用 003f 的 immutable v2
contact asset/profile/stance，不修改 reward、command、PPO cadence、asset、stance 或 contact
material，不使用旧 checkpoint、Task048 checkpoint、外部下载、H200、Task073 或 Task074。

## Route

003f single-ground CPU verifier -> v3 2048/4096/6144 CUDA capacity smoke -> 4096x24 one-update
smoke -> 4096x24x650 random-init proof training -> fixed-command checkpoint eval -> numeric gate
通过后再生成 8 s no-reset video。任一 lineage、hash、single-ground audit、capacity、OOM、NaN/Inf 或
GPU lock 证据失败则停止；4096 capacity 未通过不得降级训练；numeric gate 未通过不得生成伪 passing
video、不得 freeze、不得调 reward 或参数搜索。

## Log

- 2026-08-31：建立 003g 执行记录。初始状态为 `in_progress / not_passed`；等待重新运行 CPU
  `verify-runtime-binding`、v3 capacity smoke、one-update smoke、正式训练和固定命令 eval。

## Review

状态：**not_passed**。003g 尚未完成训练与 walking gate 验收。
