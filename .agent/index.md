# Agent Index

H200 Locomotion Lab 的 agent 入口。

## Docs

- `doc/project.md`
- `doc/h200_strategy.md`
- `doc/sonic.md`
- `doc/locoformer.md`
- `doc/runtime_architecture.md`
- `../../README.md`

## Active Tasks

- `task/task001-agent-setup/task.md`
- `task/task002-sonic-mujoco-smoke/task.md`
- `task/task003-h200-simulator-smoke/task.md`
- `task/task004-genesis-g1-baseline/task.md`
- `task/task005-locoformer-min-reproduction/task.md`
- `task/task006-sonic-genesis-action-policy/task.md`
- `task/task007-sonic-g1-deployment-bridge/task.md`
- `task/task008-runtime-profile-foundation/task.md`
- `task/task009-sonic-action-rollout-matrix/task.md`
- `task/task014-minimal-ppo-smoke/task.md`
- `task/task015-g1-curriculum-longer-horizon-ppo/task.md`
- `task/task016-g1-long-horizon-tilt-reset-ablation/task.md`
- `task/task017-g1-action-control-semantics-diagnosis/task.md`
- `task/task018-g1-no-update-ppo-causality-diagnosis/task.md`
- `task/task019-g1-zero-action-standing-causality-diagnosis/task.md`
- `task/task020-standing-ppo-stabilization/task.md`
- `task/task021-genesis-alignment-bundle/task.md`
- `task/task022-ankle-roll-contact-patch-ablation/task.md`
- `task/task023-base-attitude-height-stabilization/task.md`
- `task/task023-franka-current-payload-estimation/task.md`
- `task/task027-unitree-mjlab-g1-from-scratch/task.md`
- `task/task028-randomized-wholebody-morphology-env/task.md`
- `task/task029-motor-only-randomization-and-failure-baseline/task.md`

## Consensus

本项目复用 HeadPose 项目的工作方式：

- `.agent/doc` 放长期共识和项目判断。
- `.agent/task` 放可执行任务，任务必须拆成小的 closed unit。
- 每个 subtask 固定 `Route / Log / Review`。
- 不把 H200 上完整 Isaac Sim/RTX 路线当主线。
- 所有训练和仿真路径都必须有 smoke test、失败条件和退出规则。

## Lessons

- H200 是训练卡，不是 RTX 仿真工作站。
- 先复现官方最短链路，再改算法。
- SONIC 先跑 MuJoCo sim2sim，再试 Isaac Lab headless。
- LocoFormer 先做最小可验证版本，不从 full scale 开始。
