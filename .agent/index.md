# Agent Index

Locomotion Lab 的 agent 入口。当前主力硬件是 RTX 5060 Ti；H200 profile
已暂停。

## Docs

- `doc/project.md`
- `doc/h200_strategy.md`
- `doc/sonic.md`
- `doc/locoformer.md`
- `doc/runtime_architecture.md`
- `doc/component_architecture.md`
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
- `task/task030-online-dynamic-motor-failure-adaptation/task.md`
- `task/task031-unified-forward-speed-dynamic-switch-mlp/task.md`
- `task/task032-focused-deadgrid-mlp-ceiling-test/task.md`
- `task/task033-shared-history-memory-policy-bakeoff/task.md`
- `task/task034-right-knee-dead-history-curriculum/task.md`
- `task/task035-eval-gated-history-curriculum/task.md`
- `task/task036-adaptation-gru-token-policy-quality/task.md`
- `task/task037-locoformer-style-multitrial-long-context-training/task.md`
- `task/task038-locoformer-min-g1like-reproduction/task.md`
- `task/task039-true-txl-quality-training-diagnosis/task.md`
- `task/task040-sequence-aware-txl-ppo-update/task.md`
- `task/task041-sequence-aware-txl-clean-gait/task.md`
- `task/task042-txl-memory-causality-and-residual-training/task.md`
- `task/task043-memory-required-dynamic-switch-training/task.md`
- `task/task044-memory-required-fault-identification-target/task.md`
- `task/task045-left-knee-continuous-stability-tail-gate/task.md`
- `task/task046-retry-after-fall-adaptation-eval/task.md`
- `task/task047-rtx5060ti-mjlab-setup/task.md`
- `task/task048-rtx-normal-walking-baseline/task.md`
- `task/task049-component-boundary-redesign/task.md`
- `task/task050-whole-body-contract/task.md`
- `task/task051-procedural-whole-body-generator/task.md`
- `task/task052-whole-body-adapter-motor-process/task.md`
- `task/task053-specialist-normal-walk/task.md`
- `task/task054-shared-mlp-cross-morphology/task.md`
- `task/task055-hidden-online-motor-process/task.md`
- `task/task056-gru-transformer-xl/task.md`
- `task/task057-heldout-ood-evaluation/task.md`
- `task/task058-scale-and-flow-algorithms/task.md`
- `task/task059-rtx5060ti-primary/task.md`
- `task/task060-2000-usability-gate/task.md`
- `task/task061-rtx-specialist-normal-walk/task.md`
- `task/task062-shared-mlp-train-topologies/task.md`
- `task/task063-hidden-motor-process-training/task.md`
- `task/task064-gru-txl-adaptation/task.md`
- `task/task065-heldout-ood-final-gate/task.md`
- `task/task066-scale-flow-independent/task.md`
- `task/task067-biped-stance-contract/task.md`
- `task/task068-repository-ruff-cleanup/task.md`
- `task/task069-locoformer-paper-faithful-morphology/task.md`
- `task/task070-archetype-constrained-standable-morphology/task.md`
- `task/task071-multimorphology-training-readiness/task.md`
- `task/task072-bound-g1-go2-locomotion-proof/task.md`
- `task/task073-explicit-terminal-contact-assets/task.md`
- `task/task074-all-configuration-training/task.md`

## Consensus

本项目复用 HeadPose 项目的工作方式：

- `.agent/doc` 放长期共识和项目判断。
- `.agent/task` 放可执行任务，任务必须拆成小的 closed unit。
- 每个 subtask 固定 `Route / Log / Review`。
- 不把 H200 路线当主线；默认只在 RTX 5060 Ti 上开发和训练。
- 所有训练和仿真路径都必须有 smoke test、失败条件和退出规则。

## Lessons

- RTX 5060 Ti 是当前主力开发和训练卡；H200 仅保留历史记录。
- 先复现官方最短链路，再改算法。
- SONIC 先跑 MuJoCo sim2sim，再试 Isaac Lab headless。
- LocoFormer 先做最小可验证版本，不从 full scale 开始。
