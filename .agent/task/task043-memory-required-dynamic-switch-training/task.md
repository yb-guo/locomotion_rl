# Task 043: Memory-Required Dynamic Switch Training

## Route

Task042 closed as a negative memory-causality result. The true-TXL path is
plumbed, trainable, and measurable, but the residual-only checkpoint does not
need memory in the tested cases:

- easy right-knee 0.4 m/s dynamic eval passes even when the residual is zeroed
  or stateful memory is disabled;
- harder 1.6 m/s dynamic switch does not fall, but normal mode misses the
  strict quality gate and remains behaviorally tied with ablations.

Task043 changes the target instead of tuning the same checkpoint. The goal is a
memory-required dynamic-switch training loop:

- train on the True-TXL dynamic-failure MJLab task;
- preserve the Task041/Task042 runner, actor, action, observation, and reward
  contracts;
- evaluate normal vs zero-residual vs stateless-memory modes on the same
  dynamic switch;
- only call memory causality positive if normal passes and ablations degrade.

## Fixed Scope

- G1-like MJLab topology only.
- Runner: `Task038TrueTxlMemoryK160Runner`.
- Actor: `Task038TrueTxlMemoryModel`.
- PPO algorithm: `Task040SequenceAwareTrueTxlPPO`.
- First task id:
  `Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6`.
- Training base: reuse the Task041 train wrapper machinery; do not fork a new
  PPO implementation.
- Eval base: reuse Task042 memory-ablation dynamic eval unless a missing field
  blocks the Task043 gate.
- No LocoFormer reproduction, superiority, or held-out morphology claim.
- Do not touch `.test_tmp_task021/`.

Out of scope:

- morphology randomization;
- link-length/domain topology generalization;
- full LocoFormer transformer parity;
- changing reward/action/obs contracts before the first Task043 train/eval loop
  is measured.

## Planned Slices

1. `001-dynamic-train-entrypoint.md`
   - Add a Task043 train CLI and MJLab registry patcher.
   - Local tests must prove the new task id uses
     `Task038TrueTxlMemoryK160Runner` and the train wrapper keeps no-overclaim
     fields false.

2. `002-h200-train-smoke.md`
   - Sync the Task043 train CLI and registry patcher to H200.
   - Run a short H200 train smoke from the Task041/Task042 prior.
   - Record JSON proving `Task040SequenceAwareTrueTxlPPO`,
     `Task038TrueTxlMemoryModel`, active sequence updates, no stateless fallback,
     and a checkpoint path.

3. `003-dynamic-ablation-eval-gate.md`
   - Evaluate the produced checkpoint through the Task043 eval wrapper on
     `dynamic_case=switch` at 1.6 m/s with normal, zero-residual, and
     stateless-memory modes.
   - Compare final-trial quality and ablation deltas.

4. `004-scale-to-quality-pass.md`
   - If smoke works but normal quality fails, continue training chunks and
     evaluate candidate checkpoints until normal dynamic switch passes or a
     blocker diagnosis is documented.

5. `005-memory-causality-review.md`
   - Close the task only with a documented result:
     - positive memory causality, if normal passes and ablations degrade; or
     - negative/blocked result with a next-policy-contract recommendation.

6. `006-fixed-speed-train-target-alignment.md`
   - If range-based dynamic training does not improve the fixed 1.6 m/s eval,
     patch the Task043 H200 registration to train on fixed
     `lin_vel_x=(1.6, 1.6)` while keeping the dynamic failure schedule and
     policy contract unchanged.

7. `007-model5349-true-txl-bridge.md`
   - If fixed-speed training from the Task042 prior still misses quality,
     migrate the known high-speed MLP prior `model_5349` into the True-TXL
     actor and evaluate the same normal/zero/stateless dynamic-switch triplet.

8. `008-bridge-residual-trainable-scope.md`
   - Repair the residual-only trainable scope so the memory latent has a
     trainable path into the action head while the base MLP prior remains
     frozen, then test whether training the residual from the bridge creates
     ablation-sensitive memory behavior.

## Acceptance Criteria

Task043 is accepted only when current evidence proves:

- local tests cover:
  - Task043 train CLI defaults;
  - Task043 dynamic train task preflight;
  - Task043 registry patcher inserts the correct runner;
  - no-overclaim train summary fields;
- H200 train smoke records:
  - `task043_dynamic_switch_train=true`;
  - `task043_train_pipeline_pass=true`;
  - `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
  - `actor_model_class=Task038TrueTxlMemoryModel`;
  - `runner_cls=Task038TrueTxlMemoryK160Runner`;
  - `stateless_fallback_forward_batches=0`;
  - sequence update counters active;
  - checkpoint path exists;
- H200 dynamic eval records normal, zero-residual, and stateless-memory JSONs
  on the same checkpoint, seed, speed, and dynamic switch setting;
- normal mode must pass the selected dynamic-switch quality gate before any
  quality success claim;
- memory causality requires normal mode to outperform ablations materially;
- if normal does not pass or ablations remain tied, the review must state that
  and avoid a positive memory claim.

## Evidence Gate

Local:

```powershell
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task043_dynamic_switch_train --help
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task043_dynamic_switch_training_contract.py tests\test_task042_memory_ablation_contract.py tests\test_agent_inventory.py
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent
```

H200 train smoke shape:

```bash
PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab \
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task043_dynamic_switch_train \
  --resume-checkpoint /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/logs_txl_residual_only_env1024_iter5/model_4.pt \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_smoke.json \
  --log-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_smoke \
  --num-envs 64 \
  --rollout-steps 24 \
  --iterations 1 \
  --save-interval 1 \
  --num-mini-batches 4 \
  --actor-trainable-scope txl_residual_only \
  --device cuda:0
```

## Log

- 2026-05-31 Opened after Task042 harder dynamic switch ablations remained
  tied and did not support memory causality.
- 2026-05-31 Added Task043 train CLI:
  `src/h200_locomotion_lab/tools/task043_dynamic_switch_train.py`. It reuses
  Task041 training machinery with the Task043 dynamic task id and preserves
  no-overclaim summary fields.
- 2026-05-31 Added H200 registry patcher:
  `task043_register_dynamic_switch_train_stage.py`.
- 2026-05-31 Added Task043 dynamic ablation eval wrapper:
  `src/h200_locomotion_lab/tools/task043_dynamic_ablation_eval.py`.
- 2026-05-31 H200 train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_smoke_env64_iter1.json`.
  It produced checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_smoke_env64_iter1/model_0.pt`
  with `task043_train_pipeline_pass=true`,
  `algorithm_class=Task040SequenceAwareTrueTxlPPO`,
  `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`,
  `stateless_fallback_forward_batches=0`,
  `sequence_update_forward_batches=20`, and frozen baseline deltas at zero for
  `actor-trainable-scope=txl_residual_only`.
- 2026-05-31 H200 smoke checkpoint eval triplet recorded under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/`.
  Normal mode completed without falling and kept `pipeline_pass=true`, but
  `quality_gate_pass=false` because `lin_vel_error.mean=0.477243572473526`.
  Zero-residual and stateless-memory final velocity errors were `0.4860592782497406`
  and `0.4758363664150238`, respectively, so the smoke checkpoint still has no
  positive memory-causality evidence.
- 2026-05-31 Ran two 1024-env, 25-iteration scale probes. Both training
  pipelines passed, but fixed 1.6 m/s normal eval still failed quality:
  residual-only model_24 had final `lin_vel_error.mean=0.4807707965373993`;
  all-scope model_24 had final `lin_vel_error.mean=0.7321958541870117`.
- 2026-05-31 Diagnosed the next blocker as train/eval target mismatch:
  Task043 was registered with a dynamic-failure config that samples
  `lin_vel_x=(1.0, 1.6)`, while the eval gate fixes 1.6 m/s. Added 006 to patch
  the Task043 registration to fixed `lin_vel_x=(1.6, 1.6)`.
- 2026-05-31 Fixed-speed target alignment is technically working but not yet
  sufficient. H200 fixed-1.6 residual-only model_24 eval failed with final
  `lin_vel_error.mean=0.48685064911842346`; continued residual-only model_74
  improved to `0.4760773777961731` but still failed the quality gate. A
  fixed-1.6 all-scope model_24 regressed to `0.876373827457428`.
- 2026-05-31 Migrated the known high-speed MLP prior `model_5349` into the
  True-TXL actor after fixing partial actor copy and K160 history normalizer
  expansion. Warmstart JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/model5349_true_txl_warmstart/model_5349_task043_true_txl_bridge_history_norm_zero_tail.json`;
  checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/model5349_true_txl_warmstart/model_5349_task043_true_txl_bridge_history_norm_zero_tail.pt`.
- 2026-05-31 The `model_5349` True-TXL bridge passed normal 1.6 m/s
  dynamic-switch quality:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_none_vx1p6_seed4301701.json`,
  final `lin_vel_error.mean=0.4467167258262634`,
  `quality_gate_pass=true`, `fall_ratio=0.0`. Zero-residual and
  stateless-memory ablations also passed with tied velocity error
  (`0.44687220454216003` and `0.446464478969574`), so this is a base-prior
  bridge pass, not a positive memory-causality result.
- 2026-05-31 Found and fixed the next residual training blocker: old
  `txl_residual_only` training from the bridge changed no TXL parameters because
  the memory latent had no trainable connection into the action MLP. Added
  `txl_residual_and_mlp_memory_input`, which trains TXL residual parameters and
  masks `mlp.0.weight` so only the memory-latent columns can change.
- 2026-05-31 H200 training with the repaired scope now changes the memory path:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_model5349_bridge_residual_mlp_memory_input_env1024_iter25_seed4301901.json`,
  with `partial_frozen_delta_norm=0.0`,
  `memory_output_projection_delta_norm=0.9404060244560242`, and
  `txl_residual_output_norm_last=1.6352477073669434`. Its model_24 normal eval
  did not fall and had final `lin_vel_error.mean=0.44369199872016907`, but
  `quality_gate_pass=false` due posture/height regression; zero-residual and
  stateless ablations remained tied. A shorter 5-iteration run also failed
  quality with final `lin_vel_error.mean=0.4562378227710724`.

## Review

Status: open.

The train/eval smoke is closed. Range-based and fixed-speed training from the
Task042 residual prior did not pass the strict 1.6 m/s quality gate. The
`model_5349` True-TXL bridge now passes normal dynamic-switch quality, which
removes the frozen-base-speed-prior blocker.

Memory causality remains negative: the bridge residual is zero, and
zero-residual/stateless ablations are tied with normal mode. The next
experiment repaired and tested the residual/memory trainable path, but it also
failed to produce a memory-required pass: the residual becomes active, yet
quality does not pass and ablations remain tied.
