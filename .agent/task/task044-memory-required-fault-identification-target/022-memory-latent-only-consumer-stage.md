# 022: Memory Latent Only Consumer Stage

## Route

Subtask 021 proved that the current checkpoint is not memory-required even
after removing Task037 physical inner resets. The strongest evidence is that
`zero_memory_latent` sets the policy memory latent to zero and still does not
degrade post-fault behavior.

The likely cause is architectural: the current Task041/044 actor uses
`base_obs_passthrough=true` and `adaptation_warmstart=true`, so the MLP can
consume newest proprioception and a K160 adaptation encoder without needing the
stateful TXL memory path. This subtask tests a more LocoFormer-like consumer:
the actor policy latent is memory-only, with no base-observation passthrough and
no adaptation warmstart.

## Acceptance

- H200 smoke training must run with:
  - `--no-base-obs-passthrough`;
  - `--no-adaptation-warmstart`;
  - a memory latent dimension large enough to carry control state.
- The train JSON must record train-pipeline status, checkpoint path, actor
  config, and no overclaim fields.
- A short continuous eval must run on the smoke checkpoint in normal and
  `zero_memory_latent` modes.
- Do not claim Task044 passed unless a later long train gives normal quality
  pass and degraded ablations.

## Log

- 2026-06-01 H200 memory-latent-only smoke training ran with
  `--no-base-obs-passthrough`, `--no-adaptation-warmstart`, and
  `--memory-latent-dim 128`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_memory_only_smoke_env64_iter1_seed4418001.json`.
  It recorded `task044_train_pipeline_pass=true`, actor class
  `Task038TrueTxlMemoryModel`, algorithm class
  `Task040SequenceAwareTrueTxlPPO`, and checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_memory_only_smoke_env64_iter1_seed4418001/model_0.pt`.
- 2026-06-01 H200 continuous eval on the smoke checkpoint did not pass
  locomotion quality, as expected for a one-iteration checkpoint. Normal mode:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/memory_only_smoke_model0_continuous_normal_seed4418101.json`
  had post-fault `lin_vel_error.mean=1.2672992944717407`,
  `lin_vel_actual.mean_x=0.34125807881355286`, and `fall_ratio=0.0625`.
  `zero_memory_latent`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/memory_only_smoke_model0_continuous_zero_memory_latent_seed4418101.json`
  set `policy_memory_latent_norm=0.0` and still did not degrade
  (`lin_vel_error.mean=1.2400236129760742`).
- 2026-06-01 Interpretation: the memory-only consumer path is runnable, but
  hidden-fault training from scratch is too hard. The next minimal closed loop
  is a memory-only clean-gait prior using the Task041 clean train/eval tools,
  then reuse that checkpoint for Task044 hidden-fault training.
- 2026-06-01 Added a bridge-compatible staged-bypass mechanism:
  `base_obs_passthrough_scale` and `adaptation_warmstart_scale`. These default
  to `1.0`, preserve old checkpoint shape, and can be lowered in later chunks
  without switching from `104+32` actor input to a new `128` memory-only shape.
  Local validation:
  `python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task041_sequence_txl_clean_eval.py tests/test_task044_continuous_fault_eval.py --tb=short`
  passed with `22 passed, 6 skipped`. H200 validation passed with
  `28 passed`.
- 2026-06-01 Started two H200 clean-prior probes:
  - GPU0: continue the strict memory-only `128` latent checkpoint for 300 more
    iterations from
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/logs_env4096_iter100_lr1e4_seed4418201/model_99.pt`.
  - GPU1: bridge-compatible `104+32` checkpoint from
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/adaptk160_true_txl_warmstart/model_5467_task041_true_txl_bridge.pt`,
    training only TXL residual and MLP memory input columns with
    `--actor-trainable-scope txl_residual_and_mlp_memory_input`.
- 2026-06-01 Strict memory-only clean-prior continuation was stopped after the
  first 100-iteration checkpoint failed clean gait. Eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_memory_only_clean_prior_env4096_iter100_lr1e4_model99_seed4418301.json`
  had final `fall_ratio=1.0`, `root_z.min=0.20132026076316833`, and
  `lin_vel_error.mean=0.5985420346260071`.
- 2026-06-01 Bridge-compatible staged training produced a clean-stable,
  memory-dependent checkpoint. Stage-1 memory-column train:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/train_bridge_memory_columns_env4096_iter120_lr1e4_seed4418501.json`.
  The `model_40` normal clean eval passed, but zero-memory also passed, so it
  was still bypass-driven. The subsequent scale-0.5 anneal train:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/train_bridge_anneal_scale0p5_env4096_iter80_lr1e4_seed4418701.json`
  produced
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/logs_bridge_anneal_scale0p5_env4096_iter80_lr1e4_seed4418701/model_79.pt`.
  Normal clean eval passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_bridge_anneal_scale0p5_model79_seed4418803.json`
  with final `fall_ratio=0.0`, `root_z.min=0.7412417531013489`,
  `lin_vel_error.mean=0.18367353081703186`, and
  `policy_memory_latent_norm=4.093388557434082`.
- 2026-06-01 The same scale-0.5 checkpoint became memory-dependent under clean
  gait ablations. `zero_memory_latent` failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_bridge_anneal_scale0p5_model79_zero_memory_seed4418803.json`
  with final `fall_ratio=0.578125` and `policy_memory_latent_norm=0.0`.
  Setting both bypass scales to zero also failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_bridge_anneal_scale0p5_model79_scale0_seed4418803.json`
  with final `fall_ratio=1.0`.
- 2026-06-01 Continuous hidden left-knee eval on the clean scale-0.5 checkpoint
  failed normal quality:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_anneal_scale0p5_model79_continuous_left_knee_post_fault_triplet_seed4419001.json`.
  Normal post-fault metrics were `fall_ratio=0.2265625` and
  `lin_vel_error.mean=0.8197673559188843`; `zero_memory_latent` was much worse
  (`fall_ratio=1.96875`), while stateless memory stayed close to normal. This
  shows the clean prior needs hidden-fault training before a Task044 pass is
  possible.
- 2026-06-01 H200 hidden-fault training from the scale-0.5 clean checkpoint
  completed for two variants:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_scale0p5_hidden_pose_tight_memcols_env4096_iter80_lr1e4_seed4419101.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_scale0p5_hidden_pose_tight_all_env4096_iter80_lr5e5_seed4419102.json`.
  Both train pipelines passed. The all-scope variant was better for post-fault
  stability and speed; its `model_79` normal continuous eval had
  `fall_ratio=0.24609375` and `lin_vel_error.mean=0.508424699306488`.
- 2026-06-01 Continued all-scope training on `PersistentHiddenSpeedPoseBalance1p6`
  improved speed but not enough stability:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_scale0p5_hidden_speed_pose_balance_all_env4096_iter80_lr3e5_seed4419302.json`.
  Normal continuous eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/scale0p5_hidden_speed_pose_balance_all_model79_continuous_normal_left_knee_seed4419401.json`
  had `fall_ratio=0.1875` and `lin_vel_error.mean=0.4146566390991211`.
- 2026-06-01 A short fixed left-knee pose-forward curriculum from that
  checkpoint gave the best normal eval so far:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_scale0p5_immediate_leftknee_pose_forward_all_env2048_iter40_lr1e5_seed4419502.json`.
  Its `model_39` normal continuous evals across three seeds stayed in the same
  failure band:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/scale0p5_immediate_leftknee_pose_forward_all_model39_continuous_normal_left_knee_seed4419601.json`
  (`fall_ratio=0.10546875`, `lin_vel_error.mean=0.40938737988471985`),
  seed `4419602` (`fall_ratio=0.125`, `lin_vel_error.mean=0.4212338924407959`),
  and seed `4419603` (`fall_ratio=0.10546875`,
  `lin_vel_error.mean=0.41330718994140625`).
- 2026-06-01 Additional short continuations from the best fixed-left-knee
  checkpoint regressed fall ratio instead of closing the gate:
  `train_scale0p5_immediate_leftknee_pose_forward_all_cont2_env2048_iter40_lr1e5_seed4419701.json`
  eval had `fall_ratio=0.20703125`, and
  `train_scale0p5_from_immediate_hidden_speed_pose_balance_all_env2048_iter40_lr1e5_seed4419702.json`
  eval had `fall_ratio=0.21875`. No Task044 pass claim is made.

## Review

Status: active, not passed.

This is a policy-consumer correction stage. It does not yet implement full
LocoFormer morphology tokens or morphology randomization; it only removes the
current bypass that lets the policy avoid using the history/TXL memory latent.
The current evidence is pipeline-only. It proves the memory-only actor can be
constructed, trained for a smoke iteration, checkpointed, and evaluated, but it
does not prove useful locomotion or memory-required behavior.

The current working interpretation is now narrower. Direct strict memory-only
training cuts off the proven gait prior too early, but staged bypass annealing
does create a clean-gait checkpoint where zeroing the memory latent breaks the
policy. The remaining blocker is hidden left-knee fault robustness: the best
scale-0.5 checkpoint tracks speed and keeps root/pose within thresholds, but
still has a stable `fall_ratio` around `0.105-0.125` in the continuous
post-fault window, above the `0.05` gate. More blind same-stage training is not
enough evidence for a pass; the next route needs either a targeted stability
objective/termination change or an explicit decision to revise the quality gate
before memory-causality triplets can be meaningful.
