# Task 039: True-TXL Quality Training Diagnosis

## Route

User decision: Task039 should handle the gap between Task038's runnable
LocoFormer-min G1-like plumbing and an actual policy-quality result.

This task starts with the recommended branch:

```text
A. clean gait quality first
```

The task is a diagnosis task, not a reproduction-success task. It must answer:

```text
Does the current Task038 true-TXL policy train into a useful clean G1-like gait,
and if not, is the blocker training duration, baseline difficulty, curriculum,
or the current PPO/TXL memory update boundary?
```

Task039 must not jump directly to full held-out morphology, full LocoFormer
matrix, or TXL superiority. Those are later tasks after a clean-gait feedback
loop is trustworthy.

## Fixed Scope

- Robot family: current Task038 G1-like setup only.
- First quality target: clean walking on one train variant.
- Baselines: MLP clean baseline and Task038 true-TXL.
- Eval: short, repeatable H200 JSON metrics and optional small videos.
- Diagnosis: log whether true-TXL inference memory is active and whether PPO
  update uses real memory or falls back to stateless minibatches.

Out of scope:

- claiming full LocoFormer reproduction;
- claiming TXL superiority;
- running the full seen/heldout morphology matrix;
- changing reward/env/action/obs contracts before the feedback loop proves the
  change is necessary;
- downloading checkpoints, datasets, simulator assets, or upstream repos;
- touching `.test_tmp_task021/`.

## Planned Slices

1. `001-quality-feedback-loop.md`
   - Define the clean-gait JSON/video feedback loop, quality metrics, pass/fail
     thresholds, and no-overclaim fields.

2. `002-mlp-clean-baseline.md`
   - Run or prepare the smallest comparable MLP clean baseline on the same
     train variant and eval gate.

3. `003-true-txl-clean-longtrain.md`
   - Run true-TXL clean training long enough to distinguish learning trend from
     random or one-iteration smoke.

4. `004-txl-update-memory-diagnostics.md`
   - Measure whether training actually uses long memory or mostly uses the
     Task038 stateless PPO minibatch fallback.

5. `005-clean-speed-curriculum-probe.md`
   - If clean 0.4 m/s improves, probe a small speed curriculum toward 1.2 and
     2.0 m/s without opening held-out claims.

6. `006-one-heldout-sanity-probe.md`
   - Only after clean train quality improves, run one held-out G1-like variant
     sanity eval as a diagnostic, not a full morphology claim.

## Minimal Closed Loop

Task039 is accepted only when the router can show:

- a clean-gait feedback loop that can fail a poor checkpoint and pass a clearly
  improved checkpoint on the same metric schema;
- one MLP baseline JSON or an explicit blocked reason;
- one true-TXL clean training/eval JSON or an explicit blocked reason;
- memory-update diagnostics showing the stateless fallback rate and whether
  sequence-aware memory was used during PPO update;
- a decision note saying which next route is justified:
  - continue true-TXL longtrain;
  - implement sequence-aware TXL PPO update;
  - tune clean-gait curriculum/reward;
  - move to held-out morphology probes.

## Evidence Gate

Local evidence:

```powershell
python -m h200_locomotion_lab.tools.inspect_agent
python -m pytest tests/test_agent_inventory.py
```

Any new CLI or contract helper must include focused local tests and `--help`
coverage.

H200 evidence must be small JSON/log metadata, not large checkpoints or videos
checked into git. Every H200 JSON must record:

- task id and variant label;
- checkpoint path;
- policy class;
- action dimension;
- command speed;
- seed count;
- wall time;
- reward trend or eval metrics;
- fall ratio, root z, gravity xy, velocity tracking;
- memory diagnostic fields for true-TXL;
- explicit claim flags that keep this task diagnostic-only.

## Subagent Ownership

Router owns:

- keeping Task039 limited to clean-gait diagnosis first;
- assigning one implementation subagent per slice;
- assigning a separate review subagent before closing each slice;
- approving H200 runs and interpreting evidence.

Implementation subagents own only the files named in their slice.

Review subagents are read-only unless the router explicitly assigns a fix.

## Failure Exit

Stop and return to the router if:

- clean-gait eval cannot produce deterministic JSON;
- MLP and true-TXL cannot be compared under the same task/variant/seed/metric
  schema;
- true-TXL training mostly uses stateless fallback and therefore cannot support
  a long-memory training claim;
- training/eval outputs only show pipeline health but no quality trend;
- a proposed fix requires changing env/reward/action contracts without a
  minimal diagnostic proving that need.

## Log

- 2026-05-30 Opened after Task038 closed the runnable G1-like LocoFormer-min
  smoke/pipeline loop but left policy-quality, long training, full matrix, and
  TXL superiority unproven. User accepted the clean-gait-first route.
- 2026-05-30 Closed `001` locally. Added
  `src/h200_locomotion_lab/training/task039_quality_feedback.py` and
  `tests/test_task039_quality_feedback.py`. The helper keeps `pipeline_pass`
  separate from `quality_gate_pass`, rejects missing/non-finite or bad gait
  metrics, and requires all no-overclaim flags to be explicitly present and
  false. Router verification returned `13 passed in 0.07s` for
  `tests\test_task039_quality_feedback.py tests\test_agent_inventory.py`, and
  `python -m h200_locomotion_lab.tools.inspect_agent` completed successfully.
  Independent review found no blockers.
- 2026-05-30 Added positive calibration for `001` using the existing Task037
  AdaptK160 `model_5467` clean 0.4 eval, without treating it as a Task039
  baseline. H200 JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/quality_calibration/task039_adaptk160_model5467_clean_vx0p4_positive_quality_calibration.json`.
  Result: `task039_quality_calibration_only=true`, `pipeline_pass=true`,
  `quality_gate_pass=true`, `pass=true`,
  `final_trial.fall_ratio=0.0`,
  `final_trial.gravity_xy.max=0.06800129264593124`,
  `final_trial.root_z.min=0.7846410870552063`, and all no-overclaim flags
  false. This calibrates the gate as able to pass a clearly improved clean
  checkpoint while `002`/`003` fail poor checkpoints.
- 2026-05-30 Implemented `002` local plumbing for the MLP clean baseline, but
  did not run H200 and did not make a quality/training/eval/reproduction or
  superiority claim. Added the Task039 MJLab registration patcher for
  `Unitree-G1-Gripper-Flat-Task039-MlpClean-Train`, the Task039 MLP eval
  wrapper, focused local tests, and non-breaking Task037 eval metadata fields
  for runner class, actor model class, action dim, and total action dim.
- 2026-05-30 Ran `002` on H200 for a bounded 30-iteration MLP diagnostic
  baseline. The checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task039_mlp_clean_train/2026-05-30_16-52-26_039_mlp_clean_env4096_iter30_gpu0_seed3900201/model_29.pt`
  evaluated through
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/mlp_clean_baseline/mlp_clean_env4096_iter30_model29_vx0p4_eval_v2.json`.
  Train metadata is recorded at
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/mlp_clean_baseline/mlp_clean_env4096_iter30_model29_train_metadata.json`.
  Result: `pipeline_pass=true`, `quality_gate_pass=false`, `pass=false`,
  `final_trial.fall_ratio=0.953125`,
  `final_trial.gravity_xy.max=0.9452863335609436`,
  `final_trial.root_z.min=0.2165585458278656`. This closes the MLP evidence
  requirement as a failed-quality diagnostic sample only.
- 2026-05-30 Fixed Task039 eval pass semantics after read-only review. The
  Task039 MLP and true-TXL wrappers now set top-level
  `pass = pipeline_pass and quality_gate_pass`; pipeline-only health is
  recorded only through `pipeline_pass`. Added
  `src/h200_locomotion_lab/tools/task039_train_metadata.py` and tests so train
  evidence can be recorded as provenance JSON without making a training success
  claim.
- 2026-05-30 Implemented `003` local plumbing for true-TXL clean eval without
  running H200 or recording H200 evidence. Added
  `src/h200_locomotion_lab/tools/task039_true_txl_clean_eval.py` and
  `tests/test_task039_true_txl_clean_eval.py`. The wrapper allow-lists only
  `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`, reuses the
  Task037 multitrial evaluator and Task039 quality gate, separates
  `pipeline_pass` from `quality_gate_pass`, requires active true-TXL memory
  debug, and keeps all no-overclaim flags false. Focused local verification
  returned `48 passed in 0.24s`.
- 2026-05-30 Ran `003` on H200 for a bounded 30-iteration true-TXL diagnostic
  baseline. The checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task039_true_txl_clean_train/2026-05-30_16-58-19_039_true_txl_clean_env4096_iter30_gpu0_seed3900301/model_29.pt`
  evaluated through
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/true_txl_clean_longtrain/true_txl_clean_env4096_iter30_model29_vx0p4_eval.json`.
  Train metadata is recorded at
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/true_txl_clean_longtrain/true_txl_clean_env4096_iter30_model29_train_metadata.json`.
  Result: `pipeline_pass=false`, `quality_gate_pass=false`, `pass=false`.
  The pipeline failure is `memory_debug_missing`, with
  `memory_debug_present=false` and `memory_debug_active=false`. Gait quality
  also fails: `final_trial.fall_ratio=1.0`,
  `final_trial.gravity_xy.max=0.9480438828468323`,
  `final_trial.root_z.min=0.1402791440486908`. This routes the task to `004`
  before any true-TXL quality or held-out claim.
- 2026-05-30 Independent read-only review found no blockers for corrected
  `002` and `003`. The only non-blocking note was a stale pending-evidence
  sentence in `003`, which was removed.
- 2026-05-30 Implemented and ran `004` TXL update-memory diagnostics. H200 JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/txl_update_memory_diagnostics/true_txl_ppo_update_env8_steps2_iter1_memory_diagnostic.json`.
  Result: diagnostic `pass=true`, `long_memory_training_claim_supported=false`,
  `total_actor_forward_batches=3`, `env_cache_stateful_forward_batches=2`,
  `stateless_fallback_forward_batches=1`,
  `stateless_fallback_ratio_by_batches=0.3333333333333333`,
  `stateless_fallback_ratio_by_samples=0.5`,
  `minibatches_preserve_temporal_segments=false`, and router decision
  `sequence_aware_txl_ppo_update_required_next`. This is diagnostic validity
  only, not a quality/training/eval/reproduction/superiority claim.
- 2026-05-30 Independent read-only review found no blockers for `004`. Reviewer
  confirmed the H200 diagnostic supports `long_memory_training_claim_supported=false`
  from stateless fallback evidence and that diagnostic `pass=true` is not
  written as quality/training/eval pass.
- 2026-05-30 Gated `005` and `006` as not run. The prerequisite clean
  train-variant quality is not established for current MLP/true-TXL runs, and
  `004` routes to sequence-aware TXL PPO update before speed curriculum or
  held-out morphology. No speed expansion, held-out generalization, or
  superiority claim is made.

## Review

Status: planning opened and `001` local quality feedback loop is closed.
`002` has corrected local verification and bounded H200 train/eval evidence.
`003` has bounded H200 evidence and exits toward `004` because active memory
debug is missing. Corrected `002` and `003` passed independent read-only review.
`004` has H200 diagnostic evidence and passed independent read-only review.
`005` and `006` are gated not run. Router decision: implement sequence-aware TXL
PPO update next; do not continue longtrain, speed curriculum, or held-out
morphology under the current stateless-fallback update path. No Task039
quality, training-success, reproduction, eval success, or superiority claim has
been made.

Final status: Task039 complete as a diagnosis task.
