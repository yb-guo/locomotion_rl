# Task 044: Memory-Required Fault Identification Target

## Route

Task043 proved a useful negative result: the `model_5349` True-TXL bridge can
walk the 1.6 m/s dynamic-switch eval, but zero-residual and stateless-memory
ablations remain behaviorally tied. That means the current target can be solved
without useful TXL memory.

Task044 changes the target before spending more H200 time. The goal is a
fault-identification eval/train loop where memory is required by contract:

- the actor must not receive direct fault identity, severity, onset, or recovery
  labels in its observation;
- the eval must compare normal, zero-residual, and stateless-memory modes on
  the same checkpoint, seed, speed, and hidden-fault schedule;
- normal mode must pass locomotion quality;
- both ablations must materially degrade against normal mode;
- if the high-speed MLP prior or stateless mode passes tied with normal, the
  target is invalid rather than a memory success.

## Fixed Scope

- Keep G1-like MJLab topology.
- Keep Task038 True-TXL actor/runner and Task040 sequence-aware PPO.
- Keep action, reward, and visible observation contracts unchanged.
- Only add hidden-fault target metadata, eval comparison, and training targets
  required to make memory causality testable.
- No LocoFormer reproduction, morphology-generalization, or superiority claim.
- Do not touch `.test_tmp_task021/`.

Out of scope:

- link-length randomization;
- unified robot action-space remapping;
- transformer architecture parity beyond the existing True-TXL path;
- declaring Task044 passed from a single normal eval without ablation evidence.

## Planned Slices

1. `001-hidden-fault-eval-contract.md`
   - Add a pure local contract for normal/zero-residual/stateless eval triplets.
   - Fake JSON tests must prove tied ablations fail and degraded ablations pass.

2. `002-triplet-summary-cli.md`
   - Add a CLI that reads three H200 eval JSON files and writes one Task044
     triplet summary.
   - The CLI must preserve no-overclaim fields and record all source paths.

3. `003-h200-baseline-negative-gate.md`
   - Run the Task044 triplet gate on the current `model_5349` True-TXL bridge.
   - Expected result: fail because stateless/zero-residual remain tied. If it
     passes, the target is not memory-required enough and must be tightened.

4. `004-hidden-fault-train-target.md`
   - Train with hidden fault schedules that require within-episode or
     cross-trial identification from observation/action response history.
   - Keep the base MLP prior stable unless a deliberate task review changes the
     policy contract.

5. `005-memory-causality-evidence.md`
   - Run full triplet eval on candidate checkpoints.
   - Accept only if normal passes quality and both ablations degrade materially.

6. `006-immediate-hidden-fault-triplet-probe.md`
   - Before spending more H200 time on long training, run a tighter eval target
     where the final-trial hidden fault is active from the first control step.
   - Use the same checkpoint/seed/speed across normal, zero-residual, and
     stateless-memory modes.
   - Treat tied ablations as a target-design failure, not a memory success.

7. `007-windowed-final-trial-memory-gap.md`
   - Add a `final_trial_window` diagnostic to evaluate only the first N seconds
     of the final trial.
   - Use `--metric-scope final_trial_window` in the triplet summary to check
     whether memory helps before stateless within-trial re-identification can
     catch up.

8. `008-action-level-memory-influence-diagnostic.md`
   - Record action statistics in the eval JSON.
   - Compare normal, zero-residual, and stateless-memory action statistics to
     separate "memory does not affect actions" from "memory affects actions but
     not useful behavior."
   - Keep this as a diagnostic only; action differences are not Task044 success.

9. `009-clear-visible-history-inner-reset-runner.md`
   - Add a Task044-only runner that clears actor-visible K160 history on inner
     trial reset while preserving the TXL cache.
   - Use it to remove stateless-memory leakage from preserved visible history.
   - Verify the current checkpoint still fails, then retrain under this runner.

10. `010-memory-latent-scale-coupling-probe.md`
    - Add a `memory_latent_scale` diagnostic/training knob that does not change
      checkpoint parameter shapes.
    - Use it to test whether memory latent reaches action output but is too
      weakly coupled.

11. `011-fault-aux-conditioning-probe.md`
    - Add a Task044-only privileged fault-label auxiliary loss on memory-only
      latent.
    - Keep fault labels out of actor-visible observations.
    - Use H200 smoke, long train, and triplet eval to test whether explicit
      fault identification helps the memory-required gate.

12. `012-eval-aligned-left-knee-stage.md`
    - Add an eval-aligned training task for fixed left-knee dead motor at
      `vx=1.6` with 2.0 s trials.
    - Use it to test whether the current failure is train/eval mismatch rather
      than memory plumbing.

13. `013-velocity-pressure-left-knee-stage.md`
    - Keep the eval-aligned schedule but increase training-only linear velocity
      reward pressure.
    - Use it to test whether the remaining failure is a slow stable gait caused
      by weak speed-tracking pressure.

14. `014-persistent-hidden-scheduler-stage.md`
    - Keep hidden randomized faults fixed across inner trial resets while still
      randomizing across outer episodes.
    - Use it to make cross-trial fault identification useful again after fixed
      left-knee training removed the stateless ablation gap.

15. `015-immediate-left-knee-curriculum-stage.md`
    - Add the narrow immediate-onset curriculum: fixed left-knee dead motor from
      `t=0.0` to `2.0 s`.
    - Use it to recover stability before returning to randomized hidden-fault
      identity.

16. `016-persistent-hidden-speed-stability-stage.md`
    - Return to the persistent-hidden randomized target.
    - Add mild speed pressure and stronger posture/termination penalties to
      improve the closest checkpoint without collapsing speed.

17. `017-root-height-guard-stage.md`
    - Add root-height and posture guards after speed pressure exposed low-root
      and pose outliers.

18. `018-speed-pose-balance-stage.md`
    - Test whether stronger speed pressure plus pose constraints can close the
      strict normal gate without destabilizing the gait.

19. `019-forward-speed-floor-stage.md`
    - Add a forward-under-speed penalty after velocity-component diagnostics
      showed the remaining error is mostly x-direction speed, not lateral drift.

20. `020-reset-startup-eval-boundary.md`
    - Diagnose whether the remaining strict 2.0 s failure is a trainable speed
      issue or an eval/reset startup boundary.

21. `021-continuous-hidden-fault-no-reset-eval.md`
    - Add an eval-only continuous hidden-fault route with no Task037 inner
      physical resets.
    - Evaluate a `post_fault_window` after excluding a short fault-onset
      transient.
    - Add `zero_memory_latent` to ablate the full memory/history latent rather
      than only the TXL residual.

22. `022-memory-latent-only-consumer-stage.md`
    - Remove the base-observation and adaptation-history bypasses during
      training.
    - Verify a memory-latent-only actor can train and can be evaluated by the
      continuous post-fault gate.

## Acceptance Criteria

Task044 is accepted only when current evidence proves:

- local tests cover:
  - hidden-fault observation metadata;
  - normal quality failure;
  - missing/incorrect ablation record;
  - tied ablations;
  - materially degraded zero-residual and stateless ablations;
- H200 baseline-negative gate records a failed triplet for the current
  `model_5349` bridge or a documented target-tightening action;
- H200 train/eval records normal, zero-residual, and stateless JSONs for the
  same checkpoint, seed, speed, and fault schedule;
- normal mode passes the quality gate;
- zero-residual and stateless-memory modes both materially degrade by final
  trial metrics;
- no result claims LocoFormer reproduction, morphology generalization, or policy
  superiority.

## Evidence Gate

Local:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task044_memory_required_contract.py tests\test_agent_inventory.py
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent
```

H200 triplet summary shape:

```bash
PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src \
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task044_memory_required_triplet_summary \
  --normal-json /path/to/normal.json \
  --zero-residual-json /path/to/zero.json \
  --stateless-json /path/to/stateless.json \
  --confirm-hidden-fault-labels \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/summary.json
```

## Log

- 2026-05-31 Opened after Task043 showed the current dynamic-switch target is
  not memory-required: the `model_5349` bridge passed quality, but zero-residual
  and stateless-memory ablations stayed tied with normal mode.
- 2026-05-31 Added pure Task044 triplet contract:
  `src/h200_locomotion_lab/training/task044_memory_required_contract.py`.
  Local tests prove normal quality failure, visible fault labels, missing
  ablation modes, tied ablations, and degraded ablations are handled.
- 2026-05-31 Added triplet-summary CLI:
  `src/h200_locomotion_lab/tools/task044_memory_required_triplet_summary.py`.
  It reads normal/zero-residual/stateless JSONs, requires explicit
  `--confirm-hidden-fault-labels` before annotating missing hidden-fault
  metadata, and preserves no-overclaim fields.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task044_memory_required_contract.py tests\test_task044_triplet_summary_cli.py tests\test_agent_inventory.py --tb=short --basetemp pytest_tmp_task044_contract3`
  with sandbox escalation for pytest temp creation: 11 passed.
- 2026-05-31 CLI help and agent inventory passed locally:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task044_memory_required_triplet_summary --help`
  and
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent`.
- 2026-05-31 H200 baseline-negative gate recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_bridge_task044_baseline_negative_seed4301701.json`.
  Result: `task044_memory_required_pass=false` with
  `zero_residual_ablation_not_degraded` and
  `stateless_memory_ablation_not_degraded`. Normal final
  `lin_vel_error.mean=0.4467167258262634`; zero-residual delta was
  `0.00015547871589660645`; stateless delta was
  `-0.0002522468566894531`.
- 2026-05-31 H200 hidden-fault train smoke passed from the Task043 bridge:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_smoke_env64_iter1_seed4400101.json`.
  This proved the Task044 task registration, True-TXL runner, sequence-aware PPO,
  and hidden-fault train wrapper can execute; it did not claim memory success.
- 2026-05-31 H200 hidden-fault 25-iteration train completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_model5349_hidden_fault_env1024_iter25_seed4400301.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt`.
  Sequence updates were active and residual output was nonzero, but this is
  training-pipeline evidence only.
- 2026-05-31 H200 triplet eval for the 25-iteration checkpoint failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_hidden_fault_env1024_iter25_model24_triplet_seed4400401.json`.
  Result: `task044_memory_required_pass=false` because normal quality failed
  and both ablations stayed tied. Normal final `lin_vel_error.mean` was
  `0.4572296142578125`; zero-residual delta was `0.004590749740600586`;
  stateless delta was `0.00046581029891967773`.
- 2026-05-31 H200 immediate hidden left-knee triplet probe failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_hidden_fault_env1024_iter25_model24_immediate_left_knee_triplet_seed4400601.json`.
  Result: `task044_memory_required_pass=false` because normal quality failed
  and both ablations stayed tied. Normal final `lin_vel_error.mean` was
  `0.46178966760635376`; zero-residual delta was `0.0030305981636047363`;
  stateless delta was `0.0001862645149230957`.
- 2026-05-31 H200 0.5s windowed final-trial triplet probe failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_hidden_fault_env1024_iter25_model24_window0p5_left_knee_triplet_seed4400701.json`.
  Result: `task044_memory_required_pass=false`. Windowed normal
  `lin_vel_error.mean` was `1.2125095129013062`; zero-residual delta was
  `-0.0018093585968017578`; stateless delta was
  `-0.00041615962982177734`.
- 2026-05-31 H200 action-level memory influence diagnostic recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/model5349_hidden_fault_env1024_iter25_model24_actionstats_left_knee_joint_vx1p6_seed4400801.json`.
  Result: zero-residual changed action stats slightly, but stateless-memory
  stayed tied with normal. `stateless_memory_action_stats_tied` remained in the
  failure reasons. This indicates the stateful TXL memory path is not being
  used as a meaningful cross-trial control signal by the current checkpoint.
- 2026-05-31 Added Task044 clear-visible-history runner:
  `Task044TrueTxlMemoryK160ClearHistoryRunner`. It clears `actor_history` on
  inner trial reset but preserves TXL cache through the reset hook. H200
  registration was patched to use this runner for
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6`.
- 2026-05-31 Re-ran the action diagnostic with the clear-history runner:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/model5349_hidden_fault_env1024_iter25_model24_actionstats_left_knee_joint_vx1p6_seed4400901.json`.
  Result: still failed diagnostically with
  `stateless_memory_action_stats_tied`. Stateless action deltas were
  `mean_abs_l1_delta=0.0016770426544450944` and
  `mean_l2_delta=-0.004179716110229492`. This rules out visible K160 history
  leakage as the only cause for the tied stateless behavior on the current
  checkpoint.
- 2026-05-31 H200 clear-history train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4401001.json`.
  This verifies the Task044 clear-history runner can train with
  `Task040SequenceAwareTrueTxlPPO`.
- 2026-05-31 H200 clear-history long train completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_env1024_iter50_seed4401101.json`.
  Result: `train_pipeline_pass=true`; checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_clear_history_env1024_iter50_seed4401101/model_49.pt`.
- 2026-05-31 H200 eval of that checkpoint failed the Task044 triplet:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/clear_history_env1024_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4401201.json`.
  Result: `task044_memory_required_pass=false` with normal quality failure and
  both ablations not degraded.
- 2026-05-31 Added `memory_latent_scale` and ran scale-4 eval-only probe:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/clear_history_env1024_iter50_model49_scale4_actionstats_left_knee_joint_vx1p6_seed4401301.json`.
  Result: action influence became detectable with no action-summary failure
  reasons. Zero-residual `mean_abs_l1_delta=0.06077523926092732`; stateless
  `mean_abs_l1_delta=0.014938026426299926`. Task044 triplet still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/clear_history_env1024_iter50_model49_scale4_actionstats_triplet_left_knee_joint_vx1p6_seed4401301.json`.
- 2026-05-31 Started H200 scale-4 continuation training from the clear-history
  `model_49` checkpoint. Background PID: `587141`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_env1024_iter50_scale4_from_model49_seed4401401.json`.
- 2026-05-31 H200 scale-4 continuation training completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_env1024_iter50_scale4_from_model49_seed4401401.json`.
  Result: `train_pipeline_pass=true`; checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_clear_history_env1024_iter50_scale4_from_model49_seed4401401/model_49.pt`.
- 2026-05-31 H200 triplet eval for the scale-4 trained checkpoint failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/clear_history_scale4_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4401501.json`.
  Result: `task044_memory_required_pass=false` with
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`.
- 2026-05-31 The corresponding action influence summary showed memory can
  reach actions but still does not produce Task044 success:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/clear_history_scale4_iter50_model49_actionstats_left_knee_joint_vx1p6_seed4401501.json`.
  Zero-residual `mean_abs_l1_delta=0.11061656042452782`; stateless-memory
  `mean_abs_l1_delta=0.042740989836954305`. Normal final-window
  `lin_vel_error.mean=1.7986218929290771`, `fall_ratio=0.0`,
  `gravity_xy.max=0.2589489817619324`, and `root_z.min=0.7099651098251343`,
  so the scale-4 policy avoided falling in that window but did not track the
  1.6 m/s command.
- 2026-05-31 Added Task044 fault auxiliary conditioning probe. Local tests
  passed with 9 passed and 1 skipped. H200 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4401602.json`.
  Result: `train_pipeline_pass=true`, `task044_fault_aux_updates=20`, and
  `task044_fault_aux=1.7369526267051696` in the loss dict.
- 2026-05-31 Started H200 scale-4 aux continuation train. Background PID:
  `589546`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_scale4_aux002_env1024_iter50_from_scale4_model49_seed4401701.json`.
- 2026-05-31 H200 scale-4 aux continuation and bridge aux continuation both
  failed the Task044 triplet. The bridge scale-1 run preserved gait better but
  stateless memory remained tied, showing that whole-trial fault classification
  can be solved from visible/current response and does not force cross-trial
  cache use.
- 2026-05-31 Added early post-inner-reset auxiliary filtering:
  `task044_trial_step`, `task044_trial_index`,
  `task044_fault_aux_max_trial_step`, and
  `task044_fault_aux_min_trial_index`. Local validation passed with targeted
  pytest (9 passed, 1 skipped), `task044_hidden_fault_train --help`, and
  `inspect_agent`.
- 2026-05-31 H200 early-window plumbing smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4402102.json`.
  One-iteration smoke with `min_trial_index=1` produced no aux updates because
  the rollout had not reached trial 1; this is a smoke-design boundary, not a
  training crash.
- 2026-05-31 H200 early-window bridge continuation passed train pipeline:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_bridge_aux002_early4_trial1_scale1_env1024_iter50_seed4402201.json`.
  It had `task044_fault_aux_updates=205`. Triplet eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402301.json`.
- 2026-05-31 H200 early-window bridge continuation to 100 total iterations
  passed train pipeline:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_bridge_aux002_early4_trial1_scale1_env1024_iter50_cont2_seed4402401.json`.
  Action-level influence passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/bridge_aux002_early4_trial1_scale1_iter100_model49_actionstats_left_knee_joint_vx1p6_seed4402501.json`.
  Behavior-level triplet still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_aux002_early4_trial1_scale1_iter100_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402501.json`.
  The policy changed actions, but normal quality and ablation degradation were
  still below acceptance.
- 2026-05-31 Added Task044 eval-aligned left-knee train stage:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKnee1p6`.
  It fixes `episode_length_s=2.0`, `vx=1.6`, and a deterministic left-knee dead
  template from `0.0` to `2.0` s. Local test
  `tests\test_task044_hidden_fault_target.py` passed with 5 passed and 1
  skipped. H200 registry patch applied, and smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4402701.json`.
- 2026-05-31 Started H200 eval-aligned left-knee continuation train. Background
  PID: `597946`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter50_seed4402801.json`.
- 2026-05-31 H200 eval-aligned left-knee continuation completed and failed the
  behavior triplet:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402901.json`.
  The train pipeline passed with `task044_fault_aux_updates=750`, and action
  influence passed, but normal quality and both behavior ablations still failed.
  The policy was stable under left-knee dead (`fall_ratio=0.0`,
  `root_z.min=0.7717834115028381`) but slow (`lin_vel_error.mean=0.9456071853637695`).
- 2026-05-31 Started a longer eval-aligned continuation. Background PID:
  `599403`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403001.json`.
- 2026-05-31 H200 longer eval-aligned continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403001.json`.
  Triplet eval still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_aux002_early4_trial1_scale1_iter150_model99_actionstats_triplet_left_knee_joint_vx1p6_seed4403101.json`.
  Final trial stayed stable, but speed tracking only improved to
  `lin_vel_error.mean=0.9032137989997864`, still above the strict quality gate.
- 2026-05-31 Added velocity-pressure left-knee training stage:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKneeVelBoost1p6`.
  It keeps the eval-aligned schedule and increases training
  `track_linear_velocity` pressure. Local test passed with 5 passed and 1
  skipped. H200 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4403201.json`.
- 2026-05-31 Started H200 velocity-pressure continuation. Background PID:
  `601174`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4403301.json`.
- 2026-05-31 H200 velocity-pressure continuation completed and failed the
  triplet:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4403401.json`.
  Final trial stayed stable, and speed improved to
  `lin_vel_error.mean=0.8055592775344849`, but normal quality still failed.
  Eval-only `memory_latent_scale=2.0` destabilized normal mode, so scale1 remains
  the viable path.
- 2026-05-31 Started a longer velocity-pressure continuation. Background PID:
  `603653`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403601.json`.
- 2026-05-31 Added persistent-hidden scheduler stage:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenVelBoost1p6`.
  The new H200 scheduler patch keeps existing tasks defaulted to
  `preserve_schedule_across_inner_resets=False`, while the new task sets it to
  `True` so the hidden fault schedule persists across inner multi-trial resets.
  Local gate passed with 34 passed and 2 skipped, and `inspect_agent` passed.
- 2026-05-31 H200 persistent-hidden smoke and two continuation trains completed.
  Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4403901.json`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4404001.json`,
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4404201.json`.
  The best full-final linear velocity error improved to roughly `0.53-0.55`,
  but normal quality still failed on instability extremes and the 0.5 s window
  ablations remained tied:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_hidden_velboost_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4404101.json`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_hidden_velboost_aux002_early4_trial1_scale1_iter100_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4404301.json`.
- 2026-05-31 Diagnosed the delayed-onset mismatch in H200 dynamic-single
  training: randomized training used `onset=1.0-4.0 s`, while Task044 eval uses
  `onset=0.0 s`. Added the immediate-onset scheduler knobs and task id
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateVelBoost1p6`.
  H200 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4404501.json`.
  Immediate-onset continuation from the persistent checkpoint was aborted due
  to severe early instability; no pass claim was made.
- 2026-06-01 Added the narrower immediate left-knee curriculum task:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadVelBoost1p6`.
  It fixes target, failure type, onset, and duration before reintroducing
  randomized hidden-fault identity. H200 validation pending.
- 2026-06-01 H200 immediate left-knee curriculum smoke and 25-iteration train
  completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4404701.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_immediate_leftknee_aux002_early4_trial1_scale1_env1024_iter25_seed4404801.json`.
  It stabilized immediate left-knee dead motor but remained too slow:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_immediate_leftknee_aux002_early4_trial1_scale1_iter25_model24_actionstats_triplet_left_knee_joint_vx1p6_seed4404901.json`.
- 2026-06-01 Speed-push and all-actor probes failed. Speed-push normal eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_immediate_leftknee_speedpush_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4405401.json`
  had `fall_ratio=1.0`. All-actor continuation was aborted after early
  `fell_over=10.2917`.
- 2026-06-01 H200 persistent-hidden continuations completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter150_cont_seed4405501.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_velboost_aux002_early4_trial1_scale1_env2048_iter150_parallel_gpu1_seed4405601.json`.
  GPU0 normal eval improved root height but still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_velboost_iter150_model49_normal_probe_left_knee_joint_vx1p6_seed4405701.json`
  with `lin_vel_error.mean=0.5498349666595459`, `gravity_xy.max=0.9325734972953796`,
  and `root_z.min=0.43122363090515137`. GPU1 normal eval was worse:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_velboost_env2048_iter150_model49_normal_probe_left_knee_joint_vx1p6_seed4405801.json`.
- 2026-06-01 Added the persistent-hidden speed-stability task:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedStability1p6`.
  H200 validation pending.
- 2026-06-01 Root-height and pose-termination diagnostics narrowed the current
  blocker. Hard low-root termination fixed the root-height outlier but
  regressed pose. Pose-tight training then produced the closest normal eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4411401.json`
  with `fall_ratio=0.0078125`, `gravity_xy.max=0.7357999682426453`,
  `root_z.min=0.5745701193809509`, and
  `lin_vel_error.mean=0.4907352030277252`. It is not passed; the remaining
  strict blocker is speed tracking under hidden left-knee dead motor.
- 2026-06-01 Short pose-tight continuation did not close the speed gap. LR1e-5
  continuation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_cont_from_tight_iter5_model4_normal_probe_left_knee_joint_vx1p6_seed4411701.json`
  failed with `lin_vel_error.mean=0.5035483837127686`. LR2e-5 continuation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr2e5_cont_from_tight_iter5_model4_normal_probe_left_knee_joint_vx1p6_seed4411801.json`
  improved velocity to `0.4884888529777527` but failed
  `gravity_xy.max=0.7584195137023926`.
- 2026-06-01 Opened subtask 018,
  `PersistentHiddenSpeedPoseBalance1p6`, to test stricter pose termination plus
  stronger speed reward before changing policy architecture or relaxing the
  quality gate.
- 2026-06-01 Velocity-component diagnostics showed the current best checkpoint
  is forward-under-speed limited: final trial command `vx=1.6`, actual
  `vx=1.268424153327942`, x error `0.46766161918640137`, y error
  `0.07159683853387833`. Opened subtask 019,
  `PersistentHiddenForwardFloor1p6`, to add a targeted under-speed penalty
  while keeping the strict eval gate unchanged.
- 2026-06-01 Forward-floor, forward-target, speed-curriculum, and immediate
  left-knee variants did not materially improve the original 2.0 s strict
  normal eval. The best checkpoint remains the pose-tight model:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_persistent_hidden_pose_tight_lr1e5_from_pose_aux002_early4_trial1_scale1_env1024_iter10_seed4411201/model_9.pt`.
- 2026-06-01 Opened subtask 020 after tail-window and 3.0 s trial diagnostics
  changed the interpretation of the blocker. The same pose-tight checkpoint
  still fails the original 2.0 s full-final strict gate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4415101_tail.json`
  with full-final `lin_vel_error.mean=0.4931211471557617`, but its last 0.5 s
  final-trial tail has `lin_vel_actual.mean_x=1.660741925239563` and
  `lin_vel_error.mean=0.2633638083934784`. A 3.0 s trial diagnostic:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_trial3s_seed4415701_tail.json`
  passed final thresholds but still failed trend checks. Code inspection shows
  inner reset preserves command/fault condition tensors but resets physical
  robot state to standing/near-zero speed, so the original 2.0 s full-trial
  metric is measuring startup acceleration as much as hidden-fault adaptation.
- 2026-06-01 H200 tail-scope triplet summary recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_hidden_pose_tight_model9_tail_scope_triplet_seed4415101.json`.
  Result: `task044_memory_required_pass=false`. Tail normal quality was good
  (`lin_vel_error.mean=0.2633638083934784`), but zero-residual and stateless
  ablations were not degraded (`delta=-0.004982590675354004` and
  `delta=-0.011602520942687988`). This rules out claiming memory-required
  behavior from steady-state tail metrics on the current checkpoint.
- 2026-06-01 Opened subtask 021. Local code now has an eval-only
  `Task044TrueTxlMemoryK160ContinuousRunner`, a
  `task044_continuous_fault_eval` CLI, `post_fault_window` triplet metric
  scope, and `zero_memory_latent` as a stronger ablation that zeros the combined
  TXL/adaptation-history latent.
- 2026-06-01 H200 continuous no-reset triplet recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/pose_tight_model9_continuous_post_fault_triplet_seed4417001.json`.
  It failed with `normal_pipeline_not_passed`,
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`. Normal continuous post-fault speed
  was good (`lin_vel_error.mean=0.2691524028778076`,
  `lin_vel_actual.mean_x=1.6762081384658813`) but `fall_ratio=0.12109375`.
  `zero_memory_latent` set `policy_memory_latent_norm=0.0` and still did not
  degrade, so the current checkpoint is not memory-required even after removing
  Task037 inner physical resets.
- 2026-06-01 Opened subtask 022 to test a memory-latent-only policy consumer:
  `--no-base-obs-passthrough` and `--no-adaptation-warmstart`. This is the
  next correction because subtask 021 showed the current base-observation /
  adaptation-history bypass can solve or improve the post-fault window without
  stateful TXL memory.
- 2026-06-01 H200 memory-latent-only smoke training passed pipeline checks:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_memory_only_smoke_env64_iter1_seed4418001.json`.
  The checkpoint is:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_memory_only_smoke_env64_iter1_seed4418001/model_0.pt`.
  Continuous eval on that one-iteration checkpoint failed locomotion quality
  (`lin_vel_error.mean=1.2672992944717407`, `lin_vel_actual.mean_x=0.34125807881355286`)
  and `zero_memory_latent` did not degrade it. This confirms the corrected
  consumer path is runnable but not yet useful; the next route is a
  memory-only clean-gait prior before retrying hidden-fault training.
- 2026-06-01 Strict memory-only clean-prior training from scratch reached a
  checkpoint but failed clean eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_memory_only_clean_prior_env4096_iter100_lr1e4_model99_seed4418301.json`
  with final `fall_ratio=1.0`, `root_z.min=0.20132026076316833`, and
  `lin_vel_error.mean=0.5985420346260071`. This supports the diagnosis that
  cutting off the gait prior too early is the current blocker.
- 2026-06-01 Added bridge-compatible bypass scales:
  `base_obs_passthrough_scale` and `adaptation_warmstart_scale`. They default
  to `1.0`, keep checkpoint shapes stable, and enable chunked annealing toward
  memory-only behavior after memory columns are trained. Local related tests
  passed with `22 passed, 6 skipped`; H200 related tests passed with
  `28 passed`.
- 2026-06-01 Started H200 staged probes under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/`:
  GPU0 continues the strict `128` memory-only checkpoint, while GPU1 trains the
  bridge-compatible `104+32` checkpoint with
  `--actor-trainable-scope txl_residual_and_mlp_memory_input`.
- 2026-06-01 Staged bypass annealing recovered a clean-gait, memory-dependent
  checkpoint while strict memory-only remained failed. The scale-0.5 train
  summary is:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/train_bridge_anneal_scale0p5_env4096_iter80_lr1e4_seed4418701.json`.
  Normal clean eval on `model_79` passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_bridge_anneal_scale0p5_model79_seed4418803.json`.
  The same checkpoint failed when the full memory latent was zeroed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_bridge_anneal_scale0p5_model79_zero_memory_seed4418803.json`
  with final `fall_ratio=0.578125`, and also failed with both bypass scales
  set to zero:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/memory_only_clean_prior/eval_bridge_anneal_scale0p5_model79_scale0_seed4418803.json`
  with final `fall_ratio=1.0`.
- 2026-06-01 Continuous hidden-fault eval from that clean checkpoint failed
  normal quality, so hidden-fault training was required before a Task044 pass
  could be considered. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_anneal_scale0p5_model79_continuous_left_knee_post_fault_triplet_seed4419001.json`.
  Normal post-fault had `fall_ratio=0.2265625` and
  `lin_vel_error.mean=0.8197673559188843`.
- 2026-06-01 Hidden-fault training from the scale-0.5 clean checkpoint narrowed
  the failure to post-fault fall/reset robustness. The best candidate so far is
  the fixed left-knee pose-forward curriculum:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_scale0p5_immediate_leftknee_pose_forward_all_env2048_iter40_lr1e5_seed4419502.json`.
  Its `model_39` repeated normal continuous evals stayed at
  `fall_ratio=0.105-0.125` with speed passing:
  `lin_vel_error.mean=0.40938737988471985`,
  `0.4212338924407959`, and `0.41330718994140625` for seeds
  `4419601`, `4419602`, and `4419603`. This still fails the Task044
  post-fault fall gate (`<=0.05`) and physical-continuity pipeline gate.
- 2026-06-01 Further short continuations from the best fixed-left-knee
  checkpoint regressed fall ratio to `0.20703125` and `0.21875`. The current
  blocker is not path lookup, training execution, or velocity tracking; it is
  the residual left-knee dead-motor stability tail under the continuous
  post-fault gate.

## Review

Status: open.

Subtasks 001, 002, 003, 008, and the diagnostic part of 009 are closed. Task044
now has a local triplet contract, a JSON-summary CLI, and H200 negative evidence
proving the current `model_5349` bridge and `model_24` hidden-fault checkpoint
do not pass as memory-required evidence.

The task is not complete. Subtask 004 has train-pipeline evidence, but subtasks
005, 006, and 007 failed to show memory causality. Subtask 008 narrows the
failure: stateless-memory actions were initially tied with normal actions, so
the old checkpoint was not using stateful TXL memory as a meaningful cross-trial
control signal. Clearing visible K160 history on inner reset did not make the
old checkpoint use stateful memory. Scale-4 coupling made memory affect actions,
but the trained checkpoint still failed normal locomotion quality and ablation
degradation. Subtask 011 proved a privileged fault-label auxiliary loss can make
memory affect action statistics, especially with early post-reset filtering, but
it still did not create behavior-level memory-required locomotion. Subtask 012
matched the train distribution to the triplet gate; subtasks 013 and 014 showed
velocity pressure helps speed but creates low-root and pose outliers. Subtasks
017, 018, and 019 are closed diagnostics: hard root/pose guards fixed height and
made posture nearly pass, but additional speed pressure did not fix the
original strict 2.0 s full-trial normal eval. Subtask 020 is the current
interpretation boundary: the policy reaches the 1.6 m/s target in the final
tail and a longer-trial diagnostic passes final thresholds, while the original
2.0 s full-trial gate still fails because each inner trial restarts from a
standing physical state. However, tail-scope triplet evidence also shows the
current checkpoint does not use TXL memory in a behaviorally necessary way. No
Task044 pass claim exists yet because the original normal quality gate has not
passed and the normal/zero-residual/stateless evidence remains tied even under
the tail metric scope. Subtask 021 removed Task037 inner physical resets from
eval and closed as a negative diagnostic: the confound was removed, but the old
checkpoint still had post-fault falls and remained tied under full
memory-latent ablation. Subtask 022 then reduced the
base-observation/current-observation bypass during training. Strict
memory-only clean gait failed, but bridge-compatible staged annealing to scale
`0.5` produced a clean-gait checkpoint whose zero-memory ablation fails. That
is useful evidence that the policy can be made memory-dependent without
changing checkpoint shape.

The task is still not complete. Once the scale-0.5 clean checkpoint is trained
on hidden faults, speed and posture can pass, but the best repeated left-knee
continuous eval still has `fall_ratio=0.105-0.125`, above the `0.05` gate. The
current blocker is therefore a residual stability-tail problem under left-knee
dead motor, not missing H200 execution or memory wiring. No Task044 pass claim
exists yet; a full normal/zero-memory/stateless triplet is only meaningful
after the normal continuous post-fault quality gate passes.
