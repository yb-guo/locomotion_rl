# Task 038: LocoFormer-Min G1-like Reproduction

## Route

User decision: first reproduce LocoFormer-style behavior on a G1-like robot
family, not quadrupeds, wheeled robots, or arbitrary omni-bodied control.

This task is the minimum router/subagent-executable plan for turning the current
Task037 single-G1 multi-trial adaptation work into a cross-morphology
reproduction benchmark. The implementation may proceed only through the
subtasks below, each with its own closed loop and evidence gate:

```text
G1-like morphology distribution
  + unified joint/action/observation slot contract
  + multi-trial final-trial evaluation
  + true TXL long-memory policy
  + baseline comparison
```

The task must stay claim-driven. A component is useful only if it helps answer:

```text
Does a TXL long-memory policy adapt better than MLP/GRU/AdaptK baselines on
held-out G1-like morphology and dynamics variants?
```

Concrete output of Task038:

- a claim contract that defines baselines, splits, metrics, thresholds, and
  failure modes;
- a fixed G1-like semantic slot schema and G1 compatibility mapping;
- a reproducible procedural morphology manifest path that does not download
  assets;
- a true TXL memory policy contract with reset/cache semantics;
- a held-out multi-trial eval contract that writes comparable JSON evidence;
- a router/subagent execution plan with write scopes and review gates.

## Fixed Scope

- G1-like humanoid variants only.
- Keep high-level topology fixed for the first reproduction: same semantic leg,
  waist, arm, and gripper slots.
- Vary G1-like physical parameters: link length, mass, COM/inertia, motor
  strength/delay/damping/friction/failure.
- Keep action dimension fixed through a unified slot contract.
- Keep actor blind to exact morphology parameters unless a subtask explicitly
  opens a morphology-conditioned baseline.
- Use multi-trial evaluation: inner reset preserves memory, outer reset clears
  memory.
- Do not claim full LocoFormer reproduction, quadruped transfer, or real-robot
  deployment.

Out of scope:

- training a final policy inside this planning task;
- downloading checkpoints, robot assets, datasets, simulator assets, or upstream
  repos;
- changing implementation outside the active subtask write scope;
- touching `.test_tmp_task021/`;
- expanding Isaac Lab/Isaac Sim as the primary route on H200;
- broadening beyond fixed-topology G1-like humanoids.

## Baseline Evidence

Task037 provides the first adaptation benchmark:

- AdaptK160 clean prior `model_5467.pt` passes clean multi-trial eval at
  `0.4`, `1.2`, and `2.0 m/s`.
- Arbitrary single-joint dynamic onset multi-trial eval reaches `35/36`.
- Stable blocker: `left_hip_pitch_joint @ 2.0 m/s`, `0/5` seeds.

Task038 should preserve this benchmark as the single-G1 control comparison while
adding held-out morphology variants. Task037 evidence cannot pass a Task038
held-out morphology claim by itself.

Initial claim matrix before any superiority claim:

- variants: at least `2` seen G1-like ids and at least `4` held-out ids, with at
  least one link-length, one mass/COM/inertia, one motor/dynamics, and one
  combined morphology+dynamics held-out case;
- speeds: `0.4`, `1.2`, and `2.0 m/s`;
- seeds: at least `3` eval seeds per variant/speed condition;
- trials: at least `4` inner trials per outer episode, with final-trial metrics
  reported separately from aggregate metrics;
- baselines: MLP/specialist where available, GRU history, AdaptK160, and TXL
  must be represented as `attempted`, `not_attempted`, or `not_applicable`;
- TXL claim gate: TXL must beat the best attempted non-TXL baseline on held-out
  final-trial pass rate by at least `10` percentage points and must not regress
  seen/clean final-trial pass rate by more than `5` percentage points.

## Planned Slices

1. `001-claim-contract-and-feedback-loop.md`
   - Define the minimal LocoFormer-style claim, baselines, eval splits,
     thresholds, failure criteria, fake-env evidence, and JSON evidence shape.

2. `002-unified-g1like-slot-contract.md`
   - Define fixed semantic joint/action/observation slots, masks, and selectors
     for G1-like variants, with local schema round-trip evidence.

3. `003-procedural-g1like-morphology-generator.md`
   - Generate and validate train/held-out G1-like asset/config variants without
     changing slot semantics, with local manifest evidence and H200 load smoke.

4. `004-true-txl-memory-policy-contract.md`
   - Implement a real TXL cache policy consumer with inner-reset memory
     preservation and outer-reset clearing, verified first by local fake-env
     cache tests.

5. `005-g1like-heldout-multitrial-eval.md`
   - Run baseline and TXL policies on seen/held-out G1-like variants, with
     per-trial improvement and final-trial gates in comparable JSON.

6. `006-router-subagent-execution-plan.md`
   - Assign router, worker, and reviewer responsibilities, parallel boundaries,
     write scopes, review gates, and failure exits.

7. `007-g1like-mjcf-patch-load-smoke.md`
   - Convert one generated train and one held-out G1-like manifest variant into
     conservative patched MJCF artifacts and a local parse/contract JSON
     summary. This closes only the local artifact loop; full H200 load smoke
     remains pending.

8. `008-g1like-mujoco-compile-load-smoke.md`
   - Preserve source-relative MJCF mesh assets when patched XML is copied to an
     output directory, and add an optional MuJoCo compile/load smoke path that
     is off by default and records structured readiness evidence only when
     explicitly requested.

9. `009-mjlab-variant-env-load-smoke.md`
   - Register external MJLab env-load-only task ids for the patched train and
     held-out XML artifacts, and add a probe that checks env construction,
     action manager dimension, reset, observation finiteness, and zero-action
     steps without training or eval claims.

10. `010-mjlab-runner-smoke.md`
    - Register separate external MJLab runner-smoke task ids for the same
      Task038 train and held-out XML env cfg helpers, and add a probe that
      constructs the Task037 TXL-memory runner, forwards one inference policy
      action, and short-steps zero actions without training, eval, quality, or
      reproduction claims.

11. `011-true-txl-runner-consumer-smoke.md`
    - Register separate true-TXL runner-consumer smoke task ids for the same
      Task038 train and held-out XML env cfg helpers, and add a no-training
      probe that constructs `Task038TrueTxlMemoryK160Runner`, forwards the
      policy twice, and gates only on the new stateful cache actor attending
      previous segment memory to the attention key/value path.

12. `012-true-txl-reset-hook-integration-smoke.md`
    - Wire Task038 true-TXL actor reset hooks into the runner-visible
      `env.step()`/`env.reset()` path, and add a no-training probe that gates
      only on inner reset preserving selected env memory and outer reset
      clearing selected env memory before the next policy forward.

13. `013-true-txl-ppo-update-smoke.md`
    - Run one tiny train-variant-only
      `Task038TrueTxlMemoryK160Runner.learn(num_learning_iterations=1,
      init_at_random_ep_len=False)` path with small env count and rollout
      steps, gating only on PPO update return and runner/model/action/log
      summary fields plus positive stateless minibatch fallback debug. Flattened
      PPO update minibatches use stateless current-segment attention when their
      batch size differs from the env cache count; this is not full
      sequence-aware TXL training, not heldout training, and makes no
      quality/eval/reproduction/superiority claim.

14. `014-true-txl-inference-cache-safety-smoke.md`
    - Close the post-`013` inference-cache mutation boundary by proving Task038
      true-TXL actor cache tensors and counters are cloned before writes when
      they were created or preserved as PyTorch inference tensors. Router H200
      policy-forward evidence may close only this cache-safety boundary; no
      eval/quality/reproduction/superiority claim is made.

15. `015-true-txl-checkpoint-eval-load-smoke.md`
    - Load a checkpoint produced by the Task038 true-TXL runner path into the
      train or heldout Task038 true-TXL runner smoke task id, get the inference
      policy, and run a short policy-action rollout JSON gate. This proves only
      checkpoint load and rollout plumbing, not quality eval, reproduction, or
      superiority.

16. `016-true-txl-multitrial-metric-eval-smoke.md`
    - Run the Task038 true-TXL checkpoint through the multi-trial metric eval
      pipeline and emit comparable per-trial JSON, while keeping the top-level
      smoke pass independent of the delegated final-trial quality field.

## Minimal Closed Loop

Each subtask must close independently before a router can mark it reviewable:

1. `001`: local fake-env/schema test or JSON fixture proves seen/held-out split,
   final-trial improvement, baseline comparison, and failure classification.
2. `002`: local slot-schema test proves unique names, fixed action dimension,
   missing-slot masks, present-slot selectors, and G1 round-trip compatibility.
3. `003`: local manifest validation proves deterministic train/held-out variant
   generation; H200 smoke evidence is required before any load claim.
4. `004`: local fake-env cache test proves inner reset preserves TXL memory,
   outer reset clears it, and per-env masks prevent leakage; H200 smoke is
   required before runner compatibility is claimed.
5. `005`: JSON evidence compares attempted baselines and TXL on the same variant
   ids and reports final-trial pass/fail plus trial0-to-final improvement.
6. `006`: router plan names owner roles, parallel boundaries, write scopes,
   review gates, and stop conditions for all subtasks.
7. `007`: local XML patch tests prove train/held-out variants produce patched
   MJCF artifacts with unchanged body/joint/actuator topology, audited
   patched/skipped counts, limitation notes, and local ElementTree parse. H200
   simulator load remains pending.
8. `008`: local XML tests prove relative source meshdirs are rewritten for
   copied outputs, absolute meshdirs are preserved, CLI default does not import
   or compile MuJoCo, and optional compile/load reporting can be exercised
   without requiring MuJoCo to be installed. H200 compile/load remains pending
   unless an explicit H200 run records it.
9. `009`: local MemoryPath tests prove the external MJLab patcher inserts
   constants, XML cfg helpers, and two task registers exactly once; probe CLI
   tests prove defaults, expected action dimension, and structured failure JSON
   without requiring MJLab. H200 env-load is closed only after router-run train
   and held-out probes in external MJLab write passing JSON evidence.
10. `010`: local MemoryPath tests prove the external MJLab patcher inserts two
    runner-smoke task ids exactly once and imports the runner idempotently; probe
    tests prove defaults, pass-gate behavior, fake finite obs/action summaries,
    optional inner/outer reset gating, and no overclaim fields without requiring
    MJLab. H200 runner smoke remains pending until train and held-out probes
    show the expected runner class, `action_dim=31`, finite policy action,
    required extras, finite non-empty obs, and short zero-action stepping.
11. `011`: local MemoryPath tests prove the external MJLab patcher inserts two
    true-TXL runner-smoke task ids exactly once and imports the new runner
    idempotently; probe tests prove defaults, pass-gate behavior, true-TXL debug
    schema checks, previous-memory cache exposure gating, finite obs/action
    summaries, and no overclaim fields without requiring MJLab. H200 smoke
    remains pending until train and held-out probes show the expected runner and
    actor classes, `action_dim=31`, finite `[actual_num_envs,31]` policy action,
    required step extras, finite non-empty obs, and positive previous-memory
    cache exposure after repeated policy forward.
12. `012`: local fake env/actor tests prove the Task038 reset-hook wrapper
    dispatches `inner_reset` / `task037_inner_reset` to actor inner-reset hooks,
    dispatches `outer_reset` / `episode_done` / `task037_outer_reset` to actor
    outer-reset hooks, and clears actor cache on full env reset without changing
    Task037 runner semantics; probe tests prove defaults, pass-gate behavior,
    reset-event false-pass rejection, inner-preserve and outer-clear gates,
    wrong runner/model rejection, and no overclaim fields without requiring
    MJLab. H200 smoke remains pending until train and held-out probes show the
    expected runner/model/action dims, saw inner and outer reset extras, actor
    debug reset events, inner reset preserving selected env memory lengths
    before the next policy forward, and outer reset clearing selected env memory
    lengths before the next policy forward.
13. `013`: local tests prove CLI defaults, smoke config mutation, JSON writing,
    failure summary shape, pass-gate behavior, train-only task rejection for
    heldout variants, wrong runner/model/action dim rejection, learn-return
    rejection, missing log-dir rejection, and no overclaim fields without
    requiring MJLab. H200 smoke remains pending until one train-only probe shows
    `runner_cls=Task038TrueTxlMemoryK160Runner`,
    `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
    `total_action_dim=31`, `learn_returned=true`, `iterations=1`, positive
    `num_envs` and `rollout_steps`, positive
    `txl_debug.stateless_forward_batches`, `log_dir_exists=true`,
    `wall_time_s` recorded, and all claim flags false with
    `ppo_update_smoke_only=true`.
14. `014`: local fake-tensor tests simulate cache tensors with PyTorch
    inference-mode write restrictions and prove Task038 true-TXL layer-memory
    append, memory lengths, reset counters, incremental counters, segment
    counters, and token counters clone before mutation. Repeated direct
    `_append_layer_memory()` must grow memory lengths without the inference
    tensor RuntimeError, and inner/outer/full reset paths must record/clear the
    selected envs without mutating inference tensors in place. H200 policy
    forward remains pending router execution.
15. `015`: local tests prove the checkpoint eval-load smoke CLI defaults,
    train/heldout task allow-list, unrelated task rejection, missing checkpoint
    preflight, structured failure JSON, pass-gate behavior, JSON writing,
    `--help`, and no-overclaim fields without requiring MJLab. H200 smoke
    remains pending until a Task038 true-TXL checkpoint loads with strict actor
    load, returns `runner_cls=Task038TrueTxlMemoryK160Runner`,
    `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
    `total_action_dim=31`, finite `[actual_num_envs,31]` policy actions,
    finite non-empty observations, `policy_error=null`, positive policy step
    count, and all claim flags false with
    `checkpoint_eval_load_smoke_only=true`.
16. `016`: local tests prove the multi-trial metric eval smoke CLI defaults,
    train/heldout task allow-list, unrelated task rejection, missing checkpoint
    preflight, structured failure JSON, JSON writing, `--help`,
    no-overclaim fields, and the key pass-gate distinction where
    `quality_metric_final_trial_pass=false` can still yield top-level
    `pass=true` when the pipeline JSON is complete and finite. H200 metric
    eval smoke remains pending until router execution writes JSON with
    `trial_0`, `final_trial`, `aggregate`,
    `quality_metric_final_trial_pass`, `task037_pass`,
    `promotion_gate=pipeline_smoke_only`, `pipeline_pass=true`,
    `eval_pipeline_smoke_pass=true`, `failure_reasons=[]`, and all claim flags
    false with `eval_pipeline_smoke_only=true`.

## Final Acceptance

Task038 is accepted only when:

- `task.md` and `001-016` all keep `Route / Log / Review`;
- every subtask has a Minimal Closed Loop, Evidence Gate, Subagent Ownership,
  write scope, review gate, and failure exit;
- claim/eval docs preserve the minimum matrix above or explicitly record a
  router-approved stronger replacement before any H200 claim run;
- no subtask is marked passed without the evidence path or command output it
  names;
- all evidence is small text/JSON/log metadata, not large checkpoints, datasets,
  generated assets, or training artifacts;
- the final claim remains limited to a minimal G1-like LocoFormer reproduction
  benchmark plan, not a completed reproduction result.

## Log

- 2026-05-29 Opened after user chose the first LocoFormer-style reproduction to
  be G1-like only.
- 2026-05-29 Strengthened as a router/subagent-executable minimal closed-loop
  plan. No implementation, training, or asset download performed.
- 2026-05-29 Local planning/contracts now cover `001-006` with reviewer
  confirmation. Task038 still makes no H200 load, runner, eval, video,
  reproduction, or TXL superiority claim.
- 2026-05-29 Added `007` local G1-like MJCF patch artifact loop. It writes a
  pure-Python ElementTree patcher, local CLI, tests, and a contract JSON shape
  for later H200 load smoke. Verification recorded in `007`; full H200 load
  remains pending and is not claimed.
- 2026-05-29 Added `008` MuJoCo compile-load readiness slice. It preserves
  source-relative meshdir resolution for copied patched XML and adds optional
  structured MuJoCo compile/load smoke reporting. No H200 compile/load pass is
  claimed by this local readiness work.
- 2026-05-29 Review subagent accepted `008` with no blocking findings. H200
  compile/load evidence still requires an actual `--compile-mujoco` run.
- 2026-05-29 Added `009` MJLab variant env-load gate. It provides an
  idempotent external MJLab patcher for the Task038 train/held-out XML task ids
  and a zero-action env-load probe. Local tests do not require MJLab. H200
  external env-load evidence was recorded by the later router-run probes.
- 2026-05-29 Ran the `009` external MJLab env-load gate on H200 for one train
  and one held-out patched XML. Both probes passed on `cuda:0` with
  `action_dim=31`, `total_action_dim=31`, matched registered XML paths,
  finite actor/critic observations, and 10 zero-action steps. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/mjlab_variant_env_load/train_env_load.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/mjlab_variant_env_load/heldout_env_load.json`.
- 2026-05-29 Added `010` MJLab runner-smoke local wiring. It registers separate
  train/held-out runner-smoke task ids using the Task038 XML env cfg helpers and
  `Task037TxlMemoryK160DeterministicRunner`, plus a probe for runner
  construction, one inference-policy forward, required extras, finite obs, and
  short zero-action stepping. H200 runner-smoke evidence is pending; no training,
  eval, quality, or reproduction claim is made.
- 2026-05-29 Ran the `010` external MJLab runner-smoke gate on H200 for one
  train and one held-out patched XML. Both probes passed on `cuda:0` with
  `runner_cls=Task037TxlMemoryK160DeterministicRunner`, `action_dim=31`,
  `total_action_dim=31`, `policy_action_shape=[8,31]`,
  `step_required_extras_missing=[]`, finite actor/actor_history/critic
  observations, and 8 zero-action steps. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/runner_smoke/train_runner_smoke_env8.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/runner_smoke/heldout_runner_smoke_env8.json`.
- 2026-05-29 Added `011` true-TXL runner-consumer smoke local wiring. It
  registers separate true-TXL train/held-out runner-smoke task ids, adds
  `Task038TrueTxlMemoryK160Runner` with `Task038TrueTxlMemoryModel`, and adds a
  no-training probe that requires repeated policy forward to show positive
  previous-memory cache exposure in `txl_debug`. No training, eval, quality,
  reproduction, or superiority claim is made.
- 2026-05-29 Ran the `011` H200 true-TXL runner-consumer smoke gate on `cuda:0`
  for the Task038 train and held-out XML variants. Both env8 probes passed with
  `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, finite `[8,31]` policy action,
  `step_required_extras_missing=[]`, finite actor/actor_history/critic
  observations, and positive previous-memory cache exposure after repeated
  policy forward. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_runner_smoke/train_true_txl_runner_smoke_env8.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_runner_smoke/heldout_true_txl_runner_smoke_env8.json`.
- 2026-05-29 Added `012` true-TXL reset-hook integration smoke wiring. It adds
  a Task038-only runner env wrapper that dispatches Task037 multi-trial reset
  extras to `Task038TrueTxlMemoryModel` hooks, clears actor cache on full
  runner env reset, and adds a no-training probe/test gate for inner-preserve
  and outer-clear behavior. No training, eval, quality, reproduction, or
  superiority claim is made.
- 2026-05-29 Ran the `012` H200 reset-hook integration smoke gate on `cuda:0`
  for Task038 train and held-out true-TXL runner tasks. Both env8 probes passed
  with `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, finite `[8,31]` policy action,
  `step_required_extras_missing=[]`, `saw_inner_reset=true`,
  `saw_outer_reset=true`, positive actor reset event totals,
  `inner_reset_preserved_memory_before_next_policy=true`,
  `outer_reset_cleared_memory_before_next_policy=true`, and no claim flags.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_reset_hook/train_true_txl_reset_hook_env8.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_reset_hook/heldout_true_txl_reset_hook_env8.json`.
- 2026-05-29 Added planned slice `013` for a train-only true-TXL PPO update
  smoke. The local CLI/test gate mutates the loaded agent config down to one
  tiny update and rejects heldout task ids, wrong runner/model/action dims,
  missing `learn_returned`, missing log dir, and overclaim flags. H200 execution
  is pending router run; no heldout training, long training, eval, quality,
  reproduction, or superiority claim is made.
- 2026-05-30 Updated `013` after H200 exposed the env-cache/PPO-minibatch batch
  mismatch. Task038 true-TXL actor keeps stateful cache for env-batch
  rollout/inference forwards, but uses stateless current-segment attention for
  flattened PPO update minibatches whose batch size differs from the cache env
  count. The smoke gate now records and requires positive
  `txl_debug.stateless_forward_batches`. This is not full sequence-aware TXL
  training and makes no quality/eval/reproduction/superiority claim.
- 2026-05-30 Ran the `013` H200 train-only PPO update smoke on `cuda:0` after
  making post-learn policy action summary diagnostic-only. The probe returned
  `pass=true`, `learn_returned=true`,
  `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, `iterations=1`, `num_envs=8`, `rollout_steps=2`,
  `log_dir_exists=true`, `failure_reasons=[]`,
  `txl_debug.stateless_forward_batches=1`, and
  `txl_debug.stateless_forward_samples=16`. The optional post-learn policy
  action forward still recorded a PyTorch inference-tensor `policy_error`, but
  that field is not part of this PPO update return gate. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_ppo_update_smoke/train_true_txl_ppo_update_env8_iter1_postlearn_optional.json`.
- 2026-05-30 Added planned slice `014` for the true-TXL inference-cache safety
  boundary exposed by the optional post-learn `policy(obs)` diagnostic in
  `013`. The local implementation clones cache tensors and counters before
  mutation when they are PyTorch inference tensors, with fake-tensor tests for
  repeated layer-memory append and reset hook writes. No eval, quality,
  reproduction, or superiority claim is made.
- 2026-05-30 Ran the `014` local inference-cache safety gate:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_inference_cache_safety.py`
  returned `4 passed in 0.06s`. Regression checks also passed for
  `tests\test_task038_true_txl_reset_hook.py` (`9 passed in 0.06s`) and
  `tests\test_task038_true_txl_ppo_update_smoke.py` (`16 passed in 0.08s`).
- 2026-05-30 Ran the `014` H200 real policy-forward cache-safety smoke by
  rerunning the Task038 train-only true-TXL PPO update probe after syncing the
  cache clone fix. The post-learn policy action forward returned
  `policy_action_shape=[8,31]`, `policy_action_finite=true`, and
  `policy_error=null`; the probe also returned `pass=true`,
  `learn_returned=true`, `failure_reasons=[]`,
  `txl_debug.stateless_forward_batches=1`, and
  `txl_debug.stateless_forward_samples=16`. This closes only the cache-safety
  boundary. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_inference_cache_safety/train_true_txl_ppo_update_env8_iter1_policy_forward_safe.json`.
- 2026-05-30 Added planned slice `015` for Task038 true-TXL checkpoint
  eval-load smoke. The local CLI/test gate rejects missing checkpoints and
  unrelated task ids before MJLab import, then the H200 route will load a
  Task038 true-TXL checkpoint into the train or heldout true-TXL runner smoke
  task and run a short policy-action rollout JSON. No quality eval,
  reproduction, or superiority claim is made.
- 2026-05-30 Ran the `015` local checkpoint eval-load smoke gate:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_checkpoint_eval_smoke.py`
  returned `14 passed in 0.11s` after the optional `txl_debug` snapshot fix.
  Router focused regression returned `48 passed in 0.18s`, and the full
  Task038 regression set returned `154 passed in 0.78s`.
- 2026-05-30 Ran the `015` H200 checkpoint eval-load smoke on `cuda:0` for
  both the train and heldout Task038 true-TXL runner task ids using the
  Task038-generated `model_0.pt` from the `014` smoke. Both probes returned
  `pass=true`, `load_returned=true`, `step_count=10`,
  `policy_action_shape=[8,31]`, `policy_action_finite=true`,
  `policy_error=null`, finite actor/history/critic observation summaries,
  `failure_reasons=[]`, and all claim flags false. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_checkpoint_eval_load/train_true_txl_model0_eval_load_env8_steps10_optional_txl_debug.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_checkpoint_eval_load/heldout_true_txl_model0_eval_load_env8_steps10_optional_txl_debug.json`.
- 2026-05-30 Added planned slice `016` for Task038 true-TXL multi-trial
  metric eval pipeline smoke. The local CLI reuses the Task037 multi-trial
  evaluator and wraps its output so top-level `pass` means pipeline health,
  while the delegated `final_trial_pass` is preserved only as
  `quality_metric_final_trial_pass`. H200 execution is pending router run; no
  quality/eval/reproduction/superiority claim is made.
- 2026-05-30 Ran the `016` local multi-trial metric eval smoke gate:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_multitrial_eval_smoke.py`
  returned `13 passed in 0.11s`. Focused regression with `015` also passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_multitrial_eval_smoke.py tests\test_task038_true_txl_checkpoint_eval_smoke.py`
  returned `27 passed in 0.15s`.
- 2026-05-30 Closed the `016` P1 schema-gate review blocker and reran router
  local verification. The focused post-fix regression returned `40 passed in
  0.20s`, the full Task038 regression returned `174 passed in 1.13s`, and
  `python -m h200_locomotion_lab.tools.inspect_agent` completed successfully.
  Review subagent found no blockers, with the residual risk that the smoke
  schema gate validates `trial_0`, `final_trial`, and `aggregate`, not every
  intermediate `trial_N`.
- 2026-05-30 Ran the `016` H200 true-TXL multi-trial metric pipeline smoke on
  `cuda:0` for both train and heldout Task038 true-TXL runner task ids using
  the Task038-generated `model_0.pt` from `014`. Both JSON files returned
  `pass=true`, `pipeline_pass=true`, `eval_pipeline_smoke_pass=true`,
  `quality_metric_final_trial_pass=false`, `task037_pass=false`,
  `promotion_gate=pipeline_smoke_only`, `failure_reasons=[]`, and all claim
  flags false. This closes only the Task038/016 metric pipeline plumbing. It
  does not promote a policy-quality, eval, reproduction, or superiority claim.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_multitrial_eval_smoke/train_true_txl_model0_multitrial_metric_env8_steps60_schema_gate.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_multitrial_eval_smoke/heldout_true_txl_model0_multitrial_metric_env8_steps60_schema_gate.json`.

## Review

Status: local planning/contracts `001-006`, `007` local MJCF patch artifacts,
`008` compile-load readiness, `009` local/H200 env-load gate artifacts, and
`010` local/H200 runner-smoke wiring are implemented. `011` local/H200
true-TXL runner-consumer smoke evidence is implemented and closed by final
reviewer confirmation. `012` local reset-hook integration wiring and local/H200
reset-hook smoke evidence are implemented and closed by final reviewer
confirmation. `013` local/H200 train-only PPO update smoke evidence is
implemented and closed by reviewer confirmation. `014` local/H200
inference-cache safety smoke is implemented and closed by reviewer
confirmation. `015` local/H200 checkpoint eval-load smoke is implemented and
closed by reviewer confirmation. `016` local/H200 multi-trial metric pipeline
smoke is implemented and closed by reviewer confirmation.
`009`, `010`, `011`, `012`, `013`, `014`, `015`, and `016` are closed by final
reviewer confirmation. Full eval matrix, representative videos, actual policy
training/evaluation beyond the tiny `013` update-path smoke, and any
reproduction/superiority claim remain pending.
