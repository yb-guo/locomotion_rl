# Task 042: TXL Memory Causality and Residual Training

## Route

Task041 closed the clean 0.4 m/s gate through an AdaptK160 -> true-TXL
warmstart bridge. That proves the true-TXL plumbing can run and eval, but it
does not prove the TXL memory path is behaviorally useful.

Task042 goal is the next minimal LocoFormer-reproduction step:

- keep the Task041 bridge checkpoint as the clean-gait prior;
- continue training with Task040 sequence-aware PPO so the TXL residual can
  learn;
- add explicit memory ablations;
- compare normal eval against ablated evals without changing env/reward/action
  contracts.

## Fixed Scope

- Use the same G1-like MJLab task and action space as Task041.
- Keep reward, env, action, observation, and morphology contracts unchanged.
- Keep the AdaptK160 bridge available as the gait prior.
- Keep no-overclaim flags false until evidence supports the claim.
- Do not touch `.test_tmp_task021/`.

Out of scope:

- morphology randomization;
- arbitrary robot topology;
- full LocoFormer architecture parity;
- dynamic motor-failure robustness as the primary acceptance gate;
- claiming superiority over AdaptK/MLP baselines.

## Planned Slices

1. `001-memory-ablation-contract.md`
   - Add test-covered eval controls for memory ablations and JSON evidence.
   - Required modes start with:
     - `none`;
     - `zero_txl_residual`;
     - `stateless_txl_memory`.

2. `002-residual-warmstart-train.md`
   - Continue training from the Task041 bridge checkpoint with sequence-aware
     PPO.
   - Record projection/parameter deltas proving TXL residual parameters can
     change during update.

3. `003-clean-memory-ablation-eval.md`
   - Evaluate clean 0.4 m/s normal vs ablated modes on the same checkpoint.
   - Pass requires normal eval to remain clean-pass and ablation evidence to be
     recorded without overclaiming.

4. `004-speed-and-dynamic-probe.md`
   - Only after clean memory-ablation eval is stable, probe 1.2/2.0 m/s and a
     small dynamic-switch case to decide the next task boundary.

5. `005-residual-only-train-guardrail.md`
   - If unrestricted residual continuation breaks clean gait, freeze the
     AdaptK/warmstart baseline path and train only the TXL residual path.

6. `006-harder-dynamic-switch-probe.md`
   - Probe a harder 1.6 m/s dynamic switch case with normal, zero-residual, and
     stateless-memory modes before deciding the next training task boundary.

## Acceptance Criteria

Task042 is accepted only when current evidence proves:

- local tests cover all memory ablation modes and summary fields;
- train summary records:
  - `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
  - `actor_model_class=Task038TrueTxlMemoryModel`;
  - `train_pipeline_pass=true`;
  - `stateless_fallback_forward_batches=0`;
  - active sequence update counters;
  - TXL residual parameter delta metrics;
- eval summaries record:
  - `memory_ablation_mode`;
  - `memory_debug_active`;
  - `memory_residual_enabled`;
  - `txl_residual_output_norm` or an equivalent measurable proxy;
  - no-overclaim flags;
- normal clean 0.4 m/s eval keeps `pipeline_pass=true`,
  `quality_gate_pass=true`, and `pass=true`;
- ablated evals are recorded and compared against normal eval;
- the review states one of:
  - memory causality evidence is positive and ready for a broader eval task; or
  - memory causality evidence is absent/weak and the next task must change the
    policy/training contract.
- if residual-only training is used, summaries must prove frozen baseline
  parameters and actor normalizer buffers did not change.

## Evidence Gate

Local:

```powershell
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task042_memory_ablation_eval --help
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task042_memory_ablation_contract.py tests\test_agent_inventory.py
```

H200:

```bash
PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab \
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task042_memory_ablation_eval \
  --checkpoint /path/to/task042/model.pt \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/eval.json \
  --memory-ablation-mode none \
  --num-envs 64 \
  --steps 360 \
  --lin-vel-x 0.4 \
  --device cuda:0
```

## Log

- 2026-05-30 Opened after Task041 clean 0.4 m/s pass via the AdaptK160
  warmstart bridge. Current gap: active true-TXL memory plumbing is proven, but
  memory causality is not.
- 2026-05-30 Subtask 001 ablation contract passed locally and on H200. Normal,
  zero-residual, and stateless-memory modes all produce JSON summaries. The
  bridge checkpoint reports `txl_residual_raw_norm=0.0`, so current evidence is
  explicitly not memory causality.
- 2026-05-30 Subtask 002 bounded residual train passed on H200:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_warmstart_train/env1024_iter5_summary.json`.
  The produced checkpoint is
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_warmstart_train/logs_env1024_iter5/model_4.pt`;
  `memory_output_projection_delta_norm=0.1979231983423233` and
  `txl_residual_output_norm_last=0.377013623714447`.
- 2026-05-30 Subtask 003 failed for that unrestricted residual checkpoint:
  normal clean eval and zero-residual eval both failed. The residual path became
  non-zero, but the warmstart baseline also drifted, so Task042 moved to
  subtask 005 residual-only training guardrails.
- 2026-05-30 Subtask 005 guardrail passed locally and on H200. Projection-only
  training changed `memory_output_projection` while keeping frozen actor
  parameters and actor normalizer buffers unchanged:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/projection_only_env1024_iter5_summary.json`.
  Clean ablation evals are recorded under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/projection_only_env1024_iter5_*`.
  They preserve final clean gait, but zeroing the residual does not meaningfully
  hurt behavior; no memory-causality claim is supported.
- 2026-05-30 `txl_residual_only` also passed the frozen-baseline guardrail and
  clean ablation evals were recorded. It updates token, attention, norm,
  position, and projection parameters while keeping frozen baseline state
  unchanged, but zero-residual/stateless ablations still do not materially hurt
  clean 0.4 m/s walking.
- 2026-05-30 Subtask 004 speed probe recorded: 1.2 m/s clean eval passes
  quality, while 2.0 m/s completes without falling but fails the strict quality
  gate on speed error and trial0-regression checks. The dynamic right-knee probe
  is blocked by env setup because the current True-TXL eval task does not
  register `dynamic_motor_failure`.
- 2026-05-31 Fixed the dynamic eval boundary by adding a Task042-specific
  True-TXL dynamic MJLab task id:
  `Unitree-G1-Gripper-Flat-Task042-TrainTrueTxlDynamicMotorFailure-Fast1p6`.
  H200 right-knee dynamic single-onset eval now runs with
  `Task038TrueTxlMemoryK160Runner` and passes in normal mode:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/speed_dynamic_probe/txl_residual_only_right_knee_dynamic_task042_none_vx0p4_seed4104204.json`.
  Zero-residual and stateless-memory ablations also preserve locomotion quality,
  so this is still negative memory-causality evidence.
- 2026-05-31 Subtask 006 recorded a harder 1.6 m/s dynamic switch ablation
  triplet under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/harder_dynamic_probe/`.
  Normal mode does not fall and keeps `pipeline_pass=true`, but misses the
  strict quality gate on forward speed tracking. Zero-residual and stateless
  ablations produce near-identical final linear velocity error (`0.4865` and
  `0.4801` vs normal `0.4828`), so the current checkpoint still does not support
  a memory-causality claim.

## Review

Status: closed as a negative memory-causality result. No reproduction,
superiority, or memory-causality pass claim is made.

The engineering fix for stateless fallback/baseline drift is in place: guarded
training records `stateless_fallback_forward_batches=0` and proves frozen
baseline parameters plus actor normalizer buffers remain unchanged. The current
negative result is scientific rather than plumbing: clean constant-speed walking
does not show memory causality, because zero-residual/stateless ablations do not
materially hurt the final gait.

Next boundary: do not keep tuning clean 0.4 m/s ablations as if they prove
LocoFormer-style adaptation. The compatible True-TXL dynamic failure task now
exists and passes the right-knee dynamic quality gate, but the ablations show
the behavior does not rely on stateful TXL memory. The harder 1.6 m/s switch
triplet is also negative: normal mode fails strict quality, and ablations remain
behaviorally tied. The next task should train directly on memory-required
dynamic-switch/multi-trial scenarios rather than keep measuring this checkpoint.
