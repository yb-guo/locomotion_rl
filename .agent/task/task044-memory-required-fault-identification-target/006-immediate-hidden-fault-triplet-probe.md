# 006: Immediate Hidden-Fault Triplet Probe

## Route

The current hidden-fault train/eval loop failed to create a memory-causality
gap. The strongest current hypothesis is target design: final-trial eval resets
the failure timeline, so stateless inference can remain tied when the final
2-second window does not force a remembered fault at the start.

Run a smaller, deterministic probe before long training:

- same checkpoint, speed, seed, env count, and steps for all three modes;
- hidden fault labels remain absent from actor observations;
- set a single hidden dead motor active from the first control step of each
  trial with `--dynamic-dead-joint`, `--dynamic-onset-s 0.0`, and
  `--dynamic-recovery-s 2.0`;
- compare normal, zero-residual, and stateless-memory JSONs through the Task044
  triplet summary contract.

Primary candidate:

- checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt`
- speed: `1.6`
- joint: `left_knee_joint` first, then `right_knee_joint` if the left-knee
  probe exposes a degradation gap.

## Acceptance

- Normal, zero-residual, and stateless JSON paths are recorded.
- The command records immediate onset and matching recovery in all three files.
- Triplet summary is recorded.
- If normal quality fails, do not claim memory causality.
- If normal quality passes but ablations remain tied, mark the target invalid
  and redesign the task rather than lowering thresholds.
- If normal quality passes and both ablations degrade materially, promote the
  checkpoint/schedule to a longer Task044 eval matrix.

## Log

- 2026-05-31 Created after the 25-iteration hidden-fault triplet failed with
  tied ablations.
- 2026-05-31 First H200 attempt failed before simulation because the remote
  `PYTHONPATH` included the adapter `src` but not the MJLab repo root required
  for `import src.tasks`. This JSON is diagnostic only and is not a policy
  result:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_immediate_left_knee_none_vx1p6_seed4400601.json`.
- 2026-05-31 Re-ran with both adapter `src` and MJLab repo root on
  `PYTHONPATH`. Normal:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_immediate_left_knee_none_vx1p6_seed4400601_rerun.json`.
  Final trial: completion `1.0`, fall `0.0`,
  `lin_vel_error.mean=0.46178966760635376`, `quality_gate_pass=false`.
- 2026-05-31 Zero residual:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_immediate_left_knee_zero_vx1p6_seed4400601.json`.
  Final `lin_vel_error.mean=0.4648202657699585`.
- 2026-05-31 Stateless memory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_immediate_left_knee_stateless_vx1p6_seed4400601.json`.
  Final `lin_vel_error.mean=0.46197593212127686`.
- 2026-05-31 Triplet summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_hidden_fault_env1024_iter25_model24_immediate_left_knee_triplet_seed4400601.json`.
  Result: `task044_memory_required_pass=false` with
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`. Zero-residual lin-vel-error delta
  was `0.0030305981636047363`; stateless delta was
  `0.0001862645149230957`.

## Review

Status: failed, not accepted.

Immediate hidden left-knee failure does not create a memory-causality gap for
the current checkpoint and target. The normal policy still misses the stricter
quality gate, and both ablations remain effectively tied with normal mode. More
iterations on the same target are unlikely to prove LocoFormer-style memory
unless the target is redesigned to make historical identification produce an
observable advantage.
