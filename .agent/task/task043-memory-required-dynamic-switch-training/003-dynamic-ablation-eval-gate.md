# 003: Dynamic Ablation Eval Gate

## Route

Evaluate each Task043 candidate checkpoint on the same dynamic switch setting
with three modes:

- normal;
- `zero_txl_residual`;
- `stateless_txl_memory`.

This is the memory-causality gate. Normal quality alone is not enough; ablation
must degrade materially before we call the memory path behaviorally useful.

Use `src/h200_locomotion_lab/tools/task043_dynamic_ablation_eval.py` so the
summary records the Task043 task id instead of reusing Task042 labels.

## Acceptance

- Three JSON summaries exist for the same checkpoint, seed, speed, steps, and
  dynamic switch setting.
- Normal mode passes the selected quality gate before any quality success claim.
- Ablated modes are compared to normal with concrete metrics.
- Review explicitly states positive, weak, or negative memory-causality
  evidence.

## Log

- 2026-05-31 Opened.
- 2026-05-31 Added the Task043 eval wrapper.
- 2026-05-31 H200 `--help` for
  `python -m h200_locomotion_lab.tools.task043_dynamic_ablation_eval` passed.
- 2026-05-31 Smoke checkpoint eval triplet completed:
  - normal:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/smoke_model0_none_vx1p6_seed4300201.json`;
  - zero residual:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/smoke_model0_zero_residual_vx1p6_seed4300201.json`;
  - stateless memory:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/smoke_model0_stateless_vx1p6_seed4300201.json`.
  All three complete the final trial without falling. Normal mode has
  `pipeline_pass=true`, `quality_gate_pass=false`,
  `lin_vel_error.mean=0.477243572473526`, `gravity_xy.max=0.09868066757917404`,
  and `root_z.min=0.7612224817276001`. Zero-residual final linear velocity
  error is `0.4860592782497406`; stateless-memory final linear velocity error
  is `0.4758363664150238`.

## Review

Status: passed for smoke eval coverage, failed for positive quality and memory
causality. The smoke checkpoint is stable but still too slow, and ablations are
behaviorally tied with normal mode. Continue with larger training chunks.
