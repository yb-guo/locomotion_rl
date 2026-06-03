# 006 Long-Context Training Decision

## Route

Train only after the multi-trial reset/memory/eval semantics are proven.

Acceptance criteria:

- Run short training smoke before any longtrain.
- Full quality eval uses speeds `0.4`, `1.2`, and `2.0`.
- Full quality eval includes dynamic switch and forced deadgrid.
- Multi-trial final-trial metrics are the default pass gate.
- Velocity tracking gate prevents stand-still/high-pose local optima.
- Compare against Task036 AdaptK4 partial.
- Decision states promoted, partial, or rejected with JSON evidence.

## Log

- 2026-05-29 Planned.
- 2026-05-29 Started from the already-passed 005 short training smoke before
  any quality claim.
- 2026-05-29 Added Task037 TXL K160 full-validation task ids:
  - `Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DynamicMotorFailure-Fast1p6`
  - `Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-FocusedDeadGrid-Fast2p0`
- 2026-05-29 Extended
  `h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint` with
  `--dynamic-case switch` and `--force-dead-joint`, while preserving
  `final_trial_pass` as the promotion gate.
- 2026-05-29 Added
  `task037_validate_multitrial_checkpoint_matrix.sh` for speeds `0.4`, `1.2`,
  and `2.0`, with dynamic switch plus 12-joint forced-dead grid.
- 2026-05-29 Local validation:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task037_mjlab_smoke_scripts.py tests/test_task037_multitrial_contract.py tests/test_agent_inventory.py`
    -> `7 passed, 4 skipped in 0.10s`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint --help`
    -> passed.
- 2026-05-29 H200 validation for updated registry/eval script:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter`,
  `tests/test_task037_mjlab_smoke_scripts.py` -> `5 passed in 0.05s`.
- 2026-05-29 H200 TXL K160 scratch train completed:
  - log:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/long_context_train/037_txl_k160_scratch_env8192_iter60_gpu0_seed3700601.stdout.log`
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_long_context_train/2026-05-29_14-54-42_037_txl_k160_scratch_env8192_iter60_gpu0_seed3700601/model_59.pt`
  - training telemetry already looked bad: final iteration reward about
    `-37.78`, mean episode length about `166`, and frequent `fell_over`.
- 2026-05-29 H200 full validation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/full_validation_txl_k160_model59_iter60/task037_full_validation_summary.json`.
  Local copy:
  `task037_full_validation_txl_k160_model59_iter60_summary.json`.
  Result: `pass=false`.
  - `0.4 m/s`: dynamic switch `pass=false`,
    final-trial fall ratio `1.0`, deadgrid `0/12`.
  - `1.2 m/s`: dynamic switch `pass=false`,
    final-trial fall ratio `1.0`, deadgrid `0/12`.
  - `2.0 m/s`: dynamic switch `pass=false`,
    final-trial fall ratio `1.0`, deadgrid `0/12`.
- 2026-05-29 Comparison against Task036 AdaptK4 partial:
  - AdaptK4 `model_5408` is still the better partial reference.
  - Its Task036 full eval is still not promoted (`pass=false`), but it passed
    the Task037 clean multi-trial final-trial eval in 004
    (`final_trial_pass=true`, fall ratio `0.0`).
  - TXL K160 scratch `model_59` fails even the full matrix at `0.4 m/s`, so it
    is rejected for this route.
- 2026-05-29 Local decision summary:
  `task037_decision_summary.json`.

## Review

Status: complete.

Decision: reject the current TXL-style K160 scratch policy-quality route. Keep
the Task037 infrastructure: multi-trial contract, deterministic inner reset,
per-trial JSON eval, and K160 construction smoke are useful. Do not promote the
trained K160 checkpoint. Next work should change the objective/curriculum or
warm-start strategy before further long-context policy training.
