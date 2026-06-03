# 007 Warmstart Clean Gait Prior

## Route

The Task037 TXL K160 scratch checkpoint failed even in clean multi-trial eval:

- `0.4`, `1.2`, and `2.0 m/s` clean final-trial fall ratio were all `1.0`;
- Task036 AdaptK4 `model_5408` passed the same clean Task037 eval at `2.0 m/s`.

This subtask stops full failure-matrix training until the long-context policy
has a clean locomotion prior.

Fixed boundaries:

- Keep Task037 multi-trial contract unchanged.
- Use clean unified-speed training first: no motor failure, no dynamic switch,
  no forced dead-grid.
- Keep actor blind to explicit fault labels, motor scales, active joint ids,
  trial index, and final-trial flag.
- Keep `3.2s` history target (`K=160`, 50 Hz).
- Do not claim dynamic-failure adaptation from this subtask.

Implementation route:

1. Add a clean-only H200 task id:
   `Unitree-G1-Gripper-Flat-Task037-AdaptK160-CleanUnified-Fast2p0`.
2. Use a K160 adaptation-conditioned consumer, not the current TXL token
   scratch actor, for the first warmstart gate:
   - actor history input: `160 * (104 obs + 31 action) = 21600D`;
   - adaptation latent: `32D`;
   - actor head input remains `104D newest obs + 32D latent = 136D`;
   - warmstart can preserve the AdaptK4 actor head from `model_5408`;
   - K160 encoder first layer is fresh because K4 and K160 history dimensions
     differ.
3. Train only clean gait prior from the AdaptK4 warmstart.
4. Evaluate clean multi-trial speeds `0.4`, `1.2`, and `2.0 m/s`.
5. Only if clean gait passes, reopen failure curriculum in a later subtask.

Acceptance:

- Local tests cover the K4-to-K160 warmstart migration shape contract.
- H200 registration includes the clean-only AdaptK160 task id.
- H200 smoke confirms construction with the clean-only task id.
- H200 training log and checkpoint path are recorded.
- Clean multi-trial eval JSON covers speeds `0.4`, `1.2`, and `2.0`.
- Pass requires all clean final-trial gates to pass:
  - completion ratio `>= 0.95`;
  - fall ratio `<= 0.50`;
  - linear velocity error `<= 1.20`;
  - yaw velocity error `<= 1.00`;
  - max gravity xy `<= 0.90`;
  - min root z `>= 0.35`.

Fail:

- The clean-only AdaptK160 run falls in clean multi-trial eval.
- The actor receives explicit fault/debug labels.
- Full failure-matrix or dynamic-switch claims are made before clean gait
  passes.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/clean_gait_prior/`

## Log

- 2026-05-29 Opened after diagnosing that TXL K160 scratch failed clean
  multi-trial gait at `0.4`, `1.2`, and `2.0 m/s`, while AdaptK4 passed clean
  Task037 eval at `2.0 m/s`.
- 2026-05-29 Added the local implementation route:
  `Task037AdaptK160DeterministicInnerResetRunner`, K4-to-K160 adaptation
  warmstart migration, clean-only registration block, and H200 launch script.
- 2026-05-29 Local validation:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task037_multitrial_contract.py tests/test_task037_mjlab_smoke_scripts.py tests/test_agent_inventory.py`
    -> `8 passed, 5 skipped`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m h200_locomotion_lab.tools.inspect_agent`
    -> passed.
- 2026-05-29 H200 registration/test validation:
  - `task037_register_multitrial_stages.py` patched the external MJLab registry.
  - `tests/test_task037_multitrial_contract.py tests/test_task037_mjlab_smoke_scripts.py tests/test_task033_history_buffer.py`
    -> `18 passed`.
  - `task037_launch_clean_gait_prior.sh --help` and `DRY_RUN=1` completed.
- 2026-05-29 H200 env64 warmstart construction smoke completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/clean_gait_prior/037_adapt_k160_clean_smoke_env64_iter1_gpu0_seed3700703.stdout.log`.
  The runner loaded AdaptK4 `model_5408` into AdaptK160, preserving matching
  actor-head weights, expanding normalizers from K4 to K160, and leaving
  `adaptation_encoder.0.weight` fresh.
- 2026-05-29 H200 clean-only env8192 training completed:
  - log:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/clean_gait_prior/037_adapt_k160_clean_from_adaptk4_env8192_iter60_gpu0_seed3700705.stdout.log`
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_clean_gait_prior_train/2026-05-29_16-44-07_037_adapt_k160_clean_from_adaptk4_env8192_iter60_gpu0_seed3700705/model_5467.pt`
  - throughput was about `99.6k` steps/s near the end, with `fell_over=0.0`.
- 2026-05-29 H200 clean multi-trial eval for `model_5467.pt` completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/clean_gait_prior/task037_adaptk160_model5467_clean_eval_summary.json`.
  Result: `pass=true`.
  - `0.4 m/s`: pass, fall ratio `0.0`, linear velocity error `0.116389`.
  - `1.2 m/s`: pass, fall ratio `0.0`, linear velocity error `0.317614`.
  - `2.0 m/s`: pass, fall ratio `0.0`, linear velocity error `0.676361`.

## Review

Status: pass for the clean gait prior gate. AdaptK160 warmstart plus clean-only
training preserves clean multi-trial gait over `0.4`, `1.2`, and `2.0 m/s`.
No dynamic-switch or dead-motor adaptation claim is made by this subtask.
