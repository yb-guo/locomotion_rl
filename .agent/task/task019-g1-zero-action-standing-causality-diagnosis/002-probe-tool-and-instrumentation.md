# 002: Probe Tool And Instrumentation

## Goal

Add a standalone zero-action standing causality probe with control-mode and
pose-profile switches for the six-case gate.

## Route

1. Add `src/h200_locomotion_lab/tools/g1_zero_action_standing_causality.py`.
2. Implement control modes:
   - `genesis_position`;
   - `genesis_position_resend_physics`;
   - `custom_pd_torque`.
3. Implement pose profiles:
   - `current`;
   - `unitree_gym`.
4. Keep the probe below PPO: no actor-critic model, no rollout collection, no
   GAE, no PPO update.
5. Write `config.json`, `metrics.jsonl`, and `summary.json`.
6. Add focused tests for parsing, pose profiles, action target semantics,
   control-mode calls, summary classification, and forbidden PPO imports.

## Log

- 2026-05-11 Coding subagent added
  `src/h200_locomotion_lab/tools/g1_zero_action_standing_causality.py`.
- 2026-05-11 Probe is standalone below policy optimization paths: it imports
  `VectorizedGenesisBackend` directly, writes `config.json`, `metrics.jsonl`,
  and `summary.json` under
  `outputs/task019/zero_action_standing_causality/<run_id>`, and keeps the
  `/root/agent_workspace/project` output guard plus `physical_gpu=1` /
  `logical_cuda_device=cuda:0` CUDA isolation guard.
- 2026-05-11 Implemented control modes `genesis_position`,
  `genesis_position_resend_physics`, and `custom_pd_torque`; pose profiles
  `current` and `unitree_gym`; per-chunk reset/tilt/height/root/upright/joint
  error/joint velocity/control/saturation/contact/throughput metrics.
- 2026-05-11 Router semantic correction: `pose_profile=current` means the
  task018 failing operational baseline, not the raw YAML default. It uses
  task018 `tall_crouch` leg values `hip_pitch=-0.06`, `knee=0.12`,
  `ankle_pitch=-0.07` while preserving profile non-leg joint values.
- 2026-05-11 Added
  `tests/test_g1_zero_action_standing_causality.py` covering argument parsing,
  pose profile values, forbidden source strings, summary pass/fail, output path
  guard, artifact writing, and fake control-mode call behavior.
- 2026-05-11 Local focused test command:
  `PYTHONPATH=src python -m pytest tests/test_g1_zero_action_standing_causality.py -p no:cacheprovider`
  passed with `7 passed, 1 skipped`. The skipped test is the torch-backed fake
  artifact-writing runtime because local Windows Python could not import torch.
- 2026-05-11 `python -m compileall` was attempted but blocked by local
  `__pycache__` write permissions; pytest import/collection/execution covered
  the focused module without bytecode cache writes.
- 2026-05-11 Router fix: probe `main()` now exits nonzero only for execution
  errors/blockers. Diagnostic gate outcomes still print `status` as `passed` or
  `failed`, while exceptions print `status=error` with a blocker. Added focused
  tests for diagnostic failure exiting zero and exception path exiting nonzero.
- 2026-05-11 Local focused test command after router fix:
  `PYTHONPATH=src python -m pytest tests/test_g1_zero_action_standing_causality.py -p no:cacheprovider`
  passed with `9 passed, 1 skipped`.

## Review

Status: passed for probe implementation.

- Pre-H200 read-only reviewer found no blocking findings.
- Final read-only reviewer found no blocking findings for the first gate.
- Probe remains isolated to task019 diagnostics and below PPO/update paths.
