# 004 Checkpoint Sweep

## Route

Evaluate non-final checkpoints from the curriculum run.

Fast gates:

- `2.0 m/s` right-knee forced dead;
- `2.0 m/s` full forced dead-grid;
- `2.0 m/s` canonical dynamic switch.

Escalate only the best candidates to representative multi-speed eval.

Selection rule:

- choose by eval score;
- reject checkpoints that regress clean or dynamic-switch gates;
- never accept a checkpoint only because it is the final iteration.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Added sweep artifact:
  `task035_sweep_checkpoints.sh`.
- 2026-05-28 Fast-gate sweep evaluated `model_5352.pt`, `model_5360.pt`, and
  `model_5369.pt` from the Task035 mixed run. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/checkpoint_sweep_mixed_seed3503601/task035_checkpoint_sweep_fast_gate_summary.json`.
  All three passed `2.0 m/s` dynamic switch and `right_knee_joint` forced-dead.

## Review

Status: passed for fast-gate sweep. `model_5369.pt` was escalated to
representative full validation.
