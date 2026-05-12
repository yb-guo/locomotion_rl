# 005: Action Energy Ablation

## Goal

Find early standing action energy that can learn without triggering full reset
waves.

## Route

1. Keep reward/reset config fixed from previous subtasks.
2. Run small H200 matrix:
   - `action_scale_mult`: `0.10`, `0.20`, `0.25`, `0.35`;
   - `log_std_init`: `-2.0`, `-1.5`, `-1.0`.
3. Record:
   - reset rate;
   - episode length;
   - action saturation ratio;
   - `log_std_mean/min/max`;
   - KL and clip fraction;
   - actor/value param delta.
4. Choose smallest action energy that still changes policy params and improves
   survival.

## Acceptance

- Matrix is bounded.
- One candidate is selected with evidence, or env/contact blocker is declared.
- No yaw/vx is introduced.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Added
  `h200_locomotion_lab.tools.g1_action_energy_ablation`, a bounded standing
  PPO action-energy matrix driver around `g1_ppo_smoke.run_smoke`. Defaults
  match the task route: `action_scale_mults=0.10,0.20,0.25,0.35`,
  `log_std_inits=-2.0,-1.5,-1.0`, `seeds=0`.
- 2026-05-12 The driver fixes the subtask004 standing reward/reset config for
  every candidate: `base_height_reward_scale=0.20`,
  `joint_velocity_penalty_scale=0.001`, `termination_penalty=-1.0`,
  `termination_height_min=0.20`, `root_z=1.20`, and
  `command_mode=standing`. It does not introduce yaw/vx, assets, importers,
  MuJoCo, datasets, checkpoints, or downloads.
- 2026-05-12 Each candidate writes under the ablation parent run dir and the
  parent writes `config.json`, `candidates.jsonl`, and `summary.json`.
  Candidate summaries capture pass/fail status, blocker, run dir,
  throughput, reward, reset/height/tilt/timeout rates, episode length,
  action saturation, log std stats, KL, clip fraction, and actor/value
  parameter-change flags. Failed candidates are recorded and the matrix
  continues.
- 2026-05-12 Candidate selection sorts viable candidates by: passed, no final
  reset/wave, low action saturation, actor/value changes, higher reward and
  episode length, then smaller action scale and lower log std energy.
- 2026-05-12 Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_action_energy_ablation.py -q -p no:cacheprovider`
  -> 5 passed.
- 2026-05-12 Local smoke-tool compatibility verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_action_energy_ablation.py tests\test_g1_ppo_smoke.py -q -p no:cacheprovider`
  -> 16 passed, 1 skipped.
- 2026-05-12 Read-only review found two blocking issues before H200 execution:
  candidate selection did not explicitly require survival/no-reset evidence,
  and user-provided matrix values could escape the task005 bounded set. Fixed by
  requiring viable candidates to have actor/value changes, survival_rate 1.0,
  reset_rate 0.0, and no full-env reset wave; candidate sorting now ranks
  survival and episode length before reward/action energy. The tool also now
  rejects action/log-std values outside the task005 matrix.
- 2026-05-12 H200 execution found the matrix evidence invalid because
  `g1_action_energy_ablation` called `g1_ppo_smoke.run_smoke` repeatedly inside
  one Python process; Genesis cannot initialize more than once, so all real
  candidates after the first failed with `GenesisException:Genesis already
  initialized.` Fixed by making the default candidate runner launch
  `sys.executable -m h200_locomotion_lab.tools.g1_ppo_smoke` in a fresh
  subprocess per candidate, with equivalent CLI args, captured stdout/stderr,
  and summary loading from the candidate `summary.json`. `run_ablation` now
  accepts an injectable runner so tests can still use fake smoke summaries
  without spawning subprocesses.
- 2026-05-12 Local focused verification after subprocess isolation:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_action_energy_ablation.py -q -p no:cacheprovider`
  -> 11 passed.
- 2026-05-12 Reviewer P1 evidence-integrity fix: parent ablation summaries now
  stay `blocked` if any candidate has `status=failed` from runner/runtime or
  subprocess failure, even when another completed viable candidate is selected.
  Completed but non-viable candidates can still coexist with a selected
  candidate.

## Review

Status: Genesis reinitialization blocker fixed locally; H200 matrix evidence must
be rerun because previous multi-candidate process evidence was invalid.
