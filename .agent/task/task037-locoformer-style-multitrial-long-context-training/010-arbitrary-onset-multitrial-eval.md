# 010 Arbitrary-Onset Multi-Trial Eval

## Route

Evaluate whether Task037 AdaptK160 can use retained history across inner trials
to adapt to arbitrary single-joint dynamic dead onset.

Fixed boundaries:

- Evaluation only; no training changes.
- Use the Task037 multi-trial final-trial gate.
- Do not clear actor history on inner reset.
- Do not expose joint id, failure label, trial index, or final-trial flag to the
  actor.
- Keep one latent condition fixed across all inner trials: speed, failed joint,
  onset time, recovery time.
- Use `model_5467.pt` as the first checkpoint because it is the clean AdaptK160
  prior and the current best unpromoted baseline for Task037 dynamic probes.

Eval matrix:

- Speeds: `0.4`, `1.2`, `2.0 m/s`.
- Joints: the 12 Task029 leg motor targets.
- Per-trial schedule: `0.0-0.5s normal`, `0.5-1.5s dead`, then recovery.
- Trial length: `2.0s`, so the `3.2s` AdaptK160 history can carry previous
  trial failure evidence into the next attempt.

Acceptance:

- Local CLI parse/help checks pass.
- H200 one-case smoke writes a JSON with `eval_mode=dynamic_single_onset`.
- H200 full matrix writes per-case JSON and an aggregate summary.
- Review must classify the result as diagnostic unless all 36 final-trial cases
  pass. Do not claim random dynamic-failure robustness from partial pass.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/arbitrary_onset_multitrial_model5467/`

## Log

- 2026-05-29 Opened after user asked to try multi-trial eval for random dynamic
  switching instead of single-attempt eval.
- 2026-05-29 Added `--dynamic-dead-joint`, `--dynamic-onset-s`, and
  `--dynamic-recovery-s` to `task037_multitrial_eval_checkpoint`.
- 2026-05-29 Local validation:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint --help`
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task037_mjlab_smoke_scripts.py tests/test_task037_multitrial_contract.py tests/test_agent_inventory.py`
  - result: `9 passed, 5 skipped`.
- 2026-05-29 H200 smoke passed for `right_knee_joint @ 2.0 m/s`, env8:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/arbitrary_onset_multitrial_model5467/smoke/task037_adaptk160_model5467_dynamic_single_right_knee_vx2p0_env8_smoke.json`.
  Result: `eval_mode=dynamic_single_onset`, `final_trial_pass=true`.
- 2026-05-29 H200 full single-seed matrix completed for `model_5467.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/arbitrary_onset_multitrial_model5467/full_grid_seed3700902/task037_adaptk160_model5467_arbitrary_onset_multitrial_summary.json`.
  Result: `35/36` final-trial pass. Speeds `0.4` and `1.2` were `12/12`;
  `2.0` was `11/12`, failing only `left_hip_pitch_joint`.
- 2026-05-29 H200 multi-seed repeat for the failed case completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/arbitrary_onset_multitrial_model5467/left_hip_pitch_vx2p0_s5/task037_adaptk160_model5467_left_hip_pitch_vx2p0_s5_summary.json`.
  Result: `0/5` pass for `left_hip_pitch_joint @ 2.0 m/s`. Final-trial
  velocity error improved versus trial 0 on every seed but stayed around
  `1.26-1.31 > 1.20`; some seeds also had posture/root-z outliers.

## Review

Status: completed diagnostic. Multi-trial final-trial evaluation is much
stronger than the earlier single-attempt arbitrary-onset diagnostic: AdaptK160
`model_5467.pt` reaches `35/36` on the single-seed grid and shows final-trial
improvement over trial 0 in many cases. It is still not a full random dynamic
failure pass because `left_hip_pitch_joint @ 2.0 m/s` is a stable blocker
(`0/5` seeds). Do not promote arbitrary-onset robustness as solved.
