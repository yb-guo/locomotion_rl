# 002 Local Registration And Launch

## Route

Create clean speed-bin tasks directly from the currently installed G1 gripper
environment, without depending on the unavailable Task030/Task031 external
patch chain. Keep separate training and deterministic multi-trial eval task
ids where the runner contract differs.

## Log

- Added `task048_register_previous_gait_reproduction.py` for the current local
  Unitree MJLab tree.
- Added `task048_launch_reproduction_stage.sh` with `mlp-prior`,
  `mlp-speed-bins`, `adaptk4`, and `adaptk160` stages.
- The `historical` profile retains 8192 envs and 60 updates for each migration
  stage. The default `4090` profile matches transition count with smaller env
  batches and more updates.
- Warmstart stages require an explicit `SOURCE_CHECKPOINT`; no implicit remote
  path or download exists.
- A 4096-env clean-bin scratch ablation was stopped at iteration 330 after
  `model_300.pt`: it learned stable standing but fixed-command eval showed near
  zero forward velocity. `mlp-prior` now uses the official command curriculum;
  clean bins are a separate continuation stage.
- Registry inspection on the installed tree proved Task048 train configs use
  `episode_length_s=20.0`, while their separately registered play configs use
  `1e9`. Clean-bin configs expose only reset events, disable actor corruption,
  and have no curriculum.
- Added a dedicated `Unitree-G1-Gripper-Flat-Task048-TrueTxl-CleanBins-Eval`
  task. The first True-TXL matrix had accidentally used the randomized Task038
  train task and was discarded; regression coverage now rejects that mapping.
- Focused registration/launcher/matrix tests pass (`5 passed`), both shell
  launchers pass `bash -n`, and all accepted runtime stages constructed and ran
  on `cuda:0` on the local RTX 4090.

## Review

Status: passed. Registration is idempotent, train/play configs are isolated,
the strict clean True-TXL eval task is registered, and the staged launchers
completed the requested local CUDA route without downloads.
