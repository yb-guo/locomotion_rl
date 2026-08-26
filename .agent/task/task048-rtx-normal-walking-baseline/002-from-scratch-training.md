# 002: From-Scratch Training

## Route

Run official `Unitree-G1-Flat` MLP/PPO from random initialization using the
environment count selected in 001, seed 42, TensorBoard logging, no W&B upload,
and checkpoint saves frequent enough for closed-loop gating.

## Log

- 2026-08-19 Started the from-scratch run at
  `.external/unitree_rl_mjlab/logs/rsl_rl/task048_rtx_g1_normal_walk/2026-08-19_10-24-56_env4096_seed42_iter650`.
  Configuration: 4096 envs, 24 steps/env, seed 42, 650 maximum iterations,
  checkpoint interval 50, TensorBoard logger, upload disabled.
- 2026-08-19 At iteration 150, training remained finite and sustained roughly
  52-59k steps/s. Mean episode length had risen from about 64 early in training
  to roughly 300-390 steps. Checkpoints `model_0`, `model_50`, `model_100`, and
  `model_150` were saved.
- 2026-08-19 Completed all 650 iterations (`0..649`) without OOM, NaN, or an
  interrupted process. The run collected `63,897,600` simulation transitions
  in `00:20:35`; the final iteration reported `54,853` steps/s, mean reward
  `18.04`, mean episode length `977.59`, and mean action standard deviation
  `0.50`.
- 2026-08-19 Saved periodic 5.1 MiB checkpoints through `model_600.pt`, the
  final 5.1 MiB `model_649.pt`, and an 858 KiB `policy.onnx`. The final model
  was chosen by out-of-training closed-loop metrics rather than training
  reward alone.
- 2026-08-19 Copied the selected checkpoint and exported ONNX policy into the
  stable handoff directory `outputs/task048/normal_walk/deliverable`, alongside
  the exact training configs and a manifest with SHA-256 hashes. Copy identity
  checks passed and `onnx.checker.check_model` accepted the ONNX graph.

## Review

Status: passed. The complete from-scratch budget ran successfully on the new
single RTX GPU and produced loadable final checkpoint and ONNX artifacts.
