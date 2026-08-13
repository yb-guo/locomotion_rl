# 001 Historical Lineage

## Route

Recover the actual source checkpoints and commands from Task029, Task030,
Task036, Task037, and Task041 evidence.

## Log

- Base source: Task030 `model_5349.pt`, itself reached through a manually gated
  MLP curriculum rather than one deterministic command.
- AdaptK4 run: source `model_5349.pt`, task
  `Unitree-G1-Gripper-Flat-Task036-AdaptK4-Fast2p0`, 8192 envs, 60 updates,
  seed `3603630`, learning rate `3e-6`, entropy coefficient `3e-4`; output
  `model_5408.pt`.
- AdaptK160 run: source `model_5408.pt`, task
  `Unitree-G1-Gripper-Flat-Task037-AdaptK160-CleanUnified-Fast2p0`, 8192 envs,
  60 updates, seed `3700705`, learning rate `3e-6`, entropy coefficient
  `3e-4`; output `model_5467.pt`.
- `model_5467.pt` final-trial fall ratio was `0.0` at 0.4, 1.2, and 2.0 m/s.
- Task041 converted `model_5467.pt`; it did not learn that gait from scratch.
- The old files are absent from this machine, so exact historical-checkpoint
  mode is blocked on checkpoint availability. Fresh-prior mode remains
  executable without downloads.

## Review

Status: lineage established from repository evidence. No reproduction claim is
made from provenance alone.
