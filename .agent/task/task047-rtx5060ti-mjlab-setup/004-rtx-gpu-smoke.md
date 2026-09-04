# 004: RTX GPU Smoke

## Route

Verify in increasing scope:

1. project unit tests and `inspect_agent`;
2. finite PyTorch CUDA operation;
3. official Unitree G1 MJLab construction and short GPU stepping;
4. one minimal Task044/Task046 PPO update if registration is complete.

Use a conservative environment count for 16 GiB VRAM. Do not copy the old
H200 default of 8192 envs without a measured RTX throughput/VRAM probe.

## Log

- 2026-08-18 `python -m h200_locomotion_lab.tools.inspect_agent` passed.
- 2026-08-18 Initial full pytest exposed seven new-host portability failures:
  inaccessible `/root/...` default probe paths, a Windows absolute path parsed
  on POSIX, and optional newer true-TXL debug fields on legacy/fake runners.
  Added safe path probes, cross-platform Windows path recognition, and
  backwards-compatible debug defaults. Focused regression tests passed, then
  the complete suite passed: `713 passed, 35 warnings in 7.53s`.
- 2026-08-18 `scripts/run_rtx_mjlab_smoke.sh` passed end to end with the
  verified constraints. The CUDA preflight reported PyTorch `2.13.0+cu130`,
  capability `(12, 0)`, `sm_120`, and a finite matrix multiply.
- 2026-08-18 Official `Unitree-G1-Flat`, 32 envs, 8 rollout steps, and one PPO
  iteration completed on `cuda:0`: 256 samples, 574 steps/s, 0.45 s iteration.
  Evidence directory:
  `.external/unitree_rl_mjlab/logs/rsl_rl/task047_rtx5060ti_smoke/2026-08-18_23-27-28_official_g1_env32_iter1_script`.
  It contains a non-empty `model_0.pt`.
- 2026-08-18 Rebuilt 31-action `Unitree-G1-Gripper-Flat` also completed one
  PPO iteration with the same 32 x 8 budget: 256 samples, 576 steps/s, 0.44 s
  iteration. Actor/critic shapes were `104 -> 31` and `119 -> 1`. Evidence:
  `.external/unitree_rl_mjlab/logs/rsl_rl/task047_rtx5060ti_smoke/2026-08-18_23-26-15_g1_gripper_env32_iter1/model_0.pt`
  (`5,398,975` bytes).
- 2026-08-18 Task044/046 PPO was not attempted. Its prerequisite registration
  audit fails explicitly because the old Task029-031 external-source state is
  absent; running a later patcher on an incomplete base would not be valid
  evidence.

## Review

Status: passed for local tests, CUDA, official G1, and the Task028 31-action
base; not passed for Task044/046.

The workstation is ready for small RTX MJLab experiments with a conservative
32-environment default. The current hidden-fault/retry algorithm remains a
source/checkpoint migration problem rather than a GPU or dependency problem.
