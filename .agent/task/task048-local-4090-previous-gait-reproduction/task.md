# Task 048: Local 4090 Previous Gait Reproduction

## Route

Reproduce the previously observed clean walking result from its real policy
lineage rather than treating Task041 True-TXL as a scratch locomotion learner.

The proven historical lineage was:

1. MLP curriculum checkpoint `model_5349.pt` from Task029/Task030.
2. AdaptK4 warmstart, 8192 envs for 60 updates, producing `model_5408.pt`.
3. AdaptK160 clean warmstart, 8192 envs for 60 updates, producing
   `model_5467.pt`.
4. A weight bridge from `model_5467.pt` into Task041 True-TXL. The bridge
   copied the normalizer, MLP path, critic, and adaptation encoder while the
   TXL projection and attention parameters remained freshly initialized.

Therefore the first reproduction gate is an independently walking MLP or
AdaptK checkpoint. True-TXL scratch training is not an allowed substitute.

## Reproduction Modes

Historical-checkpoint mode:

- Preferred source is `model_5467.pt`; bridge it directly and evaluate.
- If only `model_5349.pt` exists, replay AdaptK4 and AdaptK160 first.
- This machine currently has none of `model_5349.pt`, `model_5408.pt`, or
  `model_5467.pt`. No checkpoint may be downloaded without user approval.

Fresh-prior mode:

- Train an MLP with the official command curriculum first, then continue it on
  clean `0.4`, `1.2`, and `2.0 m/s` bins. Require a zero-fall clean matrix
  before migration.
- Use the same migration architecture and transition budgets as the historical
  AdaptK stages.
- This reconstructs the gait route, but it is not bitwise or seed-identical to
  the manually selected Task029/Task030 curriculum lineage.

## Hardware Budget

- Historical AdaptK stage: `8192 * 24 * 60 = 11,796,480` transitions.
- RTX 4090 equivalents: 4096 envs for 120 updates, 2048 for 240, 1024 for 480,
  or 512 for 960.
- Fresh MLP reference budget: about 98.3M transitions, matching the historical
  8192-env, 500-update order of magnitude. The default local profile uses 4096
  envs for 1000 updates.
- Keep PPO at 5 learning epochs and 4 minibatches. The failed Task047 long run
  used 2 epochs and 1 minibatch and is not the reproduction profile.

## Planned Slices

1. `001-historical-lineage.md`
   - Freeze checkpoint provenance, seeds, learning rate, entropy, and the
     distinction between gait inheritance and TXL training.

2. `002-local-registration-and-launch.md`
   - Register self-contained clean MLP, AdaptK4, and AdaptK160 tasks against the
     currently installed Unitree MJLab tree.
   - Provide dry-runnable historical and transition-matched 4090 launch paths.

3. `003-clean-matrix-and-bridge.md`
   - Require zero falls at all three speeds before advancing a checkpoint.
   - Build and evaluate the Task041 bridge only after AdaptK160 passes.

## Acceptance Criteria

- Registry inspection proves train configs use `episode_length_s=20` and play
  configs are isolated.
- Each warmstart records its source checkpoint and migration report.
- MLP, AdaptK4, and AdaptK160 checkpoints are promoted only from saved JSON
  evaluation at `0.4`, `1.2`, and `2.0 m/s` with final-trial fall ratio zero.
- The Task041 bridge passes its clean evaluation before any optional TXL PPO
  update.
- No result from Task047 `model_499.pt` is used as a gait prior.
- No dataset, checkpoint, robot asset, or upstream repository is downloaded.

## Log

- 2026-08-11 Opened after Task047 root-cause analysis showed the local job used
  the play environment and that the previous walking policy came from an
  AdaptK160 bridge rather than True-TXL scratch training.
- 2026-08-11 Confirmed the three historical checkpoint files and old
  `/mnt/workspace` paths are absent locally.
- 2026-08-11 Added a current-tree registration patch, a stage launcher, and a
  clean-matrix evaluator. Runtime verification is pending.
- 2026-08-11 Stopped the first clean-bin-from-scratch ablation after
  `model_300.pt`. It reached full 1000-step episodes with zero falls but stayed
  near zero forward velocity at every fixed-speed eval. The route now starts
  from the official command curriculum and treats clean bins as a continuation
  stage, matching the historical curriculum-first structure.
- 2026-08-11 Hardware used for all accepted runtime evidence: local NVIDIA
  GeForce RTX 4090, 48 GB VRAM, CUDA device `cuda:0`. The installed driver is
  12.2, so MuJoCo Warp disabled CUDA graphs and printed its `< 12.4` warning;
  eager CUDA execution remained stable. No checkpoint, asset, dataset, or
  upstream repository was downloaded.
- 2026-08-11 Registry inspection confirmed every Task048 train config has
  `episode_length_s=20.0`; its separate play config has
  `episode_length_s=1e9`. The clean-bin train/eval configs have only
  `reset_base/reset_robot_joints`, actor corruption disabled, and no
  curriculum. This prevents the Task047 play-config failure from recurring.
- 2026-08-11 Official-curriculum MLP `model_999.pt` passed 0.4 and 1.2 m/s but
  failed 2.0 m/s (`1.236 m/s` error). Clean-bin continuation produced the
  accepted MLP gait prior `model_1200.pt`; its three final-trial errors were
  `0.149/0.412/0.876 m/s`, all with fall ratio `0.0`.
- 2026-08-11 Replayed AdaptK4 with 4096 envs for 120 updates, seed `3603630`,
  learning rate `3e-6`, and entropy `3e-4`. The accepted `model_1319.pt`
  passed all speeds with errors `0.155/0.383/0.839 m/s` and zero falls.
- 2026-08-11 Replayed AdaptK160 with 2048 envs for 240 updates, seed `3700705`,
  learning rate `3e-6`, and entropy `3e-4`. Its `11,796,480` transitions match
  the historical 8192-env, 60-update budget exactly. The accepted
  `model_1558.pt` passed with errors `0.135/0.374/0.847 m/s` and zero falls.
- 2026-08-11 Bridged `model_1558.pt` into True-TXL. The migration copied all
  17 compatible actor keys and all 12 critic keys; the 17 TXL-specific keys
  remained freshly initialized. The strict Task048 clean True-TXL matrix used
  the dedicated clean config and passed with errors `0.135/0.374/0.847 m/s`
  and zero falls. An earlier matrix against the Task038 randomized train env
  was discarded and is not acceptance evidence.
- 2026-08-11 Task047 `model_499.pt` was not used. The latest Task041 two-update
  smoke still exceeds strict replay parity (`1.296997e-4 > 1e-5`), so Task048
  stops at the verified bridge and makes no True-TXL PPO-training correctness
  or quality claim.
- 2026-08-11 Final verification: Task048 focused tests passed (`5 passed`),
  Python compilation, both shell syntax checks, `inspect_agent`, root and
  nested-repository `git diff --check`, and a machine assertion over all four
  matrix JSONs plus the bridge JSON passed. Full repository pytest completed
  with `723 passed, 3 failed`; the three failures are pre-existing unrelated
  Genesis/Task038 local-asset path expectations in
  `test_g1_genesis_alignment_bundle.py` and
  `test_task038_mjlab_variant_env_load.py`, not Task048 regressions.

## Review

Status: passed for fresh-prior reproduction on the local RTX 4090. MLP,
AdaptK4, AdaptK160, and the True-TXL bridge each have saved three-speed,
zero-fall clean-matrix evidence. This reproduces the previous gait lineage and
transition budgets, but is not bitwise or seed-identical to historical
`model_5349.pt -> model_5408.pt -> model_5467.pt`, whose files remain absent.
Long True-TXL PPO continuation remains outside this pass because its separate
Task047 strict replay-parity gate is still failing.
