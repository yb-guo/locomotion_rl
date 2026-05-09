# Task 014: Minimal PPO Smoke

## Goal

Build the first PPO smoke loop on top of task013
`G1VelocityTrackingVectorizedEnv`.

This task proves the training loop, not walking quality:

- env -> rollout -> GAE -> PPO update -> diagnostics -> artifact;
- all real PPO verification on H200 physical GPU 1;
- no LocoFormer, SONIC, ONNX, planner, render, GIF, dataset, upstream repo, or
  downloaded checkpoint.

## Scope

- Branch: `codex/task014-minimal-ppo-smoke`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task014-minimal-ppo-smoke`.
- Use task013 27DoF no-hand G1 Genesis env.
- Add reusable local PPO core under `h200_locomotion_lab.training`.
- Add H200 smoke CLI under `h200_locomotion_lab.tools`.
- Policy:
  - separate MLP actor/value;
  - hidden layers `2x128`;
  - tanh Gaussian actor;
  - learned `log_std`;
  - obs dim `90`;
  - action dim `27`.
- PPO smoke:
  - seeds `0,1,2`;
  - `1024` envs;
  - `32` rollout steps;
  - `5` PPO updates per seed;
  - `epochs=2`;
  - `minibatch_size=8192`;
  - `lr=3e-4`;
  - `gamma=0.99`;
  - `gae_lambda=0.95`;
  - `clip=0.2`;
  - `value_coef=0.5`;
  - `entropy_coef=0.0`;
  - `max_grad_norm=1.0`.
- Keep `torch` import lazy enough that local full pytest works without local
  torch installed.

## Non-Goals

- No "train to walk" claim.
- No reward tuning.
- No curriculum.
- No LocoFormer integration.
- No SONIC integration.
- No ONNX export.
- No rendering/GIF/video.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No downloads of assets, datasets, checkpoints, or upstream repos.
- No change to `GenesisG1SceneBackend`.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default GPU:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

Run dir:

```text
outputs/task014/minimal_ppo_smoke/<run_id>/
```

Run dir files:

- `config.json`
- `metrics.jsonl`
- `summary.json`
- `final_checkpoint.pt`

## Diagnose Loop

Task014 must create a sharp pass/fail loop before tuning anything:

1. Local unit tests for math seams.
2. H200 focused pytest for torch/PPO seams.
3. H200 3-seed PPO smoke.
4. If any seed fails, stop and record ranked hypotheses before fixing:
   - tensor/device bug;
   - PPO math/log-prob bug;
   - env/reset/reward bug;
   - throughput/CPU sync bug.

Do not mark passed without local tests, H200 tests, H200 PPO smoke evidence,
and read-only review.

## Acceptance

- Local full pytest passes without local torch.
- Torch-dependent local tests use `pytest.importorskip("torch")`.
- H200 focused tests pass with torch + CUDA.
- H200 PPO smoke passes all 3 seeds.
- For every seed:
  - no NaN/Inf in obs/action/reward/value/logprob/loss/KL/entropy;
  - actor/value params change after updates;
  - tensors stay on `cuda:0`;
  - rollout collection throughput is at least
    `10000 env_policy_steps_per_sec`;
  - output files stay under `/root/agent_workspace/project/.../outputs/task014`.
- `metrics.jsonl` records per-update diagnostics:
  - seed/update/env_steps;
  - reward mean;
  - done/timeout/fallen count;
  - policy loss;
  - value loss;
  - entropy;
  - approx KL;
  - clip fraction;
  - grad norm;
  - collect throughput;
  - update throughput;
  - tensor device ok.
- Review finds no blocking boundary, correctness, or evidence issue.

# Route

1. `001-task-and-feedback-loop.md`
2. `002-ppo-core-and-math.md`
3. `003-policy-and-rollout-contract.md`
4. `004-h200-smoke-cli-and-artifacts.md`
5. `005-h200-three-seed-verification.md`
6. `006-review-and-decision.md`

# Log

- 2026-05-09 Created task014 branch/worktree from `master` commit `7617e96`.
- Created task014 closed-loop docs from task13 evidence and task14 planning
  decisions.

# Review

Status: pending.
