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
7. `007-efficiency-diagnosis-and-optimization.md`

# Log

- 2026-05-09 Created task014 branch/worktree from `master` commit `7617e96`.
- Created task014 closed-loop docs from task13 evidence and task14 planning
  decisions.
- 2026-05-09 Implemented minimal PPO smoke loop:
  - reusable PPO core in `src/h200_locomotion_lab/training/ppo_loop.py`;
  - H200 CLI in `src/h200_locomotion_lab/tools/g1_ppo_smoke.py`;
  - focused tests in `tests/test_ppo_loop.py` and
    `tests/test_g1_ppo_smoke.py`.
- Batched rollout finite checks and terminal counters so the smoke loop avoids
  per-step diagnostic `.item()` GPU syncs.
- Local verification:
  - `PYTHONPATH=src python -m pytest tests/test_ppo_loop.py tests/test_g1_ppo_smoke.py -q -p no:cacheprovider`
    passed: `6 passed, 4 skipped`;
  - `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`
    passed: `178 passed, 4 skipped`.
- H200 focused verification:
  - guarded command under
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`;
  - result after final PPO core revision: `10 passed in 14.21s`.
- H200 PPO smoke:
  - command used `CUDA_VISIBLE_DEVICES=1`;
  - recorded `physical_gpu=1` and `logical_cuda_device=cuda:0`;
  - seeds `0,1,2`;
  - `n_envs=1024`, `rollout_steps=32`, `ppo_updates=5`;
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-three-seed-v2`;
  - output files: `config.json`, `metrics.jsonl`, `summary.json`,
    `final_checkpoint.pt`;
  - `metrics.jsonl`: 15 rows;
  - `summary.json`: `all_seeds_passed=true`;
  - min collect throughput:
    `12201.757460784085 env_policy_steps_per_sec`.
- Per-seed final evidence:
  - seed 0 passed, actor/value changed, final reward mean
    `1.4817044734954834`, min collect throughput
    `12246.350717997004`;
  - seed 1 passed, actor/value changed, final reward mean
    `1.4735194444656372`, min collect throughput
    `17392.21648739249`;
  - seed 2 passed, actor/value changed, final reward mean
    `1.479083776473999`, min collect throughput
    `12201.757460784085`.
- Warnings observed:
  - Genesis warned torch `<2.8.0` is unsupported;
  - Genesis/MJCF warned on `floating_base_joint` actuator parse;
  - Genesis warned on expected G1 link mass/COM geometry estimates.
  None blocked task014 smoke acceptance.
- 2026-05-09 Added subtask 007 for efficiency diagnosis and trainer-local
  optimization.
- Efficiency hypotheses tested:
  - CUDA timing needed synchronization to report real collect/update time;
  - rollout finite checks were better batched after rollout stack;
  - PPO metric scalar conversion was better batched after minibatches;
  - fall/reset frequency remains visible but was not changed in task014.
- Optimized PPO smoke evidence:
  - H200 focused tests: `10 passed in 4.73s`;
  - H200 smoke run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-three-seed-v3`;
  - `metrics.jsonl`: 15 rows, now including `collect_time_s` and
    `update_time_s`;
  - `summary.json`: `all_seeds_passed=true`;
  - synchronized min collect throughput:
    `19381.815355781637 env_policy_steps_per_sec`;
  - observed total smoke command wall time improved from about `73.3s` in v2
    to about `53.6s` in v3.
- Optimized per-seed final evidence:
  - seed 0 passed, final collect throughput `29075.040446951498`,
    collect time `1.1270147692412138s`, update time
    `0.043640587478876114s`;
  - seed 1 passed, final collect throughput `19381.815355781637`,
    collect time `1.6906569069251418s`, update time
    `0.044599851593375206s`;
  - seed 2 passed, final collect throughput `20574.187580320217`,
    collect time `1.592675281688571s`, update time
    `0.04406619444489479s`.

# Review

Status: passed.

- Acceptance met:
  - local full pytest passes without hard torch import;
  - torch-dependent tests use `pytest.importorskip("torch")`;
  - H200 focused tests pass with torch/CUDA;
  - H200 PPO smoke passes all 3 seeds;
  - tensors report `cuda:0`;
  - actor/value params changed for every seed;
  - all collect throughput values exceeded `10000`;
  - artifacts stayed under `/root/agent_workspace/project/.../outputs/task014`.
- Boundary review:
  - no `GenesisG1SceneBackend` changes;
  - no LocoFormer, SONIC, ONNX, planner, render/GIF path used by smoke;
  - no download path added;
  - no write/delete under `/mnt/workspace` or `/mnt/workspace1`.
- Efficiency review:
  - trainer-local synchronization overhead was reduced without relaxing
    task014 finite/value/device checks;
  - v3 metrics use synchronized CUDA timing and still exceed the 10k collect
    throughput threshold;
  - remaining bottleneck is collect/env/reset side, not PPO update.
