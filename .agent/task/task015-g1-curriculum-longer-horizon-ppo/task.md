# Task 015: G1 Curriculum Longer-Horizon PPO

## Goal

Turn task014's 5-update PPO smoke into a longer-horizon curriculum smoke for
the G1 27DoF no-hand Genesis training asset.

This task should prove the training runner can keep one policy alive across
ordered stages for longer than task014. It is not a walking-quality claim.

## Scope

- Branch: `codex/task015-g1-curriculum-longer-horizon-ppo`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo`.
- Start from merged task014 on `master`.
- Use `VectorizedGenesisBackend` and task014 PPO core.
- Add one independent curriculum runner under `h200_locomotion_lab.tools`.
- Keep the first curriculum simple and explicit:
  - `standing`;
  - `small_vx`;
  - `small_yaw`;
  - `small_vxyaw`.
- Use task014 stable defaults unless evidence says otherwise:
  - `root_z=1.20`;
  - `default_pose=tall_crouch`;
  - `termination_height_min=0.20`;
  - `action_scale_mult=0.10`;
  - `log_std_init=-2.5`;
  - `warmup_steps=1`.
- Dev H200 probe:
  - seed `0`;
  - `updates_per_stage=20`.
- Final H200 smoke target:
  - seeds `0,1,2`;
  - `updates_per_stage=50`.

## Non-Goals

- No walking-quality claim.
- No reward redesign beyond existing task014 env knobs.
- No LocoFormer integration.
- No SONIC integration.
- No ONNX export.
- No rendering/GIF/video.
- No domain randomization.
- No dataset/checkpoint/asset/upstream download.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
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
outputs/task015/g1_curriculum_ppo/<run_id>/
```

Expected run files:

- `config.json`
- `metrics.jsonl`
- `summary.json`
- `final_checkpoint.pt`

## Stop Rules

- A stage failing pass criteria stops later stages for that seed.
- `standing` failure stops all velocity stages.
- `small_vx` failure stops `small_yaw` and `small_vxyaw`.
- `small_yaw` failure stops `small_vxyaw`.
- Any NaN/Inf in obs/action/reward/value/logprob/loss/KL/entropy stops.
- Any tensor device mismatch stops.
- Any output path outside `/root/agent_workspace/project` stops.
- Do not mark passed without local tests, H200 focused tests, H200 run
  evidence, and read-only review.

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements runner.
- Read-only reviewer subagent reviews boundary, correctness, and evidence.
- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass.
- H200 dev probe runs `seed=0`, `updates_per_stage=20`.
- H200 final smoke runs `seeds=0,1,2`, `updates_per_stage=50`, unless dev
  evidence triggers a stop rule.
- For every completed seed/stage:
  - no NaN/Inf;
  - actor/value params change;
  - tensors stay on `cuda:0`;
  - collect throughput is at least `10000 env_policy_steps_per_sec`;
  - reset/tilt/height metrics are recorded per update.
- Final review records whether the curriculum smoke passed or where it blocked.

# Route

1. `001-task-and-curriculum-contract.md`
2. `002-curriculum-runner-and-artifacts.md`
3. `003-h200-dev-probe.md`
4. `004-h200-three-seed-final.md`
5. `005-review-and-decision.md`

# Log

- 2026-05-09 Created task015 branch/worktree from `master` commit `902644d`.
- 2026-05-09 Router created the task contract and subtask route before coding.
- 2026-05-09 Coding subagent implemented the curriculum runner and focused
  tests.
- 2026-05-09 Local full pytest passed:
  `196 passed, 6 skipped`.
- 2026-05-09 Read-only reviewer subagent found no blocking issues in the
  runner implementation. H200 focused/dev/final evidence still pending.
- 2026-05-09 H200 focused tests passed:
  `20 passed in 22.51s`.
- 2026-05-09 H200 dev probe passed:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo/outputs/task015/g1_curriculum_ppo/h200-gpu1-dev-seed0-updates20-v1`;
  - seed `0`, `updates_per_stage=20`;
  - `all_seeds_passed=true`;
  - min collect throughput `12567.562857868475`.
- 2026-05-09 H200 three-seed final passed:
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo/outputs/task015/g1_curriculum_ppo/h200-gpu1-final-seeds012-updates50-v1`;
  - seeds `0,1,2`, `updates_per_stage=50`;
  - 600 metric rows;
  - `all_seeds_passed=true`;
  - min collect throughput `10977.639073614948`;
  - artifacts: `config.json`, `metrics.jsonl`, `summary.json`,
    `final_checkpoint.pt`.
- Residual:
  - final rows still include recurring tilt resets
    (`reset_count=1024`, `tilt_bad_count=1024`,
    `termination_height_bad_count=0`);
  - task015 passed as a runner/curriculum smoke, not as a stable locomotion
    quality claim.
- 2026-05-09 Final read-only reviewer found no blocking issues.

# Review

Status: passed as longer-horizon curriculum smoke.

- Acceptance met:
  - local full pytest passed;
  - H200 focused tests passed;
  - H200 dev probe passed;
  - H200 three-seed final passed;
  - required artifacts were written;
  - runner preserved the stage order and stopped-on-failure semantics.
- Boundary review:
  - no `GenesisG1SceneBackend` change;
  - no downloads, render/GIF/video, SONIC, ONNX, LocoFormer, or
    `/mnt/workspace*` writes.
- Decision:
  - task015 proves the curriculum runner can execute longer-horizon staged PPO
    on H200;
  - the next task should reduce long-horizon tilt resets, not increase the
    curriculum length.
