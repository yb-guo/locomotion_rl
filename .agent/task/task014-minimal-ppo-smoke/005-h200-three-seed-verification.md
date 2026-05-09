# 005: H200 Three-Seed Verification

## Goal

Run the real task014 PPO smoke on H200 and record pass/fail evidence.

## Route

1. Copy task014 code to:
   `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`.
2. Run remote focused pytest through guarded command.
3. Run PPO smoke through guarded command:
   - seeds `0,1,2`;
   - `n_envs=1024`;
   - `rollout_steps=32`;
   - `ppo_updates=5`;
   - physical GPU `1`;
   - logical device `cuda:0`.
4. Record:
   - command;
   - output path;
   - summary metrics;
   - any warnings;
   - blocker if failed.

## Stop Rules

- If seed 0 fails, do not run seeds 1/2 until diagnosis or fix.
- If any seed has NaN/Inf, stop and record ranked hypotheses.
- If throughput `<10000 env_policy_steps_per_sec`, stop and diagnose CPU sync.
- If params do not change, stop and diagnose optimizer/graph/logprob.
- If output lands outside project workspace, stop and delete only if explicitly
  safe inside project workspace.

## Verification

- `metrics.jsonl` has 15 rows: 3 seeds x 5 updates.
- `summary.json` marks all seeds passed.
- `final_checkpoint.pt` exists.
- Task log includes key metrics and output paths.

## Log

- 2026-05-09 Copied task014 code and docs to:
  `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke`.
- H200 focused pytest command:
  `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke && PYTHONPATH=src python -m pytest tests/test_ppo_loop.py tests/test_g1_ppo_smoke.py -q -p no:cacheprovider'`
- H200 focused pytest result:
  `10 passed in 14.21s`.
- H200 PPO smoke command:
  `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke && mkdir -p outputs/task014 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 1200 python -m h200_locomotion_lab.tools.g1_ppo_smoke --n-envs 1024 --rollout-steps 32 --ppo-updates 5 --seeds 0,1,2 --backend cuda --physical-gpu 1 --logical-cuda-device cuda:0 --run-id h200-gpu1-three-seed-v2 | tee outputs/task014/minimal_ppo_smoke_h200_stdout_v2.txt'`
- H200 PPO smoke summary:
  - status `ok`;
  - all seeds passed;
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-three-seed-v2`;
  - min collect throughput:
    `12201.757460784085 env_policy_steps_per_sec`;
  - mean final reward mean: `1.4781025648117065`.
- Artifact check:
  - `config.json`;
  - `metrics.jsonl`;
  - `summary.json`;
  - `final_checkpoint.pt`;
  - `metrics.jsonl` row count: `15`.
- Warnings:
  - Genesis torch version warning;
  - Genesis MJCF floating base actuator parse warning;
  - Genesis G1 mass/COM geometry estimate warnings.
  Smoke criteria still passed.

## Review

Status: passed.

- No seed failed, so upstream stop rules did not block later seeds.
- No NaN/Inf, device, parameter-change, artifact, or throughput blocker was
  reported by the CLI.
