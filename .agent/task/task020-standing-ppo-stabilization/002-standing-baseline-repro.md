# 002: Standing Baseline Repro

## Goal

Run the current PPO stack in standing mode without tuning, to establish the
baseline failure or baseline stability.

## Route

1. Use current `g1_ppo_smoke`/PPO stack.
2. Run `command_mode=standing`.
3. Use conservative starting knobs:
   - `action_scale_mult=0.25`;
   - `root_z=1.20`;
   - `termination_height_min=0.20`;
   - `default_pose=tall_crouch`.
4. Record reset cause, episode length, reward, KL, entropy, grad norm, value
   loss, throughput, and device report.
5. Stop if NaN/device/throughput fails.

## Acceptance

- H200 baseline run exists.
- Baseline summary records standing-only config.
- Baseline identifies whether failure is PPO plumbing, reset semantics, reward,
  action energy, or env/contact.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Local focused tests before H200 run:
  `PYTHONPATH=src python -m pytest tests/test_g1_ppo_smoke.py
  tests/test_ppo_loop.py -q -p no:cacheprovider` -> 9 passed, 4 skipped in
  0.37s.
- 2026-05-12 H200 focused tests through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_ppo_smoke.py
  tests/test_ppo_loop.py -q -p no:cacheprovider` -> 13 passed in 2.75s.
- 2026-05-12 H200 baseline run through guarded command with
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`, `logical_cuda_device=cuda:0`:
  `python -m h200_locomotion_lab.tools.g1_ppo_smoke --output-root
  outputs/task020/standing_ppo_stabilization --run-id
  h200-gpu1-standing-baseline-v1 --command-mode standing
  --action-scale-mult 0.25 --root-z 1.20 --termination-height-min 0.20`.
  Result: status ok, 3/3 seeds passed, min collect throughput
  35053.92 env-policy steps/s, mean final reward 1.63396, final reset_count 0
  for seeds 0/1/2, final termination_height_bad_count 0, final tilt_bad_count 0,
  final root_height_mean about 0.779, final upright_mean about 0.983.
- 2026-05-12 Interpretation: baseline PPO plumbing is not the current blocker
  for 5-update standing smoke. This does not pass task020 yet because the task
  contract still needs episode-length/survival metrics, reset-rate hardening,
  deterministic standing eval, and review evidence.

## Review

Status: baseline evidence recorded; read-only review pending after metrics hardening.
