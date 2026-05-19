# Subtask 014: Hybrid PPO Standing Causality

## Route

- Continue from subtask013 after switching PPO smoke to the best current hybrid
  asset.
- Run diagnostic PPO smoke only; no long training or checkpoint search.
- Compare hybrid and source-profile assets under the same conservative
  standing smoke.
- Re-run the task020-style 20-update standing gate on the hybrid asset.
- Test whether action allocation explains the collapse:
  - `all`;
  - `legs`;
  - `legs_no_ankle_roll`.
- Add no-update hybrid asset support to the existing causality probe and test
  whether zero action reproduces the same tilt-reset horizon.

## Log

- 2026-05-13 H200 hybrid conservative standing smoke:

```text
run_id=h200-gpu1-hybrid-standing-conservative-v1
asset_variant=task023_hybrid
command_mode=standing
termination_height_min=0.20
action_scale_mult=0.10
log_std_init=-2.5
ppo_updates=5
seeds=0,1,2

status=ok
all_seeds_passed=true
mean_reward_mean=2.033922
mean_final_episode_length_mean=21.276042
max_training_episode_length_mean=94.170593
mean_final_survival_rate=1.0
max_final_tilt_reset_rate=0.0
max_training_tilt_reset_rate=0.03125
min_collect_env_policy_steps_per_sec=13073.48
```

- 2026-05-13 H200 source-profile conservative standing smoke with the same
  PPO settings:

```text
run_id=h200-gpu1-profile-standing-conservative-v1
asset_variant=profile

status=ok
all_seeds_passed=true
mean_reward_mean=1.755684
mean_final_episode_length_mean=51.561198
max_training_episode_length_mean=71.634644
mean_final_survival_rate=1.0
max_final_tilt_reset_rate=0.0
max_training_tilt_reset_rate=0.03125
min_collect_env_policy_steps_per_sec=15598.91
```

Interpretation: hybrid improves reward/upright/height quality and raises the
best training episode-length window, but a 5-update final row is not a
long-horizon standing pass.

- 2026-05-13 H200 task020-style 20-update standing gate on the hybrid asset:

```text
run_id=h200-gpu1-hybrid-standing-gate-v1
asset_variant=task023_hybrid
command_mode=standing
termination_height_min=0.20
action_scale_mult=0.10
log_std_init=-2.0
base_height_reward_scale=0.20
joint_velocity_penalty_scale=0.001
termination_penalty=-1.0
ppo_updates=20
seeds=0,1,2

status=ok
all_seeds_passed=true
mean_final_episode_length_mean=36.447316
max_training_episode_length_mean=100.75
mean_final_survival_rate=0.96875
max_final_tilt_reset_rate=0.03125
max_training_tilt_reset_rate=0.03125
```

This improves over the old task020 source-profile gate's max training episode
length of about 71, but misses the task020 2x baseline gate of about 103.84
and ends with a final tilt-reset sweep.

- 2026-05-13 H200 conservative 20-update hybrid run:

```text
run_id=h200-gpu1-hybrid-standing-conservative20-v1
asset_variant=task023_hybrid
command_mode=standing
termination_height_min=0.20
action_scale_mult=0.10
log_std_init=-2.5
ppo_updates=20
seeds=0,1,2

status=ok
all_seeds_passed=true
mean_final_episode_length_mean=39.380992
max_training_episode_length_mean=99.794922
mean_final_survival_rate=0.96875
max_final_tilt_reset_rate=0.03125
max_training_tilt_reset_rate=0.03125
```

Lower exploration noise and removing height/termination reward shaping did not
remove the late tilt-reset sweep.

- 2026-05-13 Added `legs_no_ankle_roll` to `VectorizedGenesisBackend`
  action masks for diagnosis. It freezes left/right ankle-roll action targets
  while keeping the other leg joints active. Default action group remains
  `all`.
- 2026-05-13 Local focused verification after adding the mask:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider \
  tests/test_vectorized_genesis_backend.py \
  tests/test_g1_ppo_smoke.py
Result: 35 passed, 1 skipped
```

- 2026-05-13 H200 focused verification after syncing the mask:

```text
PYTHONPATH=src python3 -m pytest -p no:cacheprovider \
  tests/test_vectorized_genesis_backend.py \
  tests/test_g1_ppo_smoke.py
Result: 36 passed
```

- 2026-05-13 H200 seed0 action-mask probes on the hybrid asset:

```text
run_id=h200-gpu1-hybrid-legs-standing-conservative20-seed0-v1
action_joint_group=legs
max_training_episode_length_mean=99.553711
mean_final_episode_length_mean=39.610321
max_final_tilt_reset_rate=0.03125

run_id=h200-gpu1-hybrid-noankleroll-standing-conservative20-seed0-v1
action_joint_group=legs_no_ankle_roll
max_training_episode_length_mean=99.568359
mean_final_episode_length_mean=39.450867
max_final_tilt_reset_rate=0.03125
```

Interpretation: freezing upper-body/waist action and freezing ankle-roll action
do not improve the 20-update collapse. This argues against direct action
allocation as the primary blocker.

- 2026-05-13 Added `--asset-variant` support to
  `g1_no_update_ppo_causality`. Default stays `profile` to preserve the
  task018 contract; `task023_hybrid` generates the same hybrid asset under the
  no-update run directory and records `asset_resolution.json`.
- 2026-05-13 Local focused verification after no-update asset selector:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider \
  tests/test_g1_no_update_ppo_causality.py \
  tests/test_g1_ppo_smoke.py \
  tests/test_vectorized_genesis_backend.py
Result: 41 passed, 4 skipped
```

- 2026-05-13 H200 focused verification after syncing the no-update selector:

```text
PYTHONPATH=src python3 -m pytest -p no:cacheprovider \
  tests/test_g1_no_update_ppo_causality.py \
  tests/test_g1_ppo_smoke.py \
  tests/test_vectorized_genesis_backend.py
Result: 45 passed
```

- 2026-05-13 H200 hybrid zero-action causality:

```text
run_id=h200-gpu1-hybrid-zero-action-v1
asset_variant=task023_hybrid
mode=zero_action
chunks=20
chunk_steps=32
termination_height_min=0.20
root_z=1.20

status=ok
mode_passed=true
first_tilt_chunk=3
max_reset_count=1024
max_tilt_bad_count=1024
mean_reset_count=256.0
final_reset_count=1024
final_tilt_bad_count=1024
final_root_height_mean=0.824248
final_root_height_min=0.309184
final_upright_mean=0.864110
final_action_abs_mean=0.0
final_action_abs_max=0.0
```

Zero action on the hybrid asset reproduces the same class of finite-horizon
tilt collapse. Compared with the old task020 source-profile no-update result,
the first tilt chunk moves from 2 to 3 and mean reset count drops from 307.2
to 256.0, but the system still falls without PPO updates.

## Review

Status: diagnostic_not_passed.

The hybrid asset is a real improvement over the source-profile asset for
Genesis standing dynamics, but it is not a root fix. PPO smoke can pass
short-run plumbing, yet the 20-update gate still ends in a tilt-reset sweep.
The failure does not require PPO updates and does not disappear when action is
restricted to legs or legs without ankle-roll. The current blocker is therefore
still passive/contact/standing-dynamics semantics around the hybrid support
asset, not PPO plumbing, upper-body action, or direct ankle-roll action.
