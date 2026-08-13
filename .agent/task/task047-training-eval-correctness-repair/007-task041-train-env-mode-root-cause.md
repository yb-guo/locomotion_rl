# 007 Task041 Train Environment Mode Root Cause

## Route

Trace the local Task041 runs from the training entry point through the MJLab
registry, compare the registered train and play configs, and reconcile that
with TensorBoard termination and episode-length evidence. Repair the loader
before any further long training.

## Log

- `task041_sequence_txl_clean_train.py` imported `_load_env_cfg` from the
  Task038 runner smoke helper.
- That helper is an eval helper and hard-coded `load_env_cfg(task, play=True)`.
- The registered train config has `episode_length_s=20`, actor corruption
  enabled, a `command_vel` curriculum, and `push_robot`. The play config has
  `episode_length_s=1e9`, actor corruption disabled, no curriculum, and no
  push event.
- TensorBoard's approximately `-4e-9` normalized fall termination value is
  exactly consistent with a `-4` termination penalty divided by the
  `1e9 s` play horizon. This independently confirms the wrong config mode.
- The 500-update continuation deteriorated between iterations 300 and 330:
  mean episode length fell from about `153.7` to `93.6`, then to `12.9` by
  iteration 499. Reward improved while behavior collapsed, so PPO selected
  three fast fall/reset trials rather than locomotion.
- `--resume-checkpoint` loaded actor and critic only. Optimizer and iteration
  state were intentionally not restored, so that job was a warmstart, not an
  exact continuation.
- Added a dedicated `_load_train_env_cfg` call using `play=False`, recorded
  `env_cfg_mode` and `env_episode_length_s`, and made the pipeline gate reject
  non-train mode or a play-like horizon.
- Focused unit verification: `tests/test_task041_sequence_txl_clean_train.py`
  passed with `22 passed`.

## Review

Status: code root cause repaired and unit-tested. A fresh local simulator smoke
must still show `env_cfg_mode=train` and `env_episode_length_s=20` before this
subtask is passed. Do not continue from Task047 `model_499.pt`.
