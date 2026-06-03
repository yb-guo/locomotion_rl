# 001 Fake Multi-Trial Contract

## Route

Build the contract before touching MJLab.

Use a fake vectorized env with scripted raw done events, reset counters, and
condition ids. The fake env must make failure obvious if the wrapper clears
history on inner trial reset or changes the latent condition.

Acceptance criteria:

- `trial_done=True` maps to runner-facing `done=False` until final trial.
- `episode_done=True` maps to runner-facing `done=True`.
- `trial_done = fall OR trial_timeout`.
- `episode_done = trial_done AND final_trial`.
- Independent vectorized envs maintain independent trial counters.
- Inner trial reset preserves history/memory.
- Outer episode reset clears history/memory.
- Command/failure/randomization condition id is unchanged across inner trials.
- Inner reset clears last action and appends post-reset obs with zero action.
- Actor/critic observations do not contain trial index or final-trial flag.
- Trial labels are emitted only through `extras`.

This subtask does not evaluate walking quality and does not require H200.

## Log

- 2026-05-29 Planned.
- 2026-05-29 Implemented `Task037MultiTrialVecEnvWrapper` with fake env
  contract tests.
- 2026-05-29 Added zero-action reset support to `Task033HistoryVecEnvWrapper`
  so inner trial reset appends post-reset obs with zero action while preserving
  history.
- 2026-05-29 Local validation:
  `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m
  h200_locomotion_lab.tools.inspect_agent` passed.
- 2026-05-29 Local validation:
  `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p
  no:cacheprovider tests/test_agent_inventory.py tests/test_task033_history_buffer.py
  tests/test_task037_multitrial_contract.py` -> `4 passed, 7 skipped`.
- 2026-05-29 H200 torch validation in temporary overlay:
  `/mnt/workspace/users/guoyubo/agent_workspace/task037_contract_tmp_20260529`;
  command `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python -m pytest
  -p no:cacheprovider tests/test_task037_multitrial_contract.py -q` ->
  `2 passed in 2.48s` after syncing the current local patch.

## Review

Status: passed for fake-env contract only.

Evidence:

- Runner-facing `done=False` for inner trials and `done=True` for final trial
  are covered by H200 torch pytest.
- `trial_done = fall OR trial_timeout`, independent vectorized trial counters,
  condition preservation across inner trials, outer condition resample, and
  extras-only trial labels are covered by H200 torch pytest.
- History preservation on inner reset, history clear on outer reset, and
  post-reset zero-action history frame are covered by H200 torch pytest.

Limit:

- No MJLab reset hook has been validated in this subtask. That remains 002/003.
