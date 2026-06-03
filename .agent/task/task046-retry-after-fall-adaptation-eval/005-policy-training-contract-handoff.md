# 005: Policy Training Contract Handoff

## Route

Move Task046 from reward-only post-reset tuning to an explicit retry policy
contract.

Stage2 from subtask 003 is the current retry baseline:

- checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/stage2_early_env1024_rollout220_iter20_lr5e6_seed4620502/logs/model_19.pt`
- matrix:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage2_early_model19_knee_onset_multiseed_baseline_seeds_v3/summary_corrected.json`

The next policy consumer needs actor-visible retry context, but must not expose
fault identity. Minimal actor-visible context:

- normalized trial index;
- final-trial flag;
- normalized time since current trial reset;
- one-step inner-reset marker;
- previous inner reset was fall;
- previous inner reset was timeout.

Implementation contract:

- Add a default-off `task046_retry_context` train config.
- Append retry context to actor observations before `Task033HistoryVecEnvWrapper`
  so it enters the existing history/true-TXL path.
- Keep `task044_fault_label` as non-actor privileged/debug observation only.
- Do not change old behavior unless the flag is enabled.
- Do not mark any quality improvement without H200 eval evidence.

## Acceptance

- Fake-env tests prove the actor-visible retry features are correct around
  reset/next-step boundaries.
- Actor retry feature names pass the no-fault-leakage field-name check.
- CLI/config mutation tests cover the new default-off switch.
- Local pytest passes for the touched training/wrapper tests.
- `inspect_agent` still passes.
- Review explicitly states that this is a contract handoff only, not a new
  quality checkpoint.

## Log

- 2026-06-02 Opened after Stage2 improved retry recovery but left the first
  post-reset second slow.
- 2026-06-02 Implemented `Task046RetryContextVecEnvWrapper` as a default-off
  actor-observation augmentation. It appends six retry context features and
  preserves hidden fault identity outside the actor group.
- 2026-06-02 Added `--task046-retry-context` CLI/config plumbing to the
  sequence-aware train path, with debug snapshots in train summaries.
- 2026-06-02 Local validation:
  - `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task044_hidden_fault_target.py --tb=short --basetemp .test_tmp_task046_policy_contract_local`
    passed: `20 passed, 8 skipped in 0.53s`.
  - `PYTHONPATH=src python -m h200_locomotion_lab.tools.inspect_agent`
    passed.
  - `PYTHONPATH=src python -m h200_locomotion_lab.tools.task041_sequence_txl_clean_train --help`
    exposes the `--task046-retry-context` flags.

## Review

Status: contract closed with local evidence.

The default behavior remains unchanged unless `task046_retry_context.enabled`
is set. The new actor features describe retry phase and reset reason, not fault
identity, and the feature names pass the existing no-fault-leakage guard.

No new H200 quality checkpoint is claimed here. Enabling this contract changes
the actor observation dimension, so the next training unit should start a new
consumer run or use an explicit checkpoint migration path rather than treating
the Stage2 checkpoint as directly strict-loadable.
