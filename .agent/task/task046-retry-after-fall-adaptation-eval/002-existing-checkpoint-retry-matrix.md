# 002: Existing Checkpoint Retry Matrix

## Route

Before changing training, test whether the subtask 001 retry signal is stable.
Use the current best Task045 checkpoint and the existing Task037 multi-trial eval
CLI.

Matrix:

- checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/logs_long_survival_all_env2048_iter40_lr5e6_seed4520402/model_39.pt`
- task:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6`
- command: `vx=1.6 m/s`
- joints: `left_knee_joint`, `right_knee_joint`
- onset times: `0.2 s`, `0.5 s`, `1.0 s`
- seeds: three seeds per case
- envs: 256
- trials: 3, `2.0 s` each

Promotion remains final-trial only. This matrix must not be used to mark the old
continuous no-physical-reset gate as passed.

## Acceptance

- H200 JSONs exist for the matrix.
- A compact summary records pass rate, final-trial fall counts, velocity error,
  and whether final-trial falls are lower than trial0 falls.
- The review states whether retry is stable enough to justify a training
  contract change.

## Log

- 2026-06-02 Opened after subtask 001 showed one positive fixed-case retry
  result. No matrix evidence yet.
- 2026-06-02 H200 matrix completed on both GPUs, two eval jobs in parallel.
  Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/model39_knee_onset_multiseed/summary.json`.
  All 18 cases completed with return code 0.
- 2026-06-02 Matrix result:
  - `pass_count=18/18`, `fail_count=0/18`.
  - `improved_final_vs_trial0_count=11/18`.
  - `not_worse_final_vs_trial0_count=17/18`.
  - One case worsened slightly: `left_knee_joint`, onset `0.5`,
    seed `4620210`, fall counts `3 -> 3 -> 4`.
  - Worst final fall count: `left_knee_joint`, onset `0.2`,
    seed `4620200`, fall counts `9 -> 9 -> 9`, still under the current
    final-trial gate.
  - Right knee is not uniquely broken under this retry contract; it passed all
    9 cases. Its weakest group is onset `0.5`, average fall counts
    `7.67 -> 7.00 -> 5.00`.
- 2026-06-02 Aggregates by group:
  - left knee onset `0.2`: pass `3/3`, improved `1/3`, not worse `3/3`,
    fall mean `4.67 -> 4.67 -> 4.33`, final tail velocity error `0.284`.
  - left knee onset `0.5`: pass `3/3`, improved `1/3`, not worse `2/3`,
    fall mean `3.00 -> 3.33 -> 3.00`, final tail velocity error `0.283`.
  - left knee onset `1.0`: pass `3/3`, improved `2/3`, not worse `3/3`,
    fall mean `5.00 -> 6.67 -> 4.00`, final tail velocity error `0.287`.
  - right knee onset `0.2`: pass `3/3`, improved `1/3`, not worse `3/3`,
    fall mean `3.33 -> 3.67 -> 3.00`, final tail velocity error `0.283`.
  - right knee onset `0.5`: pass `3/3`, improved `3/3`, not worse `3/3`,
    fall mean `7.67 -> 7.00 -> 5.00`, final tail velocity error `0.285`.
  - right knee onset `1.0`: pass `3/3`, improved `3/3`, not worse `3/3`,
    fall mean `3.33 -> 2.67 -> 2.00`, final tail velocity error `0.281`.

## Review

Status: evidence complete for subtask 002.

The retry contract is stable enough to justify a training-contract experiment:
all tested knee/onset/seed cases pass the final-trial retry gate, and final
trial falls are not worse than trial0 in 17/18 cases.

Do not overclaim this result. The policy still has nonzero falls, and the full
final-trial velocity error remains around `0.61`; the tail-window velocity error
around `0.28` suggests the policy recovers later in the final trial rather than
being immediately stable after reset. This supports training a retry-weighted
final-trial/post-reset-tail objective, not declaring continuous deployment
stability solved.
