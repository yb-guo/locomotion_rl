# Task 046: Retry-After-Fall Adaptation Eval

## Route

Task045 showed that the unchanged single-continuous left-knee gate is not
closed by local reward/curriculum repair. This task opens a separate eval
contract that matches the user's proposed question:

> If the robot falls after a hidden fault, can the policy keep its memory and
> improve on retry attempts?

This must not relax or replace the old continuous gate. It is a different
measurement: online adaptation after one or more failed attempts.

## Planned Slices

1. `001-existing-checkpoint-retry-eval.md`
   - Reuse the existing multi-trial eval path if it can represent retry after
     fall with memory preserved.
   - Run the current best Task045 checkpoint on H200.
   - Compare trial 0 / trial 1 / final trial metrics.

2. `002-existing-checkpoint-retry-matrix.md`
   - Reuse the existing multi-trial eval path across left/right knee, multiple
     onset times, and multiple seeds.
   - Decide whether the retry improvement from subtask 001 is stable or just a
     single-seed artifact.

3. `003-post-reset-recovery-shaping-train.md`
   - Add a default-off final-trial post-reset reward wrapper.
   - Warm-start from the current best checkpoint and train a short targeted
     recovery stage.
   - Evaluate whether final-trial early-window velocity and fall counts improve.

4. `004-dedicated-retry-contract-if-needed.md`
   - If the existing multi-trial eval cannot express a later desired retry
     contract, add a dedicated Task046 retry eval CLI.
   - Preserve fault identity as hidden actor information.

5. `005-policy-training-contract-handoff.md`
   - Add a default-off actor-visible retry context contract.
   - Keep fault identity hidden from actor observations.
   - Hand future quality work to an explicit policy consumer instead of more
     reward-only tuning.

6. `006-retry-context-consumer-smoke.md`
   - Run the first H200 consumer smoke with `task046_retry_context` enabled.
   - Do not direct strict-resume from Stage2 unless a shape-aware migration is
     added.
   - Record pipeline/debug evidence separately from quality eval claims.

## Acceptance Criteria

- The old continuous gate remains unchanged and is not marked passed by retry
  eval.
- H200 JSON exists for the current best checkpoint under retry-after-fall eval.
- The JSON shows per-trial fall/reset/completion metrics and final-trial gate.
- The task records whether improvement across retries is observed.
- Post-reset recovery training changes must be default-off and must not alter
  old Task044/Task045 behavior unless the train cfg explicitly enables them.
- If a new CLI is added, local pytest and `inspect_agent` pass.

## Log

- 2026-06-02 Opened after deciding to test the user's proposal: eval may allow
  fall/retry with memory retained, because Task044/Task045 training already has
  a multi-trial structure while the old continuous eval requires zero physical
  resets.
- 2026-06-02 Added Task037 eval CLI compatibility for the current best
  Task045 checkpoint:
  - optional local `IPython.display` / `wandb` / `wcwidth` stubs, so H200 slim
    env can run the eval path;
  - optional true-TXL actor cfg overrides for `memory_latent_dim`,
    `base_obs_passthrough`, `adaptation_warmstart`, `action_dim`, and scale
    fields.
- 2026-06-02 H200 retry eval completed for
  `model_39.pt` at hidden `left_knee_joint` dead, onset `0.5 s`, command
  `vx=1.6 m/s`, 256 envs, 3 trials, 2.0 s each. JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_eval/current_best_pose_tight_left_knee_onset0p5_retry_seed4620103.json`.
  Result: `pass=true`, `final_trial_pass=true`. Fall counts improved from
  trial0 `4` to trial1 `3` to final trial `2`.
- 2026-06-02 Replaced the old provisional subtask 002 with an existing-CLI
  retry matrix, because subtask 001 showed the current Task037 eval can already
  represent the first retry-after-fall contract.
- 2026-06-02 H200 retry matrix completed for current `model_39.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/model39_knee_onset_multiseed/summary.json`.
  Result: `18/18` pass under the retry final-trial gate. Final falls improved
  over trial0 in `11/18` cases and were not worse in `17/18` cases. This is
  evidence for trying a retry-weighted training contract, not evidence that the
  old continuous no-reset gate has passed.
- 2026-06-02 Opened subtask 003 for a default-off post-reset recovery reward
  wrapper. The wrapper only affects training when `train_cfg` enables
  `task046_post_reset_recovery_reward`.
- 2026-06-02 Completed subtask 003 H200 post-reset recovery training. Stage1
  improved the baseline-seed retry matrix modestly; Stage2 used a stronger
  early recovery weight and is the current best retry checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/stage2_early_env1024_rollout220_iter20_lr5e6_seed4620502/logs/model_19.pt`.
  Stage2 matrix summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage2_early_model19_knee_onset_multiseed_baseline_seeds_v3/summary_corrected.json`.
  Result: `18/18` pass, final falls mean `2.83`, full final velocity error
  `0.5742`, early-window velocity error `0.9033`, tail-window velocity error
  `0.2430`. Compared with current `model_39.pt`, all `18/18` cases improved
  early and tail velocity error; mean deltas were final falls `-0.72`, full
  velocity error `-0.0364`, early velocity error `-0.0315`, tail velocity
  error `-0.0410`.
- 2026-06-02 Opened subtask 005 and implemented the first default-off retry
  context contract. This appends reset/retry phase features to actor
  observations only when `task046_retry_context.enabled=true`; it does not
  expose fault identity and does not claim a new quality checkpoint.
- 2026-06-02 Opened subtask 006 for the first H200 retry-context consumer
  smoke. This is pipeline evidence only; enabling retry context changes actor
  observation shape, so direct strict resume from the Stage2 checkpoint is not
  accepted without an explicit migration path.

## Review

Status: open; subtasks 001, 002, and 003 have H200 eval/train evidence.
Subtask 005 has local policy-contract evidence. Subtask 006 has H200 consumer
smoke pipeline evidence.

This task is an eval-contract experiment. It is not a claim that the old
continuous deployment-style gate has passed.

The current best retry-after-fall checkpoint is Stage2 subtask 003. It improves
reset-after-fall recovery under the baseline-seed retry matrix, but the first
post-reset second is still slow. More reward-only tuning is now lower leverage;
the next closed unit should be the policy/training-contract handoff from
subtask 005.

Subtask 005 adds a default-off actor-visible retry context contract and passes
local pytest plus `inspect_agent`. It does not claim a new quality checkpoint;
because it changes actor observation shape, future H200 training should start a
new consumer run or use explicit checkpoint migration.

Subtask 006 proves the Task044 hidden-fault train path can consume the retry
context on H200 for a one-iteration smoke. JSON:
`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_context_consumer_smoke/smoke_train_summary_fixed.json`.
Result: `train_pipeline_pass=true`, checkpoint exists, retry-context debug
`feature_dim=6`. This remains pipeline evidence only; no eval quality or
recovery improvement is claimed.
