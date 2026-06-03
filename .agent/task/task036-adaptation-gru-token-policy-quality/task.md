# Task 036: Adaptation Conditioning and Memory Policy Quality

## Route

Task035 showed that short StackMLP curriculum can protect either low/mid speed
or high speed, but did not produce one checkpoint that passes all protected
gates together. Task033 already wired GRU and token memory consumers, but only
to smoke/overhead depth.

This task executes the policy-quality comparison that is still missing:

1. add adaptation conditioning as a new policy consumer;
2. long-train and evaluate GRU K4;
3. long-train and evaluate token K4;
4. compare all candidates with the same gates and evidence format.

Fixed boundaries:

- Reuse `Task033HistoryVecEnvWrapper` and the shared actor-visible history
  stream.
- Keep env, reward, action `31D`, failure scheduler, and eval thresholds
  unchanged.
- Do not expose fault ids, masks, motor scales, or speed-bin ids to the actor.
- Debug JSON may record hidden failure state.
- Do not claim pass from train reward, construction smoke, or fast sweep alone.

Protected full gates:

- speeds `0.4`, `1.2`, `2.0 m/s`;
- canonical dynamic switch at each speed;
- forced persistent 12-joint dead-grid at each speed.

Compatibility gates:

- retain Task033 blocker subset over `0.4`, `1.6`, `2.0 m/s` when cheap;
- use full matrix only for candidates that improve the blocker subset.

Baseline references:

- scoped high-speed StackMLP checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5350.pt`;
- base MLP warmstart for non-stack policy consumers:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt`.

## Planned Slices

1. `001-contract-and-registration.md`
   - Define consumer contracts and register Task036 H200 tasks.

2. `002-adaptation-conditioning-consumer.md`
   - Implement `history -> adaptation encoder -> latent z -> actor`.

3. `003-smoke-and-overhead.md`
   - Run local tests, env64 train smoke, and env8192 overhead smoke.

4. `004-adaptation-train-eval.md`
   - Train adaptation consumer, checkpoint sweep, blocker/full eval.

5. `005-gru-token-longtrain-eval.md`
   - Complete GRU and token longtrain plus matching eval.

6. `006-comparison-decision.md`
   - Compare StackMLP, adaptation, GRU, token and decide promote/partial/reject.

## Minimal Closed Loop

1. Add and smoke the adaptation consumer locally and on H200.
2. Confirm GRU/token/adaptation all have matching train and eval task ids.
3. Run at least one bounded env8192 training pass for adaptation, GRU, and
   token.
4. Sweep non-final checkpoints for each consumer.
5. Run full `0.4/1.2/2.0` dynamic + 12-joint dead-grid matrix for any candidate
   that passes the blocker fast gate.
6. Write a summary JSON with checkpoint paths, eval paths, failed speeds/joints,
   and promotion decision.

## Log

- 2026-05-28 Opened after user requested both adaptation conditioning and the
  unfinished GRU/token policy-quality eval.
- 2026-05-28 Completed AdaptK4 implementation, 60-iteration warmstart train,
  GRU/token scratch plus continuation training to about 300 iterations, and
  full matrix eval for final checkpoints.

## Review

Status: evidence complete for the initial policy-consumer bakeoff. No Task036
policy checkpoint passed the protected full matrix. AdaptK4 is the best partial
candidate; GRU/token K4 are not promoted.
