# 002 Adaptation Conditioning Consumer

## Route

Add a new consumer that reuses the Task033 shared history stream:

```text
actor_history K4
        |
        +--> newest base obs
        |
        +--> adaptation encoder -> latent z
                          |
                          v
              [newest base obs, z] -> MLP actor
```

The adaptation encoder can see only actor-visible history frames:

- actor observation;
- previous action.

It cannot see debug fault ids, failure masks, motor scales, or active joint ids.

Warmstart policy:

- load base MLP `104D` actor checkpoints into the newest-base-obs path;
- initialize latent columns at zero;
- use a fresh optimizer after migration.

## Log

- 2026-05-28 Added local migration/helper tests and initial implementation in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`.
- 2026-05-28 Fixed warmstart migration so the adaptation actor keeps a
  `540D` history normalizer while migrating the base `104D` actor into the
  `[104D newest obs, 32D latent]` actor input.
- 2026-05-28 H200 env64 construction smoke completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/policy_train/036_adapt_smoke_env64_iter1_gpu0_seed3603602.stdout.log`.
  The log records `Task036AdaptationConditionedMlpModel`,
  `Linear(in_features=136, out_features=512)` for the actor MLP, and
  `Linear(in_features=540, out_features=128)` for the adaptation encoder.

## Review

Status: implementation smoke passed. Policy quality and env8192 overhead are
not established yet.
