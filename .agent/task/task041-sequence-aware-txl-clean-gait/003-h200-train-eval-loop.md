# 003: H200 Train/Eval Loop

## Route

Run H200 sequence-aware true-TXL training and evaluate checkpoints until clean
0.4 m/s eval passes.

## Loop

1. Run a bounded smoke-size train to verify the train CLI works.
2. Eval the produced checkpoint through the Task041 eval wrapper.
3. If quality fails, continue training from the current route with a larger
   budget and evaluate the latest saved checkpoint.
4. Stop only when eval passes or evidence shows a different bottleneck that
   must be diagnosed before more training is useful.

## Acceptance

Task041 goal is eval pass:

```text
pipeline_pass=true
quality_gate_pass=true
pass=true
```

The loop is not complete if only train pipeline or update counters pass.

## Log

- 2026-05-30 Loop opened. No H200 train/eval evidence recorded yet.
- 2026-05-30 H200 smoke train/eval plumbing was exercised while closing
  Task041. The initial model-0 eval failed quality as expected but proved the
  eval wrapper now records active `txl_debug`.
- 2026-05-30 Started two scratch env4096 / 100-iter candidates on H200
  (`seed4100101` on cuda:0 and `seed4100102` on cuda:1). They used the older
  token-only 128-D true-TXL architecture and were stopped after the AdaptK160
  bridge route passed. These jobs are not promoted as Task041 evidence.
- 2026-05-30 Task041 clean eval passed through the AdaptK160 -> true-TXL
  warmstart bridge:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_eval/model_5467_task041_true_txl_bridge_vx0p4_eval_tolerance1e3.json`.
  Evidence: `pipeline_pass=true`, `task041_pipeline_pass=true`,
  `quality_gate_pass=true`, `pass=true`, `memory_debug_present=true`, and
  `memory_debug_active=true`. Final metrics: `fall_ratio=0.0`,
  `gravity_xy.max=0.095986507833004`, `root_z.min=0.7557933330535889`,
  `lin_vel_error.mean=0.14921148121356964`, and
  `yaw_vel_error.mean=0.09237571805715561`.
- 2026-05-30 H200 warmstart-bridge train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/warmstart_bridge_smoke_train_env8_iter1.json`.
  Evidence: `train_pipeline_pass=true`, `algorithm_class=Task040SequenceAwareTrueTxlPPO`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `checkpoint_exists=true`, `sequence_update_forward_batches=1`,
  `sequence_update_forward_samples=16`, and
  `stateless_fallback_forward_batches=0`.

## Review

Status: passed for the clean 0.4 m/s Task041 gate through the AdaptK160
warmstart bridge, with a separate H200 smoke proving the bridge checkpoint can
enter Task040 sequence-aware PPO update. This does not claim TXL superiority or
prove memory causality; that belongs in the next task.
