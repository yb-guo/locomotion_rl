# Task 041: Sequence-Aware TXL Clean-Gait Convergence

## Route

Task040 fixed the true-TXL PPO update boundary: the update smoke passed with
`stateless_fallback_forward_batches=0` and sequence update counters active.

Task041 goal is eval pass. The task is not complete until a sequence-aware
true-TXL checkpoint passes the clean 0.4 m/s Task039 quality gate on H200.

## Fixed Scope

- Use the Task038 G1-like train variant:
  `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`.
- Actor: `Task038TrueTxlMemoryModel`.
- Runner: `Task038TrueTxlMemoryK160Runner`.
- PPO algorithm: `Task040SequenceAwareTrueTxlPPO`.
- First target: clean 0.4 m/s eval only.
- Keep reward/env/action/obs contracts unchanged.
- Keep no-overclaim flags false until quality evidence passes.
- Do not touch `.test_tmp_task021/`.

Out of scope:

- held-out morphology;
- dynamic motor failure;
- speed curriculum beyond what is needed to get clean 0.4 eval passing;
- LocoFormer reproduction or superiority claims.

## Planned Slices

1. `001-sequence-aware-clean-train-entrypoint.md`
   - Add a repeatable train CLI that forces the Task040 sequence-aware PPO
     algorithm and records train/update evidence.

2. `002-clean-eval-gate.md`
   - Reuse the Task039 clean quality gate through a Task041 wrapper.
   - Pass requires `pipeline_pass=true`, `quality_gate_pass=true`, and
     top-level `pass=true`.

3. `003-h200-train-eval-loop.md`
   - Run H200 training chunks, evaluate checkpoints, and continue until clean
     eval passes or a stronger diagnosis is needed.

4. `004-adaptk160-warmstart-bridge.md`
   - If scratch true-TXL does not quickly produce a clean gait, create a
     Task041-compatible true-TXL checkpoint from the proven AdaptK160
     `model_5467` clean prior and continue training/eval from there.

## Acceptance Criteria

Task041 is accepted only when current evidence proves:

- train CLI help works;
- eval CLI help works;
- local tests cover train config mutation, sequence-aware gates, eval wrapping,
  train-summary linkage, and no-overclaim flags;
- H200 training summary records:
  - `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
  - `runner_cls=Task038TrueTxlMemoryK160Runner`;
  - `actor_model_class=Task038TrueTxlMemoryModel`;
  - `train_pipeline_pass=true`;
  - `stateless_fallback_forward_batches=0`;
  - `sequence_update_forward_batches > 0`;
  - checkpoint path exists;
- if the AdaptK160 warmstart bridge is used:
  - `warmstart_pipeline_pass=true`;
  - source is the documented Task037 AdaptK160 `model_5467.pt`;
  - final quality is still proven only by `task041_sequence_txl_clean_eval.py`;
- H200 eval summary records:
  - `task041_sequence_txl_clean_eval=true`;
  - `pipeline_pass=true`;
  - `quality_gate_pass=true`;
  - `pass=true`;
  - clean command `lin_vel_x=0.4`;
  - final trial within thresholds;
  - `training_claim:false`;
  - `reproduction_claim:false`;
  - `superiority_claim:false`.

## Evidence Gate

Local:

```powershell
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task041_sequence_txl_clean_train --help
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task041_sequence_txl_clean_eval --help
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task041_sequence_txl_clean_train.py tests\test_task041_sequence_txl_clean_eval.py tests\test_task040_sequence_txl_ppo_update_smoke.py tests\test_task039_true_txl_clean_eval.py tests\test_agent_inventory.py
python -m h200_locomotion_lab.tools.inspect_agent
```

H200 train command shape:

```bash
PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab \
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task041_sequence_txl_clean_train \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/train_summary.json \
  --log-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/logs \
  --num-envs 4096 \
  --iterations 100 \
  --save-interval 50 \
  --num-mini-batches 4 \
  --device cuda:0
```

H200 eval command shape:

```bash
PYTHONPATH=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab \
/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task041_sequence_txl_clean_eval \
  --checkpoint /path/to/model.pt \
  --train-summary-json /path/to/train_summary.json \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_eval/eval_model.json \
  --num-envs 64 \
  --steps 360 \
  --trial-length-s 2.0 \
  --lin-vel-x 0.4 \
  --device cuda:0
```

## Log

- 2026-05-30 Opened after Task040 sequence-aware PPO update smoke passed.
- 2026-05-30 Added local train/eval CLI wrappers:
  `task041_sequence_txl_clean_train.py` and
  `task041_sequence_txl_clean_eval.py`. Verification pending.
- 2026-05-30 Completed the Task041 clean 0.4 m/s eval gate through the
  AdaptK160 warmstart bridge:
  - warmstart JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/adaptk160_true_txl_warmstart/model_5467_task041_true_txl_bridge.json`;
  - warmstart checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/adaptk160_true_txl_warmstart/model_5467_task041_true_txl_bridge.pt`;
  - eval JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_eval/model_5467_task041_true_txl_bridge_vx0p4_eval_tolerance1e3.json`;
  - eval result: `pipeline_pass=true`, `task041_pipeline_pass=true`,
    `quality_gate_pass=true`, `pass=true`;
  - final metrics: `fall_ratio=0.0`, `gravity_xy.max=0.095986507833004`,
    `root_z.min=0.7557933330535889`,
    `lin_vel_error.mean=0.14921148121356964`;
  - active true-TXL eval memory debug present.
- 2026-05-30 H200 warmstart-bridge train smoke passed:
  - train JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/warmstart_bridge_smoke_train_env8_iter1.json`;
  - train result: `train_pipeline_pass=true`,
    `sequence_update_forward_batches=1`,
    `sequence_update_forward_samples=16`,
    `stateless_fallback_forward_batches=0`;
  - produced checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/warmstart_bridge_smoke_logs_env8_iter1/model_0.pt`.

## Review

Status: passed for clean 0.4 m/s Task041 eval, with bridge-train smoke evidence.
This closes the clean-gait convergence gate, but only through the documented
AdaptK160 warmstart bridge. It does not claim TXL superiority, held-out
robustness, morphology generalization, or a full LocoFormer reproduction.
