# 005: Memory Causality Review

## Route

Close Task043 by reviewing the evidence against the original target: a
memory-required dynamic-switch training loop that lets True-TXL memory matter.

## Acceptance

- Review cites exact train and eval JSON paths.
- Review does not rely on training intent or architectural plausibility.
- Positive result requires:
  - normal dynamic-switch eval passes;
  - zero-residual/stateless evals degrade materially.
- Negative result must state whether the next move is training-contract,
  policy-architecture, or environment-target change.

## Log

- 2026-05-31 Opened.
- 2026-05-31 Reviewed the fixed-speed training branch:
  residual-only continued model_74 still failed normal quality with final
  `lin_vel_error.mean=0.4760773777961731`; all-scope model_24 regressed to
  `0.876373827457428`.
- 2026-05-31 Reviewed the `model_5349` True-TXL bridge branch:
  normal mode passed quality, but zero-residual and stateless-memory ablations
  stayed tied. Evidence:
  - normal:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_none_vx1p6_seed4301701.json`;
  - zero-residual:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_zero_residual_vx1p6_seed4301701.json`;
  - stateless:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_stateless_vx1p6_seed4301701.json`.
- 2026-05-31 Reviewed repaired residual training from the bridge. The new
  `txl_residual_and_mlp_memory_input` scope makes the memory path trainable,
  but trained checkpoints did not close Task043:
  - 25-iteration train:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_model5349_bridge_residual_mlp_memory_input_env1024_iter25_seed4301901.json`;
  - 25-iteration normal eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter25_model24_switch_none_vx1p6_seed4302001.json`;
  - 25-iteration zero-residual eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter25_model24_switch_zero_residual_vx1p6_seed4302001.json`;
  - 25-iteration stateless eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter25_model24_switch_stateless_vx1p6_seed4302001.json`;
  - 5-iteration train/eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_model5349_bridge_residual_mlp_memory_input_env1024_iter5_seed4302101.json`,
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter5_model4_switch_none_vx1p6_seed4302201.json`.

## Review

Status: negative for memory causality under the current Task043 target.

Task043 now has a normal-mode dynamic-switch quality pass through the
`model_5349` True-TXL bridge, but it is not memory-required. The bridge has
`txl_residual_output_norm=0.0`, zero-residual eval passes at
`lin_vel_error.mean=0.44687220454216003`, and stateless-memory eval passes at
`0.446464478969574`.

The training-contract experiment has now been done. It fixed the dead residual
path mechanically, but the trained residual does not create a memory-required
policy. The 25-iteration model has active residual output and better final
velocity error, but fails strict quality on posture/height regression and
zero-residual/stateless ablations remain tied. The 5-iteration model also fails
velocity quality.

The next move should be an environment-target or evaluation-contract change,
not more iterations of the same Task043 setup. The current dynamic switch is
solvable by the high-speed MLP prior, so memory is not required by the task.
