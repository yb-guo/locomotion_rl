# 003: Clean Memory Ablation Eval

## Route

Evaluate the same checkpoint under normal and ablated memory modes on clean
0.4 m/s.

This is the core Task042 evidence. The normal eval must remain a clean pass;
the ablations must be recorded even if they do not fail.

## Acceptance

- Normal clean eval passes with `pipeline_pass=true`, `quality_gate_pass=true`,
  and `pass=true`.
- `zero_txl_residual` eval writes a separate JSON summary.
- `stateless_txl_memory` eval writes a separate JSON summary.
- A comparison JSON records metric deltas versus normal eval.
- Review explicitly states whether the deltas support memory causality.

## Log

- 2026-05-30 Opened. No H200 eval evidence yet.
- 2026-05-30 Evaluated the residual-trained 5-iteration checkpoint from
  subtask 002:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_warmstart_train/logs_env1024_iter5/model_4.pt`.
- Normal mode summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/residual_env1024_iter5_none_vx0p4_seed4100201.json`.
  Result: `pass=false`, `task041_pipeline_pass=true`,
  `quality_gate_pass=false`, `final_trial_pass=false`,
  `final_trial.fall_ratio=0.46875`, `max_gravity_xy=0.9468991160392761`,
  `root_z.min=0.3459492325782776`,
  `lin_vel_error.mean=0.6139609217643738`,
  `yaw_vel_error.mean=0.540497899055481`, and
  `txl_residual_output_norm=0.33590757846832275`.
- `zero_txl_residual` summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/residual_env1024_iter5_zero_txl_residual_vx0p4_seed4100201.json`.
  Result: `pass=false`, `quality_gate_pass=false`,
  `final_trial.fall_ratio=0.484375`,
  `max_gravity_xy=0.9482825994491577`,
  `root_z.min=0.3776216208934784`,
  `lin_vel_error.mean=0.6089855432510376`,
  `yaw_vel_error.mean=0.5452432036399841`,
  `txl_residual_raw_norm=0.34017473459243774`, and
  `txl_residual_output_norm=0.0`.
- 2026-05-30 Evaluated the projection-only residual checkpoint from subtask
  005:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/logs_projection_only_env1024_iter5/model_4.pt`.
- Projection-only normal summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/projection_only_env1024_iter5_none_vx0p4_seed4100201.json`.
  Result: `final_trial_pass=true`, `fall_ratio=0.0`,
  `gravity_xy.max=0.09752762317657471`,
  `root_z.min=0.7556986212730408`,
  `lin_vel_error.mean=0.14943057298660278`,
  `yaw_vel_error.mean=0.09481801092624664`,
  `txl_residual_output_norm=1.880397081375122`. Top-level
  `pass=false` because the strict trial0 non-regression gate reported
  `gravity_xy_max_regressed_from_trial0` and
  `root_z_min_regressed_from_trial0`.
- Projection-only zero-residual summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/projection_only_env1024_iter5_zero_txl_residual_vx0p4_seed4100201.json`.
  Result: `pass=true`, `quality_gate_pass=true`, `final_trial_pass=true`,
  `fall_ratio=0.0`, `gravity_xy.max=0.09603133797645569`,
  `root_z.min=0.7557902336120605`,
  `lin_vel_error.mean=0.14926666021347046`,
  `yaw_vel_error.mean=0.09243886917829514`,
  `txl_residual_raw_norm=1.8576686382293701`, and
  `txl_residual_output_norm=0.0`.
- Projection-only stateless-memory summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/projection_only_env1024_iter5_stateless_txl_memory_vx0p4_seed4100201.json`.
  Result: `final_trial_pass=true`, `memory_debug_active=false`,
  `stateful_memory_enabled=false`, `stateless_fallback_forward_batches=360`,
  `txl_residual_output_norm=1.7428245544433594`. Top-level `pass=false`
  because stateless mode intentionally disables the Task041 memory-debug
  contract.
- Projection-only comparison JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/projection_only_env1024_iter5_comparison_vx0p4_seed4100201.json`.
  Versus normal, zero-residual changed final metrics only by small amounts
  (`gravity_xy.max=-0.0014962852001190186`,
  `root_z.min=+0.00009161233901977539`,
  `lin_vel_error.mean=-0.00016391277313232422`,
  `yaw_vel_error.mean=-0.0023791417479515076`) while removing
  `txl_residual_output_norm=1.880397081375122`.
- 2026-05-30 Evaluated the `txl_residual_only` checkpoint from subtask 005:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/logs_txl_residual_only_env1024_iter5/model_4.pt`.
- `txl_residual_only` normal summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/txl_residual_only_env1024_iter5_none_vx0p4_seed4100201.json`.
  Result: `final_trial_pass=true`, `fall_ratio=0.0`,
  `gravity_xy.max=0.09610089659690857`,
  `root_z.min=0.7558407783508301`,
  `lin_vel_error.mean=0.1491757035255432`,
  `yaw_vel_error.mean=0.09318430721759796`,
  `txl_residual_output_norm=1.0037872791290283`. Top-level `pass=false`
  only because `root_z_min_regressed_from_trial0` under the strict trend gate.
- `txl_residual_only` zero-residual summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/txl_residual_only_env1024_iter5_zero_txl_residual_vx0p4_seed4100201.json`.
  Result: `pass=true`, `quality_gate_pass=true`, `final_trial_pass=true`,
  `fall_ratio=0.0`, `gravity_xy.max=0.09601275622844696`,
  `root_z.min=0.7557896971702576`,
  `lin_vel_error.mean=0.14921483397483826`,
  `yaw_vel_error.mean=0.09238332509994507`,
  `txl_residual_raw_norm=0.9999684691429138`, and
  `txl_residual_output_norm=0.0`.
- `txl_residual_only` stateless-memory summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/txl_residual_only_env1024_iter5_stateless_txl_memory_vx0p4_seed4100201.json`.
  Result: `quality_gate_pass=true`, `final_trial_pass=true`,
  `memory_debug_active=false`, `stateful_memory_enabled=false`,
  `stateless_fallback_forward_batches=360`,
  `txl_residual_output_norm=1.264326810836792`. Top-level `pass=false`
  because stateless mode intentionally disables the Task041 memory-debug
  contract.
- `txl_residual_only` comparison JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/txl_residual_only_env1024_iter5_comparison_vx0p4_seed4100201.json`.
  Versus normal, zero-residual changed final `lin_vel_error.mean` by only
  `+0.000039130449295043945` while removing
  `txl_residual_output_norm=1.0037872791290283`; stateless-memory changed
  final `lin_vel_error.mean` by only `+0.0001644343137741089`.

## Review

Status: failed for memory-causality evidence.

The checkpoint proves the TXL residual path can become non-zero, but it does
not prove memory causality. Normal clean eval no longer passes, and zeroing the
TXL residual does not recover the gait. The most likely root cause is that the
PPO continuation updated the AdaptK/warmstart baseline path in addition to the
residual path.

The projection-only guardrail fixes that baseline-drift failure mode and keeps
the final clean gait healthy, but zeroing the residual does not degrade behavior
in a meaningful way. Current evidence therefore says the residual path is
non-zero but behaviorally unnecessary for clean 0.4 m/s walking.

The wider `txl_residual_only` training reaches the same conclusion: the
residual modules update and final clean gait remains healthy, but zero-residual
and stateless-memory ablations do not materially degrade locomotion. Clean
constant-speed walking is not a good memory-causality target.
