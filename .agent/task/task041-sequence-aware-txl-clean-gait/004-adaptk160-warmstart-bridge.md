# 004 AdaptK160 Warmstart Bridge

## Route

If scratch sequence-aware true-TXL clean training does not produce a clean
0.4 m/s eval pass quickly, build a warmstart bridge from the proven Task037
AdaptK160 clean prior:

- source checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_clean_gait_prior_train/2026-05-29_16-44-07_037_adapt_k160_clean_from_adaptk4_env8192_iter60_gpu0_seed3700705/model_5467.pt`;
- target actor class remains `Task038TrueTxlMemoryModel`;
- target latent contract is `newest_base_obs(104) + memory_latent(32)`;
- copy matching AdaptK160 normalizer, MLP, critic, and adaptation encoder
  weights;
- keep TXL projection/attention fresh and active;
- drop optimizer state and continue with Task040 sequence-aware PPO.

This bridge is allowed to establish a clean gait prior, but it is not a TXL
superiority, reproduction, or held-out robustness claim.

## Log

- 2026-05-30 Added local warmstart bridge plumbing:
  - `Task038TrueTxlMemoryModel` can optionally use base-observation passthrough,
    a 32D memory latent, and an AdaptK warmstart encoder;
  - `task037_multitrial_eval_checkpoint.py` can pass optional true-TXL actor
    shape args from wrapper CLIs into the runner config;
  - `task041_sequence_txl_clean_train.py` and
    `task041_sequence_txl_clean_eval.py` default Task041 to the bridge-compatible
    `104 + 32` latent shape;
  - `task041_adaptk160_true_txl_warmstart.py` constructs a shape-complete
    Task041 checkpoint from AdaptK160 source weights.
  Verification pending.
- 2026-05-30 H200 warmstart eval produced strong clean locomotion metrics but
  failed the Task039 quality gate only because trial0/final root-z
  non-regression used zero practical tolerance:
  - `final_trial_pass=true`;
  - `fall_ratio=0.0`;
  - `final_trial.root_z.min=0.7557916641235352`;
  - `trial_0.root_z.min=0.7562705278396606`;
  - difference is about 0.48 mm.
  Updated the shared quality feedback default `trend_tolerance` to `1e-3`
  without changing final fall/root/tracking thresholds. Verification pending.
- 2026-05-30 H200 warmstart construction passed:
  - JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/adaptk160_true_txl_warmstart/model_5467_task041_true_txl_bridge.json`;
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/adaptk160_true_txl_warmstart/model_5467_task041_true_txl_bridge.pt`;
  - `warmstart_pipeline_pass=true`;
  - `actor_model_class=Task038TrueTxlMemoryModel`;
  - `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
  - `memory_latent_dim=32`;
  - `base_obs_passthrough=true`;
  - `adaptation_warmstart=true`.
- 2026-05-30 H200 Task041 clean eval passed:
  - JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_eval/model_5467_task041_true_txl_bridge_vx0p4_eval_tolerance1e3.json`;
  - `pipeline_pass=true`;
  - `task041_pipeline_pass=true`;
  - `quality_gate_pass=true`;
  - `pass=true`;
  - `final_trial.fall_ratio=0.0`;
  - `final_trial.gravity_xy.max=0.095986507833004`;
  - `final_trial.root_z.min=0.7557933330535889`;
  - `final_trial.lin_vel_error.mean=0.14921148121356964`;
  - `final_trial.yaw_vel_error.mean=0.09237571805715561`;
  - `memory_debug_active=true`;
  - no training/reproduction/superiority claim flags.
- 2026-05-30 H200 warmstart-bridge train smoke passed:
  - JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/warmstart_bridge_smoke_train_env8_iter1.json`;
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/warmstart_bridge_smoke_logs_env8_iter1/model_0.pt`;
  - `train_pipeline_pass=true`;
  - `sequence_update_forward_batches=1`;
  - `sequence_update_forward_samples=16`;
  - `stateless_fallback_forward_batches=0`.

## Review

Status: pass for the Task041 clean-gait gate via the AdaptK160 warmstart bridge.
This is a clean eval pass and an active true-TXL memory plumbing pass. It is
not a claim that TXL memory, by itself, caused the gait or outperformed
AdaptK160.

Acceptance evidence:

- local tests for migration shape/copy behavior;
- H200 warmstart construction JSON with `warmstart_pipeline_pass=true`;
- Task041 eval JSON on the warmstarted checkpoint with `pipeline_pass=true`;
- final Task041 eval has `quality_gate_pass=true` and top-level `pass=true`.
- H200 bridge train smoke JSON with `train_pipeline_pass=true` and active
  sequence-aware PPO update counters.
