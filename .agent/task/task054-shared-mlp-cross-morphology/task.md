# Task 054: Shared MLP Across Morphologies

## Route

Compose fixed-topology MuJoCo shards with `WholeBodyRolloutMux`, keep biped and
quadruped samples balanced, normalize reward by physical scale and active
actuator count, then expand the curriculum from narrow topology to the full
train manifest.

## Log

- 2026-08-19: Added the five-stage curriculum, 8×256 baseline shard plan,
  dynamic active-mask PPO support, and a two-family mux smoke.

## Review

The two-family mux PPO smoke completes with per-environment masks and
heterogeneous privileged motor widths.  RTX 5060 Ti throughput/memory
measurement and the shared-policy quality gates remain pending.
