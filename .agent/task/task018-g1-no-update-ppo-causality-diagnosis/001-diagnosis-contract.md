# 001: Diagnosis Contract

## Goal

Define the no-update feedback loop and causal decision tree before coding.

## Route

1. Use task017 u50 standing evidence as the PPO-update failure reference:
   - baseline final `reset_count=1024`;
   - baseline final `tilt_bad_count=1024`;
   - `termination_height_bad_count=0`;
   - `first_tilt_update=2`.
2. Build a no-update loop with the same standing env and 1600-step horizon.
3. Compare zero action, untrained deterministic mean action, and untrained
   sampled action before any actor/critic isolation.
4. Stop early if no-update probes already identify control/reset or sampling
   instability.

## Log

- 2026-05-11 Reproduction source:
  `/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis/outputs/task017/action_control_semantics/h200-gpu1-seed0-standing-u50-targeted-v1`.
- 2026-05-11 Task017 ruled out action scale down to `0.01` and exploration
  noise down to `log_std_init=-3.5` as sufficient fixes.
- 2026-05-11 Ranked hypotheses:
  - no-learning control/reset instability;
  - no-learning sampling/action-interface instability;
  - PPO-update-induced instability;
  - actor-update-specific instability.

## Review

Status: passed as diagnosis contract.

- Feedback loop is narrower than task017 and has explicit causal branches.
- Stop rules avoid running PPO-update isolation before no-update evidence.
