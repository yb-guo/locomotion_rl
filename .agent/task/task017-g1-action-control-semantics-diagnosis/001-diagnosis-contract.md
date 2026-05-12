# 001: Diagnosis Contract

## Goal

Define the fast feedback loop and ranked hypotheses before coding.

## Route

1. Use task016 H200 evidence as the known reproduction:
   - standing reaches `final_reset_count=1024`;
   - standing reaches `final_tilt_bad_count=1024`;
   - `termination_height_bad_count=0`;
   - `first_tilt_update=2`.
2. Minimize from 4-stage curriculum to standing-only.
3. Add action statistics so action/control hypotheses are observable.
4. Require one-variable H200 variants.

## Log

- 2026-05-11 Reproduction source:
  `/root/agent_workspace/project/h200-locomotion-lab-task016-g1-long-horizon-tilt-reset-ablation/outputs/task016/tilt_reset_ablation/h200-gpu1-seed0-updates50-v2`.
- 2026-05-11 Task016 ruled out LR, termination penalty, and action-rate
  penalty as sufficient fixes.
- 2026-05-11 Ranked hypotheses:
  - action amplitude too high;
  - action joint group too broad;
  - exploration noise too high;
  - upright/default-joint shaping still too weak.

## Review

Status: passed as diagnosis contract.

- Feedback loop is standing-only and cheap enough for repeated H200 runs.
- Hypotheses are falsifiable through one-variable variants.
