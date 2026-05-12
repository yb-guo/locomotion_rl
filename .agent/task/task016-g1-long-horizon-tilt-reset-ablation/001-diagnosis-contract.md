# 001: Diagnosis Contract

## Goal

Create the feedback loop and ranked hypotheses for task016 before coding.

## Route

1. Use task015 final H200 evidence as the reproduction:
   - final rows show recurring `tilt_bad` resets;
   - height termination is not the cause.
2. Rank falsifiable hypotheses.
3. Define the minimum ablation matrix.
4. Require coding subagent implementation and read-only reviewer review.

## Log

- 2026-05-09 Reproduction source:
  `/root/agent_workspace/project/h200-locomotion-lab-task015-g1-curriculum-longer-horizon-ppo/outputs/task015/g1_curriculum_ppo/h200-gpu1-final-seeds012-updates50-v1`.
- 2026-05-09 Primary symptom:
  `reset_count=1024`, `tilt_bad_count=1024`,
  `termination_height_bad_count=0` in final rows.
- 2026-05-09 Ranked hypotheses:
  - PPO schedule drift;
  - tilt/reset reward too weak;
  - action smoothness too weak;
  - command range too coarse.

## Review

Status: passed as diagnosis contract.

- Feedback loop target is explicit.
- Hypotheses are falsifiable and mapped to one-variable H200 variants.
