# 007 Multi-Objective Low/Mid/High Curriculum

## Route

Keep this work inside Task035. The curriculum issue is not solved by splitting
low-speed repair into a new task:

- broad `0.4..2.0 m/s` mixed curriculum regressed low speed;
- low-speed focused curriculum fixed `0.4/1.2 m/s`;
- right-hip-roll repair and balanced rehearsal still regressed `2.0 m/s`.

This subtask is the next curriculum design step:

```text
candidate checkpoint
        |
        v
train stage proposal
        |
        v
fast low/mid/high eval gate
        |
        v
checkpoint accepted only if all protected gates stay green
```

Starting evidence:

- `model_5350.pt` remains the scoped high-speed baseline.
- `model_5397.pt` is useful low/mid-speed evidence, not a promoted mainline.
- Task036-style evidence is folded back here:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/full_validation_model_5397/task036_full_validation_summary.json`.

Protected gates:

- `0.4 m/s` dynamic switch;
- `0.4 m/s` full 12-joint dead-grid;
- `1.2 m/s` dynamic switch;
- `1.2 m/s` full 12-joint dead-grid;
- `2.0 m/s` dynamic switch;
- `2.0 m/s` full 12-joint dead-grid.

Known hard cases to include in training and fast gates:

- low-speed baseline failures:
  `left_hip_yaw_joint`, `left_hip_roll_joint`, `right_knee_joint`;
- low-speed repair residual:
  `right_hip_roll_joint`;
- high-speed regression after repair:
  `right_hip_pitch_joint`, `right_knee_joint`;
- dynamic-switch gravity guard at `2.0 m/s`.

Stage proposal:

- keep frozen-base StackMLP K4;
- keep actor history `540D`, critic `119D`, action `31D`;
- do not expose fault ids, masks, scales, or speed-bin ids to actor;
- use an interleaved curriculum, not sequential single-target repair;
- sample speed bins with explicit weights, for example:
  - `0.4 m/s`: low-speed recovery;
  - `1.2 m/s`: bridge/protection;
  - `2.0 m/s`: high-speed protection;
- sample failure joints from the protected hard-case set plus the 12-joint
  dead-grid set;
- run short continuations with frequent checkpoints and select only by eval.

First bounded experiment:

1. patch Task035 command sampling to draw `lin_vel_x` from explicit
   `0.4/1.2/2.0 m/s` bins instead of a continuous range;
2. start from `model_5350.pt` or `model_5397.pt`, but record which one;
3. run env8192 with low LR and save every 1-2 iterations;
4. sweep non-final checkpoints on focused low/mid/high gates;
5. run full matrix only for candidates that pass focused gates.

Acceptance:

- A checkpoint can be called promoted only if all protected gates pass.
- A checkpoint that improves low/mid speed but fails `2.0 m/s` remains partial.
- A checkpoint that preserves `2.0 m/s` but regresses low speed remains partial.

## Log

- 2026-05-28 Opened after user clarified the next curriculum work should be a
  Task035 subtask.
- 2026-05-28 Folded in follow-up evidence from the prior low-speed repair run:
  `model_5397.pt` passes `0.4/1.2 m/s` full validation but fails `2.0 m/s`.
- 2026-05-28 Added an H200 patch-script update so Task035 can train on explicit
  low/mid/high speed bins while keeping the actor observation contract unchanged.
- 2026-05-28 Ran two speed-bin continuations from `model_5350.pt` and
  `model_5397.pt`. No final checkpoint passed the protected full matrix:
  - `model_5362` from `5350` passed `1.2/2.0 m/s` but failed `0.4 m/s`
    left hip yaw/roll dead-grid;
  - `model_5416` from `5397` passed `0.4/1.2 m/s` but failed `2.0 m/s`
    right hip pitch/right knee dead-grid.
- 2026-05-28 Added and ran a focused hard-case stage over
  `left_hip_yaw_joint`, `left_hip_roll_joint`, `right_hip_pitch_joint`, and
  `right_knee_joint`. It restored high-speed fast gates in several
  continuations but did not close the full matrix:
  - best hard-case candidates from `5362` still failed `0.4 m/s` dead-grid;
  - hard-case continuations from `5397` did not pass the `2.0 m/s`
    right-knee/right-hip-pitch fast gate.

## Review

Status: executed, no checkpoint promotable. Evidence supports keeping Task035
open as `candidate_only_not_promoted`; the current MLP curriculum can protect
either low/mid speed or high speed, but the tested short continuations did not
produce one checkpoint that passes all protected gates together.
