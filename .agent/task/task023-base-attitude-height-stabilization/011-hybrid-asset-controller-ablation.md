# Subtask 011: Hybrid Asset Controller Ablation

## Route

- Continue inside task023; no new top-level task.
- Genesis-only. No MuJoCo, PPO, downloads, `/mnt/workspace*`
  writes/deletes, or `GenesisG1SceneBackend` changes.
- Freeze asset at `ankle_roll_hybrid_edge_boxes_no_points`.
- Change only fixed-controller parameters.

## Feedback Loop

```text
hybrid passive baseline 116 -> fixed-controller active baseline 108 ->
max-delta/gain ablation -> compare active horizon and contact force
```

## Ranked Hypotheses

1. **The controller is over-aggressive/clipping-dominated.**
   - Prediction: lowering `max_joint_delta` from `0.08` to `0.04` or `0.02`
     lowers clipping and delays reset beyond 108.
2. **The attitude gain is too high for the hybrid contact asset.**
   - Prediction: lowering `attitude_kp/kd` delays reset while preserving low
     ankle-roll force.
3. **The fixed-controller target family is wrong, not merely too strong.**
   - Prediction: max-delta/gain reductions either do not improve reset, or
     revert toward passive collapse; a different controller decomposition is
     needed before PPO.

## Stop Rules

- Stop before H200 if local focused tests fail.
- Stop any candidate if ankle-roll force exceeds 1000 without stability gain.
- If any candidate improves reset by at least 20 steps, rerun once before
  treating it as evidence.
- Do not run PPO.

## Log

- 2026-05-13 Created after subtask010 found a Genesis asset candidate with
  passive horizon 116 but active `attitude + all` reset at 108.
- 2026-05-13 Local focused tests passed before H200:
  `22 passed in 1.89s`.
- 2026-05-13 H200 focused tests passed:
  `22 passed in 0.67s`.
- 2026-05-13 H200 Genesis controller strength matrix on
  `ankle_roll_hybrid_edge_boxes_no_points`, 140 steps, `attitude + all`,
  `pose_profile=current`, `roll_allocation=hip_only_mirrored`:

  | Candidate | First tilt/reset | Peak ankle-roll force | Clip ratio | Result |
  | --- | ---: | ---: | ---: | --- |
  | default from subtask010, `kp=1.6 kd=0.45 delta=0.08` | 108 | 173.6 @ step 11 | 0.900 | baseline active failure |
  | `kp=1.6 kd=0.45 delta=0.04` | 113 | 799.8 @ step 77 | 0.921 | closer to passive, high mid-collapse force |
  | `kp=1.6 kd=0.45 delta=0.02` | 114 | 772.0 @ step 76 | 0.950 | best continuous-controller row, still below passive 116 |
  | `kp=0.8 kd=0.225 delta=0.08` | 109 | 855.8 @ step 74 | 0.736 | gain reduction alone not useful |
  | `kp=0.8 kd=0.225 delta=0.04` | 112 | 800.1 @ step 76 | 0.814 | small improvement only |

- 2026-05-13 H200 Genesis timing probes with default gains on the hybrid asset:

  | Candidate | First tilt/reset | Peak ankle-roll force | Clip ratio | Result |
  | --- | ---: | ---: | ---: | --- |
  | `controller_stop_step=40`, seed 120 | 122 | 686.4 @ step 126 | 0.186 | best result; beats passive 116 |
  | `controller_start_step=80`, seed 120 | 115 | 826.3 @ step 90 | 0.429 | does not rescue late collapse |
  | `controller_stop_step=40`, seed 121 confirm | 122 | 678.0 @ step 126 | 0.186 | confirmed |

## Review

Status: diagnostic_partial_not_passed; short-pulse_controller_identified.

- Hypothesis 1 partially passed. Lowering `max_joint_delta` improves active
  reset from 108 to 113/114, but it does not exceed the passive hybrid horizon
  116 and leaves high mid-collapse ankle-roll force.
- Hypothesis 2 mostly failed. Halving attitude gain does not solve the failure;
  reset stays 109/112 depending on delta.
- Hypothesis 3 passed. Continuous fixed attitude control is the wrong target
  family on this asset. It either hurts versus passive or only approaches
  passive. The only improvement beyond passive comes from a short early control
  pulse followed by disabling the controller at step 40.
- Decision: the best non-PPO candidate is now
  `hybrid_edge_boxes_no_points + attitude controller stop_step=40`, confirmed
  at reset 122 on seeds 120 and 121. This is useful diagnosis, not a standing
  baseline: after step 40 the system is effectively passive, and collapse still
  occurs. Next route should investigate why early pose settling helps but
  continuous attitude feedback destabilizes, likely by comparing target joint
  traces/root pitch around steps 40-80 and/or designing a decaying controller
  rather than a constant clipped controller.
