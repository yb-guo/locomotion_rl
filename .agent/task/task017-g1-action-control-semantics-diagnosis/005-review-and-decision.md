# 005: Review And Decision

## Goal

Review evidence and decide the next engineering direction.

## Route

1. Read-only reviewer checks:
   - boundary compliance;
   - stage-selection correctness;
   - action-stat correctness;
   - ablation correctness;
   - H200 evidence.
2. Fix blocking findings through coding subagent.
3. Record final diagnosis decision.

## Log

- 2026-05-11 Read-only reviewer found no blocking findings before H200.
- 2026-05-11 H200 focused tests passed with 40 passed in 2.59s.
- 2026-05-11 H200 u10 standing-only matrix completed all variants and
  reproduced the update-2 reset wave in every variant.
- 2026-05-11 H200 u50 targeted standing-only runs confirmed that baseline,
  `action_scale_mult=0.01`, and `log_std_init=-3.5` all still end with
  `final_reset_count=1024` and `final_tilt_bad_count=1024`.
- 2026-05-11 Final read-only reviewer found no blocking findings.

Decision:

Task017 does not support action amplitude, broad action joint group, or
exploration noise as sufficient root causes.

Evidence:

- Lowering applied action scale to `0.01` did not stop the update-2 reset wave
  and did not stop the 50-update final collapse.
- Restricting action application to `legs` or `legs_waist` did not stop the
  update-2 reset wave.
- Lowering exploration noise to `log_std_init=-3.5` reduced normalized final
  action magnitude from about `0.0700/0.4371` mean/max to
  `0.0305/0.2077`, but did not stop the 50-update final collapse.
- `termination_height_bad_count=0` remains consistent with task016: this is
  still a tilt/fall reset path, not hard height termination.

Most supported interpretation:

The immediate reset wave and later standing collapse are probably not caused by
large sampled actions alone. The next diagnostic should separate reward/learning
effects from simulator/control effects by running no-update and deterministic
policy probes:

1. Untrained sampled policy rollout without PPO update.
2. Untrained deterministic mean-action rollout without PPO update.
3. Zero-action rollout through the exact PPO runner reset/metrics path.
4. PPO update with actor frozen or near-zero action head, if needed.

If no-update deterministic/zero-action paths are stable but PPO-update paths
fall, the next fix area is reward/value/advantage shaping. If no-update paths
fall, the next fix area is reset/default-pose/PD/control semantics.

## Review

Status: passed.

- Final read-only review found no blocking findings.
- Residual risk: u50 targeted evidence covers baseline,
  `action_scale_mult=0.01`, and `log_std_init=-3.5`; action-group variants
  only have u10 evidence for the update-2 reset wave.
