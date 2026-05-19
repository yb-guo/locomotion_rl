# 004: H200 Action Control Matrix

## Goal

Compare one-variable standing-only action/control variants against the
reproduced baseline.

## Route

1. Run seed-0 standing-only variants:
   - baseline;
   - `action_scale_mult=0.05`;
   - `action_scale_mult=0.03`;
   - `action_scale_mult=0.01`;
   - `action_joint_group=legs`;
   - `action_joint_group=legs_waist`;
   - lower exploration noise.
2. Compare reset wave timing, reset counts, action stats, root/upright, KL,
   entropy, and throughput.
3. Decide which hypothesis is most supported.

## Log

- 2026-05-11 H200 standing-only u10 matrix completed all variants:
  `/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis/outputs/task017/action_control_semantics/h200-gpu1-seed0-standing-u10-v1`.

Seed-0 u10 matrix:

| Variant | first tilt | max reset | mean reset | final reset | final tilt | final action abs mean | final action abs max | final root mean | final root min | final upright |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2 | 1024 | 307.20 | 0 | 0 | 0.0655 | 0.3795 | 0.807 | 0.784 | 1.000 |
| action_scale_0_05 | 2 | 1024 | 307.20 | 0 | 0 | 0.0655 | 0.3662 | 0.808 | 0.784 | 1.000 |
| action_scale_0_03 | 2 | 1024 | 307.20 | 0 | 0 | 0.0655 | 0.3658 | 0.808 | 0.784 | 1.000 |
| action_scale_0_01 | 2 | 1024 | 307.20 | 0 | 0 | 0.0655 | 0.3659 | 0.808 | 0.784 | 1.000 |
| action_group_legs | 2 | 1024 | 307.20 | 0 | 0 | 0.0655 | 0.3799 | 0.807 | 0.784 | 1.000 |
| action_group_legs_waist | 2 | 1024 | 307.20 | 0 | 0 | 0.0655 | 0.3797 | 0.807 | 0.784 | 1.000 |
| log_std_neg3_5 | 2 | 1024 | 307.20 | 0 | 0 | 0.0245 | 0.1425 | 0.808 | 0.784 | 1.000 |

- All u10 variants reproduced `first_tilt_update=2`, `max_reset_count=1024`,
  and `mean_reset_count=307.20`.
- `mean_reset_count=307.20` means 3 of 10 updates had full-env resets.
- All u10 variants recovered by the final row.
- `log_std_neg3_5` reduced normalized action magnitude but did not prevent the
  update-2 reset wave.
- Action group variants did not prevent the update-2 reset wave. Their top RMS
  entries are normalized policy actions; disabled joints may still appear in
  action RMS and must not be interpreted as applied backend motion.

Because u10 recovered by the final row, Router added a targeted u50 check for
the most important counterexamples:

Run root:

```text
/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis/outputs/task017/action_control_semantics/h200-gpu1-seed0-standing-u50-targeted-v1
```

Seed-0 u50 targeted:

| Variant | first tilt | max reset | mean reset | final reset | final tilt | final action abs mean | final action abs max | final root mean | final root min | final upright |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2 | 1024 | 348.16 | 1024 | 1024 | 0.0700 | 0.4371 | 0.806 | 0.316 | 0.836 |
| action_scale_0_01 | 2 | 1024 | 348.16 | 1024 | 1024 | 0.0700 | 0.4211 | 0.794 | 0.327 | 0.818 |
| log_std_neg3_5 | 2 | 1024 | 348.16 | 1024 | 1024 | 0.0305 | 0.2077 | 0.794 | 0.322 | 0.818 |

- `mean_reset_count=348.16` means 17 of 50 updates had full-env resets.
- `action_scale_mult=0.01` did not prevent the 50-update final collapse.
- `log_std_init=-3.5` reduced normalized action magnitude by more than half but
  did not prevent the 50-update final collapse.

## Review

Status: passed.

- H200 outputs stayed under `/root/agent_workspace/project`.
- The matrix changed one variable per non-baseline variant.
- No rendering, SONIC, ONNX, LocoFormer, downloads, or `/mnt/workspace*`
  writes were used.
