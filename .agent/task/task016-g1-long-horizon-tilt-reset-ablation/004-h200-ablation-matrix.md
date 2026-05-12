# 004: H200 Ablation Matrix

## Goal

Compare a small one-variable-at-a-time ablation matrix against the reproduced
baseline.

## Route

1. Run seed-0 variants:
   - baseline;
   - lower LR (`1e-4`);
   - stronger termination penalty (`-5`);
   - stronger action-rate penalty.
2. Compare first tilt update, max/final reset count, KL, entropy, reward, and
   throughput.
3. Decide which hypothesis is most supported.

## Log

- 2026-05-09 H200 v1 baseline reproduced, but later variants failed with
  `GenesisException:Genesis already initialized.` because Genesis cannot be
  reinitialized in the same Python process.
- 2026-05-09 Router fixed the runner to execute each variant in a separate
  subprocess.
- 2026-05-09 H200 v2 completed all variants with
  `all_variants_completed=true` and `baseline_reproduced_tilt_reset=true`.
- 2026-05-09 Run directory:
  `/root/agent_workspace/project/h200-locomotion-lab-task016-g1-long-horizon-tilt-reset-ablation/outputs/task016/tilt_reset_ablation/h200-gpu1-seed0-updates50-v2`.

Variant comparison, seed 0:

| Variant | Standing final reset | Standing final tilt | Small vx final reset | Small yaw final reset | Small vxyaw final reset | Min throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1024 | 1024 | 1024 | 1024 | 1024 | 12494.3 |
| lr_1e4 | 1024 | 1024 | 1024 | 1024 | 1022 | 15433.4 |
| termination_penalty_neg5 | 1024 | 1024 | 1024 | 1024 | 1024 | 10933.5 |
| action_rate_penalty_high | 1024 | 1024 | 1024 | 1024 | 1024 | 11314.7 |

Additional observations:

- Every variant and every stage had `first_tilt_update=2`.
- Every variant and every stage reached `max_reset_count=1024`.
- Every variant and every stage kept `mean_reset_count=348.16`.
- All final rows had `termination_height_bad_count=0`.
- Lower LR reduced final KL substantially but did not remove reset waves.
- Stronger termination penalty reduced reward but did not remove reset waves.
- Stronger action-rate penalty did not remove reset waves.

## Review

Status: passed.

- The matrix changed one variable per variant.
- H200 outputs stayed under `/root/agent_workspace/project`.
- No rendering, SONIC, ONNX, LocoFormer, downloads, or `/mnt/workspace*` writes
  were used.
