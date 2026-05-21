# Task 031: Unified Forward-Speed Dynamic Switch MLP

## Route

Continue from Task030 without changing the policy architecture.

Task030 produced a scoped MLP checkpoint that is stable for fixed `2.0 m/s`
specified dynamic switching:

`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt`

Task031 turns that fixed-speed result into a unified forward-speed model.

Layered target:

1. Level A: one MLP covers forward `lin_vel_x = 0.4..2.0 m/s`.
2. Level B: the same MLP covers unified speed plus the specified Task030
   dynamic-switch route.
3. Level C: arbitrary per-joint mid-episode dead onset is diagnostic only, not
   a Task031 pass condition.

Fixed decisions:

- Keep actor observation and action contract unchanged: `104 -> 31`.
- Keep the current MLP PPO stack for Task031.
- Actor must not observe explicit fault labels, motor scales, failure masks,
  active joint ids, or speed-bin labels.
- Critic, traces, JSON summaries, and debug artifacts may record speed bin,
  failure case, active joint, active scale, and segment id.
- Unified speed means forward `lin_vel_x` only. Do not expand lateral velocity
  or yaw commands in this task.
- Training speed range is continuous `0.4..2.0 m/s`.
- Evaluation speed bins are `0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`.
- Specified dynamic switch uses the Task030 canonical switch with small timing
  jitter during training. The canonical case remains an eval gate.

Planned slices:

1. `001-unified-speed-contract.md`
   - Lock policy information boundaries and unified-speed scope.
   - Define training range, eval bins, and accepted Task030 warm start.
   - Define what Level A, Level B, and Level C mean.

2. `002-unified-speed-baseline-eval.md`
   - Evaluate `model_5349.pt` before training across all speed bins.
   - Record clean, persistent motor-failure, and specified dynamic-switch
     baseline metrics.
   - Identify the speed bins that need training.

3. `003-unified-speed-training.md`
   - Add a unified-speed training env/config in the H200 MJLab checkout.
   - Train clean plus persistent failure rehearsal from `model_5349.pt`.
   - Preserve the Task029/Task030 robustness contract.

4. `004-specified-dynamic-switch-training.md`
   - Add specified dynamic-switch training across unified speed.
   - Use canonical switch plus timing jitter during training.
   - Gate with canonical dynamic-switch eval at every speed bin.

5. `005-arbitrary-onset-diagnostic.md`
   - Run arbitrary per-joint dead-onset diagnostics only.
   - Evaluate speeds `0.4`, `1.2`, and `2.0 m/s` across the 12 leg joints.
   - Produce a failure matrix for the next history/memory-policy decision.

6. `006-render-review.md`
   - Render representative clean and dynamic-switch videos.
   - Review Level A/B pass evidence and Level C diagnostic evidence.
   - Decide whether the next task should add history stack, GRU, or
     LocoFormer-style memory.

## Minimal Closed Loop

Feedback loop:

1. Prove the Task031 env keeps actor obs/action `104 -> 31` and does not leak
   fault labels to the actor.
2. Run baseline eval for `model_5349.pt` before training.
3. Train only after the baseline matrix is recorded.
4. Gate Level A with clean and persistent failure eval at all speed bins.
5. Gate Level B with specified dynamic-switch eval at all speed bins.
6. Run Level C arbitrary-onset diagnostics without claiming it as pass/fail for
   Task031.

Acceptance gates:

- Level A speed bins: `0.4`, `0.8`, `1.2`, `1.6`, `2.0 m/s`.
- Level A cases per speed:
  - clean walking;
  - persistent random motor-failure eval using the Task029 motor-only failure
    distribution;
  - forced persistent dead-motor grid over the 12 leg joints used by Task029 and
    Task030.
- Level A per-case thresholds:
  `zero_fall_ratio >= 0.90`, `lin_vel_error_mean <= 0.8`,
  `yaw_vel_error_mean <= 0.8`, and `gravity_xy_mean <= 0.8`.
- Level B cases per speed: Task030 canonical dynamic switch.
- Level B single-seed thresholds:
  `zero_fall_ratio >= 0.90`, `recovery_success_ratio >= 0.75`,
  `post_recovery_lin_vel_error_mean <= 0.8`,
  `post_recovery_yaw_vel_error_mean <= 0.8`, and
  `max_gravity_xy_after_onset <= 0.8`.
- Level B final multiseed quorum: all five speed bins must pass with `5/5`
  seeds using the same thresholds.
- Level C diagnostic classification uses the Level B dynamic thresholds per
  case, but Level C aggregate pass is not required for Task031.
- Render acceptance: low, middle, and high speed videos must have
  `done_count=0` for clean and specified dynamic-switch cases.

Pass:

- Actor obs/action remains `104 -> 31`.
- Actor receives no explicit failure state and no speed-bin label.
- Level A passes clean and persistent failure eval at all five speed bins.
- Level B passes specified dynamic-switch eval at all five speed bins.
- Render evidence exists for representative low, middle, and high speeds.
- Level C diagnostic JSON reports failed joints/speeds if arbitrary onset still
  fails.

Fail:

- The task changes policy architecture before closing the MLP baseline.
- The actor observes explicit fault state or active fault ids.
- Level C arbitrary onset is reported as solved without a full speed/joint
  diagnostic matrix.
- Training reward is used as acceptance without eval JSON and videos.
- A single fixed-speed checkpoint is reported as a unified-speed model.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/`

## Log

- 2026-05-21 Opened after Task030 partial close. User decision: Task031 pass
  target is Level A plus Level B. Level C is diagnostic only.
- 2026-05-21 User decision: unified speed range is continuous forward
  `0.4..2.0 m/s`, evaluated at `0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`.
- 2026-05-21 User decision: specified dynamic switch uses canonical switch plus
  timing jitter during training.
- 2026-05-21 User decision: C-level arbitrary onset diagnostics cover speeds
  `0.4`, `1.2`, and `2.0 m/s`.
- 2026-05-21 User decision: command scope is forward `vx` only.

## Review

Status: planned. No Task031 implementation or H200 training evidence yet.
