# Task 030: Online Dynamic Motor Failure Adaptation

## Route

Extend task029 from episode-start persistent motor failure robustness to
mid-episode dynamic weak/dead motor adaptation.

The research question is deliberately narrow:

Can the current MLP PPO policy, without explicit fault labels and without an
observation history stack, adapt online to motor failures that appear, recover,
and switch during an episode using only proprioceptive feedback?

Fixed decisions:

- Actor must not observe `motor_scale`, `failure_mask`, fault ids, or explicit
  current failure labels.
- Keep the current MLP policy and current actor observation contract for the
  first pass: `104 -> 31`.
- The only implicit history in the actor is the existing previous-action term,
  about one 50 Hz policy step (`0.02 s`).
- Critic, eval JSON, traces, and debug artifacts may record active fault state.
- First dynamic failure types are only `weak motor` and `dead motor`.
- First pass excludes locked joints, stuck commands, mid-episode action-delay
  changes, friction/stiction jumps, and multi-motor dynamic failures.
- Dynamic onset/switch events allow a `0.3 s` transient recovery window for
  tracking metrics, while still counting falls and instability.
- Final speed target is fixed `2.0 m/s`, reached through staged gates:
  `1.2 -> 1.6 -> 1.8 -> 2.0 m/s`.

Starting point:

- Accepted task029 `Fast1p6` checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-21_01-33-46_009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu0_seed29261/model_4700.pt`.

Planned slices:

1. `001-dynamic-failure-contract.md`
   - Define actor/critic information boundaries.
   - Define dynamic weak/dead motor semantics, timing, and recovery metrics.
   - Define speed ladder and first-pass exclusions.

2. `002-dynamic-failure-scheduler-harness.md`
   - Implement and trace a deterministic mid-episode scheduler.
   - Validate the template:
     `normal -> left_knee dead -> normal -> right_hip_yaw dead -> normal`.
   - Prove per-step active fault, joint, scale, and segment id are logged.

3. `003-existing-policy-dynamic-eval.md`
   - Evaluate task029 accepted checkpoints without training.
   - First run `1.2 m/s` for scheduler/eval sanity.
   - Then run fixed `1.6 m/s` from task029 `model_4700.pt`.
   - Record baseline failure modes before training.

4. `004-dynamic-failure-mlp-train.md`
   - Train the same MLP PPO stack on dynamic failure episodes.
   - Initial training mix:
     `20% clean`, `20% persistent`, `45% dynamic single failure`,
     `15% dynamic two-segment switch`.
   - Preserve task029 clean and persistent robustness.

5. `005-dynamic-eval-grid-and-render.md`
   - Run clean, persistent, dynamic single, and dynamic switch eval.
   - Add per-joint dynamic onset grid and switch grid.
   - Render clean, single dynamic failure, and switch dynamic failure videos.
   - Close `1.6`, then extend to `1.8`, then `2.0 m/s`.

6. `006-history-policy-decision.md`
   - Decide whether MLP without explicit history is sufficient.
   - Only if MLP fails, propose obs stack, GRU, or LocoFormer-style memory in a
     later task.

## Minimal Closed Loop

Feedback loop:

1. Inspect the dynamic-failure task and prove the actor observation contract is
   unchanged and fault labels do not leak.
2. Run a deterministic scheduler trace without training.
3. Evaluate the existing task029 accepted checkpoint to capture the pre-training
   dynamic failure baseline.
4. Train MLP PPO on the dynamic distribution only after the eval loop is
   trusted.
5. Gate every speed stage with JSON eval and render evidence before advancing.

Dynamic eval metrics:

- Existing task029 metrics:
  `zero_fall_ratio`, `max_done_count`, `lin_vel_error_mean`,
  `yaw_vel_error_mean`, `gravity_xy_mean`.
- New dynamic metrics:
  `recovery_success_ratio`, `time_to_recover_s`,
  `post_onset_fall_ratio`, `switch_fall_ratio`,
  `min_base_height_after_onset`, and `max_gravity_xy_after_onset`.

First-pass dynamic thresholds:

- `zero_fall_ratio >= 0.90`
- `recovery_success_ratio >= 0.75`
- post-recovery `lin_vel_error_mean <= 0.8`
- post-recovery `yaw_vel_error_mean <= 0.8`
- `max_gravity_xy_after_onset <= 0.8`

Pass:

- Actor observation remains 104 dim and action remains 31 dim for the MLP pass.
- Actor has no explicit fault labels, motor scales, or failure masks.
- Scheduler trace proves mid-episode onset, recovery, and switch timing.
- Existing policy baseline is recorded before dynamic training.
- Trained MLP passes dynamic eval and render gates at `1.6 m/s`.
- Speed expansion reaches fixed `2.0 m/s` with clean, persistent, dynamic
  single, dynamic switch, and render evidence.

Fail:

- Actor receives explicit failure state.
- The task silently changes policy architecture or adds history before the MLP
  baseline is tested.
- Dynamic failure is only reset-time persistent failure under a new name.
- Training reward is used as acceptance without eval JSON and videos.
- `2.0 m/s` is claimed without staged evidence at lower speeds.

Evidence:

- Planned root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/`.

## Log

- 2026-05-21 Opened after task029 accepted persistent motor-failure robustness
  through fixed `1.6 m/s`.
- 2026-05-21 User decision: Task030 target is online adaptation to dynamic
  failures, not only episode-start persistent weak/dead motors.
- 2026-05-21 User decision: actor still does not receive explicit fault labels,
  motor scales, or current fault ids.
- 2026-05-21 User decision: first pass keeps current MLP policy and current
  104-dim actor observation, with no observation history stack.
- 2026-05-21 User decision: first dynamic failure types are weak/dead motors
  only.
- 2026-05-21 User decision: onset/switch allows a `0.3 s` transient recovery
  window.
- 2026-05-21 User decision: final speed target is `2.0 m/s`, reached through
  staged gates.
- 2026-05-21 Created new local worktree and branch for execution:
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task030-online-dynamic-failure`,
  branch `codex/task030-online-dynamic-failure`.
- 2026-05-21 Added and ran H200 artifacts for subtasks 001-003. Subtask 001
  contract inspect passed with actor obs/action `104 -> 31` unchanged and no
  explicit fault labels in actor observations. Subtask 002 deterministic
  scheduler trace passed with the required mid-episode transition template.
- 2026-05-21 Existing-policy dynamic switch eval is split by speed: fixed
  `1.2 m/s` passed, fixed `1.6 m/s` failed. The accepted task029 checkpoint
  does not yet satisfy the task030 online adaptation target at `1.6 m/s`.

## Review

Status: in progress. Subtasks 001 and 002 have JSON evidence and pass. Subtask
003 has the key dynamic-switch baseline but is still partial because isolated
dynamic single-failure and clean/persistent task030 report links remain to be
recorded before training.
