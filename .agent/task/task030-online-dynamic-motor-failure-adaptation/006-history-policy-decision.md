# 006: History Policy Decision

## Route

Use the Task030 MLP results to decide whether explicit history or memory is
needed in a later task.

Do not change the policy in Task030 unless the user explicitly re-scopes it.

Decision outcomes:

- If MLP passes through `2.0 m/s`, record that current proprioceptive feedback
  is sufficient for this dynamic weak/dead motor setting.
- If MLP fails at `1.6`, `1.8`, or `2.0`, diagnose whether the failure is
  onset detection, recovery timing, switch ambiguity, or speed tracking.
- Only then propose a later task for one of:
  - observation stack, e.g. `5` or `10` frames
  - GRU/recurrent policy
  - LocoFormer-style long-context policy

Pass:

- Decision is based on dynamic eval evidence, not expectation.
- Any proposed memory policy has a concrete failure mode it is meant to fix.

Fail:

- History is added before the MLP baseline has been evaluated.
- A larger model is proposed without identifying which dynamic metric failed.

## Log

- 2026-05-21 Opened.
- 2026-05-21 MLP-only Task030 pass reached fixed `2.0 m/s` without adding
  explicit actor fault labels, motor scales, failure masks, observation stack,
  GRU state, or LocoFormer-style memory. Final accepted checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt`.
- 2026-05-21 Decision: current proprioceptive MLP feedback is sufficient for
  this task's first-pass dynamic weak/dead motor setting. No history policy is
  needed inside Task030. Defer observation stacks, GRU, or LocoFormer-style
  memory to a later task only if the scope expands to locked joints, stuck
  commands, multi-motor simultaneous dynamic faults, longer hidden actuator
  delays, or harder terrain.

## Review

Status: pass. The decision is evidence-based: final `2.0 m/s` dynamic switch
multi-seed s5 and task029 full regression both pass for the MLP checkpoint, so
Task030 should not change policy architecture.
