# Route

Task: task008-runtime-profile-foundation

Goal: implement the scalar action bridge from loaded profile data.

Pass condition:

- Raw 29D policy/IsaacLab action maps to 29D command/MuJoCo motor targets.
- Formula matches official SONIC deploy semantics:

```text
target_command[i] =
  default_angles_command[i]
  + raw_action_policy[command_to_policy[i]] * action_scale_command[i]
```

- Outputs match the current `g1_policy_bridge.py` implementation before
  migration.

Fail condition:

- The scalar bridge reads global SONIC constants instead of the loaded profile.
- The bridge clips raw actions unless explicitly configured as a separate safety
  layer.

# Log

- 2026-05-08: Pending profile loader.

# Review

Status: pending.

