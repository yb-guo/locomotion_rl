# 007: Render And Review

## Route

Render the accepted task029 checkpoint and review gait quality. This subtask is
the human-facing sanity check for whether the policy is walking or exploiting
the reward.

Render cases:

- Clean fixed-command walking.
- In-distribution motor-failure sample.
- One or more representative dead-motor grid cases.
- Optional worst passing and worst failing grid cases.

Review focus:

- Excessive high-frequency shaking.
- Foot dragging or skating.
- Upper-body flailing.
- Gripper or arm motion used as a reward exploit.
- Asymmetric gait consistent with the sampled failed motor.
- Falls or near-falls hidden by aggregate metrics.

## Minimal Closed Loop

Feedback loop:

1. Render videos from the exact checkpoint used in 006.
2. Save video, midframe, and render summary JSON.
3. Record command, seed, failure mask, motor scales, and checkpoint path.
4. Add absolute H200 artifact paths and local preview paths if copied.

Pass:

- At least one clean video and one motor-failure video are saved.
- Render summary records the exact checkpoint and fault settings.
- Video review does not show obvious reward hacking in accepted cases.
- Any visible gait defect is documented with the corresponding eval metrics.

Fail:

- Render uses a different checkpoint from eval without explanation.
- Fault settings are not recorded.
- Only still images are produced when video was requested by the task.
- Obvious reward hacking is ignored.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/`.

## Log

- 2026-05-19 Opened because task028 showed render evidence is necessary to
  interpret walking metrics and gripper/upper-body behavior.

## Review

Status: pending.

This is the final task029 acceptance check. The policy can pass numeric eval
and still be rejected here if the video shows unstable or exploitative motion.
