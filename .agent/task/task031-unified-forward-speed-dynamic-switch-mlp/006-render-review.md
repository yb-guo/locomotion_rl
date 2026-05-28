# 006 Render Review

## Route

Render representative Task031 behavior and close the task review.

Required renders:

- Clean walking at low, middle, and high speed.
- Specified dynamic switch at low, middle, and high speed.
- If Level C has a severe failure case, render one worst speed/joint pair for
  diagnosis.

Recommended speed choices:

- Low: `0.4 m/s`
- Middle: `1.2 m/s`
- High: `2.0 m/s`

Review should summarize:

- Final checkpoint path.
- Level A pass/fail evidence.
- Level B pass/fail evidence.
- Level C diagnostic matrix and next-task recommendation.
- Any residual gait artifacts visible in video.

## Log

- 2026-05-21 Planned as final evidence and review step.
- 2026-05-28 First render attempt against
  `Unitree-G1-Gripper-Flat-Task031-UnifiedDynamicSwitch-Fast2p0` failed because
  the Task030 render helper injects a `template` param, while the Task031
  canonical scheduler function does not accept that keyword. Failed summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349/task031_render_review_summary.json`.
- 2026-05-28 Re-rendered the actual Level B canonical eval route using
  `Unitree-G1-Gripper-Flat-DynamicMotorFailure-Fast1p6`, matching the Level B
  multiseed evaluator. Render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/task031_render_review_summary.json`.
  Result: `pass=true`, `render_count=6`, all `done_count=0`, `frames=500`,
  `fps=50`.
- 2026-05-28 Video paths:
  - `0.4 clean`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/vx0p4/task031-render-model5349-canonical-clean-vx0p4.mp4`;
  - `0.4 switch`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/vx0p4/task031-render-model5349-canonical-switch-vx0p4.mp4`;
  - `1.2 clean`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/vx1p2/task031-render-model5349-canonical-clean-vx1p2.mp4`;
  - `1.2 switch`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/vx1p2/task031-render-model5349-canonical-switch-vx1p2.mp4`;
  - `2.0 clean`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/vx2p0/task031-render-model5349-canonical-clean-vx2p0.mp4`;
  - `2.0 switch`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/render_review_model5349_canonical/vx2p0/task031-render-model5349-canonical-switch-vx2p0.mp4`.

## Review

Status: passed for canonical clean/switch video review. The videos support the
scoped Level B claim only; they do not resolve Level A forced persistent
dead-grid or Level C arbitrary onset failures.
