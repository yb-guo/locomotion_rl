# Subtask 015: Base Frame Semantics Probe

## Route

- Check the user hypothesis that the base link should be waist or hip rather
  than the current root.
- Do not rewrite the MJCF kinematic root.
- First inspect the source MJCF root/free-joint placement.
- Then run a light Genesis link trace on the current best hybrid asset for
  pelvis, waist, torso, and hip links.

## Log

- 2026-05-13 Source MJCF inspection:

```text
worldbody top body: pelvis pos="0 0 0.793"
free joint: pelvis / floating_base_joint type="free"
left_hip_pitch_link pos relative to pelvis: 0 0.064452 -0.1027
right_hip_pitch_link pos relative to pelvis: 0 -0.064452 -0.1027
waist_yaw_link is a child of pelvis
torso_link is above waist_yaw_link / waist_roll_link
imu site is on torso_link
```

- 2026-05-13 H200 hybrid zero-action base-frame link trace:

```text
run_id=h200-gpu1-hybrid-base-frame-link-trace-v1
asset=ankle_roll_hybrid_edge_boxes_no_points
n_envs=256
steps=130
root_z=1.20
termination_height_min=0.20

first_tilt_step=123
max_tilt_bad_count=256
max_termination_height_bad_count=256
unresolved_links=[]
```

- 2026-05-13 Candidate-frame trace around collapse:

```text
step 120:
  pelvis/waist_z=0.4466
  torso_z=0.4733
  hip_pitch_z≈0.4026
  upright=0.4279
  tilt_bad=0

step 122:
  pelvis/waist_z=0.3579
  torso_z=0.3793
  hip_pitch_z≈0.3242
  upright=0.3276
  tilt_bad=0

step 123:
  pelvis/waist_z=0.3077
  torso_z=0.3261
  hip_pitch_z≈0.2799
  upright=0.2702
  tilt_bad=256

step 125:
  pelvis/waist_z=0.1958
  torso_z=0.2073
  hip_pitch_z≈0.1815
  upright=0.1399
  termination_height_bad=256
  tilt_bad=256
```

## Review

Status: completed as diagnosis.

The current physical floating base is already the pelvis. In this asset,
`waist_yaw_link` is effectively colocated with the pelvis root for height
semantics, while torso and hip links differ by only a small vertical offset
around the collapse window. Reinterpreting the base height as waist, torso, or
hip would not remove the primary failure: `upright` crosses the tilt threshold
before termination height matters. A true MJCF root rewrite from pelvis to
torso or hip would require kinematic-tree reparenting and would invalidate the
SONIC/MuJoCo-aligned asset contract; it is not the next root fix.
