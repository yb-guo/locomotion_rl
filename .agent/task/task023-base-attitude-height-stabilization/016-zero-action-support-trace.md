# Subtask 016: Zero-Action Support Trace

## Route

- Follow up the task023 standing blocker by tracing zero-action hybrid dynamics
  from step 80 to 130.
- Do not run PPO.
- Do not change `GenesisG1SceneBackend`.
- Add a reusable trace tool that records:
  - root height, roll, pitch, upright;
  - mass-weighted link-position COM estimate;
  - foot net contact forces;
  - support polygon inferred from active ankle-roll-link contact geoms;
  - signed COM margin to the support polygon.

## Log

- 2026-05-14 Added
  `h200_locomotion_lab.tools.g1_zero_action_support_trace`.
  The tool generates the current `task023_hybrid` asset under the run dir,
  runs zero-action Genesis without resetting on failure, and writes
  `trace.jsonl`, `summary.json`, `config.json`, and `asset_resolution.json`.
- 2026-05-14 Local focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider \
  tests/test_g1_zero_action_support_trace.py \
  tests/test_g1_ppo_smoke.py \
  tests/test_g1_no_update_ppo_causality.py
Result: 28 passed, 4 skipped
```

- 2026-05-14 H200 focused verification after syncing the new tool:

```text
PYTHONPATH=src python3 -m pytest -p no:cacheprovider \
  tests/test_g1_zero_action_support_trace.py \
  tests/test_g1_ppo_smoke.py \
  tests/test_g1_no_update_ppo_causality.py
Result: 32 passed
```

- 2026-05-14 H200 zero-action support trace:

```text
run_id=h200-gpu1-hybrid-step080-130-v1
asset_variant=task023_hybrid
n_envs=1
start_step=80
end_step=130
root_z=1.20
termination_height_min=0.20

first_com_outside_support_step=77
first_height_bad_step=120
first_tilt_step=123
first_termination_height_bad_step=125
min_com_signed_margin=-0.6840 at step 124
mass_body_count=30
resolved_mass_body_count=30
support_model_point_counts:
  left_ankle_roll_link=12
  right_ankle_roll_link=12
unresolved_foot_links=[]
unresolved_mass_links=[]
```

- Key rows:

```text
step 80:
  root_pitch=0.3357 rad
  upright=0.9442
  COM=(0.1580, 0.0044, 0.7035)
  support area=0.05649
  COM margin=-0.0153
  foot forces: L=722.1, R=689.6

step 100:
  root_pitch=0.5410 rad
  upright=0.8572
  COM=(0.3044, 0.0095, 0.6853)
  support area=0.05510
  COM margin=-0.1619
  foot forces: L=689.5, R=625.7

step 110:
  root_pitch=0.7498 rad
  upright=0.7318
  COM=(0.4737, 0.0115, 0.6271)
  support area=0.04911
  COM margin=-0.3313
  foot forces: L=497.0, R=425.6

step 116:
  root_pitch=0.9488 rad
  upright=0.5826
  COM=(0.6239, 0.0129, 0.5287)
  support area=0.03859
  COM margin=-0.4812
  foot forces: L=258.8, R=154.9

step 120:
  root_pitch=1.1266 rad
  upright=0.4297
  COM=(0.7363, 0.0138, 0.4099)
  support area=0.00529
  COM margin=-0.5973
  foot forces: L=5.2, R=0.0
  height_bad_count=1

step 122:
  root_pitch=1.2348 rad
  upright=0.3297
  COM=(0.7919, 0.0142, 0.3297)
  active_foot_count=0
  support area=0.0
  tilt_bad_count=0

step 123:
  root_pitch=1.2948 rad
  upright=0.2725
  COM=(0.8191, 0.0143, 0.2843)
  active_foot_count=1
  support area=0.00264
  COM margin=-0.6718
  foot forces: L=79.0, R=0.0
  tilt_bad_count=1

step 125:
  termination_height_bad_count=1
  tilt_bad_count=1
```

## Review

Status: diagnostic_not_passed.

The passive failure is a forward pitch/COM-support problem, not a late scalar
height-threshold issue. The COM projection leaves the inferred support polygon
well before the reset event, while both feet initially remain in contact.
The robot then keeps pitching forward, contact force and support area collapse,
right-foot support disappears by step 120, no effective foot support remains at
step 122, and the tilt reset fires at step 123. Height termination follows
later at step 125.

Highest-value next fix is not longer PPO. It is to alter either the passive
support geometry/default pose so the zero-action COM projection starts and
stays inside the support polygon, or to add an explicit balance-control
objective/controller that arrests the forward pitch before step 77.
