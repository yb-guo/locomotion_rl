# Subtask 002: H200 Asset Report Evidence

## Route

- Sync the standalone report tool and tests into the H200 task021 project under
  `/root/agent_workspace/project`.
- Run focused tests through the guarded command.
- Run the JSON report against the prepared H200 G1 27DoF asset path from the
  27DoF profile.
- Record whether MJCF compiler/option/default/geom/contact fields are present.
- Do not run PPO, render, download assets, or write outside
  `/root/agent_workspace/project`.

## Log

- 2026-05-12 Remote project created under
  `/root/agent_workspace/project/h200-locomotion-lab-task021-genesis-alignment-bundle`
  by copying the task019 project directory. New task021 tool/tests/task docs
  were synchronized into that project.
- 2026-05-12 H200 focused tests through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_genesis_alignment_bundle.py
  tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q
  -p no:cacheprovider` -> 34 passed.
- 2026-05-12 H200 JSON report through guarded command:
  `PYTHONPATH=src python -m h200_locomotion_lab.tools.g1_genesis_alignment_bundle
  --output outputs/task021/genesis_alignment_bundle/h200_asset_report.json`;
  `python -m json.tool` validated the file on H200. Remote path:
  `/root/agent_workspace/project/h200-locomotion-lab-task021-genesis-alignment-bundle/outputs/task021/genesis_alignment_bundle/h200_asset_report.json`.
- 2026-05-12 H200 report summary:
  - `mapped_control_match=true`;
  - `xml_asset_present=true`;
  - `missing_count=11`;
  - Genesis timing self-consistent: `sim_dt_s=0.005`, `decimation=4`,
    `policy_rate_hz=50`, derived `50.0`;
  - MJCF asset compiler fields: `angle=radian`, `meshdir=meshes`;
  - MJCF has 6 `<default>` groups. Joint defaults include
    `armature=0.01`, `damping=0.05`, `frictionloss=0.2` for torso/leg/ankle/
    arm motors and `frictionloss=0.1` for wrist motors;
  - MJCF has no `<option>` element, no `<contact>` element, and no geom-level
    `friction/condim/solref/solimp/priority` fields extracted by the report.
- 2026-05-12 H200 missing records:
  `genesis_27dof_training_profile.contact_friction_solver_config`,
  `vectorized_genesis_backend.contact_friction_solver_config`,
  `contact_friction_solver.option`, `contact_friction_solver.contact`,
  `contact_friction_solver.geoms_with_contact_fields`,
  `contact_friction_solver.option.timestep`,
  SONIC profile timing fields, and MJCF decimation/policy-rate fields.

## Review

Status: passed. Final read-only reviewer accepted the H200 report evidence with
no blocking findings. Residual risk: the report proves the explicit fields are
absent from this MJCF/backend profile; it does not prove Genesis internal
defaults are correct for stable standing.
