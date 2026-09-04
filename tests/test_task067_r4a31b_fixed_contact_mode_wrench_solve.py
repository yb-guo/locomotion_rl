from __future__ import annotations

import json
from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    ContinuationRoute,
    _build_shard_for,
    _load_r4a3_records,
)
from h200_locomotion_lab.tools.whole_body_fixed_contact_mode_wrench_solve import (
    _DEFAULT_HORIZON_STEPS,
    _decide,
    _start_states_for_route,
    fixed_modes_from_start_states,
    rigid_contact_constraints_feasible,
    solve_fixed_contact_mode,
)

_R4A3_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_R4A31A_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31a_true_continuation_correctness_3fail.json"
)
_R4A31B_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31b_fixed_contact_mode_wrench_solve_3fail.json"
)


def _continuation_payload() -> dict:
    if not _R4A31A_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1a artifact is required for R4a.3.1b tests")
    return json.loads(_R4A31A_ARTIFACT.read_text(encoding="utf-8"))


def test_fixed_contact_modes_keep_both_feet_active() -> None:
    pytest.importorskip("mujoco")
    records = _load_r4a3_records(_R4A3_ARTIFACT, family="biped")
    shard, _ = _build_shard_for("biped", 0, 0.5)
    starts = _start_states_for_route(
        source_records=records,
        continuation_payload=_continuation_payload(),
        family="biped",
        route=ContinuationRoute(seed=0, start_range_fraction=0.0, end_range_fraction=0.5),
        shard=shard,
    )

    modes = fixed_modes_from_start_states(shard, starts)

    assert modes
    assert all(mode.is_double_foot for mode in modes)
    assert all(len(mode.foot_names) == 2 for mode in modes)


def test_candidate_contract_has_state_input_and_wrench_variables() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    records = _load_r4a3_records(_R4A3_ARTIFACT, family="biped")
    shard, _ = _build_shard_for("biped", 0, 0.5)
    starts = _start_states_for_route(
        source_records=records,
        continuation_payload=_continuation_payload(),
        family="biped",
        route=ContinuationRoute(seed=0, start_range_fraction=0.0, end_range_fraction=0.5),
        shard=shard,
    )
    mode = fixed_modes_from_start_states(shard, starts)[0]

    candidate = solve_fixed_contact_mode(
        shard,
        mode,
        starts[0],
        max_nfev=2,
        horizon_steps=1,
    )

    contract = candidate["variable_contract"]
    assert "actuated_joint_qpos" in contract["state_variables"]
    assert "actuator_position_ctrl" in contract["input_variables"]
    assert "per_selected_contact_point_vertical_normal_force" in contract["wrench_variables"]
    assert contract["force_variable_count"] == len(mode.points)
    assert contract["both_feet_forced_active"]


def test_rigid_constraint_feasibility_requires_both_feet_load_and_actual_input_bounds() -> None:
    candidate = {
        "rigid_static_constraints": {
            "root_wrench_norm": 0.0,
            "joint_tau_residual_max": 0.0,
            "selected_contact_height_abs_max": 0.0,
            "normal_force_sum_relative_error": 0.0,
            "load_deficit_sum": 1.0,
            "normal_force_sum": 100.0,
            "input": {
                "ctrl_range_violations": 0,
                "force_limit_violations": 0,
            },
        }
    }

    assert not rigid_contact_constraints_feasible(candidate)
    candidate["rigid_static_constraints"]["load_deficit_sum"] = 0.0
    candidate["rigid_static_constraints"]["input"]["force_limit_violations"] = 1
    assert not rigid_contact_constraints_feasible(candidate)


def test_solver_search_failure_is_not_promoted_to_physical_infeasible() -> None:
    summary = {
        "combined_source_records": 8,
        "combined_strict_contract_passed": 5,
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }

    decision = _decide(summary)

    assert decision["status"] == "r4a31b_fixed_contact_solver_incomplete_no_generator_certificate"
    assert "do not modify env, controller, generator, kp/kv" in decision["next_allowed_work"].lower()


def test_artifact_preserves_failure_as_diagnostic_not_physical_certificate() -> None:
    if not _R4A31B_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1b artifact is required for artifact regression")

    payload = json.loads(_R4A31B_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema"] == "task067_r4a31b_fixed_contact_mode_wrench_solve_v1"
    assert payload["assertions"]["all_modes_are_double_foot"]
    assert payload["assertions"]["search_failure_not_promoted_to_physical_infeasible"]
    assert payload["assertions"]["strict_acceptance_requires_actual_gate_and_2s_hold"]
    assert payload["provenance"]["parameters"]["horizon_steps"] == _DEFAULT_HORIZON_STEPS
