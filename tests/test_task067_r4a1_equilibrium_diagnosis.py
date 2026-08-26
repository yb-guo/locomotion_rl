from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
)
from h200_locomotion_lab.robots.procedural_morphology import MorphologyGenerator
from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _build_shard,
    _load_feasible_records,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _EQUILIBRIUM_THRESHOLDS,
    _classify_perturbation_response,
    _equilibrium_candidate_feasible,
    _velocity_impulse_perturbations,
    decide,
    solve_contact_consistent_equilibrium,
    summarize,
)

_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)


def _seed_zero_historical_record() -> dict:
    if not _R4A2_ARTIFACT.exists():
        pytest.fail("Task067 R4a.2 evidence artifact is required, not optional")
    return next(
        item
        for item in _load_feasible_records(_R4A2_ARTIFACT, family="biped")
        if float(item["range_fraction"]) == 0.0 and int(item["seed"]) == 0
    )


def test_fall_height_threshold_uses_physical_global_scale() -> None:
    pytest.importorskip("mujoco")
    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 0)
    base_physical = generator.sample_physical_params(blueprint, 10_000_000, range_fraction=0.0)
    physical = replace(base_physical, global_scale=1.25)
    config = WholeBodyMuJoCoShardConfig(fall_height_fraction=0.35)
    shard = WholeBodyMuJoCoShard(blueprint, physical=physical, num_envs=1, config=config)
    data = shard.data[0]

    unscaled = blueprint.nominal_height * config.fall_height_fraction
    scaled = unscaled * physical.global_scale
    assert shard._fall_height_threshold() == pytest.approx(scaled)

    data.qpos[2] = 0.5 * (unscaled + scaled)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    assert shard._is_fallen(data)

    data.qpos[2] = scaled + 0.02
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    assert not shard._is_fallen(data)


def test_r4a2_historical_artifact_contains_known_biped_seed0_solution() -> None:
    pytest.importorskip("mujoco")
    record = _seed_zero_historical_record()
    shard = _build_shard(record)
    equilibrium = record["contact_equilibrium"]

    assert equilibrium["schema"] == "contact_consistent_joint_aware_dynamic_equilibrium_v3"
    assert equilibrium["status"] == "feasible"
    assert equilibrium["feasible_candidate_count"] > 0
    assert equilibrium["candidate_count"] > 0
    best = equilibrium["best"]
    assert best["solver_kind"] == "joint_aware_least_squares_with_slsqp_contact_qp"
    assert _equilibrium_candidate_feasible(shard, best)
    assert "mj_forward" in best
    assert best["mj_forward"]["qacc_root_norm"] <= _EQUILIBRIUM_THRESHOLDS["qacc_root_norm"]
    assert best["mj_forward"]["qacc_joint_max"] <= _EQUILIBRIUM_THRESHOLDS["qacc_joint_max"]
    assert best["mj_forward"]["contact"]["non_foot_contacts"] == 0
    assert best["mj_forward"]["actuator_saturation_events"] == 0
    assert best["max_joint_adjustment"] <= _EQUILIBRIUM_THRESHOLDS["max_joint_adjustment"]
    force_solution = best["static"]["contact_force_solution"]
    assert force_solution["method"] == "slsqp_qp"
    assert "normal_force_by_foot" in force_solution
    assert "joint_force" in force_solution
    assert force_solution["min_foot_load"] > 0.0
    assert all(value >= force_solution["min_foot_load"] for value in force_solution["normal_force_by_foot"].values())


def test_r4a2_quadruped_positive_control_uses_same_solver() -> None:
    pytest.importorskip("mujoco")
    generator = MorphologyGenerator()
    blueprint = generator.generate("quadruped", 0)
    physical = generator.sample_physical_params(blueprint, 10_000_000, range_fraction=0.0)
    shard = WholeBodyMuJoCoShard(blueprint, physical=physical, num_envs=1)

    equilibrium = solve_contact_consistent_equilibrium(shard)

    assert equilibrium["schema"] == "contact_consistent_joint_aware_dynamic_equilibrium_v3"
    assert equilibrium["status"] == "feasible"
    assert _equilibrium_candidate_feasible(shard, equilibrium["best"])


def _rollout(mode: str, *, survived: bool = True, skipped: bool = False) -> dict[str, object]:
    if skipped:
        return {
            "mode": mode,
            "skipped": True,
            "skip_reason": "no_feasible_equilibrium",
            "equilibrium_feasible": False,
        }
    return {
        "mode": mode,
        "skipped": False,
        "equilibrium_feasible": mode.startswith("contact_equilibrium"),
        "survived": survived,
        "first_fall_step": None if survived else 7,
        "tilt_max_rad": 0.1,
        "base_angular_acc_max": 0.2,
        "com_cop_distance_median": 0.01,
        "actuator_saturation_events": 0,
    }


def _equilibrium(status: str) -> dict[str, object]:
    return {
        "status": status,
        "feasible_candidate_count": 1 if status == "feasible" else 0,
        "best": {
            "static": {
                "root_residual_force_norm": 0.0,
                "root_residual_torque_norm": 0.0,
                "support_bottom_abs_max": 0.0,
                "joint_force": {
                    "over_force_limit": 0,
                    "out_of_ctrl_range": 0,
                },
                "contact_force_solution": {
                    "residual_norm": 0.0,
                    "foot_load_deficit": 0.0 if status == "feasible" else 1.0,
                },
            },
            "mj_forward": {
                "qacc_root_norm": 0.1,
                "qacc_joint_max": 1.0,
                "contact": {
                    "non_foot_contacts": 0,
                },
                "actuator_saturation_events": 0,
                "actuator_force_max": 1.0,
            },
            "max_joint_adjustment": 0.0,
            "actual_ctrl_range": {
                "violations": 0,
            },
        },
    }


def test_r4a1_summary_uses_feasible_equilibrium_only_for_hold_and_perturbations() -> None:
    records = [
        {
            "family": "biped",
            "range_fraction": 0.0,
            "contact_equilibrium": _equilibrium("feasible"),
            "rollouts": [_rollout("baseline"), _rollout("contact_equilibrium_hold")],
            "perturbation_rollouts": [
                {
                    **_rollout("contact_equilibrium_perturbed"),
                    "response": "decayed",
                }
            ],
        },
        {
            "family": "biped",
            "range_fraction": 0.0,
            "contact_equilibrium": _equilibrium("infeasible"),
            "rollouts": [_rollout("baseline"), _rollout("contact_equilibrium_hold", skipped=True)],
            "perturbation_rollouts": [],
        },
    ]

    summary = summarize(records)

    assert summary["biped:rf0:contact_equilibrium"]["seeds"] == 2
    assert summary["biped:rf0:contact_equilibrium"]["feasible"] == 1
    hold = summary["biped:rf0:contact_equilibrium_hold_feasible"]
    assert hold["seeds"] == 1
    assert hold["skipped"] == 1
    assert hold["fall_ratio"] == 0.0
    perturb = summary["biped:rf0:contact_equilibrium_perturbations_feasible"]
    assert perturb["probes"] == 1
    assert perturb["decayed"] == 1


def test_r4a1_candidate_feasibility_requires_forward_gate_and_each_foot_loaded() -> None:
    class FakeShard:
        _foot_geoms = ("left_foot", "right_foot")

    candidate = {
        "static": {
            "root_residual_force_norm": 0.0,
            "root_residual_torque_norm": 0.0,
            "support_bottom_abs_max": 0.0,
            "joint_force": {
                "over_force_limit": 0,
                "out_of_ctrl_range": 0,
            },
            "contact_force_solution": {
                "status": "feasible",
                "normal_force_by_foot": {"left_foot": 12.0, "right_foot": 13.0},
                "min_foot_load": 10.0,
            },
        },
        "mj_forward": {
            "qacc_root_norm": 0.5,
            "qacc_joint_max": 5.0,
            "contact": {
                "contacts_by_foot": {"left_foot": 1, "right_foot": 1},
                "non_foot_contacts": 0,
            },
            "actuator_saturation_events": 0,
        },
        "actual_ctrl_range": {
            "violations": 0,
            "max_violation": 0.0,
        },
    }

    assert _equilibrium_candidate_feasible(FakeShard(), candidate)

    candidate["mj_forward"]["qacc_root_norm"] = _EQUILIBRIUM_THRESHOLDS["qacc_root_norm"] + 0.1
    assert not _equilibrium_candidate_feasible(FakeShard(), candidate)
    candidate["mj_forward"]["qacc_root_norm"] = 0.5

    candidate["static"]["contact_force_solution"]["normal_force_by_foot"]["right_foot"] = 0.0
    assert not _equilibrium_candidate_feasible(FakeShard(), candidate)
    candidate["static"]["contact_force_solution"]["normal_force_by_foot"]["right_foot"] = 13.0

    candidate["mj_forward"]["contact"]["contacts_by_foot"]["right_foot"] = 0
    assert not _equilibrium_candidate_feasible(FakeShard(), candidate)
    candidate["mj_forward"]["contact"]["contacts_by_foot"]["right_foot"] = 1

    candidate["mj_forward"]["contact"]["non_foot_contacts"] = 1
    assert not _equilibrium_candidate_feasible(FakeShard(), candidate)


def test_r4a1_velocity_and_impulse_perturbation_semantics() -> None:
    perturbations = _velocity_impulse_perturbations()
    assert {item["kind"] for item in perturbations} == {"velocity", "impulse"}

    assert _classify_perturbation_response(survived=False, initial_error=1.0, final_error=0.0) == "fell"
    assert _classify_perturbation_response(survived=True, initial_error=1.0, final_error=0.5) == "decayed"
    assert _classify_perturbation_response(survived=True, initial_error=1.0, final_error=1.3) == "grew"
    assert _classify_perturbation_response(survived=True, initial_error=1.0, final_error=1.0) == "neutral"


def test_r4a2_decision_uses_quadruped_positive_control_without_requiring_every_quad_seed() -> None:
    summary = {
        "biped:rf0:baseline": {"fall_ratio": 1.0},
        "biped:rf0.5:baseline": {"fall_ratio": 1.0},
        "biped:rf0:zero_gravity": {"fall_ratio": 0.0},
        "biped:rf0.5:zero_gravity": {"fall_ratio": 0.0},
        "biped:rf0:root_locked": {"fall_ratio": 0.0},
        "biped:rf0.5:root_locked": {"fall_ratio": 0.0},
        "biped:rf0:contact_equilibrium": {"seeds": 4, "feasible": 3},
        "biped:rf0.5:contact_equilibrium": {"seeds": 4, "feasible": 2},
        "biped:rf0:contact_equilibrium_hold_feasible": {"seeds": 3, "fall_ratio": 1.0},
        "biped:rf0.5:contact_equilibrium_hold_feasible": {"seeds": 2, "fall_ratio": 1.0},
        "biped:rf0:contact_equilibrium_perturbations_feasible": {"fell": 30, "grew": 0},
        "biped:rf0.5:contact_equilibrium_perturbations_feasible": {"fell": 20, "grew": 0},
        "quadruped:rf0:contact_equilibrium": {"seeds": 4, "feasible": 3},
        "quadruped:rf0.5:contact_equilibrium": {"seeds": 4, "feasible": 1},
        "quadruped:rf0:contact_equilibrium_hold_feasible": {"seeds": 3, "fall_ratio": 0.0},
        "quadruped:rf0.5:contact_equilibrium_hold_feasible": {"seeds": 1, "fall_ratio": 0.0},
    }

    decision = decide(summary)

    assert decision["status"] == "equilibrium_exists_but_perturbation_diverges"
