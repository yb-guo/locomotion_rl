from __future__ import annotations

from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _GAIN_GRID,
    _MODES,
    _apply_feedback_ctrl,
    _build_shard,
    _load_feasible_records,
    _lower_body_actuator_ids,
    decide_r4b1,
    paired_early_growth,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import _reset_to_qpos

_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)


def test_r4b1_feedback_delta_is_bounded_and_lower_body_only() -> None:
    pytest.importorskip("mujoco")
    if not _R4A2_ARTIFACT.exists():
        pytest.skip("R4a.2 artifact is required for R4b-1 controller-bound regression")
    record = next(item for item in _load_feasible_records(_R4A2_ARTIFACT, family="biped") if item["seed"] == 0)
    shard = _build_shard(record)
    data = shard.data[0]
    equilibrium = record["contact_equilibrium"]
    qpos_eq = shard.np.asarray(equilibrium["best"]["qpos"], dtype=shard.np.float64)
    ctrl_eq = shard.np.asarray(equilibrium["best"]["ctrl"], dtype=shard.np.float64)
    _reset_to_qpos(shard, data, qpos_eq)
    data.ctrl[:] = ctrl_eq

    mode = next(item for item in _MODES if item.name == "attitude_com_combined")
    gain = next(item for item in _GAIN_GRID if item.name == "bounded_high")
    _apply_feedback_ctrl(shard, data, ctrl_eq, mode, gain)
    delta = shard.np.asarray(data.ctrl, dtype=shard.np.float64) - ctrl_eq
    lower_body = _lower_body_actuator_ids(shard)

    assert max(abs(float(value)) for value in delta) <= gain.max_delta + 1e-12
    assert all(abs(float(delta[index])) <= 1e-12 for index in range(shard.model.nu) if index not in lower_body)


def test_r4b1_paired_growth_uses_nominal_trajectory_difference() -> None:
    nominal = {"trace": [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]], "survived": True}
    perturbed_growing = {"trace": [[0.0, 0.0], [0.11, 0.0], [0.25, 0.0], [0.5, 0.0]], "survived": True}
    perturbed_decaying = {"trace": [[0.0, 0.0], [0.2, 0.0], [0.25, 0.0], [0.32, 0.0]], "survived": True}

    assert paired_early_growth(nominal, perturbed_growing, window_steps=4)["response"] == "grew"
    assert paired_early_growth(nominal, perturbed_decaying, window_steps=4)["response"] == "decayed"


def test_r4b1_decision_requires_full_gate_and_controller_off_degradation() -> None:
    summary = {
        "biped:attitude_com_combined:bounded_low:nominal": {
            "seeds": 5,
            "survived": 5,
            "actuator_saturation_events": 0,
            "non_foot_contact_steps": 0,
            "unloaded_foot_steps": 0,
        },
        "biped:attitude_com_combined:bounded_low:perturbations": {
            "probes": 50,
            "survived": 50,
            "actuator_saturation_events": 0,
            "non_foot_contact_steps": 0,
            "unloaded_foot_steps": 0,
        },
        "quadruped:attitude_com_combined:bounded_low:nominal": {"seeds": 4, "survived": 4},
        "biped:attitude_com_combined:bounded_low:controller_off": {"off_degraded": 0},
        "biped:com_cop_oracle:bounded_low:nominal": {"seeds": 5, "survived": 5},
        "biped:com_cop_oracle:bounded_low:perturbations": {"probes": 50, "survived": 50},
        "biped:attitude_only:bounded_low:nominal": {"seeds": 5, "survived": 4},
    }
    selected = {
        "attitude_com_combined": "bounded_low",
        "com_cop_oracle": "bounded_low",
        "attitude_only": "bounded_low",
    }

    decision = decide_r4b1(summary, selected)

    assert decision["status"] == "oracle_passes_deployable_observation_missing"
    summary["biped:attitude_com_combined:bounded_low:controller_off"]["off_degraded"] = 4
    assert decide_r4b1(summary, selected)["status"] == "deployable_bounded_feedback_passed"
