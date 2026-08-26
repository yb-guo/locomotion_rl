"""Task067 R4b-1 bounded feedback causality diagnosis.

This diagnostic consumes feasible R4a.2 equilibrium records and applies small
position-target deltas only inside private MuJoCo rollouts. It does not change
the public environment control chain, actuator gains, rewards, observations,
action schema, motor strength, latency, or failure semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    MorphologyGenerator,
    morphology_instance_key,
)
from h200_locomotion_lab.robots.whole_body_stance import (
    STANCE_SOLUTION_CONTRACT_HASH,
    STANCE_SOLUTION_CONTRACT_VERSION,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _center_of_mass,
    _com_cop_distance,
    _contact_report,
    _fall_reason,
    _projected_gravity,
    _reset_to_qpos,
    _roll_pitch,
    _velocity_impulse_perturbations,
)

_DEFAULT_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)
_TASK067_R4A2_HISTORICAL_CONTRACT_VERSION = "procedural_whole_body_v1_footpad_static_stance"
_TASK067_R4A2_HISTORICAL_CONTRACT_HASH = "37f1e0bce3af26db1d7f5499f01bf28ced9faa4621670f5aac501f6d0f354579"
_ALLOWED_SOURCE_REPLAY_CONTRACTS = {
    (
        PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    ): "current_runtime_contract",
    (
        _TASK067_R4A2_HISTORICAL_CONTRACT_VERSION,
        _TASK067_R4A2_HISTORICAL_CONTRACT_HASH,
    ): "task067_r4a2_historical_artifact_replay",
}
_GATE = {
    "nominal_biped_survival": 1.0,
    "perturb_survival_min": 45,
    "perturb_probe_count": 50,
    "actuator_saturation_events": 0,
    "non_foot_contact_steps": 0,
    "unloaded_foot_steps": 0,
    "controller_off_degraded_min": 4,
}


@dataclass(frozen=True, slots=True)
class FeedbackMode:
    name: str
    attitude: bool = False
    com_cop_oracle: bool = False

    def manifest(self) -> dict[str, bool | str]:
        return {
            "name": self.name,
            "attitude": self.attitude,
            "com_cop_oracle": self.com_cop_oracle,
        }


@dataclass(frozen=True, slots=True)
class FeedbackGain:
    name: str
    attitude_kp: float
    attitude_kd: float
    com_kp: float
    com_kd: float
    max_delta: float

    def manifest(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "attitude_kp": self.attitude_kp,
            "attitude_kd": self.attitude_kd,
            "com_kp": self.com_kp,
            "com_kd": self.com_kd,
            "max_delta": self.max_delta,
        }


@dataclass(frozen=True, slots=True)
class ReplayContractBinding:
    """Validated binding between a source artifact identity and this runtime.

    Task067 R4a/R4b diagnostics may replay historical artifact states, but the
    source artifact contract must stay visible and separate from the current
    public runtime contract.
    """

    replay_mode: str
    source_contract_version: str
    source_contract_hash: str
    runtime_contract_version: str
    runtime_contract_hash: str
    blueprint_hash: str
    physical_hash: str

    @property
    def source_matches_runtime(self) -> bool:
        return (
            self.source_contract_version == self.runtime_contract_version
            and self.source_contract_hash == self.runtime_contract_hash
        )

    def manifest(self) -> dict[str, str | bool]:
        return {
            "replay_mode": self.replay_mode,
            "source_contract_version": self.source_contract_version,
            "source_contract_hash": self.source_contract_hash,
            "runtime_contract_version": self.runtime_contract_version,
            "runtime_contract_hash": self.runtime_contract_hash,
            "source_matches_runtime": self.source_matches_runtime,
            "blueprint_hash": self.blueprint_hash,
            "physical_hash": self.physical_hash,
        }


_MODES = (
    FeedbackMode("hold_baseline"),
    FeedbackMode("attitude_only", attitude=True),
    FeedbackMode("com_cop_oracle", com_cop_oracle=True),
    FeedbackMode("attitude_com_combined", attitude=True, com_cop_oracle=True),
)
_GAIN_GRID = (
    FeedbackGain("bounded_low", attitude_kp=0.025, attitude_kd=0.004, com_kp=0.25, com_kd=0.020, max_delta=0.05),
    FeedbackGain("bounded_mid", attitude_kp=0.050, attitude_kd=0.008, com_kp=0.50, com_kd=0.040, max_delta=0.06),
    FeedbackGain("bounded_high", attitude_kp=0.080, attitude_kd=0.012, com_kp=0.80, com_kd=0.060, max_delta=0.08),
)
_BASELINE_GAIN = FeedbackGain("no_feedback", 0.0, 0.0, 0.0, 0.0, 0.0)


def _mode_gain_pairs() -> list[tuple[FeedbackMode, FeedbackGain]]:
    pairs: list[tuple[FeedbackMode, FeedbackGain]] = []
    for mode in _MODES:
        if not mode.attitude and not mode.com_cop_oracle:
            pairs.append((mode, _BASELINE_GAIN))
        else:
            pairs.extend((mode, gain) for gain in _GAIN_GRID)
    return pairs


def _load_feasible_records(path: Path, *, family: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in data["records"]
        if record["family"] == family and record["contact_equilibrium"]["status"] == "feasible"
    ]
    return sorted(records, key=lambda item: (float(item["range_fraction"]), int(item["seed"])))


def _record_instance_key(record: dict[str, Any]) -> dict[str, str]:
    key = record.get("morphology_instance_key")
    if not isinstance(key, dict):
        raise TypeError("Task067 replay record must contain a morphology_instance_key object")
    required = (
        "blueprint_hash",
        "physical_hash",
        "embodiment_contract_version",
        "embodiment_contract_hash",
    )
    missing = [field for field in required if field not in key]
    if missing:
        raise ValueError(f"Task067 replay record is missing instance-key fields: {missing}")
    result = {field: str(key[field]) for field in required}
    if any(not value for value in result.values()):
        raise ValueError("Task067 replay instance-key fields must be non-empty")
    return result


def _bind_replay_contract(
    record: dict[str, Any],
    *,
    actual_key: dict[str, str],
) -> ReplayContractBinding:
    source_key = _record_instance_key(record)
    for field in ("blueprint_hash", "physical_hash"):
        if actual_key[field] != source_key[field]:
            raise ValueError(
                "Task067 replay record does not match regenerated "
                f"{field} for {record['family']} seed={record['seed']}: "
                f"artifact={source_key[field]} runtime={actual_key[field]}"
            )

    source_contract = (
        source_key["embodiment_contract_version"],
        source_key["embodiment_contract_hash"],
    )
    replay_mode = _ALLOWED_SOURCE_REPLAY_CONTRACTS.get(source_contract)
    if replay_mode is None:
        raise ValueError(
            "Task067 replay record uses an unsupported embodiment contract: "
            f"{source_contract[0]} / {source_contract[1]}"
        )

    return ReplayContractBinding(
        replay_mode=replay_mode,
        source_contract_version=source_key["embodiment_contract_version"],
        source_contract_hash=source_key["embodiment_contract_hash"],
        runtime_contract_version=actual_key["embodiment_contract_version"],
        runtime_contract_hash=actual_key["embodiment_contract_hash"],
        blueprint_hash=actual_key["blueprint_hash"],
        physical_hash=actual_key["physical_hash"],
    )


def _build_shard_with_replay_binding(
    record: dict[str, Any],
) -> tuple[WholeBodyMuJoCoShard, ReplayContractBinding]:
    generator = MorphologyGenerator()
    seed = int(record["seed"])
    range_fraction = float(record["range_fraction"])
    blueprint = generator.generate(str(record["family"]), seed)
    physical = generator.sample_physical_params(blueprint, seed + 10_000_000, range_fraction=range_fraction)
    shard = WholeBodyMuJoCoShard(
        blueprint,
        physical=physical,
        num_envs=1,
        config=WholeBodyMuJoCoShardConfig(seed=seed),
    )
    actual_key = morphology_instance_key(blueprint, physical).manifest()
    binding = _bind_replay_contract(record, actual_key=actual_key)
    return shard, binding


def _build_shard(record: dict[str, Any]) -> WholeBodyMuJoCoShard:
    shard, _ = _build_shard_with_replay_binding(record)
    return shard


def _unique_replay_bindings(bindings: list[ReplayContractBinding]) -> list[dict[str, str | bool]]:
    unique: dict[tuple[str, str, str, str, str], dict[str, str | bool]] = {}
    for binding in bindings:
        manifest = binding.manifest()
        key = (
            binding.replay_mode,
            binding.source_contract_version,
            binding.source_contract_hash,
            binding.runtime_contract_version,
            binding.runtime_contract_hash,
        )
        unique.setdefault(key, manifest)
    return [unique[key] for key in sorted(unique)]


def _lower_body_actuator_ids(shard: WholeBodyMuJoCoShard) -> set[int]:
    return {
        int(actuator_id)
        for joint, actuator_id in zip(shard.blueprint.joints, shard._actuator_ids)
        if joint.semantic_slot.startswith("limb")
    }


def _limb_lateral_sign(joint_name: str, semantic_slot: str) -> float:
    name = joint_name.lower()
    if "left" in name:
        return 1.0
    if "right" in name:
        return -1.0
    try:
        limb = int(semantic_slot.split("_", 1)[0].replace("limb", ""))
    except ValueError:
        return 0.0
    return 1.0 if limb % 2 == 0 else -1.0


def _foot_load_threshold(shard: WholeBodyMuJoCoShard) -> float:
    total_mass = float(shard.np.sum(shard.model.body_mass))
    return 0.05 * total_mass * abs(float(shard.model.opt.gravity[2]))


def _controller_delta(
    shard: WholeBodyMuJoCoShard,
    data: Any,
    mode: FeedbackMode,
    gain: FeedbackGain,
) -> tuple[Any, dict[str, float | int]]:
    np = shard.np
    raw = np.zeros(shard.model.nu, dtype=np.float64)
    if not mode.attitude and not mode.com_cop_oracle:
        return raw, {"raw_max_abs": 0.0, "clipped_components": 0, "active_components": 0}

    gravity = _projected_gravity(tuple(float(value) for value in data.qpos[3:7]))
    roll_command = 0.0
    pitch_command = 0.0
    if mode.attitude:
        roll_command += -gain.attitude_kp * float(gravity[1]) - gain.attitude_kd * float(data.qvel[3])
        pitch_command += gain.attitude_kp * float(gravity[0]) - gain.attitude_kd * float(data.qvel[4])
    if mode.com_cop_oracle:
        contact = _contact_report(shard, data)
        cop = contact["center_of_pressure_xy"]
        if cop is not None:
            com = _center_of_mass(shard, data)
            roll_command += -gain.com_kp * (float(com[1]) - float(cop[1])) - gain.com_kd * float(data.qvel[1])
            pitch_command += -gain.com_kp * (float(com[0]) - float(cop[0])) - gain.com_kd * float(data.qvel[0])

    for joint, actuator_id in zip(shard.blueprint.joints, shard._actuator_ids):
        actuator_id = int(actuator_id)
        if actuator_id not in _lower_body_actuator_ids(shard):
            continue
        slot = joint.semantic_slot
        lateral = _limb_lateral_sign(joint.name, slot)
        delta = 0.0
        if slot.endswith("ankle_pitch"):
            delta += pitch_command
        elif slot.endswith("knee_pitch"):
            delta += -0.35 * pitch_command
        elif slot.endswith("hip_pitch"):
            delta += -0.55 * pitch_command
        elif slot.endswith("ankle_roll"):
            delta += lateral * roll_command
        elif slot.endswith("hip_roll"):
            delta += -0.55 * lateral * roll_command
        elif slot.endswith("hip_yaw"):
            delta += 0.15 * lateral * roll_command
        raw[actuator_id] = delta

    clipped = np.clip(raw, -gain.max_delta, gain.max_delta)
    return clipped, {
        "raw_max_abs": float(np.max(np.abs(raw))) if len(raw) else 0.0,
        "clipped_components": int(np.sum(np.abs(raw) > gain.max_delta + 1e-12)),
        "active_components": int(np.sum(np.abs(raw) > 1e-12)),
    }


def _apply_feedback_ctrl(
    shard: WholeBodyMuJoCoShard,
    data: Any,
    ctrl_eq: Any,
    mode: FeedbackMode,
    gain: FeedbackGain,
) -> dict[str, float | int]:
    delta, stats = _controller_delta(shard, data, mode, gain)
    ctrl = shard.np.asarray(ctrl_eq, dtype=shard.np.float64) + delta
    for actuator_id in shard._actuator_ids:
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[int(actuator_id)])
        ctrl[int(actuator_id)] = min(upper, max(lower, float(ctrl[int(actuator_id)])))
    data.ctrl[:] = ctrl
    stats["max_abs_delta"] = float(shard.np.max(shard.np.abs(delta))) if len(delta) else 0.0
    return stats


def _trace_state(data: Any) -> list[float]:
    roll, pitch = _roll_pitch(data.qpos[3:7])
    return [
        float(data.qpos[0]),
        float(data.qpos[1]),
        float(roll),
        float(pitch),
        float(data.qvel[0]),
        float(data.qvel[1]),
        float(data.qvel[3]),
        float(data.qvel[4]),
    ]


def run_feedback_rollout(
    shard: WholeBodyMuJoCoShard,
    equilibrium: dict[str, Any],
    mode: FeedbackMode,
    gain: FeedbackGain,
    *,
    horizon_steps: int,
    perturbation: dict[str, Any] | None = None,
    disable_after_steps: int | None = None,
) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    qpos_eq = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl_eq = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    _reset_to_qpos(shard, data, qpos_eq)
    data.ctrl[:] = ctrl_eq
    impulse: dict[str, Any] | None = None
    if perturbation is not None:
        if perturbation["kind"] == "velocity":
            data.qvel[int(perturbation["dof"])] += float(perturbation["value"])
        elif perturbation["kind"] == "impulse":
            impulse = perturbation
        else:
            raise ValueError(f"unknown perturbation kind: {perturbation['kind']}")
        shard.mujoco.mj_forward(shard.model, data)

    trace: list[list[float]] = []
    foot_load_threshold = _foot_load_threshold(shard)
    unloaded_foot_steps = 0
    non_foot_contact_steps = 0
    saturation_events = 0
    clipping_events = 0
    active_delta_events = 0
    controller_substeps = 0
    max_abs_delta = 0.0
    raw_max_abs_delta = 0.0
    actuator_force_max = 0.0
    first_fall_step = None
    first_fall_reason = None
    survived_on_phase = True
    for step_index in range(horizon_steps):
        controller_enabled = disable_after_steps is None or step_index < disable_after_steps
        for substep_index in range(shard.config.substeps):
            data.qfrc_applied[:] = 0.0
            if impulse is not None and step_index == 0 and substep_index == 0:
                data.qfrc_applied[int(impulse["dof"])] = float(impulse["impulse"]) / shard.model.opt.timestep
            if controller_enabled:
                stats = _apply_feedback_ctrl(shard, data, ctrl_eq, mode, gain)
                clipping_events += int(stats["clipped_components"])
                active_delta_events += int(stats["active_components"])
                max_abs_delta = max(max_abs_delta, float(stats["max_abs_delta"]))
                raw_max_abs_delta = max(raw_max_abs_delta, float(stats["raw_max_abs"]))
                controller_substeps += 1
            else:
                data.ctrl[:] = ctrl_eq
            shard.mujoco.mj_forward(shard.model, data)
            for actuator_id in shard._actuator_ids:
                force = abs(float(data.actuator_force[int(actuator_id)]))
                actuator_force_max = max(actuator_force_max, force)
                limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[int(actuator_id)])
                saturation_events += int(force >= 0.995 * limit)
            shard.mujoco.mj_step(shard.model, data)

        contact = _contact_report(shard, data)
        non_foot_contact_steps += int(contact["non_foot_contacts"] > 0)
        if any(float(contact["normal_force_by_foot"].get(name, 0.0)) < foot_load_threshold for name in shard._foot_geoms):
            unloaded_foot_steps += 1
        trace.append(_trace_state(data))
        reason = _fall_reason(shard, data)
        if reason is not None:
            first_fall_step = step_index + 1
            first_fall_reason = reason
            if disable_after_steps is not None and first_fall_step <= disable_after_steps:
                survived_on_phase = False
            break

    clipping_denominator = max(1, controller_substeps * len(shard._actuator_ids))
    final_contact = _contact_report(shard, data)
    final_com = _center_of_mass(shard, data)
    return {
        "mode": mode.name,
        "gain": gain.manifest(),
        "perturbation": perturbation,
        "disable_after_steps": disable_after_steps,
        "survived": first_fall_step is None,
        "first_fall_step": first_fall_step,
        "first_fall_reason": first_fall_reason,
        "survived_on_phase": survived_on_phase,
        "steps_run": len(trace),
        "trace": trace,
        "actuator_saturation_events": saturation_events,
        "actuator_force_max": actuator_force_max,
        "non_foot_contact_steps": non_foot_contact_steps,
        "unloaded_foot_steps": unloaded_foot_steps,
        "controller_clipping_ratio": clipping_events / clipping_denominator,
        "controller_active_ratio": active_delta_events / clipping_denominator,
        "controller_max_abs_delta": max_abs_delta,
        "controller_raw_max_abs_delta": raw_max_abs_delta,
        "com_cop_distance_final": _com_cop_distance(final_com, final_contact),
        "final_contact": final_contact,
    }


def paired_early_growth(nominal: dict[str, Any], perturbed: dict[str, Any], *, window_steps: int = 25) -> dict[str, Any]:
    np_errors = []
    for nominal_state, perturbed_state in zip(nominal["trace"], perturbed["trace"]):
        error = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(nominal_state, perturbed_state)))
        np_errors.append(error)
    if not np_errors:
        return {"initial_error": 0.0, "window_error": 0.0, "growth_ratio": float("inf"), "response": "fell"}
    initial_index = min(1, len(np_errors) - 1)
    window_index = min(max(0, window_steps - 1), len(np_errors) - 1)
    initial_error = max(np_errors[initial_index], 1e-12)
    window_error = np_errors[window_index]
    ratio = window_error / initial_error
    if not perturbed["survived"]:
        response = "fell"
    elif ratio <= 0.8:
        response = "decayed"
    elif ratio >= 1.2:
        response = "grew"
    else:
        response = "neutral"
    return {
        "initial_error": float(initial_error),
        "window_error": float(window_error),
        "growth_ratio": float(ratio),
        "response": response,
        "window_steps": window_steps,
    }


def _record_label(record: dict[str, Any]) -> str:
    return f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"


def _nominal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"seeds": 0}
    falls = [row for row in rows if not row["rollout"]["survived"]]
    return {
        "seeds": len(rows),
        "survived": len(rows) - len(falls),
        "survival_ratio": (len(rows) - len(falls)) / len(rows),
        "first_fall_step_min": min((row["rollout"]["first_fall_step"] for row in falls), default=None),
        "actuator_saturation_events": sum(row["rollout"]["actuator_saturation_events"] for row in rows),
        "non_foot_contact_steps": sum(row["rollout"]["non_foot_contact_steps"] for row in rows),
        "unloaded_foot_steps": sum(row["rollout"]["unloaded_foot_steps"] for row in rows),
        "controller_clipping_ratio_max": max(row["rollout"]["controller_clipping_ratio"] for row in rows),
        "controller_max_abs_delta_max": max(row["rollout"]["controller_max_abs_delta"] for row in rows),
        "actuator_force_max": max(row["rollout"]["actuator_force_max"] for row in rows),
    }


def _perturbation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"probes": 0}
    survived = sum(1 for row in rows if row["rollout"]["survived"])
    return {
        "probes": len(rows),
        "survived": survived,
        "fell": len(rows) - survived,
        "grew": sum(1 for row in rows if row["paired_growth"]["response"] == "grew"),
        "decayed": sum(1 for row in rows if row["paired_growth"]["response"] == "decayed"),
        "neutral": sum(1 for row in rows if row["paired_growth"]["response"] == "neutral"),
        "growth_ratio_median": statistics.median(row["paired_growth"]["growth_ratio"] for row in rows),
        "actuator_saturation_events": sum(row["rollout"]["actuator_saturation_events"] for row in rows),
        "non_foot_contact_steps": sum(row["rollout"]["non_foot_contact_steps"] for row in rows),
        "unloaded_foot_steps": sum(row["rollout"]["unloaded_foot_steps"] for row in rows),
    }


def _select_gain_for_mode(summary: dict[str, Any], family: str, mode_name: str) -> str:
    candidates = [
        (key, value)
        for key, value in summary.items()
        if key.startswith(f"{family}:{mode_name}:") and value.get("seeds", 0) > 0
    ]
    if not candidates:
        return "no_feedback"
    key, _ = max(
        candidates,
        key=lambda item: (
            item[1]["survived"],
            -item[1]["actuator_saturation_events"],
            -item[1]["non_foot_contact_steps"],
            -item[1]["unloaded_foot_steps"],
            -item[1]["controller_clipping_ratio_max"],
        ),
    )
    return key.split(":")[-2]


def decide_r4b1(summary: dict[str, Any], selected: dict[str, str]) -> dict[str, Any]:
    combined_gain = selected.get("attitude_com_combined", "no_feedback")
    combined_nominal = summary.get(f"biped:attitude_com_combined:{combined_gain}:nominal", {})
    combined_perturb = summary.get(f"biped:attitude_com_combined:{combined_gain}:perturbations", {})
    combined_quad = summary.get(f"quadruped:attitude_com_combined:{combined_gain}:nominal", {})
    causal = summary.get(f"biped:attitude_com_combined:{combined_gain}:controller_off", {})
    deployable_pass = (
        combined_nominal.get("seeds") == 5
        and combined_nominal.get("survived") == 5
        and combined_perturb.get("probes") == _GATE["perturb_probe_count"]
        and combined_perturb.get("survived", 0) >= _GATE["perturb_survival_min"]
        and combined_nominal.get("actuator_saturation_events") == 0
        and combined_perturb.get("actuator_saturation_events") == 0
        and combined_nominal.get("non_foot_contact_steps") == 0
        and combined_perturb.get("non_foot_contact_steps") == 0
        and combined_nominal.get("unloaded_foot_steps") == 0
        and combined_perturb.get("unloaded_foot_steps") == 0
        and combined_quad.get("survived") == combined_quad.get("seeds")
        and causal.get("off_degraded", 0) >= _GATE["controller_off_degraded_min"]
    )
    if deployable_pass:
        return {
            "status": "deployable_bounded_feedback_passed",
            "decision": "Bounded feedback satisfies the R4b-1 causality and safety gate.",
            "next_allowed_work": "Design ZeroActionHoldSolution and upgrade embodiment contract/hash before integration.",
        }

    oracle_gain = selected.get("com_cop_oracle", "no_feedback")
    attitude_gain = selected.get("attitude_only", "no_feedback")
    oracle_nominal = summary.get(f"biped:com_cop_oracle:{oracle_gain}:nominal", {})
    oracle_perturb = summary.get(f"biped:com_cop_oracle:{oracle_gain}:perturbations", {})
    attitude_nominal = summary.get(f"biped:attitude_only:{attitude_gain}:nominal", {})
    oracle_pass = (
        oracle_nominal.get("survived") == oracle_nominal.get("seeds") == 5
        and oracle_perturb.get("probes") == _GATE["perturb_probe_count"]
        and oracle_perturb.get("survived", 0) >= _GATE["perturb_survival_min"]
    )
    attitude_pass = attitude_nominal.get("survived") == attitude_nominal.get("seeds") == 5
    if oracle_pass and not attitude_pass:
        return {
            "status": "oracle_passes_deployable_observation_missing",
            "decision": "COM/COP oracle stabilizes the feasible equilibria, but projected-gravity feedback does not.",
            "next_allowed_work": "Do not integrate; diagnose deployable observable state for R4b before ZeroActionHoldSolution.",
        }
    return {
        "status": "bounded_feedback_gate_failed",
        "decision": "The tested bounded feedback grid does not satisfy the R4b-1 survival and perturbation gate.",
        "next_allowed_work": "Continue Task067 R4b diagnostics only; Task061/Task062 remain blocked.",
    }


def run_diagnosis(
    *,
    input_json: Path,
    horizon_steps: int,
    off_step: int,
) -> dict[str, Any]:
    biped_records = _load_feasible_records(input_json, family="biped")
    quadruped_records = _load_feasible_records(input_json, family="quadruped")
    nominal_rows: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    summary: dict[str, Any] = {}
    all_records = {"biped": biped_records, "quadruped": quadruped_records}
    shards: dict[tuple[str, float, int], WholeBodyMuJoCoShard] = {}
    replay_bindings: list[ReplayContractBinding] = []
    for records in all_records.values():
        for record in records:
            shard, binding = _build_shard_with_replay_binding(record)
            shards[(record["family"], float(record["range_fraction"]), int(record["seed"]))] = shard
            replay_bindings.append(binding)

    for mode, gain in _mode_gain_pairs():
        for family, records in all_records.items():
            rows = []
            for record in records:
                shard = shards[(record["family"], float(record["range_fraction"]), int(record["seed"]))]
                rollout = run_feedback_rollout(
                    shard,
                    record["contact_equilibrium"],
                    mode,
                    gain,
                    horizon_steps=horizon_steps,
                )
                row = {
                    "label": _record_label(record),
                    "family": family,
                    "mode": mode.name,
                    "gain": gain.name,
                    "rollout": rollout,
                }
                rows.append(row)
                nominal_rows.append(row)
            summary[f"{family}:{mode.name}:{gain.name}:nominal"] = _nominal_summary(rows)

    for mode in _MODES:
        selected[mode.name] = _select_gain_for_mode(summary, "biped", mode.name)

    perturbation_rows: list[dict[str, Any]] = []
    for mode in _MODES:
        gain_name = selected[mode.name]
        gain = _BASELINE_GAIN if gain_name == _BASELINE_GAIN.name else next(item for item in _GAIN_GRID if item.name == gain_name)
        biped_nominal_by_label = {
            row["label"]: row["rollout"]
            for row in nominal_rows
            if row["family"] == "biped" and row["mode"] == mode.name and row["gain"] == gain.name
        }
        mode_perturb_rows = []
        for record in biped_records:
            shard = shards[(record["family"], float(record["range_fraction"]), int(record["seed"]))]
            label = _record_label(record)
            nominal = biped_nominal_by_label[label]
            for perturbation in _velocity_impulse_perturbations():
                rollout = run_feedback_rollout(
                    shard,
                    record["contact_equilibrium"],
                    mode,
                    gain,
                    horizon_steps=horizon_steps,
                    perturbation=perturbation,
                )
                row = {
                    "label": label,
                    "mode": mode.name,
                    "gain": gain.name,
                    "perturbation": perturbation,
                    "paired_growth": paired_early_growth(nominal, rollout),
                    "rollout": rollout,
                }
                perturbation_rows.append(row)
                mode_perturb_rows.append(row)
        summary[f"biped:{mode.name}:{gain.name}:perturbations"] = _perturbation_summary(mode_perturb_rows)

    controller_off_rows = []
    combined_gain_name = selected["attitude_com_combined"]
    combined_gain = (
        _BASELINE_GAIN
        if combined_gain_name == _BASELINE_GAIN.name
        else next(item for item in _GAIN_GRID if item.name == combined_gain_name)
    )
    combined_mode = next(item for item in _MODES if item.name == "attitude_com_combined")
    for record in biped_records:
        shard = shards[(record["family"], float(record["range_fraction"]), int(record["seed"]))]
        rollout = run_feedback_rollout(
            shard,
            record["contact_equilibrium"],
            combined_mode,
            combined_gain,
            horizon_steps=horizon_steps,
            disable_after_steps=off_step,
        )
        controller_off_rows.append({"label": _record_label(record), "rollout": rollout})
    summary[f"biped:attitude_com_combined:{combined_gain.name}:controller_off"] = {
        "seeds": len(controller_off_rows),
        "on_phase_survived": sum(1 for row in controller_off_rows if row["rollout"]["survived_on_phase"]),
        "off_degraded": sum(
            1
            for row in controller_off_rows
            if row["rollout"]["survived_on_phase"] and not row["rollout"]["survived"]
        ),
        "first_fall_step_min": min(
            (
                row["rollout"]["first_fall_step"]
                for row in controller_off_rows
                if row["rollout"]["first_fall_step"] is not None
            ),
            default=None,
        ),
    }

    payload = {
        "schema": "task067_r4b1_bounded_feedback_causality_v1",
        "source_equilibrium_artifact": str(input_json),
        "source_replay_contracts": _unique_replay_bindings(replay_bindings),
        "runtime_contract": {
            "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
            "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "horizon_steps": horizon_steps,
        "controller_off_step": off_step,
        "gate": _GATE,
        "modes": [mode.manifest() for mode in _MODES],
        "gain_grid": [gain.manifest() for gain in _GAIN_GRID],
        "selected_gains": selected,
        "summary": summary,
        "decision": decide_r4b1(summary, selected),
        "feasible_source_counts": {
            "biped": len(biped_records),
            "quadruped": len(quadruped_records),
        },
        "nominal_rows": nominal_rows,
        "perturbation_rows": perturbation_rows,
        "controller_off_rows": controller_off_rows,
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=_DEFAULT_R4A2_ARTIFACT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--off-step", type=int, default=50)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = run_diagnosis(input_json=args.input_json, horizon_steps=args.horizon_steps, off_step=args.off_step)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "selected_gains": payload["selected_gains"], "summary": payload["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
