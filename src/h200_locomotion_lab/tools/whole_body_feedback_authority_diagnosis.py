"""Task067 R4b-2 bounded feedback authority diagnosis.

This tool diagnoses why the R4b-1 private bounded feedback grid failed. It
keeps the R4a.2 equilibrium pose and ctrl fixed, probes only bounded lower-body
target deltas, and records whether the tested mapping has local restoring
authority before any public controller integration is considered.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
)
from h200_locomotion_lab.robots.whole_body_stance import (
    STANCE_SOLUTION_CONTRACT_HASH,
    STANCE_SOLUTION_CONTRACT_VERSION,
)
from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _BASELINE_GAIN,
    _GAIN_GRID,
    FeedbackGain,
    FeedbackMode,
    _build_shard_with_replay_binding,
    _controller_delta,
    _foot_load_threshold,
    _limb_lateral_sign,
    _load_feasible_records,
    _lower_body_actuator_ids,
    _record_label,
    _unique_replay_bindings,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _center_of_mass,
    _com_cop_distance,
    _contact_report,
    _fall_reason,
    _projected_gravity,
    _quat_from_roll_pitch_yaw,
    _reset_to_qpos,
    _roll_pitch,
    _roll_pitch_yaw,
)

_DEFAULT_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)
_DEFAULT_R4B1_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/r4b1_bounded_feedback_diagnosis_5eq.json"
)
_ANGLE_PROBE_RAD = math.radians(2.0)
_RATE_PROBE_RAD_PER_SEC = 0.10
_COM_OFFSET_METERS = 0.01
_STATIC_AXIS_DELTAS = (0.02, 0.05, 0.08)
_TILT_WARNING_RAD = math.radians(15.0)
_AUTHORITY_IMPROVEMENT_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class AuthorityProbe:
    kind: str
    axis: str
    sign: int
    value: float
    qacc_dof: int
    control_axis: str

    def manifest(self) -> dict[str, float | int | str]:
        return {
            "kind": self.kind,
            "axis": self.axis,
            "sign": self.sign,
            "value": self.value,
            "qacc_dof": self.qacc_dof,
            "control_axis": self.control_axis,
        }


@dataclass(frozen=True, slots=True)
class TimelineVariant:
    name: str
    mode: FeedbackMode
    gain: FeedbackGain
    invert_delta: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.manifest(),
            "gain": self.gain.manifest(),
            "invert_delta": self.invert_delta,
        }


_COMBINED_MODE = FeedbackMode("attitude_com_combined", attitude=True, com_cop_oracle=True)
_LOW_GAIN = next(gain for gain in _GAIN_GRID if gain.name == "bounded_low")
_HIGH_GAIN = next(gain for gain in _GAIN_GRID if gain.name == "bounded_high")
_TIMELINE_VARIANTS = (
    TimelineVariant("hold_baseline", FeedbackMode("hold_baseline"), _BASELINE_GAIN),
    TimelineVariant("current_combined_low", _COMBINED_MODE, _LOW_GAIN),
    TimelineVariant("current_combined_high", _COMBINED_MODE, _HIGH_GAIN),
    TimelineVariant("inverted_combined_high", _COMBINED_MODE, _HIGH_GAIN, invert_delta=True),
)


def _authority_probes() -> list[AuthorityProbe]:
    return [
        AuthorityProbe("angle", "roll", 1, _ANGLE_PROBE_RAD, 3, "roll"),
        AuthorityProbe("angle", "roll", -1, _ANGLE_PROBE_RAD, 3, "roll"),
        AuthorityProbe("angle", "pitch", 1, _ANGLE_PROBE_RAD, 4, "pitch"),
        AuthorityProbe("angle", "pitch", -1, _ANGLE_PROBE_RAD, 4, "pitch"),
        AuthorityProbe("rate", "roll", 1, _RATE_PROBE_RAD_PER_SEC, 3, "roll"),
        AuthorityProbe("rate", "roll", -1, _RATE_PROBE_RAD_PER_SEC, 3, "roll"),
        AuthorityProbe("rate", "pitch", 1, _RATE_PROBE_RAD_PER_SEC, 4, "pitch"),
        AuthorityProbe("rate", "pitch", -1, _RATE_PROBE_RAD_PER_SEC, 4, "pitch"),
        AuthorityProbe("offset", "x", 1, _COM_OFFSET_METERS, 0, "pitch"),
        AuthorityProbe("offset", "x", -1, _COM_OFFSET_METERS, 0, "pitch"),
        AuthorityProbe("offset", "y", 1, _COM_OFFSET_METERS, 1, "roll"),
        AuthorityProbe("offset", "y", -1, _COM_OFFSET_METERS, 1, "roll"),
    ]


def restoring_score(*, perturb_sign: int, qacc_value: float) -> float:
    """Positive means acceleration is opposite the perturbation sign."""

    return -float(perturb_sign) * float(qacc_value)


def _axis_delta_vector(shard: Any, axis: str, command: float) -> Any:
    np = shard.np
    raw = np.zeros(shard.model.nu, dtype=np.float64)
    lower_body = _lower_body_actuator_ids(shard)
    for joint, actuator_id in zip(shard.blueprint.joints, shard._actuator_ids):
        actuator_id = int(actuator_id)
        if actuator_id not in lower_body:
            continue
        slot = joint.semantic_slot
        lateral = _limb_lateral_sign(joint.name, slot)
        delta = 0.0
        if axis == "pitch":
            if slot.endswith("ankle_pitch"):
                delta += command
            elif slot.endswith("knee_pitch"):
                delta += -0.35 * command
            elif slot.endswith("hip_pitch"):
                delta += -0.55 * command
        elif axis == "roll":
            if slot.endswith("ankle_roll"):
                delta += lateral * command
            elif slot.endswith("hip_roll"):
                delta += -0.55 * lateral * command
            elif slot.endswith("hip_yaw"):
                delta += 0.15 * lateral * command
        else:
            raise ValueError(f"unknown control axis: {axis}")
        raw[actuator_id] = delta
    return raw


def _bounded_ctrl(shard: Any, ctrl_eq: Any, delta: Any) -> tuple[Any, int]:
    np = shard.np
    ctrl = np.asarray(ctrl_eq, dtype=np.float64) + np.asarray(delta, dtype=np.float64)
    clipped_components = 0
    for actuator_id in shard._actuator_ids:
        actuator_id = int(actuator_id)
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
        before = float(ctrl[actuator_id])
        ctrl[actuator_id] = min(upper, max(lower, before))
        clipped_components += int(abs(float(ctrl[actuator_id]) - before) > 1e-12)
    return ctrl, clipped_components


def _apply_probe_state(shard: Any, data: Any, qpos_eq: Any, probe: AuthorityProbe) -> None:
    np = shard.np
    qpos = np.asarray(qpos_eq, dtype=np.float64).copy()
    if probe.kind == "angle":
        roll, pitch, yaw = _roll_pitch_yaw(qpos[3:7])
        if probe.axis == "roll":
            roll += probe.sign * probe.value
        elif probe.axis == "pitch":
            pitch += probe.sign * probe.value
        else:
            raise ValueError(f"angle probe does not support axis {probe.axis}")
        qpos[3:7] = _quat_from_roll_pitch_yaw(roll, pitch, yaw)
    elif probe.kind == "offset":
        if probe.axis == "x":
            qpos[0] += probe.sign * probe.value
        elif probe.axis == "y":
            qpos[1] += probe.sign * probe.value
        else:
            raise ValueError(f"offset probe does not support axis {probe.axis}")
    _reset_to_qpos(shard, data, qpos)
    if probe.kind == "rate":
        data.qvel[probe.qacc_dof] = probe.sign * probe.value
        shard.mujoco.mj_forward(shard.model, data)


def _actuator_force_snapshot(shard: Any, data: Any) -> dict[str, float | int]:
    actuator_force_max = 0.0
    saturation_events = 0
    for actuator_id in shard._actuator_ids:
        actuator_id = int(actuator_id)
        force = abs(float(data.actuator_force[actuator_id]))
        limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[actuator_id])
        actuator_force_max = max(actuator_force_max, force)
        saturation_events += int(force >= 0.995 * limit)
    return {
        "actuator_force_max": actuator_force_max,
        "actuator_saturation_events": saturation_events,
    }


def _foot_load_snapshot(shard: Any, data: Any) -> dict[str, Any]:
    contact = _contact_report(shard, data)
    threshold = _foot_load_threshold(shard)
    unloaded_feet = [
        name
        for name in sorted(shard._foot_geoms)
        if float(contact["normal_force_by_foot"].get(name, 0.0)) < threshold
    ]
    return {
        "contact": contact,
        "foot_load_threshold": threshold,
        "unloaded_feet": unloaded_feet,
    }


def _evaluate_probe_delta(shard: Any, equilibrium: dict[str, Any], probe: AuthorityProbe, delta: Any, name: str) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    qpos_eq = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl_eq = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    _apply_probe_state(shard, data, qpos_eq, probe)
    ctrl, ctrl_range_clip_components = _bounded_ctrl(shard, ctrl_eq, delta)
    data.ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, data)
    force = _actuator_force_snapshot(shard, data)
    foot = _foot_load_snapshot(shard, data)
    qacc_value = float(data.qacc[probe.qacc_dof])
    delta_array = np.asarray(delta, dtype=np.float64)
    return {
        "name": name,
        "qacc_value": qacc_value,
        "restoring_score": restoring_score(perturb_sign=probe.sign, qacc_value=qacc_value),
        "root_qacc": [float(value) for value in data.qacc[:6]],
        "max_abs_delta": float(np.max(np.abs(delta_array))) if len(delta_array) else 0.0,
        "active_delta_components": int(np.sum(np.abs(delta_array) > 1e-12)),
        "ctrl_range_clip_components": ctrl_range_clip_components,
        "actuator_force_max": force["actuator_force_max"],
        "actuator_saturation_events": force["actuator_saturation_events"],
        "non_foot_contacts": foot["contact"]["non_foot_contacts"],
        "unloaded_feet": foot["unloaded_feet"],
        "foot_normal_force_sum": foot["contact"]["foot_normal_force_sum"],
    }


def run_authority_probe(shard: Any, equilibrium: dict[str, Any], probe: AuthorityProbe) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    qpos_eq = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    zero_delta = np.zeros(shard.model.nu, dtype=np.float64)

    _apply_probe_state(shard, data, qpos_eq, probe)
    current_delta, current_stats = _controller_delta(shard, data, _COMBINED_MODE, _HIGH_GAIN)
    current_delta = np.asarray(current_delta, dtype=np.float64)

    responses = [
        _evaluate_probe_delta(shard, equilibrium, probe, zero_delta, "baseline"),
        _evaluate_probe_delta(shard, equilibrium, probe, current_delta, "current_combined_high"),
        _evaluate_probe_delta(shard, equilibrium, probe, -current_delta, "inverted_combined_high"),
    ]
    for magnitude in _STATIC_AXIS_DELTAS:
        for sign in (-1, 1):
            command = sign * magnitude
            responses.append(
                _evaluate_probe_delta(
                    shard,
                    equilibrium,
                    probe,
                    _axis_delta_vector(shard, probe.control_axis, command),
                    f"static_{probe.control_axis}_{command:+.2f}",
                )
            )

    by_name = {response["name"]: response for response in responses}
    baseline_score = float(by_name["baseline"]["restoring_score"])
    current_score = float(by_name["current_combined_high"]["restoring_score"])
    inverted_score = float(by_name["inverted_combined_high"]["restoring_score"])
    static_responses = [response for response in responses if response["name"].startswith("static_")]
    best_static = max(static_responses, key=lambda response: float(response["restoring_score"]))
    best_any = max(responses, key=lambda response: float(response["restoring_score"]))
    return {
        "probe": probe.manifest(),
        "controller_stats": current_stats,
        "responses": responses,
        "baseline_restoring_score": baseline_score,
        "current_restoring_improvement": current_score - baseline_score,
        "inverted_restoring_improvement": inverted_score - baseline_score,
        "best_static_name": best_static["name"],
        "best_static_restoring_improvement": float(best_static["restoring_score"]) - baseline_score,
        "best_any_name": best_any["name"],
        "best_any_restoring_improvement": float(best_any["restoring_score"]) - baseline_score,
    }


def _timeline_delta(shard: Any, data: Any, variant: TimelineVariant) -> tuple[Any, dict[str, float | int]]:
    np = shard.np
    if variant.mode.name == "hold_baseline":
        return np.zeros(shard.model.nu, dtype=np.float64), {"raw_max_abs": 0.0, "clipped_components": 0, "active_components": 0}
    delta, stats = _controller_delta(shard, data, variant.mode, variant.gain)
    delta = np.asarray(delta, dtype=np.float64)
    if variant.invert_delta:
        delta = -delta
    return delta, stats


def run_timeline(
    shard: Any,
    equilibrium: dict[str, Any],
    variant: TimelineVariant,
    *,
    horizon_steps: int,
) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    qpos_eq = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl_eq = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    _reset_to_qpos(shard, data, qpos_eq)
    data.ctrl[:] = ctrl_eq
    foot_load_threshold = _foot_load_threshold(shard)
    first_unloaded_step = None
    first_tilt_warning_step = None
    first_fall_step = None
    first_fall_reason = None
    controller_clipping_events = 0
    controller_components = 0
    actuator_saturation_events = 0
    non_foot_contact_steps = 0
    unloaded_foot_steps = 0
    max_abs_delta = 0.0
    actuator_force_max = 0.0
    trace: list[dict[str, Any]] = []
    for step_index in range(horizon_steps):
        for _ in range(shard.config.substeps):
            data.qfrc_applied[:] = 0.0
            delta, stats = _timeline_delta(shard, data, variant)
            ctrl, ctrl_clip = _bounded_ctrl(shard, ctrl_eq, delta)
            data.ctrl[:] = ctrl
            controller_clipping_events += int(stats["clipped_components"]) + ctrl_clip
            controller_components += len(shard._actuator_ids)
            max_abs_delta = max(max_abs_delta, float(np.max(np.abs(delta))) if len(delta) else 0.0)
            shard.mujoco.mj_forward(shard.model, data)
            force = _actuator_force_snapshot(shard, data)
            actuator_force_max = max(actuator_force_max, float(force["actuator_force_max"]))
            actuator_saturation_events += int(force["actuator_saturation_events"])
            shard.mujoco.mj_step(shard.model, data)

        contact = _contact_report(shard, data)
        com = _center_of_mass(shard, data)
        roll, pitch = _roll_pitch(data.qpos[3:7])
        gravity = _projected_gravity(tuple(float(value) for value in data.qpos[3:7]))
        horizontal_tilt = math.hypot(float(gravity[0]), float(gravity[1]))
        unloaded = any(float(contact["normal_force_by_foot"].get(name, 0.0)) < foot_load_threshold for name in shard._foot_geoms)
        non_foot = int(contact["non_foot_contacts"] > 0)
        non_foot_contact_steps += non_foot
        unloaded_foot_steps += int(unloaded)
        if unloaded and first_unloaded_step is None:
            first_unloaded_step = step_index + 1
        if (
            first_tilt_warning_step is None
            and max(abs(roll), abs(pitch), horizontal_tilt) >= _TILT_WARNING_RAD
        ):
            first_tilt_warning_step = step_index + 1
        trace.append(
            {
                "step": step_index + 1,
                "base_xy": [float(data.qpos[0]), float(data.qpos[1])],
                "roll_pitch": [float(roll), float(pitch)],
                "root_xy_vel": [float(data.qvel[0]), float(data.qvel[1])],
                "root_ang_vel": [float(data.qvel[3]), float(data.qvel[4])],
                "root_qacc": [float(value) for value in data.qacc[:6]],
                "com_xy": [float(com[0]), float(com[1])],
                "com_cop_distance": _com_cop_distance(com, contact),
                "foot_normal_force_sum": float(contact["foot_normal_force_sum"]),
                "normal_force_by_foot": contact["normal_force_by_foot"],
                "non_foot_contacts": int(contact["non_foot_contacts"]),
                "unloaded": unloaded,
            }
        )
        reason = _fall_reason(shard, data)
        if reason is not None:
            first_fall_step = step_index + 1
            first_fall_reason = reason
            break
    clipping_denominator = max(1, controller_components)
    return {
        "variant": variant.manifest(),
        "survived": first_fall_step is None,
        "first_fall_step": first_fall_step,
        "first_fall_reason": first_fall_reason,
        "first_unloaded_step": first_unloaded_step,
        "first_tilt_warning_step": first_tilt_warning_step,
        "contact_loss_precedes_tilt_warning": (
            first_unloaded_step is not None
            and (first_tilt_warning_step is None or first_unloaded_step <= first_tilt_warning_step)
        ),
        "steps_run": len(trace),
        "non_foot_contact_steps": non_foot_contact_steps,
        "unloaded_foot_steps": unloaded_foot_steps,
        "actuator_saturation_events": actuator_saturation_events,
        "actuator_force_max": actuator_force_max,
        "controller_clipping_ratio": controller_clipping_events / clipping_denominator,
        "controller_max_abs_delta": max_abs_delta,
        "trace": trace,
    }


def summarize_authority(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"probes": 0}
    angular = [row for row in rows if row["probe"]["kind"] in {"angle", "rate"}]
    offsets = [row for row in rows if row["probe"]["kind"] == "offset"]

    def _median(key: str, source: list[dict[str, Any]]) -> float | None:
        values = [float(row[key]) for row in source]
        return statistics.median(values) if values else None

    current_improves = sum(
        float(row["current_restoring_improvement"]) > _AUTHORITY_IMPROVEMENT_EPS for row in angular
    )
    inverted_improves = sum(
        float(row["inverted_restoring_improvement"]) > _AUTHORITY_IMPROVEMENT_EPS for row in angular
    )
    best_static_improves = sum(
        float(row["best_static_restoring_improvement"]) > _AUTHORITY_IMPROVEMENT_EPS for row in angular
    )
    current_best = sum(row["best_any_name"] == "current_combined_high" for row in angular)
    inverted_best = sum(row["best_any_name"] == "inverted_combined_high" for row in angular)
    static_best = sum(str(row["best_any_name"]).startswith("static_") for row in angular)
    breakdown: dict[str, Any] = {}
    for axis in sorted({str(row["probe"]["axis"]) for row in rows}):
        for kind in sorted({str(row["probe"]["kind"]) for row in rows if row["probe"]["axis"] == axis}):
            subset = [row for row in rows if row["probe"]["axis"] == axis and row["probe"]["kind"] == kind]
            best_counts: dict[str, int] = {}
            for row in subset:
                best_name = str(row["best_any_name"])
                best_counts[best_name] = best_counts.get(best_name, 0) + 1
            breakdown[f"{axis}:{kind}"] = {
                "probes": len(subset),
                "current_improvement_median": _median("current_restoring_improvement", subset),
                "inverted_improvement_median": _median("inverted_restoring_improvement", subset),
                "best_static_improvement_median": _median("best_static_restoring_improvement", subset),
                "best_any_improvement_median": _median("best_any_restoring_improvement", subset),
                "best_any_counts": best_counts,
            }
    return {
        "probes": len(rows),
        "angular_probes": len(angular),
        "offset_probes": len(offsets),
        "current_improves_angular": current_improves,
        "inverted_improves_angular": inverted_improves,
        "best_static_improves_angular": best_static_improves,
        "current_best_angular": current_best,
        "inverted_best_angular": inverted_best,
        "static_best_angular": static_best,
        "current_improvement_median": _median("current_restoring_improvement", angular),
        "inverted_improvement_median": _median("inverted_restoring_improvement", angular),
        "best_static_improvement_median": _median("best_static_restoring_improvement", angular),
        "best_any_improvement_median": _median("best_any_restoring_improvement", angular),
        "offset_current_improvement_median": _median("current_restoring_improvement", offsets),
        "offset_best_static_improvement_median": _median("best_static_restoring_improvement", offsets),
        "breakdown": breakdown,
    }


def summarize_timelines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in _TIMELINE_VARIANTS:
        variant_rows = [row for row in rows if row["variant"]["name"] == variant.name]
        if not variant_rows:
            continue
        falls = [row for row in variant_rows if not row["rollout"]["survived"]]
        first_falls = [row["rollout"]["first_fall_step"] for row in falls if row["rollout"]["first_fall_step"] is not None]
        summary[variant.name] = {
            "seeds": len(variant_rows),
            "survived": len(variant_rows) - len(falls),
            "first_fall_step_min": min(first_falls) if first_falls else None,
            "contact_loss_precedes_tilt_warning": sum(
                int(row["rollout"]["contact_loss_precedes_tilt_warning"]) for row in variant_rows
            ),
            "first_unloaded_step_min": min(
                (
                    row["rollout"]["first_unloaded_step"]
                    for row in variant_rows
                    if row["rollout"]["first_unloaded_step"] is not None
                ),
                default=None,
            ),
            "first_tilt_warning_step_min": min(
                (
                    row["rollout"]["first_tilt_warning_step"]
                    for row in variant_rows
                    if row["rollout"]["first_tilt_warning_step"] is not None
                ),
                default=None,
            ),
            "non_foot_contact_steps": sum(row["rollout"]["non_foot_contact_steps"] for row in variant_rows),
            "unloaded_foot_steps": sum(row["rollout"]["unloaded_foot_steps"] for row in variant_rows),
            "actuator_saturation_events": sum(row["rollout"]["actuator_saturation_events"] for row in variant_rows),
            "controller_clipping_ratio_max": max(row["rollout"]["controller_clipping_ratio"] for row in variant_rows),
            "controller_max_abs_delta_max": max(row["rollout"]["controller_max_abs_delta"] for row in variant_rows),
            "actuator_force_max": max(row["rollout"]["actuator_force_max"] for row in variant_rows),
        }
    return summary


def decide_r4b2(authority_summary: dict[str, Any], timeline_summary: dict[str, Any]) -> dict[str, str]:
    angular_probes = int(authority_summary.get("angular_probes", 0))
    current_best = int(authority_summary.get("current_best_angular", 0))
    inverted_best = int(authority_summary.get("inverted_best_angular", 0))
    static_best = int(authority_summary.get("static_best_angular", 0))
    current_median = float(authority_summary.get("current_improvement_median") or 0.0)
    inverted_median = float(authority_summary.get("inverted_improvement_median") or 0.0)
    best_static_median = float(authority_summary.get("best_static_improvement_median") or 0.0)
    baseline_timeline = timeline_summary.get("hold_baseline", {})
    contact_loss_precedes = int(baseline_timeline.get("contact_loss_precedes_tilt_warning", 0))
    seeds = int(baseline_timeline.get("seeds", 0))

    if angular_probes and inverted_best > current_best and inverted_median > current_median:
        return {
            "status": "feedback_sign_or_mapping_suspect",
            "decision": "Inverted bounded feedback has stronger local restoring authority than the current mapping.",
            "next_allowed_work": "Continue R4b diagnostics with polarity/mapping repair candidates; do not integrate a controller yet.",
        }
    if seeds and contact_loss_precedes >= math.ceil(0.6 * seeds):
        return {
            "status": "contact_mode_loss_precedes_tilt",
            "decision": "Foot load loss usually precedes tilt growth, so contact support geometry or equilibrium drift remains causal.",
            "next_allowed_work": "Return to contact/equilibrium diagnostics before adding controller integration.",
        }
    if angular_probes and static_best >= math.ceil(0.6 * angular_probes) and best_static_median > max(0.0, current_median):
        return {
            "status": "bounded_mapping_or_weighting_insufficient",
            "decision": "Bounded lower-body deltas can create local restoring acceleration, but the current feedback mapping does not use that authority.",
            "next_allowed_work": "Continue R4b diagnostics on deployable mapping/weighting; do not change public control chain yet.",
        }
    return {
        "status": "bounded_authority_insufficient_or_unresolved",
        "decision": "The bounded 0.08 rad lower-body target space did not show enough current or static local authority to explain a stable hold.",
        "next_allowed_work": "Do not integrate a controller; inspect equilibrium/contact or actuator authority before R4b controller design.",
    }


def run_diagnosis(
    *,
    input_json: Path,
    source_feedback_json: Path | None,
    horizon_steps: int,
) -> dict[str, Any]:
    biped_records = _load_feasible_records(input_json, family="biped")
    shards = {}
    replay_bindings = []
    for record in biped_records:
        shard, binding = _build_shard_with_replay_binding(record)
        shards[(record["family"], float(record["range_fraction"]), int(record["seed"]))] = shard
        replay_bindings.append(binding)
    authority_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    for record in biped_records:
        shard = shards[(record["family"], float(record["range_fraction"]), int(record["seed"]))]
        label = _record_label(record)
        for probe in _authority_probes():
            authority_rows.append(
                {
                    "label": label,
                    "family": record["family"],
                    "range_fraction": float(record["range_fraction"]),
                    "seed": int(record["seed"]),
                    **run_authority_probe(shard, record["contact_equilibrium"], probe),
                }
            )
        for variant in _TIMELINE_VARIANTS:
            timeline_rows.append(
                {
                    "label": label,
                    "family": record["family"],
                    "range_fraction": float(record["range_fraction"]),
                    "seed": int(record["seed"]),
                    "variant": variant.manifest(),
                    "rollout": run_timeline(
                        shard,
                        record["contact_equilibrium"],
                        variant,
                        horizon_steps=horizon_steps,
                    ),
                }
            )

    authority_summary = summarize_authority(authority_rows)
    timeline_summary = summarize_timelines(timeline_rows)
    return {
        "schema": "task067_r4b2_feedback_authority_diagnosis_v1",
        "source_equilibrium_artifact": str(input_json),
        "source_feedback_artifact": str(source_feedback_json) if source_feedback_json else None,
        "source_replay_contracts": _unique_replay_bindings(replay_bindings),
        "runtime_contract": {
            "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
            "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "probe_config": {
            "angle_probe_rad": _ANGLE_PROBE_RAD,
            "rate_probe_rad_per_sec": _RATE_PROBE_RAD_PER_SEC,
            "com_offset_meters": _COM_OFFSET_METERS,
            "static_axis_deltas": list(_STATIC_AXIS_DELTAS),
            "timeline_horizon_steps": horizon_steps,
            "timeline_tilt_warning_rad": _TILT_WARNING_RAD,
        },
        "timeline_variants": [variant.manifest() for variant in _TIMELINE_VARIANTS],
        "feasible_source_counts": {"biped": len(biped_records)},
        "authority_summary": authority_summary,
        "timeline_summary": timeline_summary,
        "decision": decide_r4b2(authority_summary, timeline_summary),
        "authority_rows": authority_rows,
        "timeline_rows": timeline_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=_DEFAULT_R4A2_ARTIFACT)
    parser.add_argument("--source-feedback-json", type=Path, default=_DEFAULT_R4B1_ARTIFACT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--horizon-steps", type=int, default=100)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = run_diagnosis(
        input_json=args.input_json,
        source_feedback_json=args.source_feedback_json,
        horizon_steps=args.horizon_steps,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "authority_summary": payload["authority_summary"],
                "timeline_summary": payload["timeline_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
