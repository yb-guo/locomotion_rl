"""Validate all generated morphologies as usable training environments.

This is deliberately different from a passive generator smoke.  Every model
is compiled, reset into its nominal pose, checked against the 45-slot adapter,
probed for one-to-one actuator influence, and stepped with a bounded unified
action trace.  It does not claim that a policy can walk yet.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import ground_nominal_pose
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyBlueprint,
    MorphologyGenerator,
    PhysicalParams,
    compile_mjcf,
)
from h200_locomotion_lab.robots.whole_body_adapter import BoundEmbodiment


def _projected_gravity(quaternion: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Rotate world gravity into the base frame for MuJoCo's wxyz quaternion."""

    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-9:
        return (0.0, 0.0, -1.0)
    w, x, y, z = (value / norm for value in quaternion)
    return (
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )


def _finite_bounded(array: Any, bound: float) -> bool:
    import numpy as np

    values = np.asarray(array)
    return bool(np.isfinite(values).all() and np.max(np.abs(values), initial=0.0) < bound)


def _min_contact_distance(data: Any) -> float:
    if data.ncon == 0:
        return 0.0
    return min(float(data.contact[index].dist) for index in range(data.ncon))


def _reset_nominal(mujoco: Any, model: Any, data: Any, blueprint: MorphologyBlueprint,
                   physical: PhysicalParams) -> tuple[float, ...]:
    """Reset one model and return the nominal joint position vector."""

    mujoco.mj_resetData(model, data)
    data.qpos[2] = blueprint.nominal_height * physical.global_scale
    nominal_values: list[float] = []
    for joint in blueprint.joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        if joint_id < 0:
            raise RuntimeError(f"compiled model lost joint {joint.name}")
        nominal = joint.nominal + physical.nominal_offsets[joint.semantic_slot]
        lower, upper = model.jnt_range[joint_id]
        if not lower <= nominal <= upper:
            raise RuntimeError(f"illegal nominal pose for {joint.name}")
        data.qpos[model.jnt_qposadr[joint_id]] = nominal
        nominal_values.append(float(nominal))
    mujoco.mj_forward(model, data)
    ground_nominal_pose(mujoco, model, data)
    return tuple(nominal_values)


def _check_mapping_and_actuators(
    mujoco: Any,
    model: Any,
    data: Any,
    blueprint: MorphologyBlueprint,
    embodiment: BoundEmbodiment,
) -> dict[str, Any]:
    """Check unified masking plus direct generalized-force actuator influence."""

    import numpy as np

    if model.nu != len(blueprint.joints) or model.nu != embodiment.mapping.robot_action_dim:
        raise RuntimeError("compiled actuator count does not match the generated mapping")
    mask = embodiment.action_mask
    if sum(mask) != model.nu:
        raise RuntimeError("active mask count does not match compiled actuator count")

    unified = np.linspace(-1.0, 1.0, 45, dtype=np.float64)
    local = embodiment.gather_action(unified.tolist())
    round_trip = embodiment.scatter_joint_values(local)
    for index, active in enumerate(mask):
        expected = float(unified[index]) if active else 0.0
        if abs(round_trip[index] - expected) > 1e-12:
            raise RuntimeError(f"adapter round-trip mismatch at slot {index}")
    for local_index, slot_index in enumerate(embodiment.mapping.selector):
        impulse = [0.0] * 45
        impulse[slot_index] = 1.0
        projected = embodiment.gather_action(impulse)
        if projected[local_index] != 1.0 or any(
            value != 0.0 for index, value in enumerate(projected) if index != local_index
        ):
            raise RuntimeError(f"slot {slot_index} leaks into another actuator")

    actuator_ids: list[int] = []
    joint_dofs: list[int] = []
    for actuator, joint in zip(blueprint.actuators, blueprint.joints):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        if actuator_id < 0 or joint_id < 0:
            raise RuntimeError(f"compiled model lost actuator or joint for {joint.name}")
        target_joint = int(model.actuator_trnid[actuator_id, 0])
        if target_joint != joint_id:
            raise RuntimeError(f"actuator {actuator.name} targets the wrong joint")
        actuator_ids.append(actuator_id)
        joint_dofs.append(int(model.jnt_dofadr[joint_id]))

    mids = (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1]) * 0.5
    spans = model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]
    data.ctrl[:] = mids
    mujoco.mj_forward(model, data)
    baseline_force = data.qfrc_actuator.copy()
    min_target_delta = float("inf")
    max_off_target_delta = 0.0
    for actuator_id, target_dof in zip(actuator_ids, joint_dofs):
        data.ctrl[:] = mids
        data.ctrl[actuator_id] = min(
            model.actuator_ctrlrange[actuator_id, 1], mids[actuator_id] + 0.25 * spans[actuator_id]
        )
        mujoco.mj_forward(model, data)
        delta = data.qfrc_actuator - baseline_force
        target_delta = abs(float(delta[target_dof]))
        off_target = max(
            (abs(float(value)) for index, value in enumerate(delta) if index != target_dof),
            default=0.0,
        )
        min_target_delta = min(min_target_delta, target_delta)
        max_off_target_delta = max(max_off_target_delta, off_target)
        if target_delta <= 1e-4:
            raise RuntimeError(f"actuator {actuator_id} has no measurable target-joint influence")
        if off_target > max(1e-6, target_delta * 1e-6):
            raise RuntimeError(f"actuator {actuator_id} influences a non-target dof")

    return {
        "mapping_pass": True,
        "actuator_target_pass": True,
        "active_actuators": model.nu,
        "min_target_force_delta": float(min_target_delta),
        "max_off_target_force_delta": float(max_off_target_delta),
    }


def _encode_observation(mujoco: Any, model: Any, data: Any, blueprint: MorphologyBlueprint,
                        embodiment: BoundEmbodiment) -> tuple[float, ...]:
    joint_positions = []
    joint_velocities = []
    for joint in blueprint.joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        joint_positions.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
        joint_velocities.append(float(data.qvel[model.jnt_dofadr[joint_id]]))
    observation = embodiment.encode_actor_observation(
        base_linear_velocity=tuple(float(value) for value in data.qvel[:3]),
        base_angular_velocity=tuple(float(value) for value in data.qvel[3:6]),
        projected_gravity=_projected_gravity(tuple(float(value) for value in data.qpos[3:7])),
        command=(0.5, 0.0, 0.0),
        joint_position=joint_positions,
        joint_velocity=joint_velocities,
        previous_action=(0.0,) * 45,
        trial_start=1.0,
    )
    embodiment.validate_observation(observation)
    if not all(math.isfinite(value) for value in observation):
        raise RuntimeError("actor observation contains a non-finite value")
    return observation


def _run_bounded_action_trace(
    mujoco: Any,
    model: Any,
    data: Any,
    blueprint: MorphologyBlueprint,
    embodiment: BoundEmbodiment,
    *,
    duration_seconds: float,
    control_hz: float,
    physics_hz: float,
    action_amplitude: float,
) -> dict[str, Any]:
    import numpy as np

    substeps_float = physics_hz / control_hz
    if abs(substeps_float - round(substeps_float)) > 1e-9:
        raise ValueError("physics_hz/control_hz must be an integer")
    substeps = round(substeps_float)
    control_steps = round(duration_seconds * control_hz)
    if substeps <= 0 or control_steps <= 0:
        raise ValueError("duration and rates must produce at least one step")

    mids = (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1]) * 0.5
    spans = model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]
    min_contact_distance = 0.0
    min_height = float(data.qpos[2])
    max_abs_qpos = 0.0
    max_abs_qvel = 0.0
    for control_step in range(control_steps):
        unified_action = np.zeros(45, dtype=np.float64)
        for slot_index in embodiment.mapping.selector:
            unified_action[slot_index] = action_amplitude * math.sin(
                0.13 * control_step + 0.017 * slot_index
            )
        local_action = embodiment.gather_action(unified_action.tolist())
        data.ctrl[:] = mids
        for value, actuator_id in zip(local_action, range(model.nu)):
            data.ctrl[actuator_id] = mids[actuator_id] + float(value) * spans[actuator_id]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        min_contact_distance = min(min_contact_distance, _min_contact_distance(data))
        min_height = min(min_height, float(data.qpos[2]))
        max_abs_qpos = max(max_abs_qpos, float(np.max(np.abs(data.qpos), initial=0.0)))
        max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel), initial=0.0)))
        if not _finite_bounded(data.qpos, 100.0) or not _finite_bounded(data.qvel, 1000.0):
            raise RuntimeError(f"non-finite or unbounded state at control step {control_step}")
        if min_contact_distance < -0.25:
            raise RuntimeError("severe contact penetration during bounded action trace")

    return {
        "controlled_step_pass": True,
        "control_steps": control_steps,
        "physics_steps": control_steps * substeps,
        "min_contact_distance": float(min_contact_distance),
        "min_height": float(min_height),
        "max_abs_qpos": float(max_abs_qpos),
        "max_abs_qvel": float(max_abs_qvel),
    }


def _validate_one(
    mujoco: Any,
    generator: MorphologyGenerator,
    family: str,
    seed: int,
    *,
    duration_seconds: float,
    control_hz: float,
    physics_hz: float,
    action_amplitude: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {"family": family, "seed": seed, "passed": False}
    try:
        blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
        second_blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
        physical = generator.sample_physical_params(blueprint, seed + 100_000)
        second_physical = generator.sample_physical_params(second_blueprint, seed + 100_000)
        if blueprint != second_blueprint or physical != second_physical:
            raise RuntimeError("generator replay is not deterministic")
        model = mujoco.MjModel.from_xml_string(compile_mjcf(blueprint, physical))
        data = mujoco.MjData(model)
        embodiment = BoundEmbodiment.from_blueprint(blueprint, physical=physical)
        _reset_nominal(mujoco, model, data, blueprint, physical)
        import numpy as np

        initial_qpos = data.qpos.copy()
        if not _finite_bounded(data.qpos, 100.0) or not _finite_bounded(data.qvel, 1000.0):
            raise RuntimeError("nominal reset produced a non-finite or unbounded state")
        mass_positive = bool((model.body_mass[1:] > 0.0).all())
        inertia_positive = bool((model.body_inertia[1:] > 0.0).all())
        if not mass_positive or not inertia_positive:
            raise RuntimeError("generated model has non-positive mass or inertia")
        observation = _encode_observation(mujoco, model, data, blueprint, embodiment)
        mapping_result = _check_mapping_and_actuators(
            mujoco, model, data, blueprint, embodiment
        )
        _reset_nominal(mujoco, model, data, blueprint, physical)
        trace_result = _run_bounded_action_trace(
            mujoco,
            model,
            data,
            blueprint,
            embodiment,
            duration_seconds=duration_seconds,
            control_hz=control_hz,
            physics_hz=physics_hz,
            action_amplitude=action_amplitude,
        )
        _reset_nominal(mujoco, model, data, blueprint, physical)
        reset_error = float(np.max(np.abs(data.qpos - initial_qpos), initial=0.0))
        record.update(
            {
                "structural_hash": blueprint.structural_hash,
                "links": len(blueprint.links),
                "joints": len(blueprint.joints),
                "actuators": model.nu,
                "has_arms": blueprint.has_arms,
                "deterministic_replay_pass": True,
                "compile_pass": True,
                "nominal_reset_pass": True,
                "mass_positive": mass_positive,
                "inertia_positive": inertia_positive,
                "observation_pass": len(observation) == 193,
                "reset_error_max": float(reset_error),
                **mapping_result,
                **trace_result,
                "passed": True,
            }
        )
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def run_usability_gate(
    *,
    seeds: int = 1000,
    duration_seconds: float = 2.0,
    control_hz: float = 50.0,
    physics_hz: float = 500.0,
    action_amplitude: float = 0.05,
) -> dict[str, Any]:
    """Run the full per-family usability gate and return JSON-safe evidence."""

    if seeds <= 0:
        raise ValueError("seeds must be positive")
    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional simulator dependency
        raise RuntimeError("MuJoCo is required for the 2k usability gate") from exc

    generator = MorphologyGenerator()
    records = []
    for family in ("biped", "quadruped"):
        for seed in range(seeds):
            records.append(
                _validate_one(
                    mujoco,
                    generator,
                    family,
                    seed,
                    duration_seconds=duration_seconds,
                    control_hz=control_hz,
                    physics_hz=physics_hz,
                    action_amplitude=action_amplitude,
                )
            )
    passed = sum(bool(record.get("passed")) for record in records)
    return {
        "schema": "whole_body_usability_gate_v1",
        "families": ("biped", "quadruped"),
        "seeds_per_family": seeds,
        "record_count": len(records),
        "duration_seconds": duration_seconds,
        "control_hz": control_hz,
        "physics_hz": physics_hz,
        "action_amplitude": action_amplitude,
        "thresholds": {
            "max_abs_qpos": 100.0,
            "max_abs_qvel": 1000.0,
            "min_contact_distance": -0.25,
            "min_target_force_delta": 1e-4,
            "max_off_target_force_delta": "max(1e-6, target_delta*1e-6)",
        },
        "passed_records": passed,
        "failed_records": len(records) - passed,
        "passed": passed == len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--physics-hz", type=float, default=500.0)
    parser.add_argument("--action-amplitude", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run_usability_gate(
        seeds=args.seeds,
        duration_seconds=args.duration_seconds,
        control_hz=args.control_hz,
        physics_hz=args.physics_hz,
        action_amplitude=args.action_amplitude,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
