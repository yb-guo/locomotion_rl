"""Locked Task071 R2 WholeBody environment-contract verification."""

from __future__ import annotations

import hashlib
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / ".agent/task/task071-multimorphology-training-readiness"
OUT = TASK_DIR / "artifacts"
ARTIFACT_PATH = OUT / "r2_env_contract_smoke.json"
ARTIFACT_VERSION = "task071_r2_whole_body_env_contract_v1"
COMMAND = (
    "UV_CACHE_DIR=/home/admin1/workspace/store/cache/uv "
    "uv run --isolated --locked --offline --python 3.11 --extra mujoco "
    "python .agent/task/task071-multimorphology-training-readiness/"
    "task071_env_contract.py"
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _record(records: list[dict[str, Any]], reference_id: str) -> dict[str, Any]:
    matches = [row for row in records if row.get("reference_id") == reference_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one record for {reference_id}")
    return matches[0]


def _bound_xml(
    physics: Any,
    mujoco: Any,
    case: Any,
    overlay_record: dict[str, Any],
) -> str:
    regenerated, xml = physics._bind_case(case, mujoco, write_artifact=False)
    _require(regenerated == overlay_record, "in-memory bound XML/overlay mismatch")
    _require(
        hashlib.sha256(xml.encode("utf-8")).hexdigest()
        == overlay_record["output_xml_sha256"],
        "bound XML SHA mismatch",
    )
    return xml


def _event_manifest(events: Any) -> list[dict[str, Any]]:
    return [
        {
            "slot": event.slot,
            "kind": event.kind,
            "onset_step": event.onset_step,
            "duration_steps": event.duration_steps,
            "value": event.value,
            "persistent": event.persistent,
        }
        for event in events
    ]


def _stance_solution(physics: Any, case: Any, overlay: dict[str, Any], r1: dict[str, Any]) -> Any:
    from h200_locomotion_lab.robots.procedural_morphology import morphology_instance_key
    from h200_locomotion_lab.robots.whole_body_stance import StanceSolution

    overlay_record = _record(overlay["records"], case.reference_id)
    r1_record = _record(r1["records"], case.reference_id)
    profile = r1_record["stance_profile"]
    profile_payload = {key: value for key, value in profile.items() if key != "sha256"}
    _require(profile["version"] == physics.STANCE_PROFILE_VERSION, "stance version mismatch")
    _require(profile["sha256"] == physics._payload_sha256(profile_payload), "stance SHA mismatch")
    _require(
        r1_record["bound_xml_sha256"] == overlay_record["output_xml_sha256"],
        "R1/overlay XML lineage mismatch",
    )
    feedforward = r1_record["inverse_static_feedforward"]
    _require(feedforward["stance_profile_sha256"] == profile["sha256"], "feedforward/profile mismatch")
    _require(
        feedforward["bound_xml_sha256"] == overlay_record["output_xml_sha256"],
        "feedforward/XML mismatch",
    )
    overrides = profile["joint_nominal_overrides"]
    qpos = {
        joint.semantic_slot: float(overrides.get(joint.semantic_slot, joint.nominal))
        for joint in case.blueprint.joints
    }
    controls = {
        str(row["semantic_slot"]): float(row["position_target_rad"])
        for row in feedforward["actuators"]
    }
    _require(len(controls) == len(feedforward["actuators"]), "duplicate feedforward slot")
    _require(set(controls) == set(case.blueprint.active_slots), "incomplete feedforward slots")
    solution = StanceSolution(
        instance_key=morphology_instance_key(case.blueprint, case.physical),
        base_height=float(r1_record["reset_pose"]["root_z"]),
        joint_qpos=qpos,
        actuator_ctrl=controls,
        model_xml_sha256=overlay_record["output_xml_sha256"],
    )
    solution.validate_for(
        case.blueprint,
        case.physical,
        expected_model_xml_sha256=overlay_record["output_xml_sha256"],
    )
    evidence = r1_record["stance_solution"]
    _require(solution.manifest() == evidence["manifest"], "stance solution manifest mismatch")
    _require(solution.solution_hash == evidence["sha256"], "stance solution SHA mismatch")
    return solution


def _finite_metrics(np: Any, metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).all()):
            return False
    return True


def _inactive_observation_zero(np: Any, observation: Any, mask: Any) -> bool:
    inactive = ~mask
    return bool(
        (observation[0, 12:57][inactive] == 0.0).all()
        and (observation[0, 57:102][inactive] == 0.0).all()
        and (observation[0, 102:147][inactive] == 0.0).all()
    )


def _evaluate_case(
    physics: Any,
    case: Any,
    overlay: dict[str, Any],
    r1: dict[str, Any],
) -> dict[str, Any]:
    import mujoco
    import numpy as np

    from h200_locomotion_lab.envs.whole_body_mujoco import (
        WholeBodyMuJoCoShard,
        WholeBodyMuJoCoShardConfig,
    )
    from h200_locomotion_lab.robots.motor_process import MotorProcessConfig
    from h200_locomotion_lab.robots.procedural_morphology import read_canonical_root_state
    from h200_locomotion_lab.robots.whole_body_slots import (
        WHOLE_BODY_ACTION_DIM,
        WHOLE_BODY_ACTOR_OBS_DIM,
        WHOLE_BODY_SCHEMA_HASH,
        WHOLE_BODY_SCHEMA_VERSION,
    )

    overlay_record = _record(overlay["records"], case.reference_id)
    r1_record = _record(r1["records"], case.reference_id)
    xml = _bound_xml(physics, mujoco, case, overlay_record)
    stance = _stance_solution(physics, case, overlay, r1)
    config = WholeBodyMuJoCoShardConfig(
        trial_seconds=0.04,
        context_trials=2,
        command_vx_range=(0.0, 0.0),
        command_vy_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        seed=7100,
    )
    shard = WholeBodyMuJoCoShard(
        case.blueprint,
        physical=case.physical,
        config=config,
        motor_config=MotorProcessConfig(control_hz=config.control_hz, no_event_probability=1.0),
        model_xml=xml,
        model_xml_sha256=overlay_record["output_xml_sha256"],
        stance_solution=stance,
    )
    mask = shard.active_action_mask[0]
    inactive = ~mask
    reset_observation = shard.reset()
    schema_ok = bool(
        WHOLE_BODY_SCHEMA_VERSION == "whole_body_v1_45"
        and len(WHOLE_BODY_SCHEMA_HASH) == 64
        and shard.spec.action.flat_dim == WHOLE_BODY_ACTION_DIM == 45
        and shard.spec.observation("policy").flat_dim == WHOLE_BODY_ACTOR_OBS_DIM == 193
        and reset_observation.shape == (1, WHOLE_BODY_ACTOR_OBS_DIM)
    )
    active_count = int(mask.sum())
    mask_ok = bool(
        active_count == len(case.blueprint.actuators)
        and np.array_equal(mask, np.asarray(shard.embodiment.action_mask, dtype=bool))
        and np.array_equal(reset_observation[0, 147:192].astype(bool), mask)
    )
    reset_finite = bool(np.isfinite(reset_observation).all())
    inactive_reset_zero = _inactive_observation_zero(np, reset_observation, mask)

    inactive_action = inactive.astype(np.float64)[None, :]
    inactive_step = shard.step(inactive_action)
    midpoint_ctrl = np.asarray(
        [stance.actuator_ctrl[actuator.semantic_slot] for actuator in case.blueprint.actuators]
    )
    actual_ctrl = np.asarray([shard.data[0].ctrl[aid] for aid in shard._actuator_ids])
    inactive_action_masked = bool(
        np.array_equal(inactive_step.actor_observation[0, 102:147], np.zeros(45))
        and np.allclose(actual_ctrl, midpoint_ctrl, rtol=0.0, atol=1e-12)
        and _inactive_observation_zero(np, inactive_step.actor_observation, mask)
    )
    finite_step = bool(
        np.isfinite(inactive_step.actor_observation).all()
        and np.isfinite(inactive_step.critic_observation).all()
        and np.isfinite(inactive_step.reward).all()
        and _finite_metrics(np, dict(inactive_step.metrics))
    )

    shard.reset()
    active_slot_index = int(shard.embodiment.mapping.selector[0])
    active_action = np.zeros((1, 45), dtype=np.float64)
    active_action[0, active_slot_index] = 1.0
    active_step = shard.step(active_action)
    actuator_id = shard._actuator_ids[0]
    lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
    expected_target = midpoint_ctrl[0] + config.action_scale * 0.5 * (upper - lower)
    observed_scaled_target = float(shard.data[0].ctrl[actuator_id])
    action_scaling_ok = bool(
        math.isclose(observed_scaled_target, expected_target, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            float(active_step.actor_observation[0, 102 + active_slot_index]),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and bool((active_step.actor_observation[0, 102:147][inactive] == 0.0).all())
    )

    shard.reset()
    data = shard.data[0]
    quaternion = np.asarray((0.91, 0.19, -0.11, 0.31), dtype=np.float64)
    data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
    data.qvel[:6] = (0.2, -0.3, 0.4, -0.5, 0.6, -0.7)
    mujoco.mj_forward(shard.model, data)
    canonical = read_canonical_root_state(shard.model, data, shard._canonical_root_site_id)
    rotated_observation = np.asarray(shard._observation(data, 0, False))
    expected_root_observation = np.asarray(
        (*canonical.local_linear_velocity, *canonical.local_angular_velocity, *canonical.projected_gravity)
    )
    shard._commands[0] = (
        canonical.local_linear_velocity[0],
        canonical.local_linear_velocity[1],
        canonical.local_angular_velocity[2],
    )
    reward, normalized_error, non_foot = shard._reward(data, 0)
    expected_reward = 1.25 + 0.25 * max(0.0, min(1.0, -canonical.projected_gravity[2])) - 0.10 * non_foot
    expected_fall = bool(
        canonical.world_position[2] < shard._fall_height_threshold()
        or -canonical.projected_gravity[2] < config.upright_threshold
    )
    rotated_root_ok = bool(
        np.allclose(rotated_observation[:9], expected_root_observation, rtol=0.0, atol=1e-9)
        and not np.allclose(rotated_observation[:6], data.qvel[:6], rtol=0.0, atol=1e-6)
        and math.isclose(reward, expected_reward, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(normalized_error, 0.0, rel_tol=0.0, abs_tol=1e-12)
        and shard._is_fallen(data) is expected_fall
    )
    target_height = shard._fall_height_threshold() * 0.5
    fall_shard = WholeBodyMuJoCoShard(
        case.blueprint,
        physical=case.physical,
        config=config,
        motor_config=MotorProcessConfig(
            control_hz=config.control_hz,
            no_event_probability=1.0,
        ),
        model_xml=xml,
        model_xml_sha256=overlay_record["output_xml_sha256"],
        stance_solution=stance,
    )
    fall_data = fall_shard.data[0]
    initial_canonical_height = read_canonical_root_state(
        fall_shard.model,
        fall_data,
        fall_shard._canonical_root_site_id,
    ).world_position[2]
    fall_data.qpos[2] += target_height - initial_canonical_height
    mujoco.mj_forward(fall_shard.model, fall_data)
    lowered_canonical = read_canonical_root_state(
        fall_shard.model,
        fall_data,
        fall_shard._canonical_root_site_id,
    )
    fall_step = fall_shard.step(np.zeros((1, 45), dtype=np.float64))
    reset_canonical_height = read_canonical_root_state(
        fall_shard.model,
        fall_shard.data[0],
        fall_shard._canonical_root_site_id,
    ).world_position[2]
    reset_control = np.asarray(
        [fall_shard.data[0].ctrl[actuator_id] for actuator_id in fall_shard._actuator_ids]
    )
    canonical_fall_reset_ok = bool(
        lowered_canonical.world_position[2] < fall_shard._fall_height_threshold()
        and bool(fall_step.metrics["fall"][0])
        and bool(fall_step.trial_done[0])
        and not bool(fall_step.context_done[0])
        and fall_step.actor_observation[0, -1] == 1.0
        and int(fall_shard._trial_step[0]) == 0
        and int(fall_shard._trial_index[0]) == 1
        and int(fall_shard._context_index[0]) == 0
        and math.isclose(
            reset_canonical_height,
            initial_canonical_height,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        and np.allclose(reset_control, midpoint_ctrl, rtol=0.0, atol=1e-12)
    )

    shard.reset()
    zero = np.zeros((1, 45), dtype=np.float64)
    first = shard.step(zero)
    terminal_action = np.zeros((1, 45), dtype=np.float64)
    terminal_action[0, active_slot_index] = 0.25
    second = shard.step(terminal_action)
    third = shard.step(zero)
    fourth = shard.step(zero)
    zero_action_nonfall = bool(
        not bool(first.metrics["fall"][0])
        and not bool(third.metrics["fall"][0])
        and np.isfinite(first.reward).all()
        and np.isfinite(third.reward).all()
    )
    final_previous_action = second.final_observation[0, 102:147]
    trial_context_ok = bool(
        not bool(first.trial_done[0])
        and not bool(first.context_done[0])
        and bool(second.trial_done[0])
        and not bool(second.context_done[0])
        and not bool(third.trial_done[0])
        and not bool(third.context_done[0])
        and bool(fourth.trial_done[0])
        and bool(fourth.context_done[0])
        and not bool(second.metrics["fall"][0])
        and not bool(fourth.metrics["fall"][0])
        and int(shard._trial_index[0]) == 0
        and int(shard._context_index[0]) == 1
        and second.actor_observation[0, -1] == 1.0
        and fourth.actor_observation[0, -1] == 1.0
        and second.final_observation[0, -1] == 0.0
        and np.allclose(final_previous_action, terminal_action[0], rtol=0.0, atol=0.0)
        and np.array_equal(second.actor_observation[0, 102:147], np.zeros(45))
    )

    fault_config = MotorProcessConfig(
        control_hz=config.control_hz,
        no_event_probability=0.0,
        max_events=2,
        onset_range_seconds=(0.0, 0.0),
        duration_range_seconds=(0.02, 0.02),
        persistent_probability=1.0,
        weak_probability=1.0,
        dead_probability=0.0,
        weak_range=(0.5, 0.5),
    )

    def make_context_probe() -> Any:
        return WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            config=config,
            motor_config=fault_config,
            model_xml=xml,
            model_xml_sha256=overlay_record["output_xml_sha256"],
            stance_solution=stance,
        )

    context_probe = make_context_probe()
    initial_events = _event_manifest(context_probe._motor[0].events)
    context_trace = [
        context_probe.step(np.zeros((1, 45), dtype=np.float64)) for _ in range(4)
    ]
    reseeded_events = _event_manifest(context_probe._motor[0].events)
    replay_probe = make_context_probe()
    replay_initial_events = _event_manifest(replay_probe._motor[0].events)
    replay_trace = [
        replay_probe.step(np.zeros((1, 45), dtype=np.float64)) for _ in range(4)
    ]
    replay_reseeded_events = _event_manifest(replay_probe._motor[0].events)
    context_fault_reseed_ok = bool(
        initial_events
        and reseeded_events
        and initial_events != reseeded_events
        and initial_events == replay_initial_events
        and reseeded_events == replay_reseeded_events
        and [bool(step.context_done[0]) for step in context_trace]
        == [False, False, False, True]
        and [bool(step.context_done[0]) for step in replay_trace]
        == [False, False, False, True]
        and int(context_probe._context_index[0]) == 1
        and int(replay_probe._context_index[0]) == 1
    )

    checks = {
        "exact_bound_model": bool(
            shard.model.nq == overlay_record["compile_evidence"]["nq"]
            and shard.model.nv == overlay_record["compile_evidence"]["nv"]
            and shard.model.nu == overlay_record["compile_evidence"]["nu"]
            and shard.xml == xml
            and shard.model_xml_sha256 == overlay_record["output_xml_sha256"]
        ),
        "instance_bound_stance": bool(stance.instance_key == shard.stance_solution.instance_key),
        "schema_45_action_193_observation": schema_ok,
        "active_mask_exact": mask_ok,
        "reset_observation_finite": reset_finite,
        "inactive_observation_slots_zero": inactive_reset_zero,
        "inactive_actions_masked_before_control": inactive_action_masked,
        "action_scaling_from_stance_target": action_scaling_ok,
        "step_observation_reward_metrics_finite": finite_step,
        "rotated_root_all_consumers_canonical": rotated_root_ok,
        "canonical_height_fall_and_reset": canonical_fall_reset_ok,
        "zero_action_nonfall": zero_action_nonfall,
        "two_trial_context_reset_semantics": trial_context_ok,
        "context_fault_reseed_and_replay": context_fault_reseed_ok,
    }
    passed = all(checks.values())
    return {
        "reference_id": case.reference_id,
        "family": case.spec["family"],
        "bound_xml": {
            "path": overlay_record["output_xml"],
            "sha256": overlay_record["output_xml_sha256"],
            "dimensions": {
                key: overlay_record["compile_evidence"][key] for key in ("nq", "nv", "nu")
            },
        },
        "stance": {
            "profile_version": r1_record["stance_profile"]["version"],
            "profile_sha256": r1_record["stance_profile"]["sha256"],
            "solution_sha256": stance.solution_hash,
            "root_z": stance.base_height,
        },
        "schema": {
            "version": WHOLE_BODY_SCHEMA_VERSION,
            "hash": WHOLE_BODY_SCHEMA_HASH,
            "action_dim": WHOLE_BODY_ACTION_DIM,
            "observation_dim": WHOLE_BODY_ACTOR_OBS_DIM,
            "active_count": active_count,
            "inactive_count": WHOLE_BODY_ACTION_DIM - active_count,
        },
        "short_contract_timing": {
            "physics_hz": config.physics_hz,
            "control_hz": config.control_hz,
            "substeps": config.substeps,
            "trial_steps": config.trial_steps,
            "context_trials": config.context_trials,
        },
        "action_scaling_probe": {
            "semantic_slot": case.blueprint.actuators[0].semantic_slot,
            "unified_slot_index": active_slot_index,
            "normalized_action": 1.0,
            "action_scale": config.action_scale,
            "stance_target": float(midpoint_ctrl[0]),
            "expected_control_target": expected_target,
            "observed_control_target": observed_scaled_target,
        },
        "rotated_root_probe": {
            "quaternion_wxyz": [float(value) for value in quaternion / np.linalg.norm(quaternion)],
            "canonical_frame_reader": "read_canonical_root_state/canonical_root_frame_v1",
            "observation_reward_fall_match_reader": rotated_root_ok,
        },
        "fall_reset_probe": {
            "lowered_canonical_height": lowered_canonical.world_position[2],
            "fall_height_threshold": fall_shard._fall_height_threshold(),
            "reset_canonical_height": reset_canonical_height,
            "trial_index_after_reset": int(fall_shard._trial_index[0]),
        },
        "context_fault_probe": {
            "initial_seed": config.seed,
            "reseeded_context_seed": config.seed + 7919,
            "initial_events": initial_events,
            "reseeded_events": reseeded_events,
            "same_seed_replay_exact": bool(
                initial_events == replay_initial_events
                and reseeded_events == replay_reseeded_events
            ),
        },
        "checks": checks,
        "passed": passed,
        "failure_reasons": [] if passed else [name for name, value in checks.items() if not value],
    }


def run_env_contract(
    *,
    write_artifact: bool = True,
    overlay: dict[str, Any] | None = None,
    r1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, str(TASK_DIR))
    sys.path.insert(0, str(ROOT / "src"))
    import mujoco
    import task071_physics_overlay as physics

    from h200_locomotion_lab.envs import whole_body_mujoco

    if (overlay is None) != (r1 is None):
        raise ValueError("overlay and R1 payloads must be supplied together")
    if overlay is None:
        overlay = physics.generate_overlay(write_artifact=write_artifact)
        r1 = physics.run_bound_r1(overlay, write_artifact=write_artifact)
    assert r1 is not None
    _require(r1["task071_r1_admission_passed"] is True, "R2 requires bound R1 admission")
    _require(
        r1["overlay_artifact_sha256"] == physics.json_artifact_sha256(overlay),
        "R1/overlay artifact lineage mismatch",
    )
    _require(set(physics.CASE_SPECS) == {"unitree_g1", "unitree_go2"}, "R2 denominator drift")
    r1_runtime = r1.get("runtime", {})
    r1_source = r1_runtime.get("source", {})
    physics_source = Path(physics.__file__).resolve()
    stance_helper_source = ROOT / r1_source.get("stance_helper_path", "missing")
    stance_contract_source = ROOT / r1_source.get("stance_contract_path", "missing")
    _require(r1_runtime.get("git_head") == _git_head(), "R1 git identity mismatch")
    _require(
        r1_source.get("physics_overlay_sha256") == _sha256_path(physics_source),
        "R1 physics-overlay source SHA mismatch",
    )
    _require(
        stance_helper_source.is_file()
        and r1_source.get("stance_helper_sha256") == _sha256_path(stance_helper_source),
        "R1 stance-helper source SHA mismatch",
    )
    _require(
        stance_contract_source.is_file()
        and r1_source.get("stance_contract_sha256")
        == _sha256_path(stance_contract_source),
        "R1 stance-contract source SHA mismatch",
    )
    if write_artifact:
        _require(
            _sha256_path(physics.OVERLAY_PATH) == physics.json_artifact_sha256(overlay),
            "persisted overlay SHA mismatch",
        )
        _require(
            _sha256_path(physics.BOUND_R1_PATH) == physics.json_artifact_sha256(r1),
            "persisted R1 SHA mismatch",
        )
    records = [
        _evaluate_case(physics, physics.load_frozen_case(reference_id), overlay, r1)
        for reference_id in physics.CASE_SPECS
    ]
    admission = len(records) == 2 and all(record["passed"] for record in records)
    gate_names = tuple(records[0]["checks"])
    payload = {
        "artifact": ARTIFACT_VERSION,
        "task": "task071-multimorphology-training-readiness",
        "runtime": {
            "command": COMMAND,
            "git_head": _git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco_version": mujoco.__version__,
            "hardware_scope": (
                "RTX 5060 Ti-first project; R2 is a bounded CPU MuJoCo environment smoke"
            ),
            "robot_asset_dataset_or_checkpoint_downloads_performed": False,
        },
        "source": {
            "probe_path": str(Path(__file__).relative_to(ROOT)),
            "probe_sha256": _sha256_path(Path(__file__)),
            "environment_path": str(Path(whole_body_mujoco.__file__).relative_to(ROOT)),
            "environment_sha256": _sha256_path(Path(whole_body_mujoco.__file__)),
        },
        "r1": {
            "path": str(physics.BOUND_R1_PATH.relative_to(ROOT)),
            "sha256": physics.json_artifact_sha256(r1),
            "admission": True,
            "git_head": r1_runtime["git_head"],
            "source": r1_source,
        },
        "overlay": {
            "path": str(physics.OVERLAY_PATH.relative_to(ROOT)),
            "sha256": physics.json_artifact_sha256(overlay),
            "version": physics.OVERLAY_VERSION,
        },
        "denominator": 2,
        "records": records,
        "summary": {name: sum(record["checks"][name] for record in records) for name in gate_names},
        "task071_r2_admission_passed": admission,
        "failure_reasons": [] if admission else [
            f"{record['reference_id']}:{reason}"
            for record in records
            for reason in record["failure_reasons"]
        ],
        "claim_boundary": {
            "bounded_environment_contract_only": True,
            "training_performed": False,
            "ppo_started": False,
            "walking_claimed": False,
            "task071_passed": False,
        },
    }
    if write_artifact:
        physics.write_json_artifact(ARTIFACT_PATH, payload)
    return payload


def main() -> int:
    payload = run_env_contract()
    passed = sum(record["passed"] for record in payload["records"])
    print(
        f"Task071 R2: admission={payload['task071_r2_admission_passed']}, "
        f"cases={passed}/{payload['denominator']}, ppo=False"
    )
    return 0 if payload["task071_r2_admission_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
