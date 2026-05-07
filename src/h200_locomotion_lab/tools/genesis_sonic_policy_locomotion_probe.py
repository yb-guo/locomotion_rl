"""Measure locomotion in the decoder-only SONIC Genesis closed loop."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.genesis_adapter import GenesisG1SceneBackend, GenesisSceneConfig
from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_DECODER_OBS_DIM,
    SONIC_TOKEN_DIM,
    SonicG1HistoryBuffer,
    sonic_g1_history_from_decoder_observation,
)
from h200_locomotion_lab.tools.sonic_policy_decoder_forward import (
    SonicOnnxReferenceDecoder,
    read_obs_csv_rows,
    vector_range,
)
from h200_locomotion_lab.tools.genesis_action_replay_smoke import read_default_joint_positions
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    apply_sonic_g1_motor_config,
    _flatten_numeric,
    _read_contact_metrics,
)


@dataclass(frozen=True)
class RootPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class FootSample:
    z: float | None
    force: float | None
    contact: bool | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC-compatible G1 29DoF MJCF.")
    parser.add_argument("--decoder", required=True, help="Path to SONIC model_decoder.onnx.")
    parser.add_argument("--obs-csv", required=True, help="Official 994D obs CSV.")
    parser.add_argument("--token-mode", choices=("fixed", "replay"), default="replay")
    parser.add_argument("--history-init", choices=("genesis", "official_obs"), default="official_obs")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--root-qpos", nargs=7, type=float)
    parser.add_argument(
        "--initial-joint-pos-csv",
        help="CSV whose selected 29D row is used as the physical reset motor pose.",
    )
    parser.add_argument("--initial-joint-pos-row", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--left-foot-link", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-link", default="right_ankle_roll_link")
    parser.add_argument("--foot-contact-force-threshold", type=float, default=5.0)
    parser.add_argument("--min-horizontal-displacement", type=float, default=0.05)
    parser.add_argument("--min-contact-switches", type=int, default=2)
    parser.add_argument("--height-ok-min", type=float, default=0.3)
    parser.add_argument("--height-ok-max", type=float, default=1.2)
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.foot_contact_force_threshold < 0:
        raise ValueError("--foot-contact-force-threshold must be non-negative")

    obs_rows = tuple(read_obs_csv_rows(Path(args.obs_csv), SONIC_DECODER_OBS_DIM, args.frames))
    token_states = tuple(tuple(obs[:SONIC_TOKEN_DIM]) for obs in obs_rows)
    fixed_token_state = token_states[0]
    initial_motor_positions = (
        read_default_joint_positions(
            Path(args.initial_joint_pos_csv),
            args.initial_joint_pos_row,
            29,
        )
        if args.initial_joint_pos_csv
        else None
    )

    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(
            asset_path=args.asset,
            backend=args.backend,
            base_pos=tuple(args.base_pos),
            base_quat=tuple(args.base_quat),
            root_qpos=tuple(args.root_qpos) if args.root_qpos else None,
            initial_motor_positions=initial_motor_positions,
            action_mode="sonic_policy_raw",
            logging_level="warning",
        )
    )
    decoder = SonicOnnxReferenceDecoder(Path(args.decoder))

    if not args.no_sonic_motor_config:
        apply_sonic_g1_motor_config(backend.robot, backend.motor_dof_indices)
        motor_config = "sonic_g1_kp_kv_force_range"
    else:
        motor_config = "genesis_default"

    backend.reset()
    if args.history_init == "official_obs":
        _, initial_frames = sonic_g1_history_from_decoder_observation(obs_rows[0])
        backend.sonic_history = SonicG1HistoryBuffer()
        for initial_frame in initial_frames:
            backend.sonic_history.append(initial_frame)

    left_link_idx = resolve_link_index(backend.robot, args.left_foot_link)
    right_link_idx = resolve_link_index(backend.robot, args.right_foot_link)

    root_poses: list[RootPose] = [read_root_pose(backend)]
    action_max_abs_values: list[float] = []
    contact_counts: list[int] = []
    max_contact_forces: list[float] = []
    left_samples: list[FootSample] = []
    right_samples: list[FootSample] = []
    observation_finite = True
    action_finite = True

    print("GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_MODE decoder_only")
    print("ASSET", Path(args.asset))
    print("DECODER", Path(args.decoder))
    print("TOKEN_SOURCE", args.obs_csv)
    print("TOKEN_MODE", args.token_mode)
    print("TOKEN_ROWS", len(token_states))
    print("HISTORY_INIT", args.history_init)
    print("MOTOR_CONFIG", motor_config)
    print("FRAMES", args.frames)
    print("ROOT_QPOS", tuple(args.root_qpos) if args.root_qpos else None)
    print("INITIAL_JOINT_POS_SOURCE", args.initial_joint_pos_csv or "default_motor_positions")
    if initial_motor_positions is not None:
        print("INITIAL_JOINT_POS_ROW", args.initial_joint_pos_row)
        print("INITIAL_JOINT_POS_MIN_MAX", min(initial_motor_positions), max(initial_motor_positions))
    print("LEFT_FOOT_LINK", args.left_foot_link, "INDEX", left_link_idx)
    print("RIGHT_FOOT_LINK", args.right_foot_link, "INDEX", right_link_idx)
    print("FOOT_CONTACT_FORCE_THRESHOLD", args.foot_contact_force_threshold)

    for frame_index in range(args.frames):
        token_state = (
            token_states[min(frame_index, len(token_states) - 1)]
            if args.token_mode == "replay"
            else fixed_token_state
        )
        observation = backend.sonic_decoder_observation(token_state)
        observation_finite = observation_finite and _is_finite(observation)
        action = decoder.run(observation)
        action_finite = action_finite and _is_finite(action)
        backend.step(action)

        root_pose = read_root_pose(backend)
        contact_count, max_contact_force = _read_contact_metrics(backend.robot)
        left_sample = read_foot_sample(
            backend.robot,
            left_link_idx,
            contact_threshold=args.foot_contact_force_threshold,
        )
        right_sample = read_foot_sample(
            backend.robot,
            right_link_idx,
            contact_threshold=args.foot_contact_force_threshold,
        )
        _, _, action_max_abs = vector_range(action)

        root_poses.append(root_pose)
        action_max_abs_values.append(action_max_abs)
        if contact_count is not None:
            contact_counts.append(contact_count)
        if max_contact_force is not None:
            max_contact_forces.append(max_contact_force)
        left_samples.append(left_sample)
        right_samples.append(right_sample)

        if frame_index % args.log_every == 0 or frame_index == args.frames - 1:
            displacement = horizontal_distance(root_poses[0], root_pose)
            print(
                "FRAME",
                frame_index,
                "root_x",
                root_pose.x,
                "root_y",
                root_pose.y,
                "root_z",
                root_pose.z,
                "yaw",
                root_pose.yaw,
                "disp_xy",
                displacement,
                "path_xy",
                path_length_xy(root_poses),
                "left_foot_z",
                left_sample.z,
                "right_foot_z",
                right_sample.z,
                "left_contact",
                left_sample.contact,
                "right_contact",
                right_sample.contact,
                "action_max_abs",
                action_max_abs,
            )

    root_summary = summarize_root_motion(root_poses, policy_rate_hz=backend.contract.policy_rate_hz)
    left_contact = tuple(sample.contact for sample in left_samples)
    right_contact = tuple(sample.contact for sample in right_samples)
    foot_available = any(value is not None for value in left_contact + right_contact)
    left_contact_switches = count_contact_switches(left_contact)
    right_contact_switches = count_contact_switches(right_contact)
    total_contact_switches = left_contact_switches + right_contact_switches
    single_support_frames = count_single_support_frames(left_contact, right_contact)
    double_support_frames = count_double_support_frames(left_contact, right_contact)
    no_support_frames = count_no_support_frames(left_contact, right_contact)
    height_ok = (
        root_summary["root_z_min"] >= args.height_ok_min
        and root_summary["root_z_max"] <= args.height_ok_max
    )
    finite_ok = (
        observation_finite
        and action_finite
        and all(math.isfinite(value) for value in root_summary.values())
    )
    translation_observed = root_summary["horizontal_displacement"] >= args.min_horizontal_displacement
    foot_alternation_observed = (
        foot_available and total_contact_switches >= args.min_contact_switches
    )
    locomotion_observed = translation_observed and (
        foot_alternation_observed or not foot_available
    )

    print("OBS_FINITE", observation_finite)
    print("ACTION_FINITE", action_finite)
    print("ROOT_X_START", root_poses[0].x)
    print("ROOT_Y_START", root_poses[0].y)
    print("ROOT_Z_START", root_poses[0].z)
    print("ROOT_X_FINAL", root_poses[-1].x)
    print("ROOT_Y_FINAL", root_poses[-1].y)
    print("ROOT_Z_FINAL", root_poses[-1].z)
    print("ROOT_Z_MIN", root_summary["root_z_min"])
    print("ROOT_Z_MAX", root_summary["root_z_max"])
    print("HORIZONTAL_DISPLACEMENT", root_summary["horizontal_displacement"])
    print("PATH_LENGTH_XY", root_summary["path_length_xy"])
    print("AVERAGE_SPEED_XY", root_summary["average_speed_xy"])
    print("YAW_DELTA", root_summary["yaw_delta"])
    print("ACTION_MAX_ABS", max(action_max_abs_values))
    if contact_counts:
        print("CONTACT_COUNT_MAX", max(contact_counts))
        print("CONTACT_COUNT_FINAL", contact_counts[-1])
    if max_contact_forces:
        print("MAX_LINK_CONTACT_FORCE_MAX", max(max_contact_forces))
        print("MAX_LINK_CONTACT_FORCE_FINAL", max_contact_forces[-1])
    print_foot_summary("LEFT", left_samples)
    print_foot_summary("RIGHT", right_samples)
    print("LEFT_CONTACT_SWITCHES", left_contact_switches)
    print("RIGHT_CONTACT_SWITCHES", right_contact_switches)
    print("TOTAL_CONTACT_SWITCHES", total_contact_switches)
    print("SINGLE_SUPPORT_FRAMES", single_support_frames)
    print("DOUBLE_SUPPORT_FRAMES", double_support_frames)
    print("NO_SUPPORT_FRAMES", no_support_frames)
    print("TRANSLATION_THRESHOLD", args.min_horizontal_displacement)
    print("CONTACT_SWITCH_THRESHOLD", args.min_contact_switches)
    print("TRANSLATION_OBSERVED", translation_observed)
    print("FOOT_ALTERNATION_OBSERVED", foot_alternation_observed)
    print("LOCOMOTION_OBSERVED", locomotion_observed)
    print("HEIGHT_OK_RANGE", args.height_ok_min, args.height_ok_max, height_ok)
    if not finite_ok or not height_ok:
        raise SystemExit("GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_FAILED")
    print("GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK")


def read_root_pose(backend: GenesisG1SceneBackend) -> RootPose:
    qpos = backend._read_root_qpos()
    return root_pose_from_qpos(qpos)


def root_pose_from_qpos(qpos: Sequence[float]) -> RootPose:
    if len(qpos) < 3:
        raise ValueError("root qpos must contain at least xyz")
    yaw = 0.0
    if len(qpos) >= 7:
        yaw = yaw_from_wxyz_quat(tuple(float(value) for value in qpos[3:7]))
    return RootPose(float(qpos[0]), float(qpos[1]), float(qpos[2]), yaw)


def yaw_from_wxyz_quat(quat: tuple[float, float, float, float]) -> float:
    qw, qx, qy, qz = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def horizontal_distance(start: RootPose, end: RootPose) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    return math.sqrt(dx * dx + dy * dy)


def path_length_xy(poses: Sequence[RootPose]) -> float:
    return sum(horizontal_distance(prev, curr) for prev, curr in zip(poses, poses[1:]))


def summarize_root_motion(
    poses: Sequence[RootPose],
    *,
    policy_rate_hz: int,
) -> dict[str, float]:
    if len(poses) < 2:
        raise ValueError("at least two root poses are required")
    duration_s = (len(poses) - 1) / float(policy_rate_hz)
    displacement = horizontal_distance(poses[0], poses[-1])
    return {
        "root_z_min": min(pose.z for pose in poses),
        "root_z_max": max(pose.z for pose in poses),
        "horizontal_displacement": displacement,
        "path_length_xy": path_length_xy(poses),
        "average_speed_xy": displacement / duration_s if duration_s > 0 else 0.0,
        "yaw_delta": wrap_angle(poses[-1].yaw - poses[0].yaw),
    }


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def resolve_link_index(robot: Any, link_name: str) -> int | None:
    if not link_name or not hasattr(robot, "get_link"):
        return None
    try:
        link = robot.get_link(link_name)
    except Exception:
        return None
    for attr in ("idx_local", "idx", "link_idx", "id"):
        if not hasattr(link, attr):
            continue
        value = getattr(link, attr)
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def read_foot_sample(
    robot: Any,
    link_idx: int | None,
    *,
    contact_threshold: float,
) -> FootSample:
    if link_idx is None:
        return FootSample(z=None, force=None, contact=None)
    pos = read_link_position(robot, link_idx)
    force = read_link_contact_force(robot, link_idx)
    contact = None if force is None else force >= contact_threshold
    return FootSample(
        z=None if pos is None else pos[2],
        force=force,
        contact=contact,
    )


def read_link_position(robot: Any, link_idx: int) -> tuple[float, float, float] | None:
    if not hasattr(robot, "get_links_pos"):
        return None
    try:
        values = _flatten_numeric(robot.get_links_pos(links_idx_local=(link_idx,)))
        if len(values) >= 3:
            return (values[0], values[1], values[2])
    except Exception:
        pass
    values = _flatten_numeric(robot.get_links_pos())
    start = link_idx * 3
    if len(values) < start + 3:
        return None
    return (values[start], values[start + 1], values[start + 2])


def read_link_contact_force(robot: Any, link_idx: int) -> float | None:
    if not hasattr(robot, "get_links_net_contact_force"):
        return None
    forces = _flatten_numeric(robot.get_links_net_contact_force())
    start = link_idx * 3
    if len(forces) < start + 3:
        return None
    fx, fy, fz = forces[start : start + 3]
    return math.sqrt(fx * fx + fy * fy + fz * fz)


def count_contact_switches(values: Sequence[bool | None]) -> int:
    cleaned = [value for value in values if value is not None]
    return sum(1 for prev, curr in zip(cleaned, cleaned[1:]) if prev != curr)


def count_single_support_frames(
    left_values: Sequence[bool | None],
    right_values: Sequence[bool | None],
) -> int:
    return sum(
        1
        for left, right in zip(left_values, right_values)
        if left is not None and right is not None and left != right
    )


def count_double_support_frames(
    left_values: Sequence[bool | None],
    right_values: Sequence[bool | None],
) -> int:
    return sum(1 for left, right in zip(left_values, right_values) if left is True and right is True)


def count_no_support_frames(
    left_values: Sequence[bool | None],
    right_values: Sequence[bool | None],
) -> int:
    return sum(
        1
        for left, right in zip(left_values, right_values)
        if left is False and right is False
    )


def print_foot_summary(label: str, samples: Sequence[FootSample]) -> None:
    z_values = [sample.z for sample in samples if sample.z is not None]
    force_values = [sample.force for sample in samples if sample.force is not None]
    contact_values = [sample.contact for sample in samples if sample.contact is not None]
    if z_values:
        print(f"{label}_FOOT_Z_MIN", min(z_values))
        print(f"{label}_FOOT_Z_MAX", max(z_values))
        print(f"{label}_FOOT_Z_RANGE", max(z_values) - min(z_values))
    else:
        print(f"{label}_FOOT_Z_AVAILABLE", False)
    if force_values:
        print(f"{label}_FOOT_FORCE_MAX", max(force_values))
    else:
        print(f"{label}_FOOT_FORCE_AVAILABLE", False)
    if contact_values:
        print(f"{label}_CONTACT_FRAMES", sum(1 for value in contact_values if value))
    else:
        print(f"{label}_CONTACT_AVAILABLE", False)


def _is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


if __name__ == "__main__":
    main()
