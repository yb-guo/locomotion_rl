"""Render a GIF for an explicit Genesis G1 action replay sequence."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from h200_locomotion_lab.envs.genesis_adapter import (
    G1_29DOF_JOINT_ORDER,
    GenesisG1Contract,
    import_genesis_module,
)
from h200_locomotion_lab.tools.genesis_action_replay_smoke import (
    action_range,
    clip_action,
    count_out_of_range_actions,
    load_action_sequence,
    read_default_joint_positions,
)
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    apply_sonic_g1_motor_config,
    _flatten_numeric,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC G1 29-motor MJCF.")
    parser.add_argument("--output", required=True, help="Output GIF path.")
    parser.add_argument("--actions-csv", help="CSV with one 29D normalized action per row.")
    parser.add_argument("--fixture", choices=("zero", "sine", "pulse"), default="sine")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--amplitude", type=float, default=0.12)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--logging-level", default="warning")
    parser.add_argument("--decimate", action="store_true")
    parser.add_argument("--convexify", action="store_true")
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.8))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--default-joint-pos-csv")
    parser.add_argument("--default-joint-pos-row", type=int, default=0)
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    parser.add_argument("--camera-pos", nargs=3, type=float, default=(3.4, -4.2, 2.2))
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.95))
    parser.add_argument("--fov", type=float, default=42.0)
    args = parser.parse_args()

    contract = GenesisG1Contract()
    actions = load_action_sequence(
        actions_csv=Path(args.actions_csv) if args.actions_csv else None,
        fixture=args.fixture,
        frames=args.frames,
        action_dim=contract.action_dim,
        amplitude=args.amplitude,
    )
    action_min, action_max, action_max_abs = action_range(actions)

    asset = Path(args.asset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    gs = import_genesis_module()
    backend = getattr(gs, args.backend, args.backend)
    print("GENESIS_ACTION_REPLAY_GIF_MODE normalized_actions")
    print("ASSET", asset)
    print("OUTPUT", output)
    print("ACTIONS_SOURCE", args.actions_csv or f"fixture:{args.fixture}")
    print("FRAMES", len(actions))
    print("ACTION_MIN_MAX", action_min, action_max)
    print("ACTION_MAX_ABS", action_max_abs)
    print("ACTION_OUT_OF_RANGE_VALUES", count_out_of_range_actions(actions))
    print("RES", (args.width, args.height))
    print("DECIMATE", args.decimate)
    print("CONVEXIFY", args.convexify)
    print("BASE_POS", tuple(args.base_pos))
    print("BASE_QUAT", tuple(args.base_quat))
    default_motor_positions_override = (
        read_default_joint_positions(
            Path(args.default_joint_pos_csv),
            args.default_joint_pos_row,
            contract.action_dim,
        )
        if args.default_joint_pos_csv
        else None
    )
    print("DEFAULT_JOINT_POS_SOURCE", args.default_joint_pos_csv or "asset_qpos0")
    if default_motor_positions_override is not None:
        print("DEFAULT_JOINT_POS_ROW", args.default_joint_pos_row)
        print(
            "DEFAULT_JOINT_POS_MIN_MAX",
            min(default_motor_positions_override),
            max(default_motor_positions_override),
        )
    print("CAMERA_POS", tuple(args.camera_pos))
    print("CAMERA_LOOKAT", tuple(args.camera_lookat))

    gs.init(backend=backend, logging_level=args.logging_level)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=contract.sim_dt_s),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(asset),
            pos=tuple(args.base_pos),
            quat=tuple(args.base_quat),
            convexify=args.convexify,
            decimate=args.decimate,
        )
    )
    camera = scene.add_camera(
        res=(args.width, args.height),
        pos=tuple(args.camera_pos),
        lookat=tuple(args.camera_lookat),
        fov=args.fov,
        GUI=False,
    )
    scene.build(n_envs=1)

    motor_dof_indices = _resolve_motor_dof_indices(robot)
    default_motor_positions = default_motor_positions_override or _flatten_numeric(
        robot.get_dofs_position(dofs_idx_local=motor_dof_indices)
    )
    if not args.no_sonic_motor_config:
        apply_sonic_g1_motor_config(robot, motor_dof_indices)
        print("MOTOR_CONFIG", "sonic_g1_kp_kv_force_range")
    else:
        print("MOTOR_CONFIG", "genesis_default")
    robot.set_dofs_position(
        default_motor_positions,
        dofs_idx_local=motor_dof_indices,
        zero_velocity=True,
    )
    robot.set_dofs_velocity(None)
    print("MOTOR_DOF_COUNT", len(motor_dof_indices))
    print("MOTOR_DOF_INDICES", motor_dof_indices)

    rendered_frames = []
    base_heights: list[float] = []
    start = time.time()
    for frame_index, action in enumerate(actions):
        clipped_action = clip_action(action)
        target = tuple(
            default + contract.action_scale_rad * delta
            for default, delta in zip(default_motor_positions, clipped_action)
        )
        robot.control_dofs_position(target, dofs_idx_local=motor_dof_indices)
        for _ in range(contract.decimation):
            scene.step()
        rgb, _, _, _ = camera.render(
            rgb=True,
            depth=False,
            segmentation=False,
            normal=False,
        )
        rendered_frames.append(rgb)
        base_pos = _flatten_numeric(robot.get_pos())
        base_heights.append(base_pos[2])
        if frame_index in {0, len(actions) - 1}:
            print("FRAME", frame_index, "base_z", base_pos[2], "rgb_shape", getattr(rgb, "shape", None))

    import imageio.v2 as imageio

    imageio.mimsave(output, rendered_frames, duration=1.0 / args.fps)
    print("BASE_HEIGHT_MIN", min(base_heights))
    print("BASE_HEIGHT_MAX", max(base_heights))
    print("BASE_HEIGHT_FINAL", base_heights[-1])
    print("RENDERED_FRAMES", len(rendered_frames))
    print("ELAPSED_S", time.time() - start)
    print("GIF_BYTES", output.stat().st_size)
    print("GENESIS_ACTION_REPLAY_GIF_OK")


def _resolve_motor_dof_indices(robot: object) -> tuple[int, ...]:
    indices: list[int] = []
    for joint_name in G1_29DOF_JOINT_ORDER:
        joint = robot.get_joint(joint_name)
        joint_indices = getattr(joint, "dofs_idx_local")
        if len(joint_indices) != 1:
            raise ValueError(f"Expected single-DoF joint {joint_name}, got {joint_indices}")
        indices.append(int(joint_indices[0]))
    return tuple(indices)


if __name__ == "__main__":
    main()
