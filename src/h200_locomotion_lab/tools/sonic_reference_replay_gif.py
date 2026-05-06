"""Render a GIF of SONIC reference joint replay in Genesis."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from h200_locomotion_lab.envs.genesis_adapter import (
    G1_29DOF_JOINT_ORDER,
    GenesisG1Contract,
    import_genesis_module,
)
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    SONIC_G1_KDS,
    SONIC_G1_KPS,
    _flatten_numeric,
    _read_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC G1 29-motor MJCF.")
    parser.add_argument("--reference-dir", required=True, help="SONIC reference clip directory.")
    parser.add_argument("--output", required=True, help="Output GIF path.")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--logging-level", default="warning")
    parser.add_argument("--decimate", action="store_true")
    parser.add_argument("--convexify", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("dynamic", "kinematic"),
        default="kinematic",
        help="dynamic uses PD position targets; kinematic sets reference root/joints per frame.",
    )
    parser.add_argument("--no-plane", action="store_true")
    parser.add_argument("--camera-pos", nargs=3, type=float, default=(2.4, -3.2, 1.45))
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.68))
    parser.add_argument("--fov", type=float, default=35.0)
    args = parser.parse_args()

    asset = Path(args.asset)
    reference_dir = Path(args.reference_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    contract = GenesisG1Contract()
    joint_rows = _read_csv_rows(reference_dir / "joint_pos.csv", contract.action_dim)
    body_rows = _read_csv_rows(reference_dir / "body_pos.csv", 42)
    body_quat_rows = _read_csv_rows(reference_dir / "body_quat.csv", 56)
    frames_to_render = min(args.frames, len(joint_rows))
    if frames_to_render <= 0:
        raise ValueError("frames must be positive")

    gs = import_genesis_module()
    backend = getattr(gs, args.backend, args.backend)
    print("SONIC_REFERENCE_REPLAY_GIF_MODE joint_pos_as_position_targets")
    print("ASSET", asset)
    print("REF_DIR", reference_dir)
    print("OUTPUT", output)
    print("FRAMES", frames_to_render)
    print("RES", (args.width, args.height))
    print("DECIMATE", args.decimate)
    print("CONVEXIFY", args.convexify)
    print("MODE", args.mode)
    print("ADD_PLANE", not args.no_plane)
    print("CAMERA_POS", tuple(args.camera_pos))
    print("CAMERA_LOOKAT", tuple(args.camera_lookat))

    gs.init(backend=backend, logging_level=args.logging_level)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=contract.sim_dt_s),
    )
    if not args.no_plane:
        scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(asset),
            pos=(0.0, 0.0, 0.8),
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
    print("MOTOR_DOF_COUNT", len(motor_dof_indices))
    print("MOTOR_DOF_INDICES", motor_dof_indices)
    robot.set_dofs_kp(SONIC_G1_KPS, dofs_idx_local=motor_dof_indices)
    robot.set_dofs_kv(SONIC_G1_KDS, dofs_idx_local=motor_dof_indices)
    robot.set_pos(tuple(body_rows[0][:3]))
    robot.set_quat(tuple(body_quat_rows[0][:4]))
    robot.set_dofs_position(
        tuple(joint_rows[0]),
        dofs_idx_local=motor_dof_indices,
        zero_velocity=True,
    )
    robot.set_dofs_velocity(None)

    rendered_frames = []
    base_heights: list[float] = []
    start = time.time()
    for frame_index, target in enumerate(joint_rows[:frames_to_render]):
        if args.mode == "kinematic":
            robot.set_pos(tuple(body_rows[frame_index][:3]))
            robot.set_quat(tuple(body_quat_rows[frame_index][:4]))
            robot.set_dofs_position(
                tuple(target),
                dofs_idx_local=motor_dof_indices,
                zero_velocity=True,
            )
            scene.step()
        else:
            robot.control_dofs_position(tuple(target), dofs_idx_local=motor_dof_indices)
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
        if frame_index in {0, frames_to_render - 1}:
            print("FRAME", frame_index, "base_z", base_pos[2], "rgb_shape", getattr(rgb, "shape", None))

    import imageio.v2 as imageio

    duration = 1.0 / args.fps
    imageio.mimsave(output, rendered_frames, duration=duration)
    print("BASE_HEIGHT_MIN", min(base_heights))
    print("BASE_HEIGHT_MAX", max(base_heights))
    print("BASE_HEIGHT_FINAL", base_heights[-1])
    print("RENDERED_FRAMES", len(rendered_frames))
    print("ELAPSED_S", time.time() - start)
    print("GIF_BYTES", output.stat().st_size)
    print("SONIC_REFERENCE_REPLAY_GIF_OK")


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
