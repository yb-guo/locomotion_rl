"""Run Genesis with online SONIC planner -> encoder -> decoder tokens."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.genesis_adapter import (
    GenesisCameraConfig,
    GenesisG1SceneBackend,
    GenesisSceneConfig,
)
from h200_locomotion_lab.envs.robot_backend import (
    read_genesis_g1_robot_state,
    robot_state_to_planner_qpos,
)
from h200_locomotion_lab.sonic.g1_observation import SONIC_TOKEN_DIM
from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_ENCODER_OBS_DIM,
    SONIC_PLANNER_CONTEXT_FRAMES,
    SONIC_PLANNER_QPOS_DIM,
    SonicPlannerMotion50Hz,
    build_g1_encoder_observation_from_planner_motion,
    build_planner_context_from_motion,
    build_planner_context_from_mujoco_qpos_history,
    resample_planner_mujoco_qpos_to_50hz,
)
from h200_locomotion_lab.tools.genesis_action_replay_smoke import read_default_joint_positions
from h200_locomotion_lab.tools.genesis_sonic_policy_locomotion_probe import (
    count_contact_switches,
    count_double_support_frames,
    count_no_support_frames,
    count_single_support_frames,
    emit,
    enforce_wall_time,
    heartbeat,
    horizontal_distance,
    path_length_xy,
    read_foot_sample,
    read_root_pose,
    resolve_link_index,
    summarize_root_motion,
)
from h200_locomotion_lab.tools.sonic_planner_encoder_decoder_forward import (
    SonicOnnxReferenceModel,
    _flatten_numeric,
    read_planner_qpos_csv,
)
from h200_locomotion_lab.tools.sonic_policy_decoder_forward import (
    SonicOnnxReferenceDecoder,
    vector_range,
)
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    _read_contact_metrics,
    apply_sonic_g1_motor_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC-compatible G1 29DoF MJCF.")
    parser.add_argument("--planner", required=True, help="Path to planner_sonic.onnx.")
    parser.add_argument("--planner-runner", required=True, help="Compiled ONNX Runtime planner runner.")
    parser.add_argument("--encoder", required=True, help="Path to SONIC model_encoder.onnx.")
    parser.add_argument("--decoder", required=True, help="Path to SONIC model_decoder.onnx.")
    parser.add_argument("--work-dir", required=True, help="Directory for planner context/qpos/log files.")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument(
        "--initial-context-source",
        choices=("initial_joint_csv", "genesis"),
        default="initial_joint_csv",
    )
    parser.add_argument(
        "--replan-context-source",
        choices=("motion", "genesis"),
        default="motion",
    )
    parser.add_argument("--planner-timeout-s", type=float, default=300.0)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--root-qpos", nargs=7, type=float)
    parser.add_argument("--initial-joint-pos-csv")
    parser.add_argument("--initial-joint-pos-row", type=int, default=0)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--target-vel", type=float, default=-1.0)
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--movement-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--facing-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--left-foot-link", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-link", default="right_ankle_roll_link")
    parser.add_argument("--foot-contact-force-threshold", type=float, default=5.0)
    parser.add_argument("--min-horizontal-displacement", type=float, default=0.05)
    parser.add_argument("--min-contact-switches", type=int, default=2)
    parser.add_argument("--height-ok-min", type=float, default=0.3)
    parser.add_argument("--height-ok-max", type=float, default=1.2)
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    parser.add_argument("--skip-foot-metrics", action="store_true")
    parser.add_argument("--heartbeat-every-frame", action="store_true")
    parser.add_argument("--progress-file")
    parser.add_argument("--max-wall-time-s", type=float)
    parser.add_argument("--output-gif", help="Optional GIF rendered from the same rollout.")
    parser.add_argument("--output-mp4", help="Optional MP4 rendered from the same rollout.")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height-px", type=int, default=320)
    parser.add_argument("--camera-pos", nargs=3, type=float, default=(3.4, -4.2, 2.2))
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.85))
    parser.add_argument("--fov", type=float, default=42.0)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.replan_interval < 0:
        raise ValueError("--replan-interval must be non-negative")
    if args.planner_timeout_s <= 0:
        raise ValueError("--planner-timeout-s must be positive")
    if args.max_wall_time_s is not None and args.max_wall_time_s <= 0:
        raise ValueError("--max-wall-time-s must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    render_enabled = args.output_gif is not None or args.output_mp4 is not None

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_file) if args.progress_file else None
    started_at = time.monotonic()

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
            camera=(
                GenesisCameraConfig(
                    res=(args.width, args.height_px),
                    pos=tuple(args.camera_pos),
                    lookat=tuple(args.camera_lookat),
                    fov=args.fov,
                    gui=False,
                )
                if render_enabled
                else None
            ),
        )
    )
    if render_enabled and backend.camera is None:
        raise RuntimeError("Genesis camera was not created")
    encoder = SonicOnnxReferenceModel(Path(args.encoder))
    decoder = SonicOnnxReferenceDecoder(Path(args.decoder))

    if not args.no_sonic_motor_config:
        apply_sonic_g1_motor_config(backend.robot, backend.motor_dof_indices)
        motor_config = "sonic_g1_kp_kv_force_range"
    else:
        motor_config = "genesis_default"

    backend.reset()
    genesis_qpos_history = [read_planner_qpos_from_genesis(backend)]
    initial_context_rows = (
        build_planner_context_from_mujoco_qpos_history(genesis_qpos_history)
        if args.initial_context_source == "genesis"
        else None
    )
    motion, planner_num_pred_frames = run_planner_runner(
        args,
        work_dir,
        tag="initial",
        context_rows=initial_context_rows,
    )
    motion_start_frame = 0
    planner_calls = 1

    left_link_idx = (
        None if args.skip_foot_metrics else resolve_link_index(backend.robot, args.left_foot_link)
    )
    right_link_idx = (
        None if args.skip_foot_metrics else resolve_link_index(backend.robot, args.right_foot_link)
    )

    root_poses = [read_root_pose(backend)]
    token_max_abs_values: list[float] = []
    action_max_abs_values: list[float] = []
    contact_counts: list[int] = []
    max_contact_forces: list[float] = []
    left_samples = []
    right_samples = []
    rendered_frames = []
    encoder_observation_finite = True
    token_finite = True
    decoder_observation_finite = True
    action_finite = True

    emit("GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_MODE online_planner_encoder")
    emit("ASSET", Path(args.asset))
    emit("PLANNER", Path(args.planner))
    emit("PLANNER_RUNNER", Path(args.planner_runner))
    emit("ENCODER", Path(args.encoder))
    emit("DECODER", Path(args.decoder))
    emit("WORK_DIR", work_dir)
    emit("REPLAY_OBS_USED", False)
    emit("REPLAY_TOKEN_USED", False)
    emit("MOTOR_CONFIG", motor_config)
    emit("FRAMES", args.frames)
    emit("REPLAN_INTERVAL", args.replan_interval)
    emit("INITIAL_CONTEXT_SOURCE", args.initial_context_source)
    emit("REPLAN_CONTEXT_SOURCE", args.replan_context_source)
    emit("INITIAL_PLANNER_NUM_PRED_FRAMES", planner_num_pred_frames)
    emit("ROOT_QPOS", tuple(args.root_qpos) if args.root_qpos else None)
    emit("INITIAL_JOINT_POS_SOURCE", args.initial_joint_pos_csv or "default_motor_positions")
    emit("LEFT_FOOT_LINK", args.left_foot_link, "INDEX", left_link_idx)
    emit("RIGHT_FOOT_LINK", args.right_foot_link, "INDEX", right_link_idx)
    emit("SKIP_FOOT_METRICS", args.skip_foot_metrics)
    emit("PROGRESS_FILE", progress_path or "not_set")
    emit("MAX_WALL_TIME_S", args.max_wall_time_s or "not_set")
    emit("OUTPUT_GIF", args.output_gif or "not_set")
    emit("OUTPUT_MP4", args.output_mp4 or "not_set")
    if render_enabled:
        emit("RENDER_RES", (args.width, args.height_px))
        emit("RENDER_FPS", args.fps)
        emit("CAMERA_POS", tuple(args.camera_pos))
        emit("CAMERA_LOOKAT", tuple(args.camera_lookat))

    for frame_index in range(args.frames):
        enforce_wall_time(started_at, args.max_wall_time_s)
        heartbeat(args, progress_path, frame_index, "begin")

        if args.replan_interval and frame_index > 0 and frame_index % args.replan_interval == 0:
            if args.replan_context_source == "genesis":
                context_rows = build_planner_context_from_mujoco_qpos_history(
                    genesis_qpos_history,
                )
            else:
                context_rows = build_planner_context_from_motion(
                    motion,
                    gen_frame=frame_index - motion_start_frame,
                )
            motion, planner_num_pred_frames = run_planner_runner(
                args,
                work_dir,
                tag=f"replan_{frame_index:04d}",
                context_rows=context_rows,
            )
            motion_start_frame = frame_index
            planner_calls += 1
            emit(
                "REPLAN",
                frame_index,
                "planner_calls",
                planner_calls,
                "num_pred_frames",
                planner_num_pred_frames,
                "motion_timesteps",
                motion.timesteps,
            )

        motion_frame = frame_index - motion_start_frame
        encoder_observation = build_g1_encoder_observation_from_planner_motion(
            motion,
            current_frame=motion_frame,
            robot_base_quat=backend._read_base_quat(),
        )
        encoder_observation_finite = encoder_observation_finite and _is_finite(encoder_observation)
        if len(encoder_observation) != SONIC_ENCODER_OBS_DIM:
            raise SystemExit("GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_FAILED")
        heartbeat(args, progress_path, frame_index, "encoder_observation")

        token_state = tuple(
            float(value)
            for value in _flatten_numeric(
                encoder.run({"obs_dict": _row_array(encoder_observation)})["encoded_tokens"]
            )
        )
        token_finite = token_finite and _is_finite(token_state)
        if len(token_state) != SONIC_TOKEN_DIM:
            raise SystemExit("GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_FAILED")
        heartbeat(args, progress_path, frame_index, "token")

        decoder_observation = backend.sonic_decoder_observation(token_state)
        decoder_observation_finite = decoder_observation_finite and _is_finite(decoder_observation)
        action = decoder.run(decoder_observation)
        action_finite = action_finite and _is_finite(action)
        heartbeat(args, progress_path, frame_index, "action")

        backend.step(action)
        heartbeat(args, progress_path, frame_index, "step")
        genesis_qpos_history.append(read_planner_qpos_from_genesis(backend))
        if render_enabled:
            rgb, _, _, _ = backend.camera.render(
                rgb=True,
                depth=False,
                segmentation=False,
                normal=False,
            )
            rendered_frames.append(rgb)
            heartbeat(args, progress_path, frame_index, "render")

        root_pose = read_root_pose(backend)
        contact_count, max_contact_force = (
            (None, None) if args.skip_foot_metrics else _read_contact_metrics(backend.robot)
        )
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
        _, _, token_max_abs = vector_range(token_state)
        _, _, action_max_abs = vector_range(action)
        heartbeat(args, progress_path, frame_index, "metrics")

        root_poses.append(root_pose)
        token_max_abs_values.append(token_max_abs)
        action_max_abs_values.append(action_max_abs)
        if contact_count is not None:
            contact_counts.append(contact_count)
        if max_contact_force is not None:
            max_contact_forces.append(max_contact_force)
        left_samples.append(left_sample)
        right_samples.append(right_sample)

        if frame_index % args.log_every == 0 or frame_index == args.frames - 1:
            emit(
                "FRAME",
                frame_index,
                "planner_motion_frame",
                motion_frame,
                "root_x",
                root_pose.x,
                "root_y",
                root_pose.y,
                "root_z",
                root_pose.z,
                "disp_xy",
                horizontal_distance(root_poses[0], root_pose),
                "path_xy",
                path_length_xy(root_poses),
                "token_max_abs",
                token_max_abs,
                "action_max_abs",
                action_max_abs,
            )

    root_summary = summarize_root_motion(root_poses, policy_rate_hz=backend.contract.policy_rate_hz)
    left_contacts = [sample.contact for sample in left_samples]
    right_contacts = [sample.contact for sample in right_samples]
    left_contact_switches = count_contact_switches(left_contacts)
    right_contact_switches = count_contact_switches(right_contacts)
    total_contact_switches = left_contact_switches + right_contact_switches
    single_support_frames = count_single_support_frames(left_contacts, right_contacts)
    double_support_frames = count_double_support_frames(left_contacts, right_contacts)
    no_support_frames = count_no_support_frames(left_contacts, right_contacts)
    finite_ok = (
        encoder_observation_finite
        and token_finite
        and decoder_observation_finite
        and action_finite
    )
    height_ok = (
        args.height_ok_min <= root_summary["root_z_min"] <= args.height_ok_max
        and args.height_ok_min <= root_summary["root_z_max"] <= args.height_ok_max
    )
    translation_observed = root_summary["horizontal_displacement"] >= args.min_horizontal_displacement
    foot_alternation_observed = total_contact_switches >= args.min_contact_switches
    locomotion_observed = translation_observed and foot_alternation_observed

    emit("PLANNER_CALLS", planner_calls)
    emit("GENESIS_QPOS_HISTORY_FRAMES", len(genesis_qpos_history))
    emit("ENCODER_OBS_FINITE", encoder_observation_finite)
    emit("TOKEN_FINITE", token_finite)
    emit("DECODER_OBS_FINITE", decoder_observation_finite)
    emit("ACTION_FINITE", action_finite)
    emit("FINITE_OK", finite_ok)
    emit("TOKEN_MAX_ABS", max(token_max_abs_values))
    emit("ACTION_MAX_ABS", max(action_max_abs_values))
    emit("ROOT_Z_MIN", root_summary["root_z_min"])
    emit("ROOT_Z_MAX", root_summary["root_z_max"])
    emit("HORIZONTAL_DISPLACEMENT", root_summary["horizontal_displacement"])
    emit("PATH_LENGTH_XY", root_summary["path_length_xy"])
    emit("AVERAGE_SPEED_XY", root_summary["average_speed_xy"])
    emit("YAW_DELTA", root_summary["yaw_delta"])
    if contact_counts:
        emit("CONTACT_COUNT_MAX", max(contact_counts))
        emit("CONTACT_COUNT_FINAL", contact_counts[-1])
    if max_contact_forces:
        emit("MAX_LINK_CONTACT_FORCE_MAX", max(max_contact_forces))
        emit("MAX_LINK_CONTACT_FORCE_FINAL", max_contact_forces[-1])
    emit("LEFT_CONTACT_SWITCHES", left_contact_switches)
    emit("RIGHT_CONTACT_SWITCHES", right_contact_switches)
    emit("TOTAL_CONTACT_SWITCHES", total_contact_switches)
    emit("SINGLE_SUPPORT_FRAMES", single_support_frames)
    emit("DOUBLE_SUPPORT_FRAMES", double_support_frames)
    emit("NO_SUPPORT_FRAMES", no_support_frames)
    emit("TRANSLATION_THRESHOLD", args.min_horizontal_displacement)
    emit("CONTACT_SWITCH_THRESHOLD", args.min_contact_switches)
    emit("TRANSLATION_OBSERVED", translation_observed)
    emit("FOOT_ALTERNATION_OBSERVED", foot_alternation_observed)
    emit("LOCOMOTION_OBSERVED", locomotion_observed)
    emit("HEIGHT_OK_RANGE", args.height_ok_min, args.height_ok_max, height_ok)
    if render_enabled:
        write_video_outputs(args, rendered_frames)
        emit("RENDERED_FRAMES", len(rendered_frames))
    if not finite_ok or not height_ok:
        raise SystemExit("GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_FAILED")
    heartbeat(args, progress_path, args.frames, "done")
    emit("GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK")


def run_planner_runner(
    args: argparse.Namespace,
    work_dir: Path,
    *,
    tag: str,
    context_rows: Sequence[Sequence[float]] | None = None,
) -> tuple[SonicPlannerMotion50Hz, int]:
    qpos_csv = work_dir / f"planner_{tag}_qpos.csv"
    stdout_log = work_dir / f"planner_{tag}.stdout.log"
    stderr_log = work_dir / f"planner_{tag}.stderr.log"
    command = [
        str(Path(args.planner_runner)),
        "--planner",
        str(Path(args.planner)),
        "--output-qpos-csv",
        str(qpos_csv),
        "--mode",
        str(args.mode),
        "--target-vel",
        str(args.target_vel),
        "--height",
        str(args.height),
        "--random-seed",
        str(args.random_seed),
        "--movement-direction",
        *(str(value) for value in args.movement_direction),
        "--facing-direction",
        *(str(value) for value in args.facing_direction),
    ]
    if context_rows is not None:
        context_csv = work_dir / f"planner_{tag}_context.csv"
        write_planner_context_csv(context_csv, context_rows)
        command.extend(["--context-qpos-csv", str(context_csv)])
    elif args.initial_joint_pos_csv:
        command.extend(
            [
                "--initial-joint-pos-csv",
                str(Path(args.initial_joint_pos_csv)),
                "--initial-joint-pos-row",
                str(args.initial_joint_pos_row),
            ]
        )

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.planner_timeout_s,
    )
    stdout_log.write_text(completed.stdout)
    stderr_log.write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"planner runner failed with code {completed.returncode}; "
            f"stdout={stdout_log} stderr={stderr_log}"
        )
    num_pred_frames = parse_num_pred_frames(completed.stdout)
    qpos_rows = read_planner_qpos_csv(qpos_csv)
    motion = resample_planner_mujoco_qpos_to_50hz(
        qpos_rows,
        num_pred_frames=num_pred_frames,
    )
    return motion, num_pred_frames


def write_video_outputs(args: argparse.Namespace, rendered_frames: Sequence[Any]) -> None:
    if not rendered_frames:
        raise ValueError("rendered_frames must not be empty")
    import imageio.v2 as imageio

    if args.output_gif:
        output_gif = Path(args.output_gif)
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, rendered_frames, duration=1.0 / args.fps)
        emit("GIF_OUTPUT", output_gif)
        emit("GIF_BYTES", output_gif.stat().st_size)
    if args.output_mp4:
        output_mp4 = Path(args.output_mp4)
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_mp4, rendered_frames, fps=args.fps)
        emit("MP4_OUTPUT", output_mp4)
        emit("MP4_BYTES", output_mp4.stat().st_size)


def read_planner_qpos_from_genesis(backend: GenesisG1SceneBackend) -> tuple[float, ...]:
    return robot_state_to_planner_qpos(read_genesis_g1_robot_state(backend))


def parse_num_pred_frames(output: str) -> int:
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] == "PLANNER_NUM_PRED_FRAMES":
            value = int(parts[1])
            if value <= 0:
                raise ValueError("PLANNER_NUM_PRED_FRAMES must be positive")
            return value
    raise ValueError("PLANNER_NUM_PRED_FRAMES not found in planner runner output")


def write_planner_context_csv(path: Path, rows: Sequence[Sequence[float]]) -> None:
    if len(rows) != SONIC_PLANNER_CONTEXT_FRAMES:
        raise ValueError(f"expected {SONIC_PLANNER_CONTEXT_FRAMES} context rows, got {len(rows)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for row_index, row in enumerate(rows, start=1):
            if len(row) != SONIC_PLANNER_QPOS_DIM:
                raise ValueError(
                    f"context row {row_index} expected {SONIC_PLANNER_QPOS_DIM} values, got {len(row)}"
                )
            writer.writerow([f"{float(value):.9g}" for value in row])


def _row_array(values: Sequence[float]) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Running SONIC encoder requires numpy") from exc
    return np.asarray(values, dtype=np.float32).reshape(1, len(values))


def _is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


if __name__ == "__main__":
    main()
