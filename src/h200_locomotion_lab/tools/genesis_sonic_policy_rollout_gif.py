"""Render a GIF for the decoder-only SONIC closed loop in Genesis."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from h200_locomotion_lab.envs.genesis_adapter import (
    GenesisCameraConfig,
    GenesisG1SceneBackend,
    GenesisSceneConfig,
)
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
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import apply_sonic_g1_motor_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC-compatible G1 29DoF MJCF.")
    parser.add_argument("--decoder", required=True, help="Path to SONIC model_decoder.onnx.")
    parser.add_argument("--output", required=True, help="Output GIF path.")
    parser.add_argument("--obs-csv", help="Official 994D obs CSV used as token/history source.")
    parser.add_argument("--token-mode", choices=("fixed", "replay"), default="replay")
    parser.add_argument("--history-init", choices=("genesis", "official_obs"), default="official_obs")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--root-qpos", nargs=7, type=float)
    parser.add_argument(
        "--initial-joint-pos-csv",
        help="CSV whose selected 29D row is used as the physical reset motor pose.",
    )
    parser.add_argument("--initial-joint-pos-row", type=int, default=0)
    parser.add_argument("--camera-pos", nargs=3, type=float, default=(3.4, -4.2, 2.2))
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.85))
    parser.add_argument("--fov", type=float, default=42.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")

    obs_rows = (
        tuple(read_obs_csv_rows(Path(args.obs_csv), SONIC_DECODER_OBS_DIM, max_rows=args.frames))
        if args.obs_csv
        else ()
    )
    token_states = tuple(tuple(obs[:SONIC_TOKEN_DIM]) for obs in obs_rows)
    if args.token_mode == "replay" and not token_states:
        raise ValueError("--token-mode replay requires --obs-csv")
    if args.history_init == "official_obs" and not obs_rows:
        raise ValueError("--history-init official_obs requires --obs-csv")
    fixed_token_state = token_states[0] if token_states else (0.0,) * SONIC_TOKEN_DIM
    initial_motor_positions = (
        read_default_joint_positions(Path(args.initial_joint_pos_csv), args.initial_joint_pos_row, 29)
        if args.initial_joint_pos_csv
        else None
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
            camera=GenesisCameraConfig(
                res=(args.width, args.height),
                pos=tuple(args.camera_pos),
                lookat=tuple(args.camera_lookat),
                fov=args.fov,
                gui=False,
            ),
        )
    )
    if backend.camera is None:
        raise RuntimeError("Genesis camera was not created")
    camera = backend.camera
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

    print("GENESIS_SONIC_POLICY_ROLLOUT_GIF_MODE decoder_only")
    print("ASSET", Path(args.asset))
    print("DECODER", Path(args.decoder))
    print("OUTPUT", output)
    print("TOKEN_SOURCE", args.obs_csv or "zero")
    print("TOKEN_MODE", args.token_mode)
    print("HISTORY_INIT", args.history_init)
    print("MOTOR_CONFIG", motor_config)
    print("FRAMES", args.frames)
    print("RES", (args.width, args.height))
    print("ROOT_QPOS", tuple(args.root_qpos) if args.root_qpos else None)
    print("INITIAL_JOINT_POS_SOURCE", args.initial_joint_pos_csv or "default_motor_positions")
    if initial_motor_positions is not None:
        print("INITIAL_JOINT_POS_ROW", args.initial_joint_pos_row)
        print("INITIAL_JOINT_POS_MIN_MAX", min(initial_motor_positions), max(initial_motor_positions))
    print("CAMERA_POS", tuple(args.camera_pos))
    print("CAMERA_LOOKAT", tuple(args.camera_lookat))

    import imageio.v2 as imageio

    rendered_frames = []
    base_heights: list[float] = []
    action_abs_max: list[float] = []
    start = time.time()
    for frame_index in range(args.frames):
        token_state = (
            token_states[min(frame_index, len(token_states) - 1)]
            if args.token_mode == "replay"
            else fixed_token_state
        )
        observation = backend.sonic_decoder_observation(token_state)
        action = decoder.run(observation)
        action_min, action_max, action_max_abs = vector_range(action)
        backend.step(action)
        rgb, _, _, _ = camera.render(
            rgb=True,
            depth=False,
            segmentation=False,
            normal=False,
        )
        rendered_frames.append(rgb)
        base_z = _read_base_z(backend)
        base_heights.append(base_z)
        action_abs_max.append(action_max_abs)
        if frame_index % args.log_every == 0 or frame_index == args.frames - 1:
            print(
                "FRAME",
                frame_index,
                "base_z",
                base_z,
                "action_min",
                action_min,
                "action_max",
                action_max,
                "action_max_abs",
                action_max_abs,
                "rgb_shape",
                getattr(rgb, "shape", None),
            )

    imageio.mimsave(output, rendered_frames, duration=1.0 / args.fps)
    print("BASE_HEIGHT_MIN", min(base_heights))
    print("BASE_HEIGHT_MAX", max(base_heights))
    print("BASE_HEIGHT_FINAL", base_heights[-1])
    print("ACTION_MAX_ABS", max(action_abs_max))
    print("RENDERED_FRAMES", len(rendered_frames))
    print("ELAPSED_S", time.time() - start)
    print("GIF_BYTES", output.stat().st_size)
    print("GENESIS_SONIC_POLICY_ROLLOUT_GIF_OK")


def _read_base_z(backend: GenesisG1SceneBackend) -> float:
    qpos = backend._read_root_qpos()
    if len(qpos) < 3:
        return 0.0
    return float(qpos[2])


if __name__ == "__main__":
    main()
