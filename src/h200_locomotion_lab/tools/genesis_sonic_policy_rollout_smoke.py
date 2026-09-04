"""Run a minimal decoder-only SONIC closed loop in Genesis."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from h200_locomotion_lab.envs.genesis_adapter import GenesisG1SceneBackend, GenesisSceneConfig
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
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
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    _read_contact_metrics,
    apply_sonic_g1_motor_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC-compatible G1 29DoF MJCF.")
    parser.add_argument("--decoder", required=True, help="Path to SONIC model_decoder.onnx.")
    parser.add_argument(
        "--obs-csv",
        help="Optional official 994D obs CSV used as token_state source.",
    )
    parser.add_argument("--token-mode", choices=("fixed", "replay"), default="fixed")
    parser.add_argument("--history-init", choices=("genesis", "official_obs"), default="genesis")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--root-qpos", nargs=7, type=float)
    parser.add_argument("--height-ok-min", type=float, default=0.3)
    parser.add_argument("--height-ok-max", type=float, default=1.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
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
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(
            asset_path=args.asset,
            backend=args.backend,
            base_pos=tuple(args.base_pos),
            base_quat=tuple(args.base_quat),
            root_qpos=tuple(args.root_qpos) if args.root_qpos else None,
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
    base_heights: list[float] = []
    max_abs_qvels: list[float] = []
    action_ranges: list[tuple[float, float, float]] = []
    observation_finite = True
    action_finite = True
    contact_counts: list[int] = []
    max_link_contact_forces: list[float] = []

    print("GENESIS_SONIC_POLICY_ROLLOUT_MODE decoder_only")
    print("ASSET", Path(args.asset))
    print("DECODER", Path(args.decoder))
    print("TOKEN_SOURCE", args.obs_csv or "zero")
    print("TOKEN_MODE", args.token_mode)
    print("TOKEN_ROWS", len(token_states) if token_states else 1)
    print("HISTORY_INIT", args.history_init)
    print("OBS_DIM", SONIC_DECODER_OBS_DIM)
    print("ACTION_MODE sonic_policy_raw")
    print("MOTOR_CONFIG", motor_config)
    print("BASE_POS", tuple(args.base_pos))
    print("BASE_QUAT", tuple(args.base_quat))
    print("ROOT_QPOS", tuple(args.root_qpos) if args.root_qpos else None)
    print("FRAMES", args.frames)

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
        if len(action) != backend.contract.action_dim:
            raise RuntimeError(
                f"Expected action_dim={backend.contract.action_dim}, got {len(action)}"
            )
        backend.step(action)
        base_z = _read_base_z(backend)
        max_abs_qvel = _read_max_abs_qvel(backend)
        contact_count, max_contact_force = _read_contact_metrics(backend.robot)
        action_min, action_max, action_max_abs = vector_range(action)
        base_heights.append(base_z)
        max_abs_qvels.append(max_abs_qvel)
        action_ranges.append((action_min, action_max, action_max_abs))
        contact_counts.append(contact_count)
        max_link_contact_forces.append(max_contact_force)
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
                "contact_count",
                contact_count,
                "max_contact_force",
                max_contact_force,
            )

    height_ok = bool(
        base_heights
        and min(base_heights) >= args.height_ok_min
        and max(base_heights) <= args.height_ok_max
    )
    finite_ok = (
        observation_finite
        and action_finite
        and all(math.isfinite(value) for value in base_heights)
    )
    print("OBS_FINITE", observation_finite)
    print("ACTION_FINITE", action_finite)
    print("BASE_HEIGHT_MIN", min(base_heights))
    print("BASE_HEIGHT_MAX", max(base_heights))
    print("BASE_HEIGHT_FINAL", base_heights[-1])
    print("MAX_ABS_QVEL", max(max_abs_qvels) if max_abs_qvels else 0.0)
    print("ACTION_MAX_ABS", max(item[2] for item in action_ranges))
    print("CONTACT_COUNT_MAX", max(contact_counts))
    print("CONTACT_COUNT_FINAL", contact_counts[-1])
    print("MAX_LINK_CONTACT_FORCE_MAX", max(max_link_contact_forces))
    print("MAX_LINK_CONTACT_FORCE_FINAL", max_link_contact_forces[-1])
    print("HEIGHT_OK_RANGE", args.height_ok_min, args.height_ok_max, height_ok)
    if not finite_ok or not height_ok:
        raise SystemExit("GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_FAILED")
    print("GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_OK")


def _read_base_z(backend: GenesisG1SceneBackend) -> float:
    qpos = backend._read_root_qpos()
    if len(qpos) < 3:
        return 0.0
    return float(qpos[2])


def _read_max_abs_qvel(backend: GenesisG1SceneBackend) -> float:
    try:
        values = backend.robot.get_dofs_velocity(
            dofs_idx_local=tuple(range(int(backend.robot.n_dofs)))
        )
        flattened = tuple(abs(value) for value in _flatten_numeric(values))
    except RECOVERABLE_RUNTIME_ERRORS:
        return 0.0
    return max(flattened, default=0.0)


def _flatten_numeric(values: object) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return (float(values),)
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def _is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


if __name__ == "__main__":
    main()
