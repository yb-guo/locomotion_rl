"""Probe decoder-only SONIC closed-loop drift against captured official obs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from h200_locomotion_lab.envs.genesis_adapter import GenesisG1SceneBackend, GenesisSceneConfig
from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_DECODER_OBS_DIM,
    SONIC_G1_DECODER_OBSERVATION_FIELDS,
    SonicG1HistoryBuffer,
    field_by_name,
    sonic_g1_history_from_decoder_observation,
)
from h200_locomotion_lab.tools.genesis_sonic_policy_rollout_smoke import (
    _read_base_z,
    _read_max_abs_qvel,
)
from h200_locomotion_lab.tools.sonic_policy_decoder_forward import (
    SonicOnnxReferenceDecoder,
    read_obs_csv_rows,
    vector_range,
)
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import apply_sonic_g1_motor_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--obs-csv", required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--root-qpos", nargs=7, type=float)
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--official-fields",
        default="",
        help="Comma-separated obs fields to replace with official captured values; use 'all'.",
    )
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    obs_rows = tuple(
        read_obs_csv_rows(Path(args.obs_csv), SONIC_DECODER_OBS_DIM, max_rows=args.frames)
    )
    if len(obs_rows) < args.frames:
        raise ValueError(f"Need {args.frames} official obs rows, got {len(obs_rows)}")
    official_fields = _parse_official_fields(args.official_fields)
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
    _, initial_frames = sonic_g1_history_from_decoder_observation(obs_rows[0])
    backend.sonic_history = SonicG1HistoryBuffer()
    for initial_frame in initial_frames:
        backend.sonic_history.append(initial_frame)

    print("GENESIS_SONIC_POLICY_ROLLOUT_PROBE decoder_only")
    print("OFFICIAL_FIELDS", tuple(official_fields))
    print("MOTOR_CONFIG", motor_config)
    print("FRAMES", args.frames)
    print("ROOT_QPOS", tuple(args.root_qpos) if args.root_qpos else None)

    base_heights: list[float] = []
    action_abs_max: list[float] = []
    action_drift_max: list[float] = []
    last_field_stats: dict[str, tuple[float, float]] = {}
    for frame_index in range(args.frames):
        official_obs = obs_rows[frame_index]
        token = official_obs[:64]
        genesis_obs = backend.sonic_decoder_observation(token)
        field_stats = _field_diff_stats(genesis_obs, official_obs)
        last_field_stats = field_stats
        decode_obs = _replace_fields(genesis_obs, official_obs, official_fields)
        action = decoder.run(decode_obs)
        official_action = decoder.run(official_obs)
        action_drift = _max_abs_diff(action, official_action)
        backend.step(action)
        base_z = _read_base_z(backend)
        max_abs_qvel = _read_max_abs_qvel(backend)
        _, _, action_max_abs = vector_range(action)
        base_heights.append(base_z)
        action_abs_max.append(action_max_abs)
        action_drift_max.append(action_drift)
        if frame_index % args.log_every == 0 or frame_index == args.frames - 1:
            print(
                "FRAME",
                frame_index,
                "base_z",
                base_z,
                "max_abs_qvel",
                max_abs_qvel,
                "action_max_abs",
                action_max_abs,
                "action_drift_vs_official",
                action_drift,
            )
            for field_name, (max_abs, mean_abs) in field_stats.items():
                print(
                    "FIELD_DIFF",
                    frame_index,
                    field_name,
                    "max_abs",
                    max_abs,
                    "mean_abs",
                    mean_abs,
                )

    print("BASE_HEIGHT_MIN", min(base_heights))
    print("BASE_HEIGHT_FINAL", base_heights[-1])
    print("ACTION_MAX_ABS", max(action_abs_max))
    print("ACTION_DRIFT_MAX", max(action_drift_max))
    for field_name, (max_abs, mean_abs) in last_field_stats.items():
        print("FINAL_FIELD_DIFF", field_name, "max_abs", max_abs, "mean_abs", mean_abs)
    print("GENESIS_SONIC_POLICY_ROLLOUT_PROBE_OK")


def _parse_official_fields(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    if value == "all":
        return tuple(field.name for field in SONIC_G1_DECODER_OBSERVATION_FIELDS)
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    for field in fields:
        field_by_name(field)
    return fields


def _replace_fields(
    genesis_obs: Sequence[float],
    official_obs: Sequence[float],
    fields: Sequence[str],
) -> tuple[float, ...]:
    obs = list(genesis_obs)
    for field_name in fields:
        field = field_by_name(field_name)
        obs[field.offset : field.offset + field.dim] = official_obs[
            field.offset : field.offset + field.dim
        ]
    return tuple(obs)


def _field_diff_stats(
    genesis_obs: Sequence[float],
    official_obs: Sequence[float],
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for field in SONIC_G1_DECODER_OBSERVATION_FIELDS:
        left = genesis_obs[field.offset : field.offset + field.dim]
        right = official_obs[field.offset : field.offset + field.dim]
        diffs = tuple(abs(float(a) - float(b)) for a, b in zip(left, right))
        stats[field.name] = (max(diffs), sum(diffs) / len(diffs))
    return stats


def _max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"shape mismatch: {len(left)} != {len(right)}")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


if __name__ == "__main__":
    main()
