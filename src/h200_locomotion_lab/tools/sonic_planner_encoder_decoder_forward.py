"""Run SONIC planner -> encoder -> decoder without replayed obs/token rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_DECODER_OBS_DIM,
    SONIC_HISTORY_FRAMES,
    SONIC_TOKEN_DIM,
    SonicG1HistoryFrame,
    build_sonic_g1_decoder_observation,
    mujoco_motor_state_to_sonic_body_state,
)
from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_PLANNER_QPOS_DIM,
    build_g1_encoder_observation_from_planner_motion,
    build_planner_inputs,
    resample_planner_mujoco_qpos_to_50hz,
)
from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_ACTION_DIM,
    SONIC_G1_DEFAULT_ANGLES,
)
from h200_locomotion_lab.tools.genesis_action_replay_smoke import read_default_joint_positions
from h200_locomotion_lab.tools.sonic_policy_decoder_forward import (
    SonicOnnxReferenceDecoder,
    vector_range,
    write_action_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", help="Path to planner_sonic.onnx.")
    parser.add_argument(
        "--planner-qpos-csv",
        help="Precomputed planner mujoco_qpos CSV; skips Python planner ONNX execution.",
    )
    parser.add_argument("--planner-num-pred-frames", type=int)
    parser.add_argument("--encoder", required=True, help="Path to model_encoder.onnx.")
    parser.add_argument("--decoder", required=True, help="Path to model_decoder.onnx.")
    parser.add_argument(
        "--initial-joint-pos-csv",
        help="Optional 29D MuJoCo-order q row used for planner context and decoder history.",
    )
    parser.add_argument("--initial-joint-pos-row", type=int, default=0)
    parser.add_argument("--mode", type=int, default=2, help="Planner locomotion mode; 2=WALK.")
    parser.add_argument("--target-vel", type=float, default=-1.0)
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--movement-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--facing-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument(
        "--token-frames",
        type=int,
        default=1,
        help="Number of consecutive planner-motion frames to encode into token rows.",
    )
    parser.add_argument("--output-token-csv", help="Write the generated 64D encoder token.")
    parser.add_argument("--output-action-csv", help="Write the generated 29D raw decoder action.")
    args = parser.parse_args()

    initial_joints = (
        read_default_joint_positions(Path(args.initial_joint_pos_csv), args.initial_joint_pos_row, 29)
        if args.initial_joint_pos_csv
        else SONIC_G1_DEFAULT_ANGLES
    )
    planner_inputs = build_planner_inputs(
        initial_joints,
        mode=args.mode,
        target_vel=args.target_vel,
        movement_direction=args.movement_direction,
        facing_direction=args.facing_direction,
        random_seed=args.random_seed,
        height=args.height,
    )

    if args.planner_qpos_csv:
        mujoco_qpos = read_planner_qpos_csv(Path(args.planner_qpos_csv))
        num_pred_frames = args.planner_num_pred_frames or len(mujoco_qpos)
        planner_source = "csv"
    else:
        if not args.planner:
            raise ValueError("--planner is required unless --planner-qpos-csv is provided")
        planner = SonicOnnxReferenceModel(Path(args.planner))
        planner_outputs = planner.run(_planner_inputs_to_numpy(planner_inputs))
        mujoco_qpos = _reshape_planner_qpos(planner_outputs["mujoco_qpos"])
        num_pred_frames = int(_flatten_numeric(planner_outputs["num_pred_frames"])[0])
        planner_source = "onnx_reference"
    motion = resample_planner_mujoco_qpos_to_50hz(
        mujoco_qpos,
        num_pred_frames=num_pred_frames,
    )

    if args.token_frames <= 0:
        raise ValueError("--token-frames must be positive")

    encoder = SonicOnnxReferenceModel(Path(args.encoder))
    encoder_observations = tuple(
        build_g1_encoder_observation_from_planner_motion(motion, current_frame=frame_index)
        for frame_index in range(args.token_frames)
    )
    token_rows = tuple(
        tuple(
            float(value)
            for value in _flatten_numeric(encoder.run({"obs_dict": _row_array(observation)})[
                "encoded_tokens"
            ])
        )
        for observation in encoder_observations
    )
    token_state = token_rows[0]

    decoder_observation = build_sonic_g1_decoder_observation(
        token_state,
        _standing_history(initial_joints),
    )
    decoder = SonicOnnxReferenceDecoder(Path(args.decoder))
    action = decoder.run(decoder_observation)

    token_range = vector_range(token_state)
    action_range = vector_range(action)
    planner_root_z = [frame[2] for frame in mujoco_qpos[: max(1, num_pred_frames)]]

    print("SONIC_PLANNER_ENCODER_DECODER_FORWARD_MODE onnx_reference")
    print("PLANNER", Path(args.planner) if args.planner else "not_run")
    print("PLANNER_SOURCE", planner_source)
    print("PLANNER_QPOS_CSV", args.planner_qpos_csv or "not_set")
    print("ENCODER", Path(args.encoder))
    print("DECODER", Path(args.decoder))
    print("REPLAY_OBS_USED", False)
    print("REPLAY_TOKEN_USED", False)
    print("INITIAL_JOINT_POS_SOURCE", args.initial_joint_pos_csv or "sonic_default_angles")
    print("PLANNER_MODE", args.mode)
    print("TARGET_VEL", args.target_vel)
    print("MOVEMENT_DIRECTION", tuple(args.movement_direction))
    print("FACING_DIRECTION", tuple(args.facing_direction))
    print("PLANNER_QPOS_DIM", len(mujoco_qpos), len(mujoco_qpos[0]))
    print("PLANNER_NUM_PRED_FRAMES", num_pred_frames)
    print("PLANNER_QPOS_FINITE", _is_finite(_flatten_rows(mujoco_qpos)))
    print("PLANNER_ROOT_Z_MIN_MAX", min(planner_root_z), max(planner_root_z))
    print("MOTION_50HZ_TIMESTEPS", motion.timesteps)
    print("ENCODER_OBS_DIM", len(encoder_observations[0]))
    print("ENCODER_OBS_ROWS", len(encoder_observations))
    print("ENCODER_OBS_FINITE", all(_is_finite(observation) for observation in encoder_observations))
    print("TOKEN_ROWS_GENERATED", len(token_rows))
    print("TOKEN_DIM", len(token_state))
    print("TOKEN_FINITE", _is_finite(token_state))
    print("TOKEN_MIN_MAX_ABS", token_range[0], token_range[1], token_range[2])
    print("TOKEN_FIRST8", tuple(token_state[:8]))
    print("DECODER_OBS_DIM", len(decoder_observation))
    print("DECODER_OBS_FINITE", _is_finite(decoder_observation))
    print("ACTION_DIM", len(action))
    print("ACTION_FINITE", _is_finite(action))
    print("ACTION_MIN_MAX_ABS", action_range[0], action_range[1], action_range[2])
    print("ACTION_FIRST10", tuple(action[:10]))

    if args.output_token_csv:
        write_rows_csv(Path(args.output_token_csv), token_rows)
        print("OUTPUT_TOKEN_CSV", Path(args.output_token_csv))
    if args.output_action_csv:
        write_action_csv(Path(args.output_action_csv), (action,))
        print("OUTPUT_ACTION_CSV", Path(args.output_action_csv))

    if len(token_state) != SONIC_TOKEN_DIM:
        raise SystemExit(f"encoder produced token dim {len(token_state)}, expected {SONIC_TOKEN_DIM}")
    if len(action) != SONIC_ACTION_DIM:
        raise SystemExit(f"decoder produced action dim {len(action)}, expected {SONIC_ACTION_DIM}")
    if len(decoder_observation) != SONIC_DECODER_OBS_DIM:
        raise SystemExit(
            f"decoder observation dim {len(decoder_observation)}, expected {SONIC_DECODER_OBS_DIM}"
        )
    if not (
        _is_finite(_flatten_rows(mujoco_qpos))
        and all(_is_finite(observation) for observation in encoder_observations)
        and all(_is_finite(row) for row in token_rows)
        and _is_finite(decoder_observation)
        and _is_finite(action)
    ):
        raise SystemExit("non-finite planner/encoder/decoder value")
    print("SONIC_PLANNER_ENCODER_DECODER_FORWARD_OK")


class SonicOnnxReferenceModel:
    """Small generic ONNX ReferenceEvaluator wrapper keyed by output name."""

    def __init__(self, model_path: Path) -> None:
        try:
            import onnx
            from onnx.reference import ReferenceEvaluator
        except ModuleNotFoundError as exc:
            raise RuntimeError("Running SONIC ONNX models requires onnx") from exc

        model = onnx.load(str(model_path))
        self._output_names = tuple(value.name for value in model.graph.output)
        self._evaluator = ReferenceEvaluator(model)

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        outputs = self._evaluator.run(None, inputs)
        return dict(zip(self._output_names, outputs, strict=True))


def _planner_inputs_to_numpy(planner_inputs: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Running SONIC ONNX models requires numpy") from exc

    return {
        "context_mujoco_qpos": np.asarray(
            planner_inputs.context_mujoco_qpos,
            dtype=np.float32,
        ).reshape(1, 4, SONIC_PLANNER_QPOS_DIM),
        "target_vel": np.asarray([planner_inputs.target_vel], dtype=np.float32),
        "mode": np.asarray([planner_inputs.mode], dtype=np.int64),
        "movement_direction": np.asarray(
            planner_inputs.movement_direction,
            dtype=np.float32,
        ).reshape(1, 3),
        "facing_direction": np.asarray(
            planner_inputs.facing_direction,
            dtype=np.float32,
        ).reshape(1, 3),
        "random_seed": np.asarray([planner_inputs.random_seed], dtype=np.int64),
        "has_specific_target": np.asarray(
            [[planner_inputs.has_specific_target]],
            dtype=np.int64,
        ),
        "specific_target_positions": np.asarray(
            planner_inputs.specific_target_positions,
            dtype=np.float32,
        ).reshape(1, 4, 3),
        "specific_target_headings": np.asarray(
            planner_inputs.specific_target_headings,
            dtype=np.float32,
        ).reshape(1, 4),
        "allowed_pred_num_tokens": np.asarray(
            [planner_inputs.allowed_pred_num_tokens],
            dtype=np.int64,
        ).reshape(1, 11),
        "height": np.asarray([planner_inputs.height], dtype=np.float32),
    }


def _standing_history(joint_positions_mujoco: Sequence[float]) -> tuple[SonicG1HistoryFrame, ...]:
    body_q, body_dq = mujoco_motor_state_to_sonic_body_state(
        joint_positions_mujoco,
        (0.0,) * SONIC_ACTION_DIM,
    )
    frame = SonicG1HistoryFrame(
        base_ang_vel=(0.0, 0.0, 0.0),
        body_q=body_q,
        body_dq=body_dq,
        last_action=(0.0,) * SONIC_ACTION_DIM,
        base_quat=(1.0, 0.0, 0.0, 0.0),
    )
    return (frame,) * SONIC_HISTORY_FRAMES


def _row_array(values: Sequence[float]) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Running SONIC ONNX models requires numpy") from exc
    return np.asarray(values, dtype=np.float32).reshape(1, len(values))


def _reshape_planner_qpos(values: Any) -> tuple[tuple[float, ...], ...]:
    flat = _flatten_numeric(values)
    if len(flat) % SONIC_PLANNER_QPOS_DIM:
        raise ValueError(
            f"planner qpos output length {len(flat)} is not divisible by {SONIC_PLANNER_QPOS_DIM}"
        )
    return tuple(
        tuple(flat[index : index + SONIC_PLANNER_QPOS_DIM])
        for index in range(0, len(flat), SONIC_PLANNER_QPOS_DIM)
    )


def read_planner_qpos_csv(path: Path) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        for row_index, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            try:
                values = tuple(float(value) for value in row if value.strip())
            except ValueError:
                if row_index == 1:
                    continue
                raise
            if len(values) != SONIC_PLANNER_QPOS_DIM:
                raise ValueError(
                    f"{path}:{row_index} expected {SONIC_PLANNER_QPOS_DIM} qpos values, "
                    f"got {len(values)}"
                )
            rows.append(values)
    if not rows:
        raise ValueError(f"{path} contains no planner qpos rows")
    return tuple(rows)


def _flatten_numeric(values: Any) -> tuple[float, ...]:
    try:
        array = values.reshape(-1)
        return tuple(float(value) for value in array.tolist())
    except AttributeError:
        if isinstance(values, (list, tuple)):
            output: list[float] = []
            for value in values:
                if isinstance(value, (list, tuple)):
                    output.extend(_flatten_numeric(value))
                else:
                    output.append(float(value))
            return tuple(output)
        return (float(values),)


def _flatten_rows(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


def _is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def write_rows_csv(path: Path, rows: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow([f"{float(value):.9g}" for value in row])


if __name__ == "__main__":
    main()
