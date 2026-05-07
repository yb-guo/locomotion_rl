"""Run the SONIC decoder ONNX policy artifact and optionally export actions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", required=True, help="Path to SONIC model_decoder.onnx.")
    parser.add_argument("--obs-csv", help="CSV containing one or more obs_dict rows.")
    parser.add_argument("--obs-dim", type=int, default=994)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum numeric obs rows to decode from --obs-csv.",
    )
    parser.add_argument("--output-actions-csv", help="Write decoded 29D action rows to this CSV.")
    parser.add_argument("--repeat-rows", type=int, default=1)
    args = parser.parse_args()

    obs_rows = (
        read_obs_csv_rows(Path(args.obs_csv), args.obs_dim, max_rows=args.max_rows)
        if args.obs_csv
        else [zero_obs(args.obs_dim)]
    )
    action_rows = [run_decoder_reference(Path(args.decoder), obs) for obs in obs_rows]
    action_min, action_max, action_max_abs = vector_range(_flatten(action_rows))

    print("SONIC_POLICY_DECODER_FORWARD_MODE onnx_reference")
    print("DECODER", Path(args.decoder))
    print("OBS_SOURCE", args.obs_csv or "zero")
    print("OBS_ROWS", len(obs_rows))
    print("OBS_DIM", len(obs_rows[0]))
    print("OBS_FINITE", all(is_finite(obs) for obs in obs_rows))
    print("ACTION_ROWS", len(action_rows))
    print("ACTION_DIM", len(action_rows[0]))
    print("ACTION_FINITE", all(is_finite(action) for action in action_rows))
    print("ACTION_MIN_MAX", action_min, action_max)
    print("ACTION_MAX_ABS", action_max_abs)
    print("ACTION_FIRST10", tuple(action_rows[0][:10]))

    if args.output_actions_csv:
        output = Path(args.output_actions_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows_to_write = action_rows
        if len(action_rows) == 1:
            rows_to_write = action_rows * args.repeat_rows
        elif args.repeat_rows != 1:
            raise ValueError("--repeat-rows can only be used with a single obs row")
        write_action_csv(output, rows_to_write)
        print("OUTPUT_ACTIONS_CSV", output)
        print("OUTPUT_ACTION_ROWS", len(rows_to_write))

    if any(len(action) != 29 for action in action_rows):
        dims = sorted({len(action) for action in action_rows})
        raise SystemExit(f"SONIC decoder produced action dims {dims}, expected 29")
    if not all(is_finite(obs) for obs in obs_rows) or not all(
        is_finite(action) for action in action_rows
    ):
        raise SystemExit("non-finite SONIC decoder input/output")
    print("SONIC_POLICY_DECODER_FORWARD_OK")


def run_decoder_reference(decoder: Path, obs: Sequence[float]) -> tuple[float, ...]:
    return SonicOnnxReferenceDecoder(decoder).run(obs)


class SonicOnnxReferenceDecoder:
    """Small ONNX ReferenceEvaluator wrapper that keeps the decoder loaded."""

    def __init__(self, decoder: Path) -> None:
        try:
            import numpy as np
            import onnx
            from onnx.reference import ReferenceEvaluator
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Running the SONIC decoder requires numpy and onnx with ReferenceEvaluator"
            ) from exc

        self._np = np
        model = onnx.load(str(decoder))
        self._evaluator = ReferenceEvaluator(model)

    def run(self, obs: Sequence[float]) -> tuple[float, ...]:
        obs_array = self._np.asarray(obs, dtype=self._np.float32).reshape(1, len(obs))
        output = self._evaluator.run(None, {"obs_dict": obs_array})[0].reshape(-1)
        return tuple(float(value) for value in output.tolist())


def _run_decoder_reference_slow(decoder: Path, obs: Sequence[float]) -> tuple[float, ...]:
    try:
        import numpy as np
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Running the SONIC decoder requires numpy and onnx with ReferenceEvaluator"
        ) from exc

    model = onnx.load(str(decoder))
    evaluator = ReferenceEvaluator(model)
    obs_array = np.asarray(obs, dtype=np.float32).reshape(1, len(obs))
    output = evaluator.run(None, {"obs_dict": obs_array})[0].reshape(-1)
    return tuple(float(value) for value in output.tolist())


def zero_obs(obs_dim: int) -> tuple[float, ...]:
    if obs_dim <= 0:
        raise ValueError("obs_dim must be positive")
    return (0.0,) * obs_dim


def read_obs_csv(path: Path, obs_dim: int) -> tuple[float, ...]:
    return read_obs_csv_rows(path, obs_dim, max_rows=1)[0]


def read_obs_csv_rows(
    path: Path,
    obs_dim: int,
    max_rows: int | None = None,
) -> list[tuple[float, ...]]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max rows must be positive")
    rows: list[tuple[float, ...]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        for row_index, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            numeric_fields = tuple(value.strip() for value in row if value.strip())
            try:
                values = tuple(float(value) for value in numeric_fields)
            except ValueError:
                if row_index == 1:
                    continue
                raise ValueError(f"{path}:{row_index} contains a non-numeric obs value")
            if len(values) != obs_dim:
                raise ValueError(
                    f"{path}:{row_index} expected {obs_dim} obs values, got {len(values)}"
                )
            rows.append(values)
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"{path} contains no observation rows")
    return rows


def write_action_csv(path: Path, actions: Sequence[Sequence[float]]) -> None:
    if not actions:
        raise ValueError("actions must not be empty")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for action in actions:
            writer.writerow([f"{float(value):.9g}" for value in action])


def vector_range(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values must not be empty")
    return min(values), max(values), max(abs(value) for value in values)


def is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _flatten(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


if __name__ == "__main__":
    main()
