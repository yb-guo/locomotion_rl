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
    parser.add_argument("--obs-csv", help="CSV containing one obs_dict row.")
    parser.add_argument("--obs-dim", type=int, default=994)
    parser.add_argument("--output-actions-csv", help="Write the 29D action row to this CSV.")
    parser.add_argument("--repeat-rows", type=int, default=1)
    args = parser.parse_args()

    obs = read_obs_csv(Path(args.obs_csv), args.obs_dim) if args.obs_csv else zero_obs(args.obs_dim)
    action = run_decoder_reference(Path(args.decoder), obs)
    action_min, action_max, action_max_abs = vector_range(action)

    print("SONIC_POLICY_DECODER_FORWARD_MODE onnx_reference")
    print("DECODER", Path(args.decoder))
    print("OBS_SOURCE", args.obs_csv or "zero")
    print("OBS_DIM", len(obs))
    print("OBS_FINITE", is_finite(obs))
    print("ACTION_DIM", len(action))
    print("ACTION_FINITE", is_finite(action))
    print("ACTION_MIN_MAX", action_min, action_max)
    print("ACTION_MAX_ABS", action_max_abs)
    print("ACTION_FIRST10", tuple(action[:10]))

    if args.output_actions_csv:
        output = Path(args.output_actions_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_action_csv(output, action, args.repeat_rows)
        print("OUTPUT_ACTIONS_CSV", output)
        print("OUTPUT_ACTION_ROWS", args.repeat_rows)

    if len(action) != 29:
        raise SystemExit(f"SONIC decoder produced {len(action)} actions, expected 29")
    if not is_finite(obs) or not is_finite(action):
        raise SystemExit("non-finite SONIC decoder input/output")
    print("SONIC_POLICY_DECODER_FORWARD_OK")


def run_decoder_reference(decoder: Path, obs: Sequence[float]) -> tuple[float, ...]:
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
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        for row_index, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            try:
                values = tuple(float(value) for value in row)
            except ValueError:
                if row_index == 1:
                    continue
                raise ValueError(f"{path}:{row_index} contains a non-numeric obs value")
            if len(values) != obs_dim:
                raise ValueError(f"{path}:{row_index} expected {obs_dim} obs values, got {len(values)}")
            return values
    raise ValueError(f"{path} contains no observation rows")


def write_action_csv(path: Path, action: Sequence[float], rows: int) -> None:
    if rows <= 0:
        raise ValueError("repeat rows must be positive")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for _ in range(rows):
            writer.writerow([f"{float(value):.9g}" for value in action])


def vector_range(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values must not be empty")
    return min(values), max(values), max(abs(value) for value in values)


def is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


if __name__ == "__main__":
    main()
