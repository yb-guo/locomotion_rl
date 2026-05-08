"""Inspect SONIC ONNX graph input/output names and shapes."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="ONNX model path(s) to inspect.")
    args = parser.parse_args()

    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError("Inspecting ONNX metadata requires the onnx package") from exc

    for model_path in args.models:
        path = Path(model_path)
        model = onnx.load(str(path))
        print("MODEL", path)
        print("INPUTS")
        for value in model.graph.input:
            print(value.name, tensor_shape(value), value.type.tensor_type.elem_type)
        print("OUTPUTS")
        for value in model.graph.output:
            print(value.name, tensor_shape(value), value.type.tensor_type.elem_type)
        print("SONIC_ONNX_IO_INSPECT_OK")


def tensor_shape(value: object) -> tuple[int | str, ...]:
    shape = value.type.tensor_type.shape
    dims: list[int | str] = []
    for dim in shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append("?")
    return tuple(dims)


if __name__ == "__main__":
    main()
