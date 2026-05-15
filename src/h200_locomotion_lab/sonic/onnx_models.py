"""Small ONNX runtime wrappers for SONIC models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class SonicOnnxReferenceModel:
    """Generic ONNX ReferenceEvaluator wrapper keyed by output name."""

    def __init__(self, model_path: Path | str) -> None:
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


class SonicOnnxEncoder:
    """Reference ONNX wrapper for `model_encoder.onnx`."""

    def __init__(self, encoder: Path | str) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("Running the SONIC encoder requires numpy") from exc

        self._np = np
        self._model = SonicOnnxReferenceModel(encoder)

    def run(self, observation: Sequence[float]) -> tuple[float, ...]:
        obs = self._np.asarray(observation, dtype=self._np.float32).reshape(1, len(observation))
        output = self._model.run({"obs_dict": obs})["encoded_tokens"]
        return tuple(float(value) for value in output.reshape(-1).tolist())


class SonicOnnxDecoder:
    """Reference ONNX wrapper for `model_decoder.onnx`."""

    def __init__(self, decoder: Path | str) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("Running the SONIC decoder requires numpy") from exc

        self._np = np
        self._model = SonicOnnxReferenceModel(decoder)

    def run(self, observation: Sequence[float]) -> tuple[float, ...]:
        obs = self._np.asarray(observation, dtype=self._np.float32).reshape(1, len(observation))
        output = self._model.run({"obs_dict": obs})["action"]
        return tuple(float(value) for value in output.reshape(-1).tolist())
