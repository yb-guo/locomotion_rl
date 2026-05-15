from __future__ import annotations

import tomllib
from pathlib import Path


def test_sonic_optional_dependency_extra_declares_python_runtime_deps() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    sonic = pyproject["project"]["optional-dependencies"]["sonic"]

    assert "numpy>=1.26" in sonic
    assert "onnx>=1.16" in sonic
    assert not any(dep.startswith("onnxruntime") for dep in sonic)
