"""Bridge the local `src` package to the vendored Unitree asset tree."""

from pathlib import Path


_REPO_SRC_PATH = Path(__file__).resolve().parent
SRC_PATH = _REPO_SRC_PATH.parent / "external" / "unitree_rl_mjlab" / "src"

if str(SRC_PATH) not in __path__:
    __path__.append(str(SRC_PATH))

__all__ = ["SRC_PATH"]
