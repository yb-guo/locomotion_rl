"""Filesystem probes that treat inaccessible paths as unavailable."""

from __future__ import annotations

from pathlib import Path


def path_exists(path: Path) -> bool:
    """Return whether *path* exists without leaking host permission failures."""

    try:
        return path.exists()
    except OSError:
        return False


def path_is_file(path: Path) -> bool:
    """Return whether *path* is a file, or false when it cannot be inspected."""

    try:
        return path.is_file()
    except OSError:
        return False
