"""Create a Task033 StackMLP history-input checkpoint from a base MLP checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h200_locomotion_lab.training.history_checkpoint_migration import (
    StackMlpHistoryMigrationConfig,
    migrate_stack_mlp_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate a 104D MLP actor checkpoint to a Task033 StackMLP history-input checkpoint."
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=104)
    parser.add_argument("--action-dim", type=int, default=31)
    parser.add_argument("--command-label", default="task033_migrate_stackmlp_checkpoint")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = _require_torch()
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    target_checkpoint = args.target_checkpoint.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    loaded = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    config = StackMlpHistoryMigrationConfig(
        history_len=args.history_len,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
    )
    migrated, report = migrate_stack_mlp_checkpoint(loaded, config)
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, target_checkpoint)
    result = {
        "pass": True,
        "status": "passed",
        "command_label": args.command_label,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "output_json": str(output_json),
        **report,
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - H200-only path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


if __name__ == "__main__":
    main()
