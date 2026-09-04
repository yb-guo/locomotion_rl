"""Write the deterministic procedural train/validation/heldout manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h200_locomotion_lab.robots.procedural_morphology import build_morphology_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_morphology_split_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(manifest.train), "validation": len(manifest.validation), "heldout": len(manifest.heldout), "structural_hash": manifest.structural_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
