"""Task038 local G1-like MJCF variant artifact smoke.

This tool does not start a simulator, access H200, or download assets. It
patches a caller-provided MJCF into one train and one held-out G1-like variant
artifact, then writes a small local parse/contract JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h200_locomotion_lab.robots.g1like_mjcf_patch import patch_g1like_mjcf_variant
from h200_locomotion_lab.robots.g1like_morphology import (
    HELDOUT_CONDITIONS,
    generate_g1like_morphology_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mjcf",
        required=True,
        type=Path,
        help="Existing source MJCF to patch. No assets are downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for patched train/heldout MJCF artifacts.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
        help="Path for the small JSON contract summary.",
    )
    parser.add_argument("--seed", type=int, default=38, help="Deterministic manifest seed.")
    parser.add_argument(
        "--heldout-condition",
        choices=HELDOUT_CONDITIONS,
        default="combined",
        help="Single held-out condition to generate alongside one train variant.",
    )
    parser.add_argument(
        "--heldout-band",
        choices=("near", "ood"),
        default="near",
        help="Held-out scale band used by the Task038 manifest generator.",
    )
    parser.add_argument(
        "--compile-mujoco",
        action="store_true",
        help=(
            "Optionally run mujoco.MjModel.from_xml_path on each patched XML. "
            "Default false: does not import mujoco or start a simulator."
        ),
    )
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest = generate_g1like_morphology_manifest(
        seed=args.seed,
        train_count=1,
        heldout_conditions=(args.heldout_condition,),
        heldout_band=args.heldout_band,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_summaries = []
    for variant in manifest["variants"]:
        output_mjcf = args.output_dir / f"{variant['variant_id']}.xml"
        variant_summaries.append(
            patch_g1like_mjcf_variant(
                source_mjcf=args.source_mjcf,
                output_mjcf=output_mjcf,
                variant=variant,
            )
        )

    compile_results = _compile_variant_summaries(variant_summaries) if args.compile_mujoco else []
    compile_ok = bool(compile_results) and all(item["compile_ok"] for item in compile_results)
    compile_status = (
        "not_requested"
        if not args.compile_mujoco
        else "ok"
        if compile_ok
        else "blocked_or_failed"
    )
    local_parse_ok = all(item["local_parse_ok"] for item in variant_summaries)
    pass_ok = local_parse_ok and (not args.compile_mujoco or compile_ok)
    summary = {
        "task": "task038_g1like_variant_asset_smoke",
        "scope": "local_mjcf_patch_and_parse_only",
        "h200_load_smoke": "pending",
        "simulator_started": False,
        "asset_downloaded": False,
        "mujoco_compile_requested": bool(args.compile_mujoco),
        "mujoco_compile": {
            "status": compile_status,
            "variants": compile_results,
        },
        "manifest": {
            key: value for key, value in manifest.items() if key != "variants"
        },
        "source_mjcf": str(args.source_mjcf),
        "output_dir": str(args.output_dir),
        "summary_json": str(args.summary_json),
        "variant_summaries": variant_summaries,
        "local_parse_ok": local_parse_ok,
        "pass": pass_ok and all(item["pass"] for item in variant_summaries),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _compile_variant_summaries(variant_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": str(item["variant_id"]),
            "output_path": str(item["output_path"]),
            **_compile_mujoco_xml(Path(item["output_path"])),
        }
        for item in variant_summaries
    ]


def _compile_mujoco_xml(path: Path) -> dict[str, Any]:
    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as error:
        return {
            "compile_ok": False,
            "blocked": True,
            "error": f"mujoco is not installed: {error}",
        }

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as error:  # pragma: no cover - depends on optional MuJoCo.
        return {
            "compile_ok": False,
            "blocked": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "compile_ok": True,
        "blocked": False,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
    }


def main(argv: list[str] | None = None) -> None:
    summary = run_smoke(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
