"""Deterministic local manifests for G1-like morphology variants."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

from h200_locomotion_lab.robots.g1like_slots import (
    G1LIKE_ACTION_DIM,
    G1LIKE_ACTION_SLOT_NAMES,
    G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER,
)


G1LIKE_SLOT_SCHEMA_ID = "g1like_slots.v1.body29.command_mujoco"
TRAIN_RANGE = (0.95, 1.05)
NEAR_HELDOUT_RANGES = ((0.90, 0.95), (1.05, 1.10))
OOD_HELDOUT_RANGES = ((0.80, 0.90), (1.10, 1.20))
HELDOUT_CONDITIONS = (
    "link_length",
    "mass_com_inertia",
    "motor_dynamics",
    "combined",
)
VALID_SPLITS = ("train", "heldout")
SCALE_FIELDS = (
    "link_scale",
    "mass_scale",
    "com_scale",
    "inertia_scale",
    "motor_strength_scale",
)
NONPHYSICAL_SCALING_NOTE = (
    "Mass/COM/inertia scales are deterministic scalar approximations for local "
    "manifest coverage only; they are nonphysical until validated against a "
    "loaded simulator asset."
)


class G1LikeMorphologyManifestError(ValueError):
    """Raised when a G1-like morphology manifest is invalid."""


def slot_schema_hash() -> str:
    """Return the stable hash of the 002 G1-like slot schema."""

    return _sha256_short(
        {
            "schema_id": G1LIKE_SLOT_SCHEMA_ID,
            "action_dim": G1LIKE_ACTION_DIM,
            "slot_names": list(G1LIKE_ACTION_SLOT_NAMES),
        }
    )


def joint_order_hash() -> str:
    """Return the stable hash of the 002 29-slot command order."""

    return _sha256_short(
        {
            "schema_id": G1LIKE_SLOT_SCHEMA_ID,
            "command_mujoco": list(G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER),
        }
    )


def generate_g1like_morphology_manifest(
    *,
    seed: int,
    train_count: int = 1,
    heldout_conditions: Sequence[str] = HELDOUT_CONDITIONS,
    heldout_band: str = "near",
) -> dict[str, Any]:
    """Generate a small deterministic train/heldout morphology manifest."""

    if train_count < 1:
        raise G1LikeMorphologyManifestError("train_count must be at least 1")
    if not heldout_conditions:
        raise G1LikeMorphologyManifestError("heldout_conditions must not be empty")
    for condition in heldout_conditions:
        if condition not in HELDOUT_CONDITIONS:
            raise G1LikeMorphologyManifestError(f"unknown heldout_condition: {condition}")
    if heldout_band not in ("near", "ood"):
        raise G1LikeMorphologyManifestError("heldout_band must be near or ood")

    rng = random.Random(seed)
    variants: list[dict[str, Any]] = []
    for index in range(train_count):
        variants.append(
            _build_variant(
                seed=seed,
                index=index,
                split="train",
                heldout_condition="none",
                link_scale=_sample_train_scale(rng),
                mass_scale=_sample_train_scale(rng),
                com_scale=_sample_train_scale(rng),
                inertia_scale=_sample_train_scale(rng),
                motor_strength_scale=_sample_train_scale(rng),
            )
        )

    heldout_ranges = NEAR_HELDOUT_RANGES if heldout_band == "near" else OOD_HELDOUT_RANGES
    for index, condition in enumerate(heldout_conditions):
        scales = {
            "link_scale": _sample_train_scale(rng),
            "mass_scale": _sample_train_scale(rng),
            "com_scale": _sample_train_scale(rng),
            "inertia_scale": _sample_train_scale(rng),
            "motor_strength_scale": _sample_train_scale(rng),
        }
        if condition in ("link_length", "combined"):
            scales["link_scale"] = _sample_heldout_scale(rng, heldout_ranges)
        if condition in ("mass_com_inertia", "combined"):
            scales["mass_scale"] = _sample_heldout_scale(rng, heldout_ranges)
            scales["com_scale"] = _sample_heldout_scale(rng, heldout_ranges)
            scales["inertia_scale"] = _sample_heldout_scale(rng, heldout_ranges)
        if condition in ("motor_dynamics", "combined"):
            scales["motor_strength_scale"] = _sample_heldout_scale(rng, heldout_ranges)
        variants.append(
            _build_variant(
                seed=seed,
                index=index,
                split="heldout",
                heldout_condition=condition,
                **scales,
            )
        )

    manifest = {
        "manifest_id": _manifest_id(seed, heldout_band, variants),
        "seed": int(seed),
        "slot_schema_id": G1LIKE_SLOT_SCHEMA_ID,
        "slot_schema_hash": slot_schema_hash(),
        "action_dim": G1LIKE_ACTION_DIM,
        "joint_order_hash": joint_order_hash(),
        "train_range": list(TRAIN_RANGE),
        "near_heldout_ranges": [list(range_) for range_ in NEAR_HELDOUT_RANGES],
        "ood_heldout_ranges": [list(range_) for range_ in OOD_HELDOUT_RANGES],
        "heldout_band": heldout_band,
        "variants": variants,
    }
    validate_g1like_morphology_manifest(manifest)
    return manifest


def validate_g1like_morphology_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a generated or fixture G1-like morphology manifest."""

    heldout_band = manifest.get("heldout_band")
    if heldout_band not in ("near", "ood"):
        raise G1LikeMorphologyManifestError("heldout_band must be near or ood")
    heldout_ranges = _heldout_ranges_for_band(str(heldout_band))

    if manifest.get("slot_schema_id") != G1LIKE_SLOT_SCHEMA_ID:
        raise G1LikeMorphologyManifestError("slot_schema_id does not match g1like_slots schema")
    if manifest.get("slot_schema_hash") != slot_schema_hash():
        raise G1LikeMorphologyManifestError("slot_schema_hash does not match g1like_slots schema")
    if manifest.get("action_dim") != G1LIKE_ACTION_DIM:
        raise G1LikeMorphologyManifestError("action_dim must be 29")
    if manifest.get("joint_order_hash") != joint_order_hash():
        raise G1LikeMorphologyManifestError(
            "joint_order_hash must derive from the 002 29-slot command order"
        )

    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise G1LikeMorphologyManifestError("variants must be a non-empty list")

    seen_splits: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise G1LikeMorphologyManifestError("each variant must be a mapping")
        _validate_variant(variant, heldout_ranges=heldout_ranges)
        seen_splits.add(str(variant["split"]))
    if "train" not in seen_splits or "heldout" not in seen_splits:
        raise G1LikeMorphologyManifestError("manifest must contain train and heldout variants")


def _build_variant(
    *,
    seed: int,
    index: int,
    split: str,
    heldout_condition: str,
    link_scale: float,
    mass_scale: float,
    com_scale: float,
    inertia_scale: float,
    motor_strength_scale: float,
) -> dict[str, Any]:
    variant_core = {
        "seed": int(seed),
        "index": int(index),
        "split": split,
        "heldout_condition": heldout_condition,
        "link_scale": round(float(link_scale), 6),
        "mass_scale": round(float(mass_scale), 6),
        "com_scale": round(float(com_scale), 6),
        "inertia_scale": round(float(inertia_scale), 6),
        "motor_strength_scale": round(float(motor_strength_scale), 6),
    }
    variant_id = f"g1like-{split}-{heldout_condition}-{_sha256_short(variant_core, size=10)}"
    return {
        "variant_id": variant_id,
        "split": split,
        "heldout_condition": heldout_condition,
        "link_scale": variant_core["link_scale"],
        "mass_scale": variant_core["mass_scale"],
        "com_scale": variant_core["com_scale"],
        "inertia_scale": variant_core["inertia_scale"],
        "motor_strength_scale": variant_core["motor_strength_scale"],
        "limitation_notes": [NONPHYSICAL_SCALING_NOTE],
    }


def _validate_variant(
    variant: Mapping[str, Any],
    *,
    heldout_ranges: Sequence[tuple[float, float]],
) -> None:
    split = variant.get("split")
    if split not in VALID_SPLITS:
        raise G1LikeMorphologyManifestError("split must be train or heldout")
    condition = variant.get("heldout_condition")
    if split == "train":
        if condition != "none":
            raise G1LikeMorphologyManifestError("train variants must use heldout_condition=none")
    elif condition not in HELDOUT_CONDITIONS:
        raise G1LikeMorphologyManifestError(
            "heldout_condition must be link_length, mass_com_inertia, motor_dynamics, or combined"
        )

    for field in SCALE_FIELDS:
        scale = variant.get(field)
        if not isinstance(scale, int | float):
            raise G1LikeMorphologyManifestError(f"{field} must be numeric")
        _validate_scale(
            field,
            float(scale),
            str(split),
            str(condition),
            heldout_ranges=heldout_ranges,
        )

    notes = variant.get("limitation_notes")
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise G1LikeMorphologyManifestError("limitation_notes must be a list of strings")
    joined_notes = " ".join(notes).lower()
    if any(float(variant[field]) != 1.0 for field in ("mass_scale", "com_scale", "inertia_scale")):
        if "nonphysical" not in joined_notes or "mass/com/inertia" not in joined_notes:
            raise G1LikeMorphologyManifestError(
                "nonphysical mass/COM/inertia scaling must be recorded in limitation_notes"
            )


def _validate_scale(
    field: str,
    scale: float,
    split: str,
    condition: str,
    *,
    heldout_ranges: Sequence[tuple[float, float]],
) -> None:
    if split == "train":
        if not _inside(scale, TRAIN_RANGE):
            raise G1LikeMorphologyManifestError(f"{field} train scale must be in 0.95-1.05")
        return

    if condition == "combined":
        expected_outside_train = True
    elif condition == "link_length":
        expected_outside_train = field == "link_scale"
    elif condition == "mass_com_inertia":
        expected_outside_train = field in ("mass_scale", "com_scale", "inertia_scale")
    elif condition == "motor_dynamics":
        expected_outside_train = field == "motor_strength_scale"
    else:
        raise G1LikeMorphologyManifestError(f"unknown heldout_condition: {condition}")

    if expected_outside_train:
        if _inside(scale, TRAIN_RANGE):
            raise G1LikeMorphologyManifestError(
                f"{field} heldout scale must be outside the train range"
            )
        if not _inside_any(scale, heldout_ranges):
            raise G1LikeMorphologyManifestError(
                f"{field} heldout scale must match the manifest heldout_band range"
            )
    elif not _inside(scale, TRAIN_RANGE):
        raise G1LikeMorphologyManifestError(
            f"{field} must remain in the train range for heldout_condition={condition}"
        )


def _manifest_id(seed: int, heldout_band: str, variants: Sequence[Mapping[str, Any]]) -> str:
    return f"g1like-manifest-{_sha256_short({'seed': seed, 'band': heldout_band, 'variants': variants})}"


def _sample_train_scale(rng: random.Random) -> float:
    return rng.uniform(*TRAIN_RANGE)


def _sample_heldout_scale(
    rng: random.Random, ranges: Sequence[tuple[float, float]]
) -> float:
    low, high = ranges[rng.randrange(len(ranges))]
    return rng.uniform(low, high)


def _heldout_ranges_for_band(heldout_band: str) -> tuple[tuple[float, float], ...]:
    if heldout_band == "near":
        return NEAR_HELDOUT_RANGES
    if heldout_band == "ood":
        return OOD_HELDOUT_RANGES
    raise G1LikeMorphologyManifestError("heldout_band must be near or ood")


def _inside(value: float, range_: tuple[float, float]) -> bool:
    low, high = range_
    return low <= value <= high


def _inside_any(value: float, ranges: Sequence[tuple[float, float]]) -> bool:
    return any(_inside(value, range_) for range_ in ranges)


def _sha256_short(payload: Any, *, size: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:size]
