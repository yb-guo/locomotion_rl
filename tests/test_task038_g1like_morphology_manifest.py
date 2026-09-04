import copy

import pytest

from h200_locomotion_lab.robots.g1like_morphology import (
    G1LIKE_SLOT_SCHEMA_ID,
    HELDOUT_CONDITIONS,
    NONPHYSICAL_SCALING_NOTE,
    TRAIN_RANGE,
    G1LikeMorphologyManifestError,
    generate_g1like_morphology_manifest,
    joint_order_hash,
    slot_schema_hash,
    validate_g1like_morphology_manifest,
)
from h200_locomotion_lab.robots.g1like_slots import (
    G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER,
    G1LIKE_ACTION_DIM,
    G1LIKE_ACTION_SLOT_NAMES,
)


def test_same_seed_is_deterministic() -> None:
    first = generate_g1like_morphology_manifest(seed=38)
    second = generate_g1like_morphology_manifest(seed=38)

    assert first == second


def test_different_seed_changes_variant_ids_or_scales() -> None:
    first = generate_g1like_morphology_manifest(seed=38)
    second = generate_g1like_morphology_manifest(seed=39)

    first_signature = [
        (
            variant["variant_id"],
            variant["link_scale"],
            variant["mass_scale"],
            variant["com_scale"],
            variant["inertia_scale"],
            variant["motor_strength_scale"],
        )
        for variant in first["variants"]
    ]
    second_signature = [
        (
            variant["variant_id"],
            variant["link_scale"],
            variant["mass_scale"],
            variant["com_scale"],
            variant["inertia_scale"],
            variant["motor_strength_scale"],
        )
        for variant in second["variants"]
    ]

    assert first_signature != second_signature


def test_generated_manifest_has_train_and_heldout_splits() -> None:
    manifest = generate_g1like_morphology_manifest(seed=38)
    splits = {variant["split"] for variant in manifest["variants"]}
    heldout_conditions = {
        variant["heldout_condition"]
        for variant in manifest["variants"]
        if variant["split"] == "heldout"
    }

    assert splits == {"train", "heldout"}
    assert heldout_conditions == set(HELDOUT_CONDITIONS)
    assert manifest["manifest_id"].startswith("g1like-manifest-")


def test_seed_5494_near_manifest_boundary_generates_and_validates() -> None:
    manifest = generate_g1like_morphology_manifest(seed=5494, heldout_band="near")

    validate_g1like_morphology_manifest(manifest)
    assert manifest["heldout_band"] == "near"


@pytest.mark.parametrize(
    ("condition", "outside_fields"),
    [
        ("link_length", {"link_scale"}),
        ("mass_com_inertia", {"mass_scale", "com_scale", "inertia_scale"}),
        ("motor_dynamics", {"motor_strength_scale"}),
        ("combined", set()),
    ],
)
def test_heldout_ranges_are_outside_train_range(condition: str, outside_fields: set[str]) -> None:
    manifest = generate_g1like_morphology_manifest(
        seed=38,
        heldout_conditions=(condition,),
    )
    heldout = next(variant for variant in manifest["variants"] if variant["split"] == "heldout")
    scale_fields = {
        "link_scale",
        "mass_scale",
        "com_scale",
        "inertia_scale",
        "motor_strength_scale",
    }
    if condition == "combined":
        outside_fields = scale_fields

    for field in scale_fields:
        scale = heldout[field]
        in_train_range = TRAIN_RANGE[0] <= scale <= TRAIN_RANGE[1]
        assert in_train_range is (field not in outside_fields)


def test_action_slot_schema_and_joint_order_hash_are_stable_from_002() -> None:
    manifest = generate_g1like_morphology_manifest(seed=38)

    assert manifest["action_dim"] == G1LIKE_ACTION_DIM == 29
    assert manifest["slot_schema_id"] == G1LIKE_SLOT_SCHEMA_ID
    assert manifest["slot_schema_hash"] == slot_schema_hash()
    assert manifest["joint_order_hash"] == joint_order_hash()
    assert len(G1LIKE_ACTION_SLOT_NAMES) == 29
    assert G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER == tuple(
        f"{slot_name}_joint" for slot_name in G1LIKE_ACTION_SLOT_NAMES
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda manifest: manifest.update({"action_dim": 28}), "action_dim must be 29"),
        (
            lambda manifest: manifest.update({"slot_schema_id": "other"}),
            "slot_schema_id does not match",
        ),
        (
            lambda manifest: manifest.update({"joint_order_hash": "bad"}),
            "joint_order_hash must derive",
        ),
        (
            lambda manifest: manifest["variants"][0].update({"split": "eval"}),
            "split must be train or heldout",
        ),
        (
            lambda manifest: manifest["variants"][0].update({"link_scale": 1.2}),
            "link_scale train scale",
        ),
        (
            lambda manifest: next(
                variant
                for variant in manifest["variants"]
                if variant["heldout_condition"] == "link_length"
            ).update({"link_scale": 1.0}),
            "link_scale heldout scale",
        ),
        (
            lambda manifest: manifest.update({"heldout_band": "far"}),
            "heldout_band must be near or ood",
        ),
    ],
)
def test_invalid_manifest_fields_are_rejected(mutator, message: str) -> None:
    manifest = generate_g1like_morphology_manifest(seed=38)

    broken = copy.deepcopy(manifest)
    mutator(broken)

    with pytest.raises(G1LikeMorphologyManifestError, match=message):
        validate_g1like_morphology_manifest(broken)


def test_near_manifest_rejects_ood_heldout_scale() -> None:
    manifest = generate_g1like_morphology_manifest(
        seed=38,
        heldout_conditions=("link_length",),
        heldout_band="near",
    )
    broken = copy.deepcopy(manifest)
    heldout = next(variant for variant in broken["variants"] if variant["split"] == "heldout")
    heldout["link_scale"] = 0.85

    with pytest.raises(G1LikeMorphologyManifestError, match="heldout_band range"):
        validate_g1like_morphology_manifest(broken)


def test_ood_manifest_rejects_near_heldout_scale() -> None:
    manifest = generate_g1like_morphology_manifest(
        seed=38,
        heldout_conditions=("link_length",),
        heldout_band="ood",
    )
    broken = copy.deepcopy(manifest)
    heldout = next(variant for variant in broken["variants"] if variant["split"] == "heldout")
    heldout["link_scale"] = 1.075

    with pytest.raises(G1LikeMorphologyManifestError, match="heldout_band range"):
        validate_g1like_morphology_manifest(broken)


def test_limitation_notes_include_nonphysical_scaling_warning() -> None:
    manifest = generate_g1like_morphology_manifest(seed=38)

    for variant in manifest["variants"]:
        assert NONPHYSICAL_SCALING_NOTE in variant["limitation_notes"]
        joined_notes = " ".join(variant["limitation_notes"]).lower()
        assert "nonphysical" in joined_notes
        assert "mass/com/inertia" in joined_notes
