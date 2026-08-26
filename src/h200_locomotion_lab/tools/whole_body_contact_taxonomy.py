"""Shared MuJoCo contact taxonomy for Task067 diagnostics."""

from __future__ import annotations

from typing import Any


def _geom_name(shard: Any, geom_id: int) -> str:
    return (
        shard.mujoco.mj_id2name(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        or f"geom_{int(geom_id)}"
    )


def _body_name(shard: Any, body_id: int) -> str:
    return (
        shard.mujoco.mj_id2name(shard.model, shard.mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        or f"body_{int(body_id)}"
    )


def _foot_geom_ids(shard: Any) -> dict[int, str]:
    return {
        int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, name)): name
        for name in sorted(shard._foot_geoms)
    }


def contact_taxonomy(shard: Any, data: Any) -> dict[str, Any]:
    """Classify actual MuJoCo contacts into stance-contract buckets.

    The stance contract treats only footpad-floor pairs as support contacts.
    Any non-foot geom touching the floor is forbidden, and any non-floor robot
    geom pair is a self-contact.
    """

    np = shard.np
    floor_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    foot_by_id = _foot_geom_ids(shard)
    support_foot_floor: list[dict[str, Any]] = []
    forbidden_nonfoot_floor: list[dict[str, Any]] = []
    self_contacts: list[dict[str, Any]] = []
    geom_pairs: list[dict[str, Any]] = []
    contacts_by_foot = {name: 0 for name in sorted(shard._foot_geoms)}
    normal_by_foot = {name: 0.0 for name in sorted(shard._foot_geoms)}
    weighted_xy = np.zeros(2, dtype=np.float64)
    foot_normal_sum = 0.0
    foot_normal_max = 0.0
    floor_min_dist = float("inf")
    self_min_dist = float("inf")

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_name = _geom_name(shard, geom1)
        geom2_name = _geom_name(shard, geom2)
        geom1_body = int(shard.model.geom_bodyid[geom1])
        geom2_body = int(shard.model.geom_bodyid[geom2])
        force = np.zeros(6, dtype=np.float64)
        shard.mujoco.mj_contactForce(shard.model, data, index, force)
        normal_force = max(0.0, float(force[0]))
        has_floor = floor_id in (geom1, geom2)
        other = geom2 if geom1 == floor_id else geom1
        foot_name = foot_by_id.get(other) if has_floor else None
        if has_floor and foot_name is not None:
            category = "support_foot_floor_contacts"
        elif has_floor:
            category = "forbidden_nonfoot_floor_contacts"
        else:
            category = "self_contacts"

        pair = {
            "contact_index": index,
            "category": category,
            "geom1": {"id": geom1, "name": geom1_name, "body": _body_name(shard, geom1_body)},
            "geom2": {"id": geom2, "name": geom2_name, "body": _body_name(shard, geom2_body)},
            "geom_pair": [geom1_name, geom2_name],
            "distance": float(contact.dist),
            "position": [float(value) for value in contact.pos],
            "normal_force": normal_force,
            "foot": foot_name,
        }
        geom_pairs.append(pair)
        if category == "support_foot_floor_contacts":
            assert foot_name is not None
            support_foot_floor.append(pair)
            contacts_by_foot[foot_name] += 1
            normal_by_foot[foot_name] += normal_force
            foot_normal_sum += normal_force
            foot_normal_max = max(foot_normal_max, normal_force)
            weighted_xy += normal_force * np.asarray(contact.pos[:2], dtype=np.float64)
            floor_min_dist = min(floor_min_dist, float(contact.dist))
        elif category == "forbidden_nonfoot_floor_contacts":
            forbidden_nonfoot_floor.append(pair)
            floor_min_dist = min(floor_min_dist, float(contact.dist))
        else:
            self_contacts.append(pair)
            self_min_dist = min(self_min_dist, float(contact.dist))

    center_of_pressure = None
    if foot_normal_sum > 1e-9:
        center_of_pressure = [float(value) for value in weighted_xy / foot_normal_sum]

    return {
        "support_foot_floor_contacts": support_foot_floor,
        "forbidden_nonfoot_floor_contacts": forbidden_nonfoot_floor,
        "self_contacts": self_contacts,
        "geom_pairs": sorted(geom_pairs, key=lambda row: row["contact_index"]),
        "counts": {
            "support_foot_floor_contacts": len(support_foot_floor),
            "forbidden_nonfoot_floor_contacts": len(forbidden_nonfoot_floor),
            "self_contacts": len(self_contacts),
            "all_contacts": len(geom_pairs),
        },
        "contacts_by_foot": contacts_by_foot,
        "normal_force_by_foot": normal_by_foot,
        "foot_normal_force_sum": foot_normal_sum,
        "foot_normal_force_max": foot_normal_max,
        "center_of_pressure_xy": center_of_pressure,
        "min_floor_contact_distance": None if floor_min_dist == float("inf") else floor_min_dist,
        "min_self_contact_distance": None if self_min_dist == float("inf") else self_min_dist,
    }


def _contact_constraint_types(shard: Any) -> set[int]:
    constraint = shard.mujoco.mjtConstraint
    names = (
        "mjCNSTR_CONTACT_FRICTIONLESS",
        "mjCNSTR_CONTACT_PYRAMIDAL",
        "mjCNSTR_CONTACT_ELLIPTIC",
    )
    return {int(getattr(constraint, name)) for name in names if hasattr(constraint, name)}


def efc_qfrc_for_contact_indices(shard: Any, data: Any, contact_indices: set[int]) -> Any:
    """Map only EFC rows associated with the requested contact ids to qfrc."""

    np = shard.np
    filtered_force = np.zeros(data.efc_force.shape, dtype=np.float64)
    contact_types = _contact_constraint_types(shard)
    for efc_index in range(int(data.nefc)):
        if int(data.efc_id[efc_index]) not in contact_indices:
            continue
        if contact_types and int(data.efc_type[efc_index]) not in contact_types:
            continue
        filtered_force[efc_index] = float(data.efc_force[efc_index])
    qfrc = np.zeros(shard.model.nv, dtype=np.float64)
    shard.mujoco.mj_mulJacTVec(shard.model, data, qfrc, filtered_force)
    return qfrc


def full_efc_qfrc(shard: Any, data: Any) -> Any:
    np = shard.np
    qfrc = np.zeros(shard.model.nv, dtype=np.float64)
    shard.mujoco.mj_mulJacTVec(shard.model, data, qfrc, data.efc_force)
    return qfrc
