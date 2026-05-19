"""Emit a G1 SONIC-to-Genesis alignment bundle as stable JSON.

The tool intentionally does not import Genesis. It only reads structured robot
profiles, vectorized backend config defaults, and optional MJCF/XML metadata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree

from h200_locomotion_lab.envs.vectorized_genesis_backend import VectorizedGenesisConfig
from h200_locomotion_lab.robots.g1_27dof_nohand import (
    DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE,
    REMOVED_FROM_G1_29DOF_COMMAND_ORDER,
    load_g1_27dof_nohand_profile,
)
from h200_locomotion_lab.robots.loader import (
    DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE,
    load_robot_profile,
)
from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_PLANNER_DEFAULT_HEIGHT,
    build_initial_planner_context,
)


CONTROL_FIELDS = (
    "default_angles_rad",
    "action_scales_rad",
    "kp",
    "kv",
    "force_limits",
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_alignment_bundle(
        sonic_profile_path=args.sonic_profile,
        genesis_profile_path=args.genesis_profile,
        asset_path=args.asset_path,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sonic-profile",
        type=Path,
        default=DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE,
        help="Path to the 29DoF SONIC robot profile YAML.",
    )
    parser.add_argument(
        "--genesis-profile",
        type=Path,
        default=DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE,
        help="Path to the 27DoF no-hand Genesis training profile YAML.",
    )
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=None,
        help="Optional MJCF/XML asset path override.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def build_alignment_bundle(
    *,
    sonic_profile_path: str | Path = DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE,
    genesis_profile_path: str | Path = DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE,
    asset_path: str | Path | None = None,
) -> dict[str, Any]:
    sonic = load_robot_profile(sonic_profile_path)
    genesis = load_g1_27dof_nohand_profile(genesis_profile_path)
    missing: list[dict[str, str]] = []

    mapped = mapped_control_comparison(sonic, genesis)
    runtime_config = VectorizedGenesisConfig(n_envs=1)
    backend_contact_solver_config = runtime_config.contact_solver_config_report()
    xml_path = asset_path if asset_path is not None else genesis.asset.path
    missing.extend(
        [
            {
                "path": "genesis_27dof_training_profile.contact_friction_solver_config",
                "reason": "not_represented_in_profile",
            },
        ]
    )
    missing.extend(backend_contact_solver_config["missing"])
    contact_friction_solver = parse_contact_friction_solver(xml_path, missing)
    control_timing = control_timing_comparison(
        genesis=genesis,
        contact_friction_solver=contact_friction_solver,
        missing=missing,
    )
    root_pose_defaults = {
        "backend_vectorized_genesis_config": {
            "action_joint_group": runtime_config.action_joint_group,
            "action_scale_mult": runtime_config.action_scale_mult,
            "add_plane": runtime_config.add_plane,
            "backend": runtime_config.backend,
            "root_qpos": list(runtime_config.root_qpos),
            "default_positions_rad": (
                None
                if runtime_config.default_positions_rad is None
                else list(runtime_config.default_positions_rad)
            ),
            "logical_cuda_device": runtime_config.logical_cuda_device,
            "motor_force_limit_mult": runtime_config.motor_force_limit_mult,
            "motor_kp_mult": runtime_config.motor_kp_mult,
            "motor_kv_mult": runtime_config.motor_kv_mult,
            "rigid_contact_solver_configured": (
                backend_contact_solver_config["configured"]
            ),
            "uses_profile_default_positions_when_none": (
                runtime_config.default_positions_rad is None
            ),
        },
        "sonic_planner": {
            "default_height": SONIC_PLANNER_DEFAULT_HEIGHT,
            "context_root_qpos": list(build_initial_planner_context()[0][:7]),
        },
        "sonic_29dof_default_angles_rad": list(sonic.control.default_angles_rad),
        "genesis_27dof_default_angles_rad": list(genesis.control.default_angles_rad),
    }
    alignment_status = {
        "status": "pass" if mapped["all_match"] else "fail",
        "mapped_control_match": mapped["all_match"],
        "xml_asset_present": contact_friction_solver["asset_present"],
        "missing_count": len(missing),
    }

    return {
        "sonic_29dof_profile": serialize(sonic),
        "genesis_27dof_training_profile": serialize(genesis),
        "mapped_control_comparison": mapped,
        "control_timing_comparison": control_timing,
        "root_pose_defaults": root_pose_defaults,
        "contact_friction_solver": contact_friction_solver,
        "vectorized_genesis_backend_contact_solver_config": backend_contact_solver_config,
        "missing": missing,
        "alignment_status": alignment_status,
    }


def control_timing_comparison(
    *,
    genesis: Any,
    contact_friction_solver: Mapping[str, Any],
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    contract = genesis.training_contract
    derived_policy_rate_hz = 1.0 / (contract.sim_dt_s * contract.decimation)
    option = contact_friction_solver.get("option")
    mjcf_timestep_s = None
    if isinstance(option, Mapping) and option.get("timestep") is not None:
        mjcf_timestep_s = parse_optional_float(str(option["timestep"]))
    else:
        missing.append(
            {
                "path": "contact_friction_solver.option.timestep",
                "reason": (
                    "asset_missing"
                    if not contact_friction_solver.get("asset_present")
                    else "xml_field_absent"
                ),
            }
        )
    missing.extend(
        [
            {
                "path": "sonic_29dof_profile.training_contract.sim_dt_s",
                "reason": "not_represented_in_profile",
            },
            {
                "path": "sonic_29dof_profile.training_contract.decimation",
                "reason": "not_represented_in_profile",
            },
            {
                "path": "sonic_29dof_profile.training_contract.policy_rate_hz",
                "reason": "not_represented_in_profile",
            },
            {
                "path": "mjcf.control_decimation",
                "reason": "not_represented_in_mjcf",
            },
            {
                "path": "mjcf.policy_rate_hz",
                "reason": "not_represented_in_mjcf",
            },
        ]
    )
    return {
        "genesis_training_contract": {
            "sim_dt_s": contract.sim_dt_s,
            "decimation": contract.decimation,
            "policy_rate_hz": contract.policy_rate_hz,
            "derived_policy_rate_hz": derived_policy_rate_hz,
            "self_consistent": round(derived_policy_rate_hz) == contract.policy_rate_hz,
        },
        "vectorized_genesis_backend": {
            "sim_dt_source": "profile.training_contract.sim_dt_s",
            "decimation_source": "profile.training_contract.decimation",
            "policy_step_source": "VectorizedGenesisBackend.step_physics decimation loop",
        },
        "sonic_29dof_profile": {
            "sim_dt_s": None,
            "decimation": None,
            "policy_rate_hz": None,
            "source": "not_represented_in_profile",
        },
        "mjcf_option": {
            "timestep_s": mjcf_timestep_s,
            "timestep_present": mjcf_timestep_s is not None,
        },
        "comparison": {
            "genesis_vs_mjcf_timestep_match": (
                None if mjcf_timestep_s is None else mjcf_timestep_s == contract.sim_dt_s
            ),
            "sonic_profile_timing_present": False,
            "mjcf_decimation_present": False,
        },
    }


def mapped_control_comparison(sonic: Any, genesis: Any) -> dict[str, Any]:
    keep_indices = [
        index
        for index, joint_name in enumerate(sonic.joint_order.command_mujoco)
        if joint_name not in REMOVED_FROM_G1_29DOF_COMMAND_ORDER
    ]
    mapped_joint_order = [sonic.joint_order.command_mujoco[index] for index in keep_indices]
    genesis_joint_order = list(genesis.actuator_order)
    field_results: dict[str, Any] = {}
    all_match = mapped_joint_order == genesis_joint_order

    for field_name in CONTROL_FIELDS:
        sonic_values = getattr(sonic.control, field_name)
        genesis_values = getattr(genesis.control, field_name)
        mapped_values = [float(sonic_values[index]) for index in keep_indices]
        genesis_list = [float(value) for value in genesis_values]
        mismatches = [
            {
                "index": index,
                "joint": genesis_joint_order[index],
                "sonic_mapped": mapped_values[index],
                "genesis": genesis_list[index],
            }
            for index in range(min(len(mapped_values), len(genesis_list)))
            if mapped_values[index] != genesis_list[index]
        ]
        length_match = len(mapped_values) == len(genesis_list)
        field_match = length_match and not mismatches
        all_match = all_match and field_match
        field_results[field_name] = {
            "match": field_match,
            "length_match": length_match,
            "mapped_sonic": mapped_values,
            "genesis": genesis_list,
            "mismatches": mismatches,
        }

    return {
        "removed_joints": list(REMOVED_FROM_G1_29DOF_COMMAND_ORDER),
        "mapped_joint_order": mapped_joint_order,
        "genesis_joint_order": genesis_joint_order,
        "joint_order_match": mapped_joint_order == genesis_joint_order,
        "fields": field_results,
        "all_match": all_match,
    }


def parse_contact_friction_solver(
    asset_path: str | Path,
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    asset_path_text = str(asset_path)
    asset_fs_path = Path(asset_path_text)
    report: dict[str, Any] = {
        "asset_path": asset_path_text,
        "asset_present": asset_fs_path.is_file(),
        "compiler": None,
        "option": None,
        "defaults": [],
        "geoms_with_contact_fields": [],
        "contact": None,
    }
    if not asset_fs_path.is_file():
        missing.append({"path": "contact_friction_solver.asset_path", "reason": "asset_missing"})
        missing.append({"path": "contact_friction_solver.compiler", "reason": "asset_missing"})
        missing.append({"path": "contact_friction_solver.option", "reason": "asset_missing"})
        missing.append({"path": "contact_friction_solver.defaults", "reason": "asset_missing"})
        missing.append({"path": "contact_friction_solver.contact", "reason": "asset_missing"})
        return report

    root = ElementTree.parse(asset_fs_path).getroot()
    report["compiler"] = element_attrs(root.find("compiler"))
    report["option"] = element_attrs(root.find("option"))
    default_elements = root.findall(".//default")
    report["defaults"] = [default_summary(element) for element in default_elements]
    contact = root.find("contact")
    report["contact"] = None if contact is None else [child_summary(child) for child in contact]
    report["geoms_with_contact_fields"] = [
        {"name": geom.get("name"), **contact_attrs}
        for geom in root.findall(".//geom")
        if (
            contact_attrs := select_attrs(
                geom,
                ("friction", "condim", "solref", "solimp", "priority"),
            )
        )
    ]

    for key in ("compiler", "option"):
        if report[key] is None:
            missing.append({"path": f"contact_friction_solver.{key}", "reason": "xml_field_absent"})
    if not report["defaults"]:
        missing.append({"path": "contact_friction_solver.defaults", "reason": "xml_field_absent"})
    if report["contact"] is None:
        missing.append({"path": "contact_friction_solver.contact", "reason": "xml_field_absent"})
    if not report["geoms_with_contact_fields"]:
        missing.append(
            {
                "path": "contact_friction_solver.geoms_with_contact_fields",
                "reason": "xml_field_absent",
            }
        )
    return report


def default_summary(element: ElementTree.Element) -> dict[str, Any]:
    return {
        "class": element.get("class"),
        "attrs": dict(sorted(element.attrib.items())),
        "children": [child_summary(child) for child in element],
    }


def child_summary(element: ElementTree.Element) -> dict[str, Any]:
    return {"tag": element.tag, "attrs": dict(sorted(element.attrib.items()))}


def element_attrs(element: ElementTree.Element | None) -> dict[str, str] | None:
    if element is None:
        return None
    return dict(sorted(element.attrib.items()))


def select_attrs(element: ElementTree.Element, names: Iterable[str]) -> dict[str, str]:
    return {name: element.attrib[name] for name in names if name in element.attrib}


def parse_optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: serialize(getattr(value, field.name))
            for field in fields(value)
            if field.name != "source_path" or getattr(value, field.name) is not None
        }
    if isinstance(value, Mapping):
        return {str(key): serialize(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [serialize(item) for item in value]
    return value


if __name__ == "__main__":
    main()
