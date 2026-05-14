"""Task023 Genesis Franka effort/Jacobian API smoke.

This script is intentionally task-local. It does not import Genesis at module
import time and does not depend on project ``src/`` modules.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_OUTPUT = Path("outputs/task023/franka_current_force_estimation/genesis_api_smoke.json")
DEFAULT_ASSET = "xml/franka_emika_panda/panda_nohand.xml"
FRANKA_JOINT_NAME_SETS = (
    tuple(f"joint{index}" for index in range(1, 8)),
    tuple(f"panda_joint{index}" for index in range(1, 8)),
)
TOOL_LINK_CANDIDATES = (
    "hand",
    "panda_hand",
    "link7",
    "panda_link7",
    "link8",
    "panda_link8",
)


class SmokeBlocked(RuntimeError):
    """Environment/API limitation that should be recorded, not hidden."""


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.perf_counter()
    report = initial_report(args)
    try:
        report.update(run_smoke(args))
        report["status"] = "ok"
        report["blocker"] = ""
    except SmokeBlocked as exc:
        report["status"] = "blocked"
        report["blocker"] = str(exc)
    except Exception as exc:  # pragma: no cover - target-only simulator failures.
        report["status"] = "failed"
        report["blocker"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        report["elapsed_s"] = round(time.perf_counter() - started, 6)
        write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["status"] != "ok":
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Genesis Franka effort and Jacobian APIs.")
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    return parser


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "blocked",
        "blocker": "not_run",
        "asset": args.asset,
        "backend": args.backend,
        "sim_dt": args.sim_dt,
        "n_envs": args.n_envs,
        "steps": args.steps,
        "output": str(args.output),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.sim_dt <= 0.0:
        raise SmokeBlocked("sim_dt_must_be_positive")
    if args.n_envs <= 0:
        raise SmokeBlocked("n_envs_must_be_positive")
    if args.steps <= 0:
        raise SmokeBlocked("steps_must_be_positive")

    gs = import_genesis()
    np = import_numpy()
    init_genesis(gs, args.backend)

    scene = make_scene(gs, args.sim_dt)
    add_plane(gs, scene)
    franka, morph_info = add_franka(gs, scene, args.asset)
    payload = add_payload_box(gs, scene)
    build_scene(scene, args.n_envs)

    joint_names, arm_dofs = resolve_arm_dofs(franka)
    tool_link, tool_link_name = resolve_tool_link(franka)
    apply_basic_pd(franka, np, arm_dofs)
    command_target = command_small_motion(franka, arm_dofs)
    for _ in range(args.steps):
        scene.step()

    q = read_dof_api(franka, "get_dofs_position", arm_dofs)
    dq = read_dof_api(franka, "get_dofs_velocity", arm_dofs)
    effort_control = read_dof_api(franka, "get_dofs_control_force", arm_dofs)
    effort_internal = read_dof_api(franka, "get_dofs_force", arm_dofs)
    jacobian = read_jacobian(franka, tool_link)
    weld_report = probe_weld(gs, np, scene, payload, tool_link)

    report: dict[str, Any] = {
        "genesis_version": str(getattr(gs, "__version__", "unknown")),
        "morph_requires_jac_and_ik_requested": True,
        "morph_requires_jac_and_ik_used": morph_info["requires_jac_and_IK"],
        "robot_n_dofs": int(getattr(franka, "n_dofs", -1)),
        "arm_joint_names": joint_names,
        "arm_dof_indices": arm_dofs,
        "tool_link_name": tool_link_name,
        "tool_link_idx": int(getattr(tool_link, "idx", -1)),
        "command_target": command_target,
        "api": {
            "get_dofs_position": describe_tensor(q),
            "get_dofs_velocity": describe_tensor(dq),
            "get_dofs_control_force": describe_tensor(effort_control),
            "get_dofs_force": describe_tensor(effort_internal),
            "get_jacobian": describe_jacobian(jacobian),
        },
        "jacobian_row_contract": {
            "translation_rows": "0:3",
            "rotation_rows": "3:6",
            "source": "Genesis official source/API docs; smoke records row norms only.",
        },
        "weld": weld_report,
    }
    require_finite_api(report["api"])
    return report


def import_genesis() -> Any:
    try:
        import genesis as gs  # type: ignore[import-not-found]
    except Exception as exc:
        raise SmokeBlocked(f"genesis_import_failed:{exc}") from exc
    return gs


def import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:
        raise SmokeBlocked(f"numpy_import_failed:{exc}") from exc
    return np


def init_genesis(gs: Any, backend: str) -> None:
    backend_value = getattr(gs, backend, backend)
    try:
        gs.init(backend=backend_value, logging_level="warning")
    except TypeError:
        gs.init(backend=backend_value)
    except Exception as exc:
        raise SmokeBlocked(f"genesis_init_failed:{exc}") from exc


def make_scene(gs: Any, sim_dt: float) -> Any:
    try:
        return gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=sim_dt),
        )
    except Exception as exc:
        raise SmokeBlocked(f"scene_create_failed:{exc}") from exc


def add_plane(gs: Any, scene: Any) -> None:
    try:
        scene.add_entity(gs.morphs.Plane())
    except Exception:
        return


def add_franka(gs: Any, scene: Any, asset: str) -> tuple[Any, dict[str, Any]]:
    candidates = (
        {"file": asset, "requires_jac_and_IK": True},
        {"file": asset},
    )
    last_error = ""
    for kwargs in candidates:
        try:
            morph = gs.morphs.MJCF(**kwargs)
            robot = scene.add_entity(morph)
            return robot, {"requires_jac_and_IK": bool(kwargs.get("requires_jac_and_IK", False))}
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            raise SmokeBlocked(f"franka_add_failed:{exc}") from exc
    raise SmokeBlocked(f"franka_mjcf_kwargs_unsupported:{last_error}")


def add_payload_box(gs: Any, scene: Any) -> Any | None:
    if not hasattr(gs.morphs, "Box"):
        return None
    candidates = (
        {"size": (0.04, 0.04, 0.04), "pos": (0.5, 0.0, 0.4)},
        {"size": (0.04, 0.04, 0.04)},
        {},
    )
    for kwargs in candidates:
        try:
            return scene.add_entity(gs.morphs.Box(**kwargs))
        except TypeError:
            continue
        except Exception:
            return None
    return None


def build_scene(scene: Any, n_envs: int) -> None:
    try:
        scene.build(n_envs=n_envs)
    except TypeError:
        scene.build()
    except Exception as exc:
        raise SmokeBlocked(f"scene_build_failed:{exc}") from exc


def resolve_arm_dofs(robot: Any) -> tuple[list[str], list[int]]:
    for names in FRANKA_JOINT_NAME_SETS:
        resolved: list[int] = []
        ok = True
        for name in names:
            try:
                joint = robot.get_joint(name)
                resolved.append(single_joint_dof_index(joint))
            except Exception:
                ok = False
                break
        if ok and len(resolved) == 7:
            return list(names), resolved

    n_dofs = int(getattr(robot, "n_dofs", 0))
    if n_dofs < 7:
        raise SmokeBlocked(f"could_not_resolve_7_arm_dofs:n_dofs={n_dofs}")
    names = list_named_children(robot, ("joints", "_joints"))
    fallback_names = names[:7] if len(names) >= 7 else [f"dof{index}" for index in range(7)]
    return fallback_names, list(range(7))


def single_joint_dof_index(joint: Any) -> int:
    if hasattr(joint, "dof_idx_local"):
        return int(getattr(joint, "dof_idx_local"))
    if hasattr(joint, "dofs_idx_local"):
        values = list(getattr(joint, "dofs_idx_local"))
        if len(values) != 1:
            raise ValueError(f"expected one dof, got {values}")
        return int(values[0])
    raise ValueError("joint has no local dof index")


def resolve_tool_link(robot: Any) -> tuple[Any, str]:
    for name in TOOL_LINK_CANDIDATES:
        try:
            link = robot.get_link(name)
            return link, str(name)
        except Exception:
            continue
    links = list_named_objects(robot, ("links", "_links"))
    if not links:
        raise SmokeBlocked("could_not_resolve_tool_link")
    link = links[-1]
    return link, str(getattr(link, "name", "last_link"))


def apply_basic_pd(robot: Any, np: Any, arm_dofs: Sequence[int]) -> None:
    kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000], dtype=float)
    kv = np.array([450, 450, 350, 350, 200, 200, 200], dtype=float)
    lower = np.array([-87, -87, -87, -87, -12, -12, -12], dtype=float)
    upper = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
    optional_dof_setter(robot, "set_dofs_kp", kp, arm_dofs)
    optional_dof_setter(robot, "set_dofs_kv", kv, arm_dofs)
    if hasattr(robot, "set_dofs_force_range"):
        try:
            robot.set_dofs_force_range(lower, upper, dofs_idx_local=arm_dofs)
        except TypeError:
            try:
                robot.set_dofs_force_range(lower, upper, arm_dofs)
            except Exception:
                return
        except Exception:
            return


def optional_dof_setter(robot: Any, method_name: str, values: Any, arm_dofs: Sequence[int]) -> None:
    if not hasattr(robot, method_name):
        return
    method = getattr(robot, method_name)
    try:
        method(values, dofs_idx_local=arm_dofs)
    except TypeError:
        try:
            method(values, arm_dofs)
        except Exception:
            return
    except Exception:
        return


def command_small_motion(robot: Any, arm_dofs: Sequence[int]) -> list[float]:
    q = first_row(flatten_or_rows(read_dof_api(robot, "get_dofs_position", arm_dofs)))
    if len(q) < len(arm_dofs):
        q = [0.0 for _ in arm_dofs]
    target = [float(value) for value in q[: len(arm_dofs)]]
    for index in range(len(target)):
        target[index] += 0.01 * math.sin(index + 1.0)
    method = getattr(robot, "control_dofs_position")
    try:
        method(target, dofs_idx_local=arm_dofs)
    except TypeError:
        method(target, arm_dofs)
    return target


def read_dof_api(robot: Any, method_name: str, arm_dofs: Sequence[int]) -> Any:
    if not hasattr(robot, method_name):
        raise SmokeBlocked(f"{method_name}_missing")
    method = getattr(robot, method_name)
    try:
        return method(dofs_idx_local=arm_dofs)
    except TypeError:
        return method(arm_dofs)
    except Exception as exc:
        raise SmokeBlocked(f"{method_name}_failed:{exc}") from exc


def read_jacobian(robot: Any, tool_link: Any) -> Any:
    if not hasattr(robot, "get_jacobian"):
        raise SmokeBlocked("get_jacobian_missing")
    try:
        return robot.get_jacobian(tool_link, local_point=None)
    except TypeError:
        return robot.get_jacobian(tool_link)
    except Exception as exc:
        raise SmokeBlocked(f"get_jacobian_failed:{exc}") from exc


def probe_weld(gs: Any, np: Any, scene: Any, payload: Any | None, tool_link: Any) -> dict[str, Any]:
    report = {
        "payload_entity_created": payload is not None,
        "rigid_solver_available": False,
        "add_weld_constraint_available": False,
        "delete_weld_constraint_available": False,
        "smoke_status": "not_run",
        "blocker": "",
    }
    rigid = getattr(getattr(scene, "sim", None), "rigid_solver", None)
    report["rigid_solver_available"] = rigid is not None
    if rigid is None:
        report["smoke_status"] = "blocked"
        report["blocker"] = "rigid_solver_missing"
        return report
    report["add_weld_constraint_available"] = hasattr(rigid, "add_weld_constraint")
    report["delete_weld_constraint_available"] = hasattr(rigid, "delete_weld_constraint")
    if payload is None:
        report["smoke_status"] = "blocked"
        report["blocker"] = "payload_entity_missing"
        return report
    if not report["add_weld_constraint_available"]:
        report["smoke_status"] = "blocked"
        report["blocker"] = "add_weld_constraint_missing"
        return report
    try:
        payload_link = resolve_payload_link(payload)
        payload_ids = np.array([int(getattr(payload_link, "idx"))], dtype=gs.np_int)
        tool_ids = np.array([int(getattr(tool_link, "idx"))], dtype=gs.np_int)
        rigid.add_weld_constraint(payload_ids, tool_ids)
        report["payload_link_name"] = str(getattr(payload_link, "name", "unknown"))
        report["payload_link_idx"] = int(getattr(payload_link, "idx", -1))
        report["smoke_status"] = "ok"
        if hasattr(rigid, "delete_weld_constraint"):
            try:
                rigid.delete_weld_constraint(payload_ids, tool_ids)
            except TypeError:
                rigid.delete_weld_constraint(payload_ids)
            except Exception as exc:
                report["delete_blocker"] = str(exc)
    except Exception as exc:
        report["smoke_status"] = "blocked"
        report["blocker"] = f"weld_smoke_failed:{exc}"
    return report


def resolve_payload_link(payload: Any) -> Any:
    for name in ("box_baselink", "base_link", "link"):
        try:
            return payload.get_link(name)
        except Exception:
            continue
    links = list_named_objects(payload, ("links", "_links"))
    if not links:
        raise SmokeBlocked("payload_link_missing")
    return links[0]


def describe_tensor(value: Any) -> dict[str, Any]:
    flat = flatten_numeric(value)
    return {
        "available": True,
        "shape": shape_of(value),
        "finite": all(math.isfinite(number) for number in flat),
        "count": len(flat),
        "min": min(flat) if flat else None,
        "max": max(flat) if flat else None,
        "mean_abs": sum(abs(number) for number in flat) / len(flat) if flat else None,
    }


def describe_jacobian(value: Any) -> dict[str, Any]:
    rows = matrix_rows(value)
    row_norms = [math.sqrt(sum(number * number for number in row)) for row in rows]
    return {
        **describe_tensor(value),
        "row_norms": row_norms,
        "translation_row_norms": row_norms[:3],
        "rotation_row_norms": row_norms[3:6],
    }


def require_finite_api(api: Mapping[str, Mapping[str, Any]]) -> None:
    missing = [name for name, desc in api.items() if not desc.get("available")]
    nonfinite = [name for name, desc in api.items() if not desc.get("finite")]
    if missing:
        raise SmokeBlocked(f"missing_api:{missing}")
    if nonfinite:
        raise SmokeBlocked(f"nonfinite_api:{nonfinite}")


def shape_of(value: Any) -> list[int] | str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(dim) for dim in shape]
        except Exception:
            return str(shape)
    data = to_python(value)
    if isinstance(data, list):
        return infer_list_shape(data)
    return []


def infer_list_shape(data: list[Any]) -> list[int]:
    if not data:
        return [0]
    if isinstance(data[0], list):
        return [len(data)] + infer_list_shape(data[0])
    return [len(data)]


def matrix_rows(value: Any) -> list[list[float]]:
    data = to_python(value)
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        return []
    if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
        data = data[0]
    rows: list[list[float]] = []
    for row in data:
        if isinstance(row, list):
            rows.append([float(number) for number in flatten_list(row)])
    return rows


def flatten_or_rows(value: Any) -> list[Any]:
    data = to_python(value)
    if isinstance(data, list):
        return data
    return [data]


def first_row(data: list[Any]) -> list[float]:
    if not data:
        return []
    first = data[0]
    if isinstance(first, list):
        return [float(value) for value in flatten_list(first)]
    return [float(value) for value in flatten_list(data)]


def flatten_numeric(value: Any) -> list[float]:
    return [float(number) for number in flatten_list(to_python(value))]


def flatten_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(flatten_list(item))
        return out
    try:
        return [float(value)]
    except Exception:
        return []


def to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def list_named_children(parent: Any, attrs: Iterable[str]) -> list[str]:
    return [str(getattr(item, "name")) for item in list_named_objects(parent, attrs)]


def list_named_objects(parent: Any, attrs: Iterable[str]) -> list[Any]:
    for attr in attrs:
        if not hasattr(parent, attr):
            continue
        values = getattr(parent, attr)
        objects = [item for item in flatten_object_iter(values) if hasattr(item, "name")]
        if objects:
            return objects
    return []


def flatten_object_iter(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return []
    try:
        iterator = iter(values)
    except TypeError:
        return [values]
    out: list[Any] = []
    for item in iterator:
        if isinstance(item, (list, tuple)):
            out.extend(flatten_object_iter(item))
        else:
            out.append(item)
    return out


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
