"""Probe Genesis official batched tensor APIs without using project backends.

This tool intentionally does not import ``genesis`` or ``torch`` at module
import time. Local unit tests cover pure helpers; the CLI is meant to run on the
H200 target with an already-prepared Genesis install and asset bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ASSET_KINDS = ("franka", "go2", "g1")
ASSET_VARIANTS = ("default", "performance_mode", "convexify", "decimate")
UNAVAILABLE = "unavailable"

METRIC_KEYS: tuple[str, ...] = (
    "status",
    "asset_kind",
    "asset_variant",
    "cuda_visible_devices",
    "physical_gpu",
    "logical_cuda_device",
    "backend",
    "n_envs",
    "build_time_s",
    "warmup_time_s",
    "measure_time_s",
    "policy_steps_per_sec",
    "sim_steps_per_sec",
    "env_policy_steps_per_sec",
    "env_sim_steps_per_sec",
    "includes_build_time",
    "includes_reset_time",
    "includes_state_read",
    "includes_action_write",
    "includes_reward",
    "includes_render",
    "action_device",
    "qpos_device",
    "dofs_pos_device",
    "dofs_vel_device",
    "root_pos_device",
    "root_quat_device",
    "root_vel_device",
    "tensor_device_ok",
    "selected_reset_supported",
    "selected_reset_changes_only_target_envs",
    "selected_reset_time_s",
    "gpu_snapshot_before",
    "gpu_snapshot_after",
    "blocker",
)


class BlockedProbe(RuntimeError):
    """Expected environment or API limitation that should be reported as blocked."""


@dataclass(frozen=True)
class CudaIsolationResult:
    ok: bool
    blocker: str
    torch_module: Any | None = None


@dataclass(frozen=True)
class StateRead:
    qpos: Any | None
    dofs_pos: Any | None
    dofs_vel: Any | None
    root_pos: Any | None
    root_quat: Any | None
    root_vel: Any | None


@dataclass(frozen=True)
class SelectedResetResult:
    supported: bool
    changes_only_target_envs: bool
    reason: str
    changed_envs: tuple[int, ...]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    metrics = initial_metrics(args)

    try:
        result = run_probe(args, metrics)
        emit_metrics(result)
    except BlockedProbe as exc:
        metrics["status"] = "blocked"
        metrics["blocker"] = normalize_blocker(str(exc))
        emit_metrics(metrics)
    except Exception as exc:  # pragma: no cover - exercised on target-only failures.
        metrics["status"] = "failed"
        metrics["blocker"] = normalize_blocker(f"{exc.__class__.__name__}: {exc}")
        emit_metrics(metrics)
        raise SystemExit(1) from None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Genesis official batched tensor API throughput and reset semantics."
    )
    parser.add_argument("--asset-kind", choices=ASSET_KINDS, required=True)
    parser.add_argument("--asset", required=True, help="Prepared local asset path on the target.")
    parser.add_argument("--n-envs", type=positive_int, default=1)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--variant", choices=ASSET_VARIANTS, default="default")
    parser.add_argument("--warmup-policy-steps", type=non_negative_int, default=20)
    parser.add_argument("--measure-policy-steps", type=positive_int, default=100)
    parser.add_argument("--decimation", type=positive_int, default=4)
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--sim-dt", type=float, default=0.005)
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def initial_metrics(args: argparse.Namespace) -> dict[str, Any]:
    metrics = {key: UNAVAILABLE for key in METRIC_KEYS}
    metrics.update(
        {
            "status": "blocked",
            "asset_kind": args.asset_kind,
            "asset_variant": args.variant,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "physical_gpu": str(args.physical_gpu),
            "logical_cuda_device": args.logical_cuda_device,
            "backend": args.backend,
            "n_envs": args.n_envs,
            "build_time_s": UNAVAILABLE,
            "warmup_time_s": UNAVAILABLE,
            "measure_time_s": UNAVAILABLE,
            "policy_steps_per_sec": UNAVAILABLE,
            "sim_steps_per_sec": UNAVAILABLE,
            "env_policy_steps_per_sec": UNAVAILABLE,
            "env_sim_steps_per_sec": UNAVAILABLE,
            "includes_build_time": False,
            "includes_reset_time": False,
            "includes_state_read": True,
            "includes_action_write": True,
            "includes_reward": False,
            "includes_render": False,
            "tensor_device_ok": False,
            "selected_reset_supported": False,
            "selected_reset_changes_only_target_envs": False,
            "selected_reset_time_s": UNAVAILABLE,
            "blocker": "",
        }
    )
    return metrics


def run_probe(args: argparse.Namespace, metrics: dict[str, Any]) -> dict[str, Any]:
    if args.sim_dt <= 0:
        raise BlockedProbe("--sim-dt must be positive")

    torch_module = None
    if args.backend == "cuda":
        cuda_result = verify_single_visible_cuda_device(
            physical_gpu=str(args.physical_gpu),
            logical_cuda_device=args.logical_cuda_device,
        )
        if not cuda_result.ok:
            raise BlockedProbe(cuda_result.blocker)
        torch_module = cuda_result.torch_module

    asset_path = Path(args.asset)
    if not asset_path.is_file():
        raise BlockedProbe(f"asset_not_found:{asset_path}")

    metrics["gpu_snapshot_before"] = gpu_snapshot()
    gs = import_genesis_module()
    init_genesis(gs, args.backend)

    build_started = time.perf_counter()
    scene, robot = build_scene(
        gs,
        asset_kind=args.asset_kind,
        asset_path=str(asset_path),
        asset_variant=args.variant,
        n_envs=args.n_envs,
        sim_dt=args.sim_dt,
    )
    metrics["build_time_s"] = elapsed_since(build_started)

    dof_indices = resolve_dof_indices(robot)
    action = make_action_target(
        torch_module=torch_module,
        backend=args.backend,
        logical_cuda_device=args.logical_cuda_device,
        n_envs=args.n_envs,
        n_dofs=len(dof_indices),
    )
    metrics["action_device"] = tensor_device_name(action)

    selected_reset = exercise_selected_reset(
        robot,
        asset_kind=args.asset_kind,
        n_envs=args.n_envs,
        dof_indices=dof_indices,
        torch_module=torch_module,
        logical_cuda_device=args.logical_cuda_device,
    )
    metrics["includes_reset_time"] = selected_reset["includes_reset_time"]
    metrics["selected_reset_supported"] = selected_reset["supported"]
    metrics["selected_reset_changes_only_target_envs"] = selected_reset["changes_only_target_envs"]
    metrics["selected_reset_time_s"] = selected_reset["time_s"]
    if args.n_envs >= 2 and not selected_reset["changes_only_target_envs"]:
        raise BlockedProbe(f"selected_reset_not_verified:{selected_reset['reason']}")

    warmup_started = time.perf_counter()
    state = run_policy_loop(
        scene,
        robot,
        action=action,
        dof_indices=dof_indices,
        policy_steps=args.warmup_policy_steps,
        decimation=args.decimation,
    )
    metrics["warmup_time_s"] = elapsed_since(warmup_started)

    measure_started = time.perf_counter()
    state = run_policy_loop(
        scene,
        robot,
        action=action,
        dof_indices=dof_indices,
        policy_steps=args.measure_policy_steps,
        decimation=args.decimation,
    )
    measure_time_s = elapsed_since(measure_started)
    metrics["measure_time_s"] = measure_time_s
    metrics.update(
        compute_throughput(
            policy_steps=args.measure_policy_steps,
            decimation=args.decimation,
            n_envs=args.n_envs,
            elapsed_s=measure_time_s,
        )
    )

    metrics.update(state_device_metrics(state))
    device_values = (
        metrics["action_device"],
        metrics["qpos_device"],
        metrics["dofs_pos_device"],
        metrics["dofs_vel_device"],
        metrics["root_pos_device"],
        metrics["root_quat_device"],
        metrics["root_vel_device"],
    )
    metrics["tensor_device_ok"] = tensor_devices_ok(
        device_values,
        backend=args.backend,
        logical_cuda_device=args.logical_cuda_device,
    )
    metrics["gpu_snapshot_after"] = gpu_snapshot()
    if not metrics["tensor_device_ok"]:
        device_labels = (
            "action",
            "qpos",
            "dofs_pos",
            "dofs_vel",
            "root_pos",
            "root_quat",
            "root_vel",
        )
        raise BlockedProbe(
            "tensor_device_mismatch:"
            + json.dumps(dict(zip(device_labels, device_values)), sort_keys=True)
        )

    metrics["status"] = "ok"
    metrics["blocker"] = ""
    return metrics


def verify_single_visible_cuda_device(
    *,
    physical_gpu: str,
    logical_cuda_device: str,
) -> CudaIsolationResult:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = tuple(
        token.strip() for token in cuda_visible_devices.split(",") if token.strip()
    )
    if len(visible_tokens) != 1:
        return CudaIsolationResult(
            False,
            f"cuda_visible_devices_not_single:{cuda_visible_devices or 'not_set'}",
        )
    if visible_tokens[0] != str(physical_gpu):
        return CudaIsolationResult(
            False,
            f"cuda_visible_devices_expected_{physical_gpu}_got_{visible_tokens[0]}",
        )
    if logical_cuda_device != "cuda:0":
        return CudaIsolationResult(
            False,
            f"logical_cuda_device_expected_cuda:0_got_{logical_cuda_device}",
        )
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        return CudaIsolationResult(False, f"torch_import_failed:{exc}")
    if not torch.cuda.is_available():
        return CudaIsolationResult(False, "torch_cuda_unavailable")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        return CudaIsolationResult(False, f"torch_cuda_device_count_expected_1_got_{device_count}")
    try:
        probe = torch.empty((1,), device=logical_cuda_device)
        if str(probe.device) != logical_cuda_device:
            return CudaIsolationResult(
                False,
                f"torch_probe_device_expected_{logical_cuda_device}_got_{probe.device}",
            )
    except Exception as exc:
        return CudaIsolationResult(False, f"torch_cuda_probe_failed:{exc}")
    return CudaIsolationResult(True, "", torch)


def import_genesis_module() -> Any:
    try:
        import genesis as gs  # type: ignore[import-not-found]
    except Exception as exc:
        raise BlockedProbe(f"genesis_import_failed:{exc}") from exc
    return gs


def init_genesis(gs: Any, backend: str) -> None:
    backend_value = getattr(gs, backend, backend)
    try_call(
        "gs.init",
        lambda kwargs: gs.init(**kwargs),
        ({"backend": backend_value, "logging_level": "warning"}, {"backend": backend_value}),
    )


def build_scene(
    gs: Any,
    *,
    asset_kind: str,
    asset_path: str,
    asset_variant: str,
    n_envs: int,
    sim_dt: float,
) -> tuple[Any, Any]:
    sim_options = make_sim_options(gs, sim_dt)
    scene_kwargs = {"show_viewer": False}
    if sim_options is not None:
        scene_kwargs["sim_options"] = sim_options
    scene = try_call(
        "gs.Scene",
        lambda kwargs: gs.Scene(**kwargs),
        (scene_kwargs, {"show_viewer": False}),
    )
    add_plane(gs, scene)
    morph = build_morph(
        gs,
        asset_kind=asset_kind,
        asset_path=asset_path,
        asset_variant=asset_variant,
    )
    robot = call_method(
        "scene.add_entity",
        scene.add_entity,
        ({"morph": morph},),
        positional=(morph,),
    )
    build_kwargs = {"n_envs": n_envs}
    try_call("scene.build", lambda kwargs: scene.build(**kwargs), (build_kwargs,))
    return scene, robot


def make_sim_options(gs: Any, sim_dt: float) -> Any | None:
    options = getattr(gs, "options", None)
    if options is None or not hasattr(options, "SimOptions"):
        return None
    try:
        return options.SimOptions(dt=sim_dt)
    except TypeError:
        return options.SimOptions()


def add_plane(gs: Any, scene: Any) -> None:
    morphs = getattr(gs, "morphs", None)
    if morphs is None or not hasattr(morphs, "Plane"):
        return
    try:
        plane = morphs.Plane()
        call_method("scene.add_entity", scene.add_entity, ({"morph": plane},), positional=(plane,))
    except Exception:
        return


def build_morph(gs: Any, *, asset_kind: str, asset_path: str, asset_variant: str) -> Any:
    morphs = getattr(gs, "morphs", None)
    if morphs is None:
        raise BlockedProbe("genesis_morphs_missing")
    morph_type_name = "URDF" if asset_kind == "go2" or asset_path.endswith(".urdf") else "MJCF"
    if not hasattr(morphs, morph_type_name):
        raise BlockedProbe(f"genesis_morph_missing:{morph_type_name}")
    morph_type = getattr(morphs, morph_type_name)
    variant_kwargs = variant_morph_kwargs(asset_variant)
    candidates: list[dict[str, Any]] = []
    if morph_type_name == "URDF":
        candidates.append({"file": asset_path, "fixed": False, **variant_kwargs})
    candidates.append({"file": asset_path, **variant_kwargs})
    return try_call(f"gs.morphs.{morph_type_name}", lambda kwargs: morph_type(**kwargs), candidates)


def variant_morph_kwargs(variant: str) -> dict[str, bool]:
    if variant == "default":
        return {}
    if variant == "performance_mode":
        return {"performance_mode": True}
    if variant == "convexify":
        return {"convexify": True}
    if variant == "decimate":
        return {"decimate": True}
    raise ValueError(f"unknown asset variant: {variant}")


def variant_build_kwargs(variant: str) -> dict[str, bool]:
    if variant in ASSET_VARIANTS:
        return {}
    raise ValueError(f"unknown asset variant: {variant}")


def resolve_dof_indices(robot: Any) -> tuple[int, ...]:
    n_dofs = getattr(robot, "n_dofs", None)
    if n_dofs is None:
        try:
            dofs_pos = read_dofs_position(robot, None)
            rows = tensor_rows(dofs_pos, n_envs=1)
            n_dofs = len(rows[0]) if rows else 0
        except Exception as exc:
            raise BlockedProbe(f"could_not_resolve_n_dofs:{exc}") from exc
    try:
        n_dofs_int = int(n_dofs)
    except (TypeError, ValueError) as exc:
        raise BlockedProbe(f"invalid_n_dofs:{n_dofs}") from exc
    if n_dofs_int <= 0:
        raise BlockedProbe(f"invalid_n_dofs:{n_dofs_int}")
    return tuple(range(n_dofs_int))


def make_action_target(
    *,
    torch_module: Any | None,
    backend: str,
    logical_cuda_device: str,
    n_envs: int,
    n_dofs: int,
) -> Any:
    if backend == "cuda":
        if torch_module is None:
            raise BlockedProbe("torch_missing_for_cuda_action_tensor")
        return torch_module.zeros((n_envs, n_dofs), device=logical_cuda_device)
    return [[0.0 for _ in range(n_dofs)] for _ in range(n_envs)]


def run_policy_loop(
    scene: Any,
    robot: Any,
    *,
    action: Any,
    dof_indices: tuple[int, ...],
    policy_steps: int,
    decimation: int,
) -> StateRead:
    state = read_state(robot, dof_indices)
    for _ in range(policy_steps):
        write_action_targets(robot, action, dof_indices)
        state = read_state(robot, dof_indices)
        for _ in range(decimation):
            scene.step()
    return state


def write_action_targets(robot: Any, action: Any, dof_indices: tuple[int, ...]) -> None:
    if not hasattr(robot, "control_dofs_position"):
        raise BlockedProbe("robot_control_dofs_position_missing")
    method = robot.control_dofs_position
    vector_action = first_env_row(action)
    candidates = (
        {"position": action, "dofs_idx_local": dof_indices},
        {"pos": action, "dofs_idx_local": dof_indices},
        {"position": action, "dofs_idx": dof_indices},
        {"position": action},
        {},
    )
    try:
        try_call("robot.control_dofs_position", lambda kwargs: method(**kwargs), candidates[:-1])
        return
    except BlockedProbe:
        pass
    positional_candidates = (
        (action, dof_indices),
        (action,),
        (vector_action, dof_indices),
        (vector_action,),
    )
    last_error = ""
    for positional in positional_candidates:
        try:
            method(*positional)
            return
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            raise BlockedProbe(f"robot.control_dofs_position_failed:{exc}") from exc
    raise BlockedProbe(f"robot.control_dofs_position_unsupported:{last_error}")


def read_state(robot: Any, dof_indices: tuple[int, ...]) -> StateRead:
    return StateRead(
        qpos=read_optional(lambda: read_qpos(robot)),
        dofs_pos=read_optional(lambda: read_dofs_position(robot, dof_indices)),
        dofs_vel=read_optional(lambda: read_dofs_velocity(robot, dof_indices)),
        root_pos=read_optional(lambda: read_root_pos(robot)),
        root_quat=read_optional(lambda: read_root_quat(robot)),
        root_vel=read_optional(lambda: read_root_vel(robot)),
    )


def read_optional(reader: Callable[[], Any]) -> Any | None:
    try:
        return reader()
    except Exception:
        return None


def read_qpos(robot: Any) -> Any:
    if hasattr(robot, "get_qpos"):
        return robot.get_qpos()
    if hasattr(robot, "qpos"):
        return getattr(robot, "qpos")
    raise AttributeError("qpos unavailable")


def read_dofs_position(robot: Any, dof_indices: tuple[int, ...] | None) -> Any:
    if not hasattr(robot, "get_dofs_position"):
        raise AttributeError("get_dofs_position unavailable")
    method = robot.get_dofs_position
    if dof_indices is None:
        return call_with_optional_dofs("robot.get_dofs_position", method, ())
    return call_with_optional_dofs("robot.get_dofs_position", method, dof_indices)


def read_dofs_velocity(robot: Any, dof_indices: tuple[int, ...]) -> Any:
    if not hasattr(robot, "get_dofs_velocity"):
        raise AttributeError("get_dofs_velocity unavailable")
    return call_with_optional_dofs("robot.get_dofs_velocity", robot.get_dofs_velocity, dof_indices)


def read_root_pos(robot: Any) -> Any:
    if hasattr(robot, "get_pos"):
        return robot.get_pos()
    if hasattr(robot, "get_links_pos"):
        return robot.get_links_pos(links_idx_local=(0,))
    raise AttributeError("root position unavailable")


def read_root_quat(robot: Any) -> Any:
    if hasattr(robot, "get_quat"):
        return robot.get_quat()
    raise AttributeError("root quat unavailable")


def read_root_vel(robot: Any) -> Any:
    if hasattr(robot, "get_vel"):
        return robot.get_vel()
    if hasattr(robot, "get_links_vel"):
        return robot.get_links_vel(links_idx_local=(0,))
    raise AttributeError("root velocity unavailable")


def call_with_optional_dofs(
    label: str,
    method: Callable[..., Any],
    dof_indices: tuple[int, ...],
) -> Any:
    candidates = (
        {"dofs_idx_local": dof_indices},
        {"dofs_idx": dof_indices},
        {},
    )
    return try_call(label, lambda kwargs: method(**kwargs), candidates)


def exercise_selected_reset(
    robot: Any,
    *,
    asset_kind: str,
    n_envs: int,
    dof_indices: tuple[int, ...],
    torch_module: Any | None,
    logical_cuda_device: str,
) -> dict[str, Any]:
    if n_envs < 2:
        return {
            "includes_reset_time": False,
            "supported": False,
            "changes_only_target_envs": False,
            "time_s": "not_applicable_n_envs_lt_2",
            "reason": "not_applicable_n_envs_lt_2",
        }
    if not hasattr(robot, "set_dofs_position"):
        return {
            "includes_reset_time": False,
            "supported": False,
            "changes_only_target_envs": False,
            "time_s": UNAVAILABLE,
            "reason": "robot_set_dofs_position_missing",
        }

    target_env_ids = (0,)
    before_state = read_state(robot, dof_indices)
    before_rows = state_rows(before_state, n_envs=n_envs)
    if not before_rows:
        return {
            "includes_reset_time": False,
            "supported": False,
            "changes_only_target_envs": False,
            "time_s": UNAVAILABLE,
            "reason": "selected_reset_state_unavailable_before",
        }

    reset_target = make_selected_dof_reset_target(
        before_state.dofs_pos,
        target_env_ids=target_env_ids,
        n_envs=n_envs,
        n_dofs=len(dof_indices),
        torch_module=torch_module,
        logical_cuda_device=logical_cuda_device,
    )
    started = time.perf_counter()
    root_supported = True
    root_reason = ""
    if asset_kind in {"go2", "g1"}:
        root_target = make_selected_root_pos_reset_target(
            before_state.root_pos,
            target_env_ids=target_env_ids,
            n_envs=n_envs,
            torch_module=torch_module,
            logical_cuda_device=logical_cuda_device,
        )
        quat_target = make_selected_root_quat_reset_target(
            before_state.root_quat,
            target_env_ids=target_env_ids,
            n_envs=n_envs,
            torch_module=torch_module,
            logical_cuda_device=logical_cuda_device,
        )
        root_supported, root_reason = call_selected_root_reset(
            robot,
            pos_target=root_target,
            quat_target=quat_target,
            target_env_ids=target_env_ids,
            torch_module=torch_module,
            logical_cuda_device=logical_cuda_device,
        )
    reset_supported, reset_reason = call_selected_dof_reset(
        robot,
        reset_target=reset_target,
        target_env_ids=target_env_ids,
        dof_indices=dof_indices,
        torch_module=torch_module,
        logical_cuda_device=logical_cuda_device,
    )
    reset_time_s = elapsed_since(started)
    if not root_supported:
        return {
            "includes_reset_time": True,
            "supported": False,
            "changes_only_target_envs": False,
            "time_s": reset_time_s,
            "reason": root_reason,
        }
    if not reset_supported:
        return {
            "includes_reset_time": True,
            "supported": False,
            "changes_only_target_envs": False,
            "time_s": reset_time_s,
            "reason": reset_reason,
        }
    after_state = read_state(robot, dof_indices)
    after_rows = state_rows(after_state, n_envs=n_envs)
    classification = classify_selected_reset_change(
        before_rows,
        after_rows,
        target_env_ids=target_env_ids,
    )
    return {
        "includes_reset_time": True,
        "supported": True,
        "changes_only_target_envs": classification.changes_only_target_envs,
        "time_s": reset_time_s,
        "reason": classification.reason,
    }


def make_selected_dof_reset_target(
    current_dofs_pos: Any,
    *,
    target_env_ids: tuple[int, ...],
    n_envs: int,
    n_dofs: int,
    torch_module: Any | None,
    logical_cuda_device: str,
) -> Any:
    rows = tensor_rows(current_dofs_pos, n_envs=n_envs)
    selected_rows = [list(rows[index]) if rows else [0.0] * n_dofs for index in target_env_ids]
    for row in selected_rows:
        for index in range(len(row)):
            row[index] = float(row[index]) + 0.01
    if torch_module is not None:
        return torch_module.tensor(selected_rows, device=logical_cuda_device)
    return selected_rows


def make_selected_root_pos_reset_target(
    current_root_pos: Any,
    *,
    target_env_ids: tuple[int, ...],
    n_envs: int,
    torch_module: Any | None,
    logical_cuda_device: str,
) -> Any:
    rows = tensor_rows(current_root_pos, n_envs=n_envs)
    selected_rows = [list(rows[index]) if rows else [0.0, 0.0, 0.0] for index in target_env_ids]
    for row in selected_rows:
        if len(row) < 3:
            row.extend([0.0] * (3 - len(row)))
        row[0] = float(row[0]) + 0.01
    if torch_module is not None:
        return torch_module.tensor(selected_rows, device=logical_cuda_device)
    return selected_rows


def make_selected_root_quat_reset_target(
    current_root_quat: Any,
    *,
    target_env_ids: tuple[int, ...],
    n_envs: int,
    torch_module: Any | None,
    logical_cuda_device: str,
) -> Any:
    rows = tensor_rows(current_root_quat, n_envs=n_envs)
    selected_rows = [
        list(rows[index]) if rows else [0.99995, 0.0, 0.0, 0.01]
        for index in target_env_ids
    ]
    for row in selected_rows:
        if len(row) < 4:
            row[:] = [1.0, 0.0, 0.0, 0.0]
        row[0] = 0.99995
        row[1] = 0.0
        row[2] = 0.0
        row[3] = 0.01
    if torch_module is not None:
        return torch_module.tensor(selected_rows, device=logical_cuda_device)
    return selected_rows


def call_selected_root_reset(
    robot: Any,
    *,
    pos_target: Any,
    quat_target: Any,
    target_env_ids: tuple[int, ...],
    torch_module: Any | None,
    logical_cuda_device: str,
) -> tuple[bool, str]:
    env_ids: Any = target_env_ids
    if torch_module is not None:
        env_ids = torch_module.tensor(target_env_ids, device=logical_cuda_device)
    root_qpos_reason = ""
    if hasattr(robot, "set_qpos"):
        qpos_target = combine_root_pos_quat(pos_target, quat_target)
        root_qpos_ok, root_qpos_reason = call_selected_root_qpos_setter(
            robot.set_qpos,
            target=qpos_target,
            target_env_ids=env_ids,
        )
        if root_qpos_ok:
            return True, ""
        # Fall through to set_pos/set_quat for Genesis versions or assets where
        # selected root qpos is unavailable but selected pose setters work.
    if not hasattr(robot, "set_pos"):
        suffix = f":{root_qpos_reason}" if root_qpos_reason else ""
        return False, f"robot_set_pos_missing{suffix}"
    if not hasattr(robot, "set_quat"):
        suffix = f":{root_qpos_reason}" if root_qpos_reason else ""
        return False, f"robot_set_quat_missing{suffix}"
    root_pos_ok, root_pos_reason = call_selected_root_setter(
        "robot.set_pos",
        robot.set_pos,
        target=pos_target,
        target_env_ids=env_ids,
    )
    if not root_pos_ok:
        return False, root_pos_reason
    root_quat_ok, root_quat_reason = call_selected_root_setter(
        "robot.set_quat",
        robot.set_quat,
        target=quat_target,
        target_env_ids=env_ids,
    )
    if not root_quat_ok:
        return False, root_quat_reason
    return True, ""


def combine_root_pos_quat(pos_target: Any, quat_target: Any) -> Any:
    pos_rows = as_rows(pos_target)
    quat_rows = as_rows(quat_target)
    combined = [
        list(pos_row[:3]) + list(quat_row[:4])
        for pos_row, quat_row in zip(pos_rows, quat_rows)
    ]
    if hasattr(pos_target, "device") and hasattr(pos_target, "new_tensor"):
        return pos_target.new_tensor(combined)
    return combined


def call_selected_root_qpos_setter(
    method: Callable[..., Any],
    *,
    target: Any,
    target_env_ids: Any,
) -> tuple[bool, str]:
    vector_target = first_env_row(target)
    candidates = (
        {"qs_idx_local": tuple(range(7)), "envs_idx": target_env_ids, "zero_velocity": True},
        {"qs_idx_local": tuple(range(7)), "envs_idx": target_env_ids},
    )
    last_error = ""
    for kwargs in candidates:
        try:
            method(target, **kwargs)
            return True, ""
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            return False, f"robot.set_qpos_failed:{exc}"
    for kwargs in candidates:
        try:
            method(vector_target, **kwargs)
            return True, ""
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            return False, f"robot.set_qpos_failed:{exc}"
    return False, f"robot.set_qpos_selected_envs_unsupported:{last_error}"


def call_selected_root_setter(
    label: str,
    method: Callable[..., Any],
    *,
    target: Any,
    target_env_ids: Any,
) -> tuple[bool, str]:
    vector_target = first_env_row(target)
    candidates = (
        {"envs_idx": target_env_ids, "zero_velocity": True},
        {"envs_idx": target_env_ids},
    )
    last_error = ""
    for kwargs in candidates:
        try:
            method(target, **kwargs)
            return True, ""
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            return False, f"{label}_failed:{exc}"
    for kwargs in candidates:
        try:
            method(vector_target, **kwargs)
            return True, ""
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            return False, f"{label}_failed:{exc}"
    return False, f"{label}_selected_envs_unsupported:{last_error}"


def call_selected_dof_reset(
    robot: Any,
    *,
    reset_target: Any,
    target_env_ids: tuple[int, ...],
    dof_indices: tuple[int, ...],
    torch_module: Any | None,
    logical_cuda_device: str,
) -> tuple[bool, str]:
    method = robot.set_dofs_position
    env_ids: Any = target_env_ids
    if torch_module is not None:
        env_ids = torch_module.tensor(target_env_ids, device=logical_cuda_device)
    vector_target = first_env_row(reset_target)
    candidates = (
        {
            "position": reset_target,
            "dofs_idx_local": dof_indices,
            "envs_idx": env_ids,
            "zero_velocity": True,
        },
        {"position": reset_target, "dofs_idx_local": dof_indices, "envs_idx": env_ids},
        {"position": reset_target, "dofs_idx_local": dof_indices, "env_ids": env_ids},
        {"position": reset_target, "dofs_idx": dof_indices, "envs_idx": env_ids},
        {
            "position": vector_target,
            "dofs_idx_local": dof_indices,
            "envs_idx": env_ids,
            "zero_velocity": True,
        },
        {"position": vector_target, "dofs_idx_local": dof_indices, "envs_idx": env_ids},
    )
    last_error = ""
    for kwargs in candidates:
        try:
            method(**kwargs)
            return True, ""
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            return False, f"robot.set_dofs_position_failed:{exc}"
    return False, f"robot.set_dofs_position_selected_envs_unsupported:{last_error}"


def state_rows(state: StateRead, *, n_envs: int) -> list[list[float]]:
    row_groups = [
        tensor_rows(value, n_envs=n_envs)
        for value in (
            state.qpos,
            state.dofs_pos,
            state.dofs_vel,
            state.root_pos,
            state.root_quat,
            state.root_vel,
        )
        if value is not None
    ]
    row_groups = [rows for rows in row_groups if rows]
    if not row_groups:
        return []
    combined = [[] for _ in range(n_envs)]
    for rows in row_groups:
        if len(rows) != n_envs:
            continue
        for env_index, row in enumerate(rows):
            combined[env_index].extend(float(value) for value in row)
    return combined


def classify_selected_reset_change(
    before: Any,
    after: Any,
    *,
    target_env_ids: Sequence[int],
    atol: float = 1e-8,
) -> SelectedResetResult:
    before_rows = as_rows(before)
    after_rows = as_rows(after)
    if len(before_rows) != len(after_rows):
        return SelectedResetResult(False, False, "row_count_mismatch", ())
    if len(before_rows) < 2:
        return SelectedResetResult(False, False, "not_applicable_n_envs_lt_2", ())
    targets = set(int(index) for index in target_env_ids)
    changed_envs = tuple(
        index
        for index, (before_row, after_row) in enumerate(zip(before_rows, after_rows))
        if row_changed(before_row, after_row, atol=atol)
    )
    target_changed = all(index in changed_envs for index in targets)
    non_target_changed = any(index not in targets for index in changed_envs)
    if target_changed and not non_target_changed:
        return SelectedResetResult(True, True, "only_target_envs_changed", changed_envs)
    if not target_changed:
        return SelectedResetResult(True, False, "target_env_unchanged", changed_envs)
    return SelectedResetResult(True, False, "non_target_env_changed", changed_envs)


def as_rows(value: Any) -> list[list[float]]:
    data = to_python_data(value)
    if data is None:
        return []
    if isinstance(data, (int, float)):
        return [[float(data)]]
    if not isinstance(data, list):
        data = list(data)
    if not data:
        return []
    if all(isinstance(item, (int, float)) for item in data):
        return [[float(item) for item in data]]
    return [flatten_numeric(row) for row in data]


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = list(value)
    flattened: list[float] = []
    for item in value:
        flattened.extend(flatten_numeric(item))
    return flattened


def tensor_rows(value: Any, *, n_envs: int) -> list[list[float]]:
    rows = as_rows(value)
    if len(rows) == n_envs:
        return rows
    if len(rows) == 1 and n_envs > 1:
        flat = rows[0]
        if len(flat) % n_envs != 0:
            return []
        width = len(flat) // n_envs
        return [flat[index * width : (index + 1) * width] for index in range(n_envs)]
    return rows


def row_changed(before: Sequence[float], after: Sequence[float], *, atol: float) -> bool:
    if len(before) != len(after):
        return True
    return any(abs(float(left) - float(right)) > atol for left, right in zip(before, after))


def to_python_data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def first_env_row(value: Any) -> Any:
    if hasattr(value, "__getitem__"):
        try:
            return value[0]
        except Exception:
            pass
    rows = as_rows(value)
    return rows[0] if rows else value


def tensor_device_name(value: Any) -> str:
    if value is None:
        return UNAVAILABLE
    device = getattr(value, "device", None)
    if device is not None:
        return str(device)
    if getattr(value, "is_cuda", False):
        return "cuda"
    return "not_tensor"


def tensor_devices_ok(
    devices: Iterable[str],
    *,
    backend: str,
    logical_cuda_device: str,
) -> bool:
    required = tuple(devices)
    if any(device in {"not_available", UNAVAILABLE} for device in required):
        return False
    if backend != "cuda":
        return True
    return all(device == logical_cuda_device for device in required)


def state_device_metrics(state: StateRead) -> dict[str, str]:
    return {
        "qpos_device": tensor_device_name(state.qpos),
        "dofs_pos_device": tensor_device_name(state.dofs_pos),
        "dofs_vel_device": tensor_device_name(state.dofs_vel),
        "root_pos_device": tensor_device_name(state.root_pos),
        "root_quat_device": tensor_device_name(state.root_quat),
        "root_vel_device": tensor_device_name(state.root_vel),
    }


def compute_throughput(
    *,
    policy_steps: int,
    decimation: int,
    n_envs: int,
    elapsed_s: float,
) -> dict[str, float]:
    if elapsed_s <= 0:
        return {
            "policy_steps_per_sec": 0.0,
            "sim_steps_per_sec": 0.0,
            "env_policy_steps_per_sec": 0.0,
            "env_sim_steps_per_sec": 0.0,
        }
    sim_steps = policy_steps * decimation
    return {
        "policy_steps_per_sec": policy_steps / elapsed_s,
        "sim_steps_per_sec": sim_steps / elapsed_s,
        "env_policy_steps_per_sec": policy_steps * n_envs / elapsed_s,
        "env_sim_steps_per_sec": sim_steps * n_envs / elapsed_s,
    }


def gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return normalize_blocker(f"nvidia_smi_unavailable:{exc}")
    output = (result.stdout or result.stderr).strip()
    if not output:
        output = f"nvidia_smi_exit_{result.returncode}"
    return normalize_blocker(output)


def try_call(
    label: str,
    caller: Callable[[dict[str, Any]], Any],
    candidates: Iterable[Mapping[str, Any]],
) -> Any:
    last_error = ""
    for kwargs in candidates:
        try:
            return caller(dict(kwargs))
        except TypeError as exc:
            last_error = str(exc)
        except Exception as exc:
            raise BlockedProbe(f"{label}_failed:{exc}") from exc
    raise BlockedProbe(f"{label}_unsupported:{last_error}")


def call_method(
    label: str,
    method: Callable[..., Any],
    candidates: Iterable[Mapping[str, Any]],
    *,
    positional: tuple[Any, ...] | None = None,
) -> Any:
    try:
        return try_call(label, lambda kwargs: method(**kwargs), candidates)
    except BlockedProbe as keyword_exc:
        if positional is None:
            raise
        try:
            return method(*positional)
        except Exception as positional_exc:
            raise BlockedProbe(f"{label}_unsupported:{keyword_exc}; positional:{positional_exc}")


def format_key_value(key: str, value: Any) -> str:
    return f"{key}={format_value(value)}"


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (int, str)):
        return normalize_value(str(value))
    return normalize_value(json.dumps(value, sort_keys=True))


def normalize_blocker(value: str) -> str:
    return normalize_value(value)


def normalize_value(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n")


def emit_metrics(metrics: Mapping[str, Any]) -> None:
    for key in METRIC_KEYS:
        print(format_key_value(key, metrics.get(key, UNAVAILABLE)), flush=True)


def elapsed_since(started_at: float) -> float:
    return time.perf_counter() - started_at


if __name__ == "__main__":
    main()
