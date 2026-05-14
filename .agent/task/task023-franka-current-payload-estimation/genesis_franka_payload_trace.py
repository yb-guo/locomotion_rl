"""Task023 Franka endpoint payload trace collector.

Runs one payload mass per process. This avoids Genesis singleton reuse and keeps
each trace reproducible. The script is intentionally task-local until task023
has a passing feasibility decision.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from genesis_franka_effort_api_smoke import (
    SmokeBlocked,
    add_franka,
    add_plane,
    apply_basic_pd,
    build_scene,
    describe_tensor,
    flatten_numeric,
    import_genesis,
    import_numpy,
    init_genesis,
    make_scene,
    matrix_rows,
    read_dof_api,
    read_jacobian,
    resolve_arm_dofs,
    resolve_payload_link,
    resolve_tool_link,
    to_python,
)


Q0 = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
AMP = (0.20, 0.15, 0.20, 0.12, 0.15, 0.12, 0.20)
PHASE = (0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2)
DEFAULT_OUTPUT_ROOT = Path("outputs/task023/franka_current_force_estimation")
DEFAULT_ASSET = "xml/franka_emika_panda/panda_nohand.xml"


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.perf_counter()
    report = initial_report(args)
    try:
        report.update(run_trace(args))
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
        write_json(Path(report["summary_path"]), report)
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["status"] != "ok":
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a Franka payload effort trace.")
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--sweep-s", type=float, default=12.0)
    parser.add_argument("--payload-mass-kg", type=float, required=True)
    parser.add_argument(
        "--force-limit-scale",
        type=float,
        default=1.0,
        help="Scale the nominal Franka force limits after the basic PD setup.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--render-gif", type=Path)
    parser.add_argument("--render-every-steps", type=int, default=50)
    parser.add_argument("--render-fps", type=float, default=12.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-pos", nargs=3, type=float, default=(1.4, -1.8, 1.25))
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.3, 0.0, 0.55))
    parser.add_argument("--camera-fov", type=float, default=42.0)
    parser.add_argument("--strict", action="store_true")
    return parser


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or mass_run_id(args.payload_mass_kg)
    summary_path = args.output_root / "summaries" / f"{run_id}.json"
    trace_path = args.output_root / "traces" / f"{run_id}.npz"
    return {
        "status": "blocked",
        "blocker": "not_run",
        "asset": args.asset,
        "backend": args.backend,
        "sim_dt": args.sim_dt,
        "hold_s": args.hold_s,
        "sweep_s": args.sweep_s,
        "payload_mass_kg": args.payload_mass_kg,
        "payload_force_z_N": -args.payload_mass_kg * 9.81,
        "force_limit_scale": args.force_limit_scale,
        "run_id": run_id,
        "output_root": str(args.output_root),
        "summary_path": str(summary_path),
        "trace_path": str(trace_path),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "render_gif": str(args.render_gif) if args.render_gif is not None else "",
        "render_every_steps": args.render_every_steps,
        "render_fps": args.render_fps,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "camera_pos": list(args.camera_pos),
        "camera_lookat": list(args.camera_lookat),
        "camera_fov": args.camera_fov,
        "q0": list(Q0),
        "amp": list(AMP),
        "phase": list(PHASE),
    }


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    if args.sim_dt <= 0.0:
        raise SmokeBlocked("sim_dt_must_be_positive")
    if args.hold_s < 0.0 or args.sweep_s <= 0.0:
        raise SmokeBlocked("invalid_hold_or_sweep_duration")
    if args.payload_mass_kg < 0.0:
        raise SmokeBlocked("payload_mass_must_be_non_negative")
    if args.force_limit_scale <= 0.0:
        raise SmokeBlocked("force_limit_scale_must_be_positive")
    if args.render_gif is not None and args.render_every_steps <= 0:
        raise SmokeBlocked("render_every_steps_must_be_positive")
    if args.render_gif is not None and args.render_fps <= 0.0:
        raise SmokeBlocked("render_fps_must_be_positive")

    gs = import_genesis()
    np = import_numpy()
    init_genesis(gs, args.backend)

    scene = make_scene(gs, args.sim_dt)
    add_plane(gs, scene)
    franka, morph_info = add_franka(gs, scene, args.asset)
    payload, payload_asset_path = add_payload_if_needed(gs, scene, args)
    camera = add_camera_if_needed(scene, args)
    build_scene(scene, 1)

    joint_names, arm_dofs = resolve_arm_dofs(franka)
    tool_link, tool_link_name = resolve_tool_link(franka)
    apply_basic_pd(franka, np, arm_dofs)
    force_limit = apply_force_limit_scale(franka, np, arm_dofs, args.force_limit_scale)
    set_arm_state(franka, np, arm_dofs, Q0)
    scene.step()

    attach_report = attach_payload_if_needed(gs, np, scene, franka, tool_link, payload)
    if payload is not None and attach_report["status"] != "ok":
        raise SmokeBlocked(f"payload_attach_failed:{attach_report['blocker']}")

    hold_steps = int(round(args.hold_s / args.sim_dt))
    sweep_steps = int(round(args.sweep_s / args.sim_dt))
    total_steps = hold_steps + sweep_steps
    if total_steps <= 0:
        raise SmokeBlocked("total_steps_must_be_positive")

    arrays = init_arrays(np, total_steps, n_dofs=len(arm_dofs))
    rendered_frames: list[Any] = []
    first_rgb_shape: list[int] | str = []
    for step in range(total_steps):
        t = step * args.sim_dt
        target = target_at_step(step, hold_steps, args.sim_dt, args.sweep_s)
        control_position(franka, target, arm_dofs)
        scene.step()
        record_step(
            np,
            arrays,
            step=step,
            t=t,
            payload_mass_kg=args.payload_mass_kg,
            target=target,
            franka=franka,
            tool_link=tool_link,
            arm_dofs=arm_dofs,
        )
        if camera is not None and (step % args.render_every_steps == 0 or step == total_steps - 1):
            rgb = render_camera_rgb(camera)
            rendered_frames.append(rgb)
            if not first_rgb_shape:
                first_rgb_shape = list(getattr(rgb, "shape", [])) or str(type(rgb))

    trace_path = args.output_root / "traces" / f"{args.run_id or mass_run_id(args.payload_mass_kg)}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **arrays)
    render_report = save_render_gif_if_needed(args, rendered_frames, first_rgb_shape)

    report = {
        "genesis_version": str(getattr(gs, "__version__", "unknown")),
        "morph_requires_jac_and_ik_used": morph_info["requires_jac_and_IK"],
        "robot_n_dofs": int(getattr(franka, "n_dofs", -1)),
        "arm_joint_names": joint_names,
        "arm_dof_indices": arm_dofs,
        "tool_link_name": tool_link_name,
        "tool_link_idx": int(getattr(tool_link, "idx", -1)),
        "payload_asset_path": str(payload_asset_path) if payload_asset_path is not None else "",
        "payload_attach": attach_report,
        "hold_steps": hold_steps,
        "sweep_steps": sweep_steps,
        "total_steps": total_steps,
        "sample_rate_hz": 1.0 / args.sim_dt,
        "force_limit": force_limit,
        "trace_path": str(trace_path),
        "render": render_report,
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "finite": all(bool(np.isfinite(value).all()) for value in arrays.values()),
        "effort_control": describe_array(np, arrays["effort_control"]),
        "effort_internal": describe_array(np, arrays["effort_internal"]),
        "tracking_error": describe_array(np, arrays["tracking_error"]),
        "jacobian_translation_row_norm_mean": row_norm_means(np, arrays["jacobian"][:, :3, :]),
        "jacobian_rotation_row_norm_mean": row_norm_means(np, arrays["jacobian"][:, 3:6, :]),
    }
    if not report["finite"]:
        raise SmokeBlocked("nonfinite_trace_array")
    return report


def add_camera_if_needed(scene: Any, args: argparse.Namespace) -> Any | None:
    if args.render_gif is None:
        return None
    try:
        return scene.add_camera(
            res=(args.camera_width, args.camera_height),
            pos=tuple(args.camera_pos),
            lookat=tuple(args.camera_lookat),
            fov=args.camera_fov,
            GUI=False,
        )
    except Exception as exc:
        raise SmokeBlocked(f"camera_create_failed:{exc}") from exc


def render_camera_rgb(camera: Any) -> Any:
    try:
        rgb, _, _, _ = camera.render(
            rgb=True,
            depth=False,
            segmentation=False,
            normal=False,
        )
        return rgb
    except Exception as exc:
        raise SmokeBlocked(f"camera_render_failed:{exc}") from exc


def save_render_gif_if_needed(
    args: argparse.Namespace,
    rendered_frames: Sequence[Any],
    first_rgb_shape: list[int] | str,
) -> dict[str, Any]:
    if args.render_gif is None:
        return {
            "enabled": False,
            "path": "",
            "frames": 0,
            "rgb_shape": [],
            "blocker": "",
        }
    if not rendered_frames:
        raise SmokeBlocked("render_enabled_but_no_frames")
    args.render_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except Exception as exc:
        raise SmokeBlocked(f"imageio_import_failed:{exc}") from exc
    try:
        imageio.mimsave(args.render_gif, list(rendered_frames), duration=1.0 / args.render_fps)
    except Exception as exc:
        raise SmokeBlocked(f"gif_save_failed:{exc}") from exc
    return {
        "enabled": True,
        "path": str(args.render_gif),
        "frames": len(rendered_frames),
        "rgb_shape": first_rgb_shape,
        "fps": args.render_fps,
        "every_steps": args.render_every_steps,
        "bytes": args.render_gif.stat().st_size,
        "blocker": "",
    }


def add_payload_if_needed(
    gs: Any,
    scene: Any,
    args: argparse.Namespace,
) -> tuple[Any | None, Path | None]:
    if args.payload_mass_kg == 0.0:
        return None, None
    payload_dir = args.output_root / "payload_assets"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"payload_{mass_token(args.payload_mass_kg)}kg.xml"
    write_payload_mjcf(payload_path, args.payload_mass_kg)
    payload = scene.add_entity(
        gs.morphs.MJCF(
            file=str(payload_path),
            pos=(0.45, 0.0, 0.45),
            requires_jac_and_IK=False,
        )
    )
    return payload, payload_path


def write_payload_mjcf(path: Path, mass_kg: float) -> None:
    half_size = 0.025
    xml = f"""<mujoco model="task023_payload_{mass_token(mass_kg)}kg">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="payload_body" pos="0 0 0">
      <joint name="payload_free" type="free"/>
      <geom name="payload_geom" type="box" size="{half_size} {half_size} {half_size}" mass="{mass_kg}"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")


def set_arm_state(franka: Any, np: Any, arm_dofs: Sequence[int], q: Sequence[float]) -> None:
    try:
        franka.set_dofs_position(q, dofs_idx_local=arm_dofs, zero_velocity=True)
    except TypeError:
        franka.set_dofs_position(q, arm_dofs)
    if hasattr(franka, "set_dofs_velocity"):
        zeros = np.zeros(len(arm_dofs), dtype=float)
        try:
            franka.set_dofs_velocity(zeros, dofs_idx_local=arm_dofs)
        except TypeError:
            try:
                franka.set_dofs_velocity(zeros, arm_dofs)
            except Exception:
                pass
        except Exception:
            pass


def apply_force_limit_scale(
    franka: Any,
    np: Any,
    arm_dofs: Sequence[int],
    scale: float,
) -> dict[str, Any]:
    nominal_upper = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
    lower = -nominal_upper * scale
    upper = nominal_upper * scale
    report = {
        "scale": float(scale),
        "lower": [float(value) for value in lower],
        "upper": [float(value) for value in upper],
        "applied": False,
        "blocker": "",
    }
    if not hasattr(franka, "set_dofs_force_range"):
        report["blocker"] = "set_dofs_force_range_missing"
        return report
    try:
        franka.set_dofs_force_range(lower, upper, dofs_idx_local=arm_dofs)
        report["applied"] = True
    except TypeError:
        try:
            franka.set_dofs_force_range(lower, upper, arm_dofs)
            report["applied"] = True
        except Exception as exc:
            report["blocker"] = str(exc)
    except Exception as exc:
        report["blocker"] = str(exc)
    return report


def attach_payload_if_needed(
    gs: Any,
    np: Any,
    scene: Any,
    franka: Any,
    tool_link: Any,
    payload: Any | None,
) -> dict[str, Any]:
    if payload is None:
        return {"status": "not_applicable", "blocker": "", "payload_link_name": ""}

    payload_link = resolve_payload_link(payload)
    tool_pos = read_link_pos(franka, tool_link)
    move_entity_to_pos(payload, tool_pos)
    scene.step()

    rigid = getattr(getattr(scene, "sim", None), "rigid_solver", None)
    if rigid is None:
        return {"status": "blocked", "blocker": "rigid_solver_missing"}
    if not hasattr(rigid, "add_weld_constraint"):
        return {"status": "blocked", "blocker": "add_weld_constraint_missing"}
    payload_ids = np.array([int(getattr(payload_link, "idx"))], dtype=gs.np_int)
    tool_ids = np.array([int(getattr(tool_link, "idx"))], dtype=gs.np_int)
    try:
        rigid.add_weld_constraint(payload_ids, tool_ids)
    except Exception as exc:
        return {"status": "blocked", "blocker": f"add_weld_constraint_failed:{exc}"}
    return {
        "status": "ok",
        "blocker": "",
        "payload_link_name": str(getattr(payload_link, "name", "unknown")),
        "payload_link_idx": int(getattr(payload_link, "idx", -1)),
        "tool_link_idx": int(getattr(tool_link, "idx", -1)),
        "tool_pos_at_attach": tool_pos,
    }


def read_link_pos(franka: Any, link: Any) -> list[float]:
    if hasattr(link, "get_pos"):
        return flatten_xyz(link.get_pos())
    if hasattr(franka, "get_links_pos"):
        idx = int(getattr(link, "idx"))
        try:
            return flatten_xyz(franka.get_links_pos(links_idx_local=[idx]))
        except TypeError:
            return flatten_xyz(franka.get_links_pos([idx]))
    raise SmokeBlocked("tool_link_position_unavailable")


def move_entity_to_pos(entity: Any, pos: Sequence[float]) -> None:
    if hasattr(entity, "set_pos"):
        try:
            entity.set_pos(pos, zero_velocity=True)
        except TypeError:
            entity.set_pos(pos)


def init_arrays(np: Any, steps: int, *, n_dofs: int) -> dict[str, Any]:
    return {
        "step": np.zeros((steps,), dtype=np.int64),
        "t": np.zeros((steps,), dtype=np.float64),
        "payload_mass_kg": np.zeros((steps,), dtype=np.float64),
        "payload_force_z_N": np.zeros((steps,), dtype=np.float64),
        "q": np.zeros((steps, n_dofs), dtype=np.float64),
        "dq": np.zeros((steps, n_dofs), dtype=np.float64),
        "q_target": np.zeros((steps, n_dofs), dtype=np.float64),
        "tracking_error": np.zeros((steps, n_dofs), dtype=np.float64),
        "effort_control": np.zeros((steps, n_dofs), dtype=np.float64),
        "effort_internal": np.zeros((steps, n_dofs), dtype=np.float64),
        "jacobian": np.zeros((steps, 6, n_dofs), dtype=np.float64),
        "tool_pos": np.zeros((steps, 3), dtype=np.float64),
    }


def target_at_step(step: int, hold_steps: int, sim_dt: float, sweep_s: float) -> list[float]:
    if step < hold_steps:
        return list(Q0)
    t = (step - hold_steps) * sim_dt
    omega = 2.0 * math.pi / sweep_s
    return [
        Q0[index] + AMP[index] * math.sin(omega * t + PHASE[index])
        for index in range(len(Q0))
    ]


def control_position(franka: Any, target: Sequence[float], arm_dofs: Sequence[int]) -> None:
    try:
        franka.control_dofs_position(target, dofs_idx_local=arm_dofs)
    except TypeError:
        franka.control_dofs_position(target, arm_dofs)


def record_step(
    np: Any,
    arrays: dict[str, Any],
    *,
    step: int,
    t: float,
    payload_mass_kg: float,
    target: Sequence[float],
    franka: Any,
    tool_link: Any,
    arm_dofs: Sequence[int],
) -> None:
    q = np.array(flatten_numeric(read_dof_api(franka, "get_dofs_position", arm_dofs)), dtype=float)
    dq = np.array(flatten_numeric(read_dof_api(franka, "get_dofs_velocity", arm_dofs)), dtype=float)
    effort_control = np.array(
        flatten_numeric(read_dof_api(franka, "get_dofs_control_force", arm_dofs)),
        dtype=float,
    )
    effort_internal = np.array(
        flatten_numeric(read_dof_api(franka, "get_dofs_force", arm_dofs)),
        dtype=float,
    )
    jacobian = np.array(matrix_rows(read_jacobian(franka, tool_link)), dtype=float)
    if jacobian.shape != (6, len(arm_dofs)):
        jacobian = jacobian.reshape(6, len(arm_dofs))
    target_array = np.array(target, dtype=float)
    arrays["step"][step] = step
    arrays["t"][step] = t
    arrays["payload_mass_kg"][step] = payload_mass_kg
    arrays["payload_force_z_N"][step] = -payload_mass_kg * 9.81
    arrays["q"][step] = q[: len(arm_dofs)]
    arrays["dq"][step] = dq[: len(arm_dofs)]
    arrays["q_target"][step] = target_array
    arrays["tracking_error"][step] = q[: len(arm_dofs)] - target_array
    arrays["effort_control"][step] = effort_control[: len(arm_dofs)]
    arrays["effort_internal"][step] = effort_internal[: len(arm_dofs)]
    arrays["jacobian"][step] = jacobian[:, : len(arm_dofs)]
    arrays["tool_pos"][step] = np.array(read_link_pos(franka, tool_link), dtype=float)[:3]


def describe_array(np: Any, array: Any) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean_abs": float(np.mean(np.abs(array))),
        "rms": float(np.sqrt(np.mean(array * array))),
    }


def row_norm_means(np: Any, jacobian_rows: Any) -> list[float]:
    norms = np.linalg.norm(jacobian_rows, axis=2)
    return [float(value) for value in np.mean(norms, axis=0)]


def flatten_xyz(value: Any) -> list[float]:
    flat = flatten_numeric(value)
    if len(flat) >= 3:
        return [float(flat[0]), float(flat[1]), float(flat[2])]
    raise SmokeBlocked(f"xyz_unavailable:{to_python(value)}")


def mass_token(mass_kg: float) -> str:
    return f"{mass_kg:g}".replace("-", "neg").replace(".", "p")


def mass_run_id(mass_kg: float) -> str:
    return f"mass_{mass_token(mass_kg)}kg"


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
