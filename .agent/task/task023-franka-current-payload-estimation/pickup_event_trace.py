"""Task023 pickup-event Franka payload trace collector.

Runs one deterministic Genesis episode per process. The payload is pre-created
and only welded to the Franka tool at the configured pickup event. This keeps
the first pickup test narrow: it diagnoses endpoint payload observability, not
contact-rich grasping.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from genesis_franka_effort_api_smoke import (
    SmokeBlocked,
    add_franka,
    add_plane,
    apply_basic_pd,
    build_scene,
    import_genesis,
    import_numpy,
    init_genesis,
    make_scene,
    resolve_arm_dofs,
    resolve_tool_link,
)
from genesis_franka_payload_trace import (
    add_camera_if_needed,
    add_payload_if_needed,
    apply_force_limit_scale,
    attach_payload_if_needed,
    control_position,
    describe_array,
    mass_token,
    move_entity_to_pos,
    record_step,
    render_camera_rgb,
    row_norm_means,
    save_render_gif_if_needed,
    set_arm_state,
)

Q_HOME_A = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
Q_HOME_VARIANTS = {
    "home_a": Q_HOME_A,
    "home_b": tuple(a + b for a, b in zip(Q_HOME_A, (0.10, 0.05, -0.08, 0.04, 0.05, -0.04, 0.06))),
    "home_c": tuple(a + b for a, b in zip(Q_HOME_A, (-0.12, 0.08, 0.10, -0.04, -0.06, 0.04, -0.05))),
}
Q_PICK_OFFSET = (0.18, -0.10, 0.14, 0.08, -0.12, 0.06, 0.10)
Q_CARRY_OFFSET = (-0.16, 0.12, -0.12, -0.06, 0.10, -0.06, -0.08)
DEFAULT_OUTPUT_ROOT = Path("outputs/task023/franka_current_force_estimation")
DEFAULT_ASSET = "xml/franka_emika_panda/panda_nohand.xml"
RECOVERABLE_TRACE_ERRORS = (
    ArithmeticError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
PHASE_TO_ID = {
    "home_hold_before": 0,
    "move_to_pick_unloaded": 1,
    "settle_after_pick": 2,
    "move_with_payload": 3,
    "return_home_loaded": 4,
    "final_hold_loaded": 5,
    "home_hold_after": 6,
}


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
    except RECOVERABLE_TRACE_ERRORS as exc:  # pragma: no cover - target-only simulator failures.
        report["status"] = "failed"
        report["blocker"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        report["elapsed_s"] = round(time.perf_counter() - started, 6)
        write_json(Path(report["summary_path"]), report)
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["status"] != "ok":
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a Franka pickup-event effort trace.")
    parser.add_argument("--scenario", choices=("pickup_transport", "return_home_diff"), required=True)
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--payload-mass-kg", type=float, required=True)
    parser.add_argument("--home-variant", choices=tuple(Q_HOME_VARIANTS), default="home_a")
    parser.add_argument("--force-limit-scale", type=float, default=1.0)
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
    run_id = args.run_id or default_run_id(args.scenario, args.payload_mass_kg, args.home_variant)
    summary_path = args.output_root / "summaries" / f"{run_id}.json"
    trace_path = args.output_root / "traces" / f"{run_id}.npz"
    return {
        "status": "blocked",
        "blocker": "not_run",
        "scenario": args.scenario,
        "asset": args.asset,
        "backend": args.backend,
        "sim_dt": args.sim_dt,
        "payload_mass_nominal_kg": args.payload_mass_kg,
        "home_variant": args.home_variant,
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
    }


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    gs = import_genesis()
    np = import_numpy()
    init_genesis(gs, args.backend)

    q_home = Q_HOME_VARIANTS[args.home_variant]
    q_pick = add_vec(q_home, Q_PICK_OFFSET)
    q_carry = add_vec(q_home, Q_CARRY_OFFSET)
    phases = build_phases(args.scenario, args.sim_dt, q_home, q_pick, q_carry)
    pickup_event_step = sum(phase["steps"] for phase in phases[:2])
    total_steps = sum(phase["steps"] for phase in phases)
    if total_steps <= 0:
        raise SmokeBlocked("total_steps_must_be_positive")

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
    set_arm_state(franka, np, arm_dofs, q_home)
    if payload is not None:
        move_entity_to_pos(payload, (1.2, 1.2, 0.8))
    scene.step()

    arrays = init_pickup_arrays(np, total_steps, n_dofs=len(arm_dofs))
    rendered_frames: list[Any] = []
    first_rgb_shape: list[int] | str = []
    attached = False
    attach_report: dict[str, Any] = {
        "status": "not_applicable" if payload is None else "not_run",
        "blocker": "",
    }
    phase_starts = phase_start_steps(phases)
    force_upper = np.array(force_limit.get("upper", [87, 87, 87, 87, 12, 12, 12]), dtype=float)

    for step in range(total_steps):
        if step == pickup_event_step and payload is not None:
            attach_report = attach_payload_if_needed(gs, np, scene, franka, tool_link, payload)
            if attach_report["status"] != "ok":
                raise SmokeBlocked(f"runtime_payload_attach_failed:{attach_report['blocker']}")
            attached = True

        phase_index, phase = phase_for_step(step, phases, phase_starts)
        target = target_for_phase_step(step, phase, phase_starts[phase_index])
        control_position(franka, target, arm_dofs)
        scene.step()

        mass_at_tool = args.payload_mass_kg if attached else 0.0
        record_step(
            np,
            arrays,
            step=step,
            t=step * args.sim_dt,
            payload_mass_kg=mass_at_tool,
            target=target,
            franka=franka,
            tool_link=tool_link,
            arm_dofs=arm_dofs,
        )
        arrays["phase"][step] = phase["name"]
        arrays["phase_id"][step] = PHASE_TO_ID[phase["name"]]
        arrays["payload_attached"][step] = 1 if attached else 0
        arrays["force_saturation"][step] = int(
            np.any(np.abs(arrays["effort_control"][step]) >= 0.98 * force_upper[: len(arm_dofs)])
        )

        if camera is not None and (step % args.render_every_steps == 0 or step == total_steps - 1):
            rgb = render_camera_rgb(camera)
            rendered_frames.append(rgb)
            if not first_rgb_shape:
                first_rgb_shape = list(getattr(rgb, "shape", [])) or str(type(rgb))

    trace_path = args.output_root / "traces" / f"{args.run_id or default_run_id(args.scenario, args.payload_mass_kg, args.home_variant)}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        **arrays,
        scenario=np.array(args.scenario),
        run_id=np.array(args.run_id or default_run_id(args.scenario, args.payload_mass_kg, args.home_variant)),
        pickup_event_step=np.array(pickup_event_step, dtype=np.int64),
        payload_mass_nominal_kg=np.array(args.payload_mass_kg, dtype=np.float64),
        q_home=np.array(q_home, dtype=np.float64),
        q_pick=np.array(q_pick, dtype=np.float64),
        q_carry=np.array(q_carry, dtype=np.float64),
    )
    render_report = save_render_gif_if_needed(args, rendered_frames, first_rgb_shape)

    numeric_arrays = {name: value for name, value in arrays.items() if name != "phase"}
    finite = all(bool(np.isfinite(value).all()) for value in numeric_arrays.values())
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
        "q_home": list(q_home),
        "q_pick": list(q_pick),
        "q_carry": list(q_carry),
        "phases": summarize_phases(phases, args.sim_dt),
        "pickup_event_step": pickup_event_step,
        "pickup_time_s": pickup_event_step * args.sim_dt,
        "total_steps": total_steps,
        "sample_rate_hz": 1.0 / args.sim_dt,
        "force_limit": force_limit,
        "force_saturation_ratio": float(np.mean(arrays["force_saturation"])),
        "trace_path": str(trace_path),
        "render": render_report,
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "finite": finite,
        "effort_control": describe_array(np, arrays["effort_control"]),
        "effort_internal": describe_array(np, arrays["effort_internal"]),
        "tracking_error": describe_array(np, arrays["tracking_error"]),
        "jacobian_translation_row_norm_mean": row_norm_means(np, arrays["jacobian"][:, :3, :]),
        "jacobian_rotation_row_norm_mean": row_norm_means(np, arrays["jacobian"][:, 3:6, :]),
    }
    if not finite:
        raise SmokeBlocked("nonfinite_trace_array")
    return report


def validate_args(args: argparse.Namespace) -> None:
    if args.sim_dt <= 0.0:
        raise SmokeBlocked("sim_dt_must_be_positive")
    if args.payload_mass_kg < 0.0:
        raise SmokeBlocked("payload_mass_must_be_non_negative")
    if args.force_limit_scale <= 0.0:
        raise SmokeBlocked("force_limit_scale_must_be_positive")
    if args.render_gif is not None and args.render_every_steps <= 0:
        raise SmokeBlocked("render_every_steps_must_be_positive")
    if args.render_gif is not None and args.render_fps <= 0.0:
        raise SmokeBlocked("render_fps_must_be_positive")


def init_pickup_arrays(np: Any, steps: int, *, n_dofs: int) -> dict[str, Any]:
    arrays = {
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
        "phase": np.empty((steps,), dtype="<U32"),
        "phase_id": np.zeros((steps,), dtype=np.int64),
        "payload_attached": np.zeros((steps,), dtype=np.int64),
        "force_saturation": np.zeros((steps,), dtype=np.int64),
    }
    return arrays


def build_phases(
    scenario: str,
    sim_dt: float,
    q_home: Sequence[float],
    q_pick: Sequence[float],
    q_carry: Sequence[float],
) -> list[dict[str, Any]]:
    if scenario == "pickup_transport":
        specs = (
            ("home_hold_before", 1.0, q_home, q_home),
            ("move_to_pick_unloaded", 3.0, q_home, q_pick),
            ("settle_after_pick", 0.5, q_pick, q_pick),
            ("move_with_payload", 4.0, q_pick, q_carry),
            ("final_hold_loaded", 1.0, q_carry, q_carry),
        )
    elif scenario == "return_home_diff":
        specs = (
            ("home_hold_before", 2.0, q_home, q_home),
            ("move_to_pick_unloaded", 3.0, q_home, q_pick),
            ("settle_after_pick", 0.5, q_pick, q_pick),
            ("return_home_loaded", 3.0, q_pick, q_home),
            ("home_hold_after", 2.0, q_home, q_home),
        )
    else:
        raise SmokeBlocked(f"unknown_scenario:{scenario}")
    phases: list[dict[str, Any]] = []
    for name, duration_s, start_q, end_q in specs:
        steps = round(duration_s / sim_dt)
        if steps <= 0:
            raise SmokeBlocked(f"phase_has_no_steps:{name}")
        phases.append(
            {
                "name": name,
                "duration_s": duration_s,
                "steps": steps,
                "start_q": tuple(start_q),
                "end_q": tuple(end_q),
            }
        )
    return phases


def phase_start_steps(phases: Sequence[Mapping[str, Any]]) -> list[int]:
    starts = []
    current = 0
    for phase in phases:
        starts.append(current)
        current += int(phase["steps"])
    return starts


def phase_for_step(
    step: int,
    phases: Sequence[Mapping[str, Any]],
    starts: Sequence[int],
) -> tuple[int, Mapping[str, Any]]:
    for index in range(len(phases) - 1, -1, -1):
        if step >= starts[index]:
            return index, phases[index]
    return 0, phases[0]


def target_for_phase_step(step: int, phase: Mapping[str, Any], phase_start: int) -> list[float]:
    local = step - phase_start
    steps = int(phase["steps"])
    if steps <= 1:
        blend = 1.0
    else:
        u = max(0.0, min(1.0, local / float(steps - 1)))
        blend = 0.5 - 0.5 * math.cos(math.pi * u)
    start_q = phase["start_q"]
    end_q = phase["end_q"]
    return [float(a + blend * (b - a)) for a, b in zip(start_q, end_q)]


def summarize_phases(phases: Sequence[Mapping[str, Any]], sim_dt: float) -> list[dict[str, Any]]:
    out = []
    start = 0
    for phase in phases:
        steps = int(phase["steps"])
        out.append(
            {
                "name": phase["name"],
                "start_step": start,
                "end_step_exclusive": start + steps,
                "duration_s": steps * sim_dt,
            }
        )
        start += steps
    return out


def add_vec(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a + b) for a, b in zip(left, right))


def default_run_id(scenario: str, mass_kg: float, home_variant: str) -> str:
    return f"{scenario}_{mass_token(mass_kg)}kg_{home_variant}"


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
