"""Run SONIC or SONIC-compatible action providers in unitree_rl_mjlab."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from h200_locomotion_lab.envs.mjlab_backend import MjlabG1RobotBackend
from h200_locomotion_lab.runtime.scalar_g1_runtime import (
    SequenceActionProvider,
    ZeroActionProvider,
)
from h200_locomotion_lab.runtime.scalar_g1_runtime import ScalarG1Runtime
from h200_locomotion_lab.sonic.controller import SonicPlannerEncoderActionProvider
from h200_locomotion_lab.sonic.onnx_models import SonicOnnxDecoder, SonicOnnxEncoder
from h200_locomotion_lab.sonic.planner_runner import (
    SonicPlannerCommand,
    SubprocessSonicPlanner,
    read_numeric_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="Unitree-G1-Flat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-prefix", default="mjlab_sonic_rollout")
    parser.add_argument(
        "--provider",
        choices=("zero", "sequence", "online"),
        default="zero",
    )
    parser.add_argument("--actions-csv")
    parser.add_argument("--planner")
    parser.add_argument("--planner-runner")
    parser.add_argument("--encoder")
    parser.add_argument("--decoder")
    parser.add_argument("--planner-work-dir")
    parser.add_argument("--replan-interval", type=int, default=10)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--target-vel", type=float, default=-1.0)
    parser.add_argument("--movement-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--facing-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--disable-terminations", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    raw_env = build_mjlab_env(args)
    backend = MjlabG1RobotBackend(raw_env)
    provider = build_provider(args)
    runtime = ScalarG1Runtime(backend, provider)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_env = attach_video_recorder(raw_env, output_dir, args)
    if video_env is not None:
        backend.raw_env = video_env

    root_positions = []
    done_steps = []
    try:
        runtime.reset()
        for step in range(args.steps):
            result = runtime.step()
            root_positions.append(result.next_state.root_qpos[:3])
            if backend.last_step_result and backend.last_step_result.done:
                done_steps.append(step)
    finally:
        close_env(backend.raw_env)

    videos = sorted(output_dir.glob(f"{args.video_prefix}-*.mp4"))
    summary = {
        "provider": args.provider,
        "task_id": args.task_id,
        "steps": args.steps,
        "done_steps": done_steps[:20],
        "root_start_xyz": root_positions[0] if root_positions else None,
        "root_end_xyz": root_positions[-1] if root_positions else None,
        "root_delta_xyz": _delta(root_positions[0], root_positions[-1])
        if root_positions
        else None,
        "videos": [str(path.resolve()) for path in videos],
        "video_bytes": [path.stat().st_size for path in videos],
        "planner_calls": getattr(provider, "planner_calls", None),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_mjlab_env(args: argparse.Namespace):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    cfg = load_env_cfg(args.task_id, play=True)
    cfg.scene.num_envs = 1
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    if args.disable_terminations:
        cfg.terminations = {}
    return ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")


def attach_video_recorder(raw_env, output_dir: Path, args: argparse.Namespace):
    try:
        from mjlab.utils.wrappers import VideoRecorder
    except ModuleNotFoundError:
        return None
    return VideoRecorder(
        raw_env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=args.video_prefix,
        disable_logger=False,
    )


def build_provider(args: argparse.Namespace):
    if args.provider == "zero":
        return ZeroActionProvider()
    if args.provider == "sequence":
        if not args.actions_csv:
            raise ValueError("--actions-csv is required for --provider sequence")
        return SequenceActionProvider(read_numeric_rows(Path(args.actions_csv)), repeat_last=True)
    required = {
        "--planner": args.planner,
        "--planner-runner": args.planner_runner,
        "--encoder": args.encoder,
        "--decoder": args.decoder,
        "--planner-work-dir": args.planner_work_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required online provider args: {', '.join(missing)}")
    planner = SubprocessSonicPlanner(
        planner=Path(args.planner),
        planner_runner=Path(args.planner_runner),
        work_dir=Path(args.planner_work_dir),
        command=SonicPlannerCommand(
            mode=args.mode,
            target_vel=args.target_vel,
            movement_direction=tuple(args.movement_direction),
            facing_direction=tuple(args.facing_direction),
        ),
    )
    return SonicPlannerEncoderActionProvider(
        planner=planner,
        encoder=SonicOnnxEncoder(Path(args.encoder)),
        decoder=SonicOnnxDecoder(Path(args.decoder)),
        replan_interval=args.replan_interval,
    )


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _delta(start, end):
    return [float(end[index] - start[index]) for index in range(3)]


if __name__ == "__main__":
    main()
