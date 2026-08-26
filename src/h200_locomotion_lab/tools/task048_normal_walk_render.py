"""Render a Task048 normal-walking checkpoint with a fixed forward command."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from h200_locomotion_lab.tools.task048_normal_walk_eval import (
    DEFAULT_LIN_VEL_X,
    DEFAULT_TASK,
    configure_clean_fixed_command,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="task048-normal-walk")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=4800302)
    parser.add_argument("--lin-vel-x", type=float, default=DEFAULT_LIN_VEL_X)
    return parser.parse_args(argv)


def run_render(args: argparse.Namespace) -> dict[str, object]:
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import mediapy as media
    import mjlab.tasks as _mjlab_tasks
    import numpy as np
    import src.tasks as _project_tasks

    del _mjlab_tasks, _project_tasks  # Imports register task packages by side effect.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    configure_clean_fixed_command(env_cfg, args.lin_vel_x)

    start = time.time()
    wrapped_env = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped_env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()

        rollout_env = runner.env
        obs, _ = rollout_env.reset()
        robot = rollout_env.unwrapped.scene["robot"]
        initial_x = float(robot.data.root_link_pos_w[0, 0].item())
        frames: list[np.ndarray] = []
        done_count = 0
        forward_velocities: list[float] = []

        for _ in range(args.steps):
            action = policy(obs)
            step_result = rollout_env.step(action)
            if len(step_result) == 4:
                obs, _reward, dones, _extras = step_result
            else:
                obs, _reward, terminated, truncated, _extras = step_result
                dones = terminated | truncated
            done_count += int(dones[0].item())
            forward_velocities.append(float(robot.data.root_link_lin_vel_b[0, 0].item()))
            frame = env.render()
            if frame is not None:
                frame0 = frame[0] if frame.ndim == 4 else frame
                if frame0.dtype != np.uint8:
                    frame0 = (np.clip(frame0, 0, 1) * 255).astype(np.uint8)
                frames.append(frame0)

        if not frames:
            raise RuntimeError("renderer produced zero frames")

        fps = round(1.0 / float(rollout_env.unwrapped.step_dt))
        video_path = output_dir / f"{args.prefix}.mp4"
        midframe_path = output_dir / f"{args.prefix}-midframe.png"
        summary_path = output_dir / f"{args.prefix}.json"
        media.write_video(str(video_path), frames, fps=fps)
        media.write_image(str(midframe_path), frames[len(frames) // 2])

        final_x = float(robot.data.root_link_pos_w[0, 0].item())
        summary: dict[str, object] = {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "device": args.device,
            "fixed_command": [args.lin_vel_x, 0.0, 0.0],
            "steps": args.steps,
            "frames": len(frames),
            "fps": fps,
            "duration_s": len(frames) / fps,
            "resolution": [int(frames[0].shape[1]), int(frames[0].shape[0])],
            "done_count": done_count,
            "world_x_displacement": final_x - initial_x,
            "forward_vel_x_mean": float(np.mean(forward_velocities)),
            "wall_time_s": time.time() - start,
            "video_path": str(video_path),
            "video_bytes": video_path.stat().st_size,
            "midframe_path": str(midframe_path),
            "midframe_bytes": midframe_path.stat().st_size,
        }
        summary["pass"] = bool(
            done_count == 0
            and len(frames) == args.steps
            and video_path.stat().st_size > 0
            and midframe_path.stat().st_size > 0
            and final_x - initial_x > 1.0
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        if wrapped_env is not None:
            wrapped_env.close()


def main() -> None:
    args = parse_args()
    summary = run_render(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
