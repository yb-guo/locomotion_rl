"""Probe Task037 deterministic MJLab inner-reset condition preservation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small MJLab probe for deterministic Task037 inner reset."
    )
    parser.add_argument(
        "--task",
        default="Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3700301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-length-s", type=float, default=0.06)
    parser.add_argument("--max-command-delta", type=float, default=1e-6)
    parser.add_argument("--max-failure-delta", type=float, default=1e-6)
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.episode_length_s

    start = time.time()
    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(outer_env, asdict(agent_cfg), device=args.device)
        rollout_env = runner.env
        _obs, _reset_extras = rollout_env.reset()
        base_env = rollout_env.unwrapped
        controller = getattr(base_env, "_task037_inner_reset_controller", None)
        if controller is None:
            raise RuntimeError("Task037 inner reset controller is not installed")

        saw_inner = False
        saw_outer = False
        done_matches_episode = True
        inner_command_max_delta = 0.0
        inner_failure_max_delta = 0.0
        inner_episode_length_max = 0
        outer_command_changed_any = False
        phase_after_inner_samples = []
        root_z_after_inner_samples = []

        for _step in range(args.steps):
            pre = _snapshot(torch, base_env)
            actions = torch.zeros(
                (rollout_env.num_envs, rollout_env.num_actions),
                device=rollout_env.device,
            )
            obs, _reward, done, extras = rollout_env.step(actions)
            post = _snapshot(torch, base_env)
            done_bool = done.to(device=args.device, dtype=torch.bool)
            episode_done = extras["episode_done"].to(device=args.device, dtype=torch.bool)
            done_matches_episode = done_matches_episode and bool(torch.equal(done_bool, episode_done))
            inner = extras["inner_reset"].to(device=args.device, dtype=torch.bool)
            outer = extras["outer_reset"].to(device=args.device, dtype=torch.bool)
            saw_inner = saw_inner or bool(inner.any().item())
            saw_outer = saw_outer or bool(outer.any().item())
            if bool(inner.any().item()):
                inner_ids = inner.nonzero(as_tuple=False).flatten()
                inner_command_max_delta = max(
                    inner_command_max_delta,
                    _max_abs(pre["command"][inner_ids] - post["command"][inner_ids]),
                )
                inner_failure_max_delta = max(
                    inner_failure_max_delta,
                    _failure_delta(pre["failure"], post["failure"], inner_ids),
                )
                inner_episode_length_max = max(
                    inner_episode_length_max,
                    int(post["episode_length"][inner_ids].max().item()),
                )
                phase = obs["actor"][inner_ids, 9:11].detach().float().cpu().tolist()
                phase_after_inner_samples.extend(phase[:4])
                root_z_after_inner_samples.extend(
                    post["root_z"][inner_ids].detach().float().cpu().tolist()[:4]
                )
            if bool(outer.any().item()):
                outer_ids = outer.nonzero(as_tuple=False).flatten()
                outer_delta = _max_abs(pre["command"][outer_ids] - post["command"][outer_ids])
                outer_command_changed_any = outer_command_changed_any or outer_delta > args.max_command_delta

        deterministic_reset_cfg = _deterministic_reset_cfg(base_env.cfg)
        result = {
            "task": args.task,
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "episode_length_s": args.episode_length_s,
            "done_matches_episode_done": done_matches_episode,
            "saw_inner_reset": saw_inner,
            "saw_outer_reset": saw_outer,
            "inner_command_max_delta": inner_command_max_delta,
            "inner_failure_max_delta": inner_failure_max_delta,
            "inner_episode_length_max": inner_episode_length_max,
            "outer_command_changed_any": outer_command_changed_any,
            "deterministic_reset_cfg": deterministic_reset_cfg,
            "controller_inner_reset_total": int(controller.inner_reset_count.sum().item()),
            "controller_outer_reset_total": int(controller.outer_reset_count.sum().item()),
            "phase_after_inner_samples": phase_after_inner_samples[:8],
            "root_z_after_inner_samples": root_z_after_inner_samples[:8],
            "wall_time_s": time.time() - start,
        }
        result["pass"] = (
            done_matches_episode
            and saw_inner
            and saw_outer
            and deterministic_reset_cfg
            and inner_command_max_delta <= args.max_command_delta
            and inner_failure_max_delta <= args.max_failure_delta
            and inner_episode_length_max == 0
            and result["controller_inner_reset_total"] > 0
            and result["controller_outer_reset_total"] > 0
        )
        return result
    finally:
        if outer_env is not None:
            outer_env.close()


def _snapshot(torch: Any, env: Any) -> dict[str, Any]:
    robot = env.scene["robot"]
    return {
        "command": env.command_manager.get_command("twist").clone(),
        "episode_length": env.episode_length_buf.clone(),
        "root_z": robot.data.root_link_pos_w[:, 2].clone(),
        "failure": _failure_tensors(env),
    }


def _failure_tensors(env: Any) -> dict[str, Any]:
    out = {}
    for name in (
        "_task029_motor_failure_selected_target_indices",
        "_task029_motor_failure_selected_ctrl_ids",
        "_task029_motor_failure_selected_scales",
        "_task029_motor_failure_selected_types",
        "_task031_dynamic_switch_transition_s",
    ):
        value = getattr(env, name, None)
        if hasattr(value, "shape") and value.shape[:1] == (env.num_envs,):
            out[name] = value.clone()
    return out


def _failure_delta(before: dict[str, Any], after: dict[str, Any], env_ids: Any) -> float:
    delta = 0.0
    for name, pre in before.items():
        post = after.get(name)
        if post is None:
            continue
        delta = max(delta, _max_abs(pre[env_ids].float() - post[env_ids].float()))
    return delta


def _max_abs(x: Any) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.detach().float().abs().max().item())


def _deterministic_reset_cfg(cfg: Any) -> bool:
    params = cfg.events["reset_base"].params
    pose_range = params["pose_range"]
    return all(
        tuple(pose_range[name]) == (0.0, 0.0)
        for name in ("x", "y", "z", "yaw")
    )


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_probe(args)
    except Exception as exc:
        result = {
            "task": args.task,
            "command": list(sys.argv),
            "seed": args.seed,
            "pass": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    result["json_path"] = str(output_json)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
