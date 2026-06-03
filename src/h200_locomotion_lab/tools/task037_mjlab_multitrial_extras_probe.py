"""Probe Task037 MJLab multi-trial extras and runner-facing done contract."""

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


REQUIRED_EXTRA_KEYS = (
    "trial_done",
    "episode_done",
    "trial_index",
    "final_trial",
    "reset_reason",
    "task037_trial_done",
    "task037_episode_done",
    "task037_trial_index",
    "task037_final_trial",
    "task037_reset_reason",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small MJLab Task037 multi-trial extras contract probe."
    )
    parser.add_argument(
        "--task",
        default="Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3700201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=0.06,
        help="Short horizon used only to force MJLab raw timeout events for this smoke.",
    )
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
    if not hasattr(env_cfg, "episode_length_s"):
        raise AttributeError("env_cfg has no episode_length_s")
    env_cfg.episode_length_s = args.episode_length_s

    start = time.time()
    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(outer_env, asdict(agent_cfg), device=args.device)
        rollout_env = runner.env
        _obs, reset_extras = rollout_env.reset()

        required_seen = {key: key in reset_extras for key in REQUIRED_EXTRA_KEYS}
        done_matches_episode = True
        saw_inner = False
        saw_outer = False
        trial_done_total = 0
        episode_done_total = 0
        max_trial_index = 0
        reset_reason_counts: dict[str, int] = {}
        last_extra_keys: list[str] = sorted(reset_extras)

        for _step in range(args.steps):
            actions = torch.zeros(
                (rollout_env.num_envs, rollout_env.num_actions),
                device=rollout_env.device,
            )
            _obs, _reward, done, extras = rollout_env.step(actions)
            last_extra_keys = sorted(extras)
            for key in REQUIRED_EXTRA_KEYS:
                required_seen[key] = required_seen[key] or key in extras
            trial_done = _bool_tensor(torch, extras["trial_done"], args.device)
            episode_done = _bool_tensor(torch, extras["episode_done"], args.device)
            trial_index = extras["trial_index"].to(device=args.device, dtype=torch.long)
            reset_reason = extras["reset_reason"].to(device=args.device, dtype=torch.long)
            done_bool = _bool_tensor(torch, done, args.device)

            done_matches_episode = done_matches_episode and bool(
                torch.equal(done_bool, episode_done)
            )
            saw_inner = saw_inner or bool((trial_done & ~episode_done).any().item())
            saw_outer = saw_outer or bool(episode_done.any().item())
            trial_done_total += int(trial_done.sum().item())
            episode_done_total += int(episode_done.sum().item())
            max_trial_index = max(max_trial_index, int(trial_index.max().item()))
            for reason in reset_reason.unique().detach().cpu().tolist():
                count = int((reset_reason == int(reason)).sum().item())
                reset_reason_counts[str(int(reason))] = reset_reason_counts.get(str(int(reason)), 0) + count

        missing = [key for key, seen in required_seen.items() if not seen]
        gpu_name = (
            torch.cuda.get_device_name(torch.device(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        result = {
            "task": args.task,
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "gpu_name": gpu_name,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "episode_length_s": args.episode_length_s,
            "required_extras_missing": missing,
            "last_extra_keys": last_extra_keys,
            "done_matches_episode_done": done_matches_episode,
            "saw_inner_trial_done": saw_inner,
            "saw_outer_episode_done": saw_outer,
            "trial_done_total": trial_done_total,
            "episode_done_total": episode_done_total,
            "max_trial_index": max_trial_index,
            "reset_reason_counts": reset_reason_counts,
            "wall_time_s": time.time() - start,
        }
        result["pass"] = (
            not missing
            and done_matches_episode
            and saw_inner
            and saw_outer
            and trial_done_total > episode_done_total
            and max_trial_index >= 1
        )
        return result
    finally:
        if outer_env is not None:
            outer_env.close()


def _bool_tensor(torch: Any, value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, dtype=torch.bool)
    return torch.as_tensor(value, device=device, dtype=torch.bool)


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
