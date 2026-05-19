from __future__ import annotations

import json
import sys
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def _jsonable(value):
  try:
    json.dumps(value)
    return value
  except TypeError:
    return str(value)


def main() -> None:
  import src.tasks  # noqa: F401

  task_id = sys.argv[1]
  output_path = Path(sys.argv[2])
  output_path.parent.mkdir(parents=True, exist_ok=True)

  configure_torch_backends()
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  try:
    actor_group = cfg.observations["actor"]
    critic_group = cfg.observations["critic"]
    report = {
      "task_id": task_id,
      "num_envs": cfg.scene.num_envs,
      "events": {
        name: {
          "mode": event.mode,
          "func": getattr(event.func, "__name__", str(event.func)),
          "params": {key: _jsonable(val) for key, val in event.params.items()},
        }
        for name, event in cfg.events.items()
      },
      "curriculum": list(cfg.curriculum.keys()),
      "actor_enable_corruption": actor_group.enable_corruption,
      "actor_observation_dim": env.observation_manager.group_obs_dim["actor"][0],
      "critic_observation_dim": env.observation_manager.group_obs_dim["critic"][0],
      "action_dim": env.action_manager.total_action_dim,
      "action_terms": {
        name: {
          "type": type(term).__name__,
          "action_dim": getattr(term, "action_dim", None),
        }
        for name, term in env.action_manager._terms.items()
      },
      "actor_terms": list(actor_group.terms.keys()),
      "critic_terms": list(critic_group.terms.keys()),
    }
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
  finally:
    env.close()


if __name__ == "__main__":
  main()
