"""Inspect one task/policy/algorithm composition without importing a simulator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from h200_locomotion_lab.experiments.config import (
    CONFIG_ROOT,
    DEFAULT_G1_FLAT_PPO_EXPERIMENT,
    load_experiment,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_G1_FLAT_PPO_EXPERIMENT,
        help="Experiment YAML path, absolute or relative to --config-root.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=CONFIG_ROOT,
        help="Root used to resolve task, policy, and algorithm component paths.",
    )
    return parser.parse_args(argv)


def inspect_experiment(path: Path, *, config_root: Path) -> dict[str, object]:
    experiment = load_experiment(path, config_root=config_root)
    return {
        "name": experiment.name,
        "task": {
            "name": experiment.task.name,
            "observations": {
                role: {"shape": space.shape, "dtype": space.dtype}
                for role, space in experiment.task.observations.items()
            },
            "action": {
                "shape": experiment.task.action.shape,
                "dtype": experiment.task.action.dtype,
            },
            "max_episode_steps": experiment.task.max_episode_steps,
        },
        "policy": {
            "name": experiment.policy.name,
            "family": experiment.policy.family,
            "capabilities": sorted(experiment.policy.capabilities),
            "action_horizon": experiment.policy.action_horizon,
        },
        "algorithm": {
            "name": experiment.algorithm.name,
            "family": experiment.algorithm.family,
            "interaction": experiment.algorithm.interaction,
            "requires": sorted(experiment.algorithm.required_policy_capabilities),
        },
        "runtime": {
            "backend": experiment.runtime.backend,
            "device": experiment.runtime.device,
            "num_envs": experiment.runtime.num_envs,
            "headless": experiment.runtime.headless,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = inspect_experiment(args.experiment, config_root=args.config_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
