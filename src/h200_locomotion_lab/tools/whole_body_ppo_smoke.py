"""Train a small procedural whole-body MLP specialist and write a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h200_locomotion_lab.core.checkpoint import WholeBodyCheckpointMetadata, make_checkpoint_payload
from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    MorphologyGenerator,
    morphology_instance_key,
)
from h200_locomotion_lab.training.whole_body_ppo import (
    WholeBodyPPOConfig,
    WholeBodyPPOTrainer,
    evaluate_specialist_gate,
    evaluate_whole_body_policy,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("biped", "quadruped"), default="biped")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--trial-seconds", type=float, default=10.0)
    parser.add_argument("--eval-trials", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-scale", type=float, default=0.65)
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    parser.add_argument(
        "--physics-range-fraction",
        type=float,
        default=1.0,
        help="centered fraction of Task051 physics ranges; 0.5 is specialist curriculum stage 1",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs <= 0 or args.updates <= 0 or args.rollout_steps <= 0 or args.eval_trials < 0:
        raise ValueError("num-envs, updates, and rollout-steps must be positive; eval-trials non-negative")
    generator = MorphologyGenerator()
    blueprint = generator.generate(args.family, args.seed)
    physical = generator.sample_physical_params(
        blueprint,
        args.seed + 10_000_000,
        range_fraction=args.physics_range_fraction,
    )
    instance_key = morphology_instance_key(blueprint, physical)
    env = WholeBodyMuJoCoShard(
        blueprint,
        physical=physical,
        num_envs=args.num_envs,
        config=WholeBodyMuJoCoShardConfig(
            trial_seconds=args.trial_seconds,
            action_scale=args.action_scale,
            seed=args.seed,
        ),
    )
    trainer = WholeBodyPPOTrainer(
        env,
        action_mask=env.active_action_mask,
        config=WholeBodyPPOConfig(
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            log_std_init=args.log_std_init,
            device=args.device,
        ),
    )
    reports = trainer.train()
    metadata = WholeBodyCheckpointMetadata(
        embodiment_contract_version=PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        embodiment_contract_hash=PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        policy_family="mlp_specialist",
        manifest_hash=instance_key.cache_key,
    )
    result: dict[str, Any] = {
        "family": args.family,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "physics_range_fraction": args.physics_range_fraction,
        "action_scale": args.action_scale,
        "log_std_init": args.log_std_init,
        "structural_hash": blueprint.structural_hash,
        "morphology_instance_key": instance_key.manifest(),
        "stance_solution": env.stance_solution.manifest(),
        "stance_solution_hash": env.stance_solution.solution_hash,
        "stance_cache_key": env.stance_solution.cache_key,
        "active_actuators": len(blueprint.actuators),
        "reports": reports,
        "checkpoint_metadata": metadata.as_dict(),
    }
    if args.eval_trials:
        evaluation = evaluate_whole_body_policy(
            lambda: WholeBodyMuJoCoShard(
                blueprint,
                physical=physical,
                num_envs=args.num_envs,
                config=WholeBodyMuJoCoShardConfig(trial_seconds=args.trial_seconds, seed=args.seed + 1),
            ),
            trainer.policy,
            trials=args.eval_trials,
            device=args.device,
        )
        evaluation["pass"], evaluation["failure_reasons"] = evaluate_specialist_gate(evaluation)
        result["evaluation"] = evaluation
    if args.checkpoint is not None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional training dependency
            raise RuntimeError("Torch is required to write a checkpoint") from exc
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(make_checkpoint_payload(trainer.policy.state_dict(), metadata), args.checkpoint)
        result["checkpoint"] = str(args.checkpoint)
    return result


def main() -> None:
    args = parse_args()
    result = run_train(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
