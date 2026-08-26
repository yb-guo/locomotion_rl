"""Task054 curriculum and shard planning for shared whole-body PPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
)
from h200_locomotion_lab.envs.whole_body_mux import WholeBodyRolloutMux
from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyGenerator,
    MorphologySplitManifest,
    build_morphology_split_manifest,
)


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    topology_count: int
    physics_half_range: float
    static_motor_randomization: bool
    contact_randomization: bool
    sensor_noise: bool


WHOLE_BODY_CURRICULUM = (
    CurriculumStage("narrow_topology", 8, 0.5, False, False, False),
    CurriculumStage("all_train_topologies", 64, 0.5, False, False, False),
    CurriculumStage("continuous_physics", 64, 1.0, False, False, False),
    CurriculumStage("static_motor_randomization", 64, 1.0, True, False, False),
    CurriculumStage("contact_and_sensor_noise", 64, 1.0, True, True, True),
)


@dataclass(frozen=True, slots=True)
class WholeBodyShardPlan:
    topology_shards: int = 8
    envs_per_shard: int = 256
    family_fraction: float = 0.5
    expanded_topology_shards: int = 16
    expanded_envs_per_shard: int = 256

    def __post_init__(self) -> None:
        if self.topology_shards <= 0 or self.envs_per_shard <= 0:
            raise ValueError("shard counts must be positive")
        if not 0.0 < self.family_fraction < 1.0:
            raise ValueError("family_fraction must be between zero and one")
        if self.topology_shards % 2 != 0:
            raise ValueError("the baseline plan allocates half of shards to each family")

    @property
    def num_envs(self) -> int:
        return self.topology_shards * self.envs_per_shard

    @property
    def expanded_num_envs(self) -> int:
        return self.expanded_topology_shards * self.expanded_envs_per_shard


class CurriculumScheduler:
    """Advance stages only after the previous stage's diagnostics are healthy."""

    def __init__(self, stages: tuple[CurriculumStage, ...] = WHOLE_BODY_CURRICULUM) -> None:
        if not stages:
            raise ValueError("curriculum requires at least one stage")
        self.stages = stages
        self.index = 0

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.index]

    def observe(self, metrics: dict[str, float]) -> bool:
        """Return true when the scheduler moved to the next stage."""

        if self.index >= len(self.stages) - 1:
            return False
        survival = float(metrics.get("zero_fall_ratio", 0.0))
        tracking = float(metrics.get("normalized_velocity_error", float("inf")))
        if survival < 0.90 or tracking > 0.30:
            return False
        self.index += 1
        return True


def evaluate_shared_mlp_gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Apply Task054 family/heldout thresholds without selecting on OOD data."""

    reasons: list[str] = []
    if metrics.get("biped_zero_fall_ratio", 0.0) < 0.90:
        reasons.append("biped_zero_fall_ratio_below_threshold")
    if metrics.get("quadruped_zero_fall_ratio", 0.0) < 0.90:
        reasons.append("quadruped_zero_fall_ratio_below_threshold")
    if metrics.get("biped_normalized_velocity_error", float("inf")) > 0.30:
        reasons.append("biped_tracking_error_above_threshold")
    if metrics.get("quadruped_normalized_velocity_error", float("inf")) > 0.30:
        reasons.append("quadruped_tracking_error_above_threshold")
    if metrics.get("heldout_worst10_survival", 0.0) < 0.75:
        reasons.append("heldout_worst10_survival_below_threshold")
    return not reasons, reasons


def normalize_whole_body_reward(
    reward: Any,
    *,
    robot_mass: float,
    robot_scale: float,
    active_actuators: int,
) -> Any:
    """Normalize reward magnitude across mass, size, and actuator count.

    The denominator is dimensionless relative to a 10 kg, unit-scale, 12-DoF
    reference body; the square-root mass term avoids over-penalizing larger
    bodies while keeping the per-actuator torque scale comparable.
    """

    if robot_mass <= 0.0 or robot_scale <= 0.0 or active_actuators <= 0:
        raise ValueError("mass, scale, and active actuator count must be positive")
    denominator = (
        (robot_mass / 10.0) ** 0.5
        * robot_scale
        * (active_actuators / 12.0) ** 0.5
    )
    return reward / denominator


class RewardNormalizedWholeBodyMux:
    """Apply per-shard physical reward normalization after mux aggregation."""

    def __init__(self, mux: WholeBodyRolloutMux, shard_factors: tuple[float, ...]) -> None:
        if len(shard_factors) != len(mux.shards):
            raise ValueError("one positive reward factor is required per shard")
        if any(factor <= 0.0 for factor in shard_factors):
            raise ValueError("reward factors must be positive")
        self.mux = mux
        self.shard_factors = shard_factors

    @property
    def num_envs(self) -> int:
        return self.mux.num_envs

    @property
    def active_action_mask(self) -> Any:
        return self.mux.active_action_mask

    def reset(self) -> Any:
        return self.mux.reset()

    def step(self, action: Any) -> WholeBodyStep:
        step = self.mux.step(action)
        reward = step.reward.copy()
        offset = 0
        for shard, factor in zip(self.mux.shards, self.shard_factors):
            reward[offset : offset + shard.num_envs] /= factor
            offset += shard.num_envs
        return WholeBodyStep(
            actor_observation=step.actor_observation,
            critic_observation=step.critic_observation,
            reward=reward,
            trial_done=step.trial_done,
            context_done=step.context_done,
            active_action_mask=step.active_action_mask,
            metrics=step.metrics,
            final_observation=step.final_observation,
        )


def build_procedural_mujoco_mux(
    *,
    manifest: MorphologySplitManifest | None = None,
    plan: WholeBodyShardPlan | None = None,
    num_envs_per_shard: int | None = None,
    config: WholeBodyMuJoCoShardConfig | None = None,
) -> WholeBodyRolloutMux:
    """Build the H200 baseline shard layout from train topologies only."""

    manifest = manifest or build_morphology_split_manifest()
    plan = plan or WholeBodyShardPlan()
    env_count = num_envs_per_shard or plan.envs_per_shard
    if env_count <= 0:
        raise ValueError("num_envs_per_shard must be positive")
    bipeds = [item for item in manifest.train if item.family == "biped"]
    quadrupeds = [item for item in manifest.train if item.family == "quadruped"]
    per_family = plan.topology_shards // 2
    if len(bipeds) < per_family or len(quadrupeds) < per_family:
        raise ValueError("train manifest does not contain enough family topologies")
    generator = MorphologyGenerator()
    shards = []
    for index, blueprint in enumerate(bipeds[:per_family] + quadrupeds[:per_family]):
        physical = generator.sample_physical_params(blueprint, blueprint.seed + 10_000_000)
        shard_config = replace(config, seed=index) if config is not None else WholeBodyMuJoCoShardConfig(seed=index)
        shards.append(
            WholeBodyMuJoCoShard(
                blueprint,
                physical=physical,
                num_envs=env_count,
                config=shard_config,
            )
        )
    return WholeBodyRolloutMux(tuple(shards))
