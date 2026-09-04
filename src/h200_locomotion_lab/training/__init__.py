"""Training namespace."""

from h200_locomotion_lab.training.sequence_ppo import (
    SequencePPOMinibatch,
    iter_sequence_minibatches,
    make_sequence_padding_mask,
    masked_sequence_ppo_loss,
)
from h200_locomotion_lab.training.whole_body_curriculum import (
    WHOLE_BODY_CURRICULUM,
    CurriculumScheduler,
    CurriculumStage,
    RewardNormalizedWholeBodyMux,
    WholeBodyShardPlan,
    build_procedural_mujoco_mux,
    evaluate_shared_mlp_gate,
    normalize_whole_body_reward,
)
from h200_locomotion_lab.training.whole_body_ppo import (
    SpecialistQualityThresholds,
    WholeBodyPPOConfig,
    WholeBodyPPOTrainer,
    evaluate_specialist_gate,
    evaluate_whole_body_policy,
)

__all__ = [
    "WHOLE_BODY_CURRICULUM",
    "CurriculumScheduler",
    "CurriculumStage",
    "RewardNormalizedWholeBodyMux",
    "SequencePPOMinibatch",
    "SpecialistQualityThresholds",
    "WholeBodyPPOConfig",
    "WholeBodyPPOTrainer",
    "WholeBodyShardPlan",
    "build_procedural_mujoco_mux",
    "evaluate_shared_mlp_gate",
    "evaluate_specialist_gate",
    "evaluate_whole_body_policy",
    "iter_sequence_minibatches",
    "make_sequence_padding_mask",
    "masked_sequence_ppo_loss",
    "normalize_whole_body_reward",
]
