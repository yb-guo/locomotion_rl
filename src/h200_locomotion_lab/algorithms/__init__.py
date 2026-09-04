"""Task-independent learning rules such as PPO, FPO, or OGPO."""

from h200_locomotion_lab.algorithms.flow_matching import (
    AdvantageWeightedFlowMatching,
    FlowMatchingPPOAdapter,
    FlowMatchingRLConfig,
)
from h200_locomotion_lab.algorithms.masked_ppo import (
    MaskedPPOLoss,
    mask_action,
    masked_entropy,
    masked_log_prob,
    masked_ppo_surrogate,
    masked_tanh_gaussian_log_prob,
    sample_masked_tanh_gaussian,
)
from h200_locomotion_lab.algorithms.ppo import (
    PPODiagnostics,
    PPORollout,
    PPOSettings,
    compute_gae,
    ppo_update,
)
from h200_locomotion_lab.core.rl import AlgorithmSpec, LearningAlgorithm, UpdateReport

__all__ = [
    "AdvantageWeightedFlowMatching",
    "AlgorithmSpec",
    "FlowMatchingPPOAdapter",
    "FlowMatchingRLConfig",
    "LearningAlgorithm",
    "MaskedPPOLoss",
    "PPODiagnostics",
    "PPORollout",
    "PPOSettings",
    "UpdateReport",
    "compute_gae",
    "mask_action",
    "masked_entropy",
    "masked_log_prob",
    "masked_ppo_surrogate",
    "masked_tanh_gaussian_log_prob",
    "ppo_update",
    "sample_masked_tanh_gaussian",
]
