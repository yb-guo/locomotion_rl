from __future__ import annotations

import pytest

from h200_locomotion_lab.policies.whole_body_mlp import (
    WholeBodyMLPActorCritic,
    WholeBodyMLPConfig,
)
from h200_locomotion_lab.robots.procedural_morphology import MorphologyGenerator


def test_whole_body_mlp_parameters_and_forward_share_device() -> None:
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = WholeBodyMLPActorCritic(
        WholeBodyMLPConfig(hidden_dim=32),
        action_mask=torch.ones(45, dtype=torch.bool, device=device),
        device=device,
    )
    devices = {parameter.device.type for parameter in policy.parameters()}
    assert devices == {device}
    observation = torch.zeros((2, 193), device=device)
    mean, value = policy.forward(observation)
    assert mean.device.type == device
    assert value.device.type == device


def test_specialist_physics_range_is_centered_and_fault_free() -> None:
    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 0)
    physical = generator.sample_physical_params(blueprint, 123, range_fraction=0.5)

    assert 0.85 <= physical.global_scale <= 1.15
    assert all(0.875 <= value <= 1.125 for value in physical.link_scales.values())
    assert all(0.75 <= value <= 1.25 for value in physical.mass_scales.values())
    assert all(abs(value) <= 0.075 for value in physical.nominal_offsets.values())
    assert physical.delay_ms == 0.0
