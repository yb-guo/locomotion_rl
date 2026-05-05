from h200_locomotion_lab.agents import (
    build_locoformer_submodules,
    build_sonic_adapter_submodules,
)


def test_sonic_adapter_submodules_are_named() -> None:
    agent = build_sonic_adapter_submodules()
    assert agent.module_names() == (
        "observation_bridge",
        "reference_motion_bridge",
        "policy_runtime",
        "action_bridge",
    )


def test_locoformer_min_has_policy_core() -> None:
    agent = build_locoformer_submodules()
    assert "transformer_policy" in agent.module_names()
    assert "actor_critic_heads" in agent.module_names()

