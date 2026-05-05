"""Agent module inventory."""

from h200_locomotion_lab.agents.base import AgentSpec, SubmoduleSpec
from h200_locomotion_lab.agents.locoformer import build_locoformer_submodules
from h200_locomotion_lab.agents.sonic_adapter import build_sonic_adapter_submodules

__all__ = [
    "AgentSpec",
    "SubmoduleSpec",
    "build_locoformer_submodules",
    "build_sonic_adapter_submodules",
]

