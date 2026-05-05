"""Small dataclasses for documenting agent boundaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SubmoduleSpec:
    """A named part of an agent and the interface it owns."""

    name: str
    responsibility: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class AgentSpec:
    """A lightweight inventory for an agent architecture."""

    name: str
    goal: str
    submodules: tuple[SubmoduleSpec, ...]

    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.submodules)

