"""Print the current agent module inventory."""

from h200_locomotion_lab.agents import (
    build_locoformer_submodules,
    build_sonic_adapter_submodules,
)


def format_agent_inventory() -> str:
    lines: list[str] = []
    for agent in (build_sonic_adapter_submodules(), build_locoformer_submodules()):
        lines.append(f"{agent.name}: {agent.goal}")
        for module in agent.submodules:
            inputs = ", ".join(module.inputs)
            outputs = ", ".join(module.outputs)
            lines.append(f"  - {module.name}: {module.responsibility}")
            lines.append(f"    inputs: {inputs}")
            lines.append(f"    outputs: {outputs}")
    return "\n".join(lines)


def main() -> None:
    print(format_agent_inventory())


if __name__ == "__main__":
    main()

