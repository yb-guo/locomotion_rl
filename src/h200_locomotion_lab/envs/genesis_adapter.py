"""Genesis adapter placeholder."""


class GenesisG1Env:
    """Boundary for future Genesis G1 locomotion experiments."""

    simulator = "genesis"
    robot = "unitree_g1"

    def describe(self) -> str:
        return "Genesis G1 headless RL environment placeholder."

