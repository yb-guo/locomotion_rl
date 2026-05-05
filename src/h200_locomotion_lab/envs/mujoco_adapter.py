"""MuJoCo adapter placeholder."""


class SonicMujocoSim2Sim:
    """Boundary for official SONIC MuJoCo sim2sim validation."""

    simulator = "mujoco"
    robot = "unitree_g1"

    def describe(self) -> str:
        return "Use upstream GR00T-WholeBodyControl MuJoCo sim2sim first."

