"""GEAR-SONIC adapter decomposition."""

from h200_locomotion_lab.agents.base import AgentSpec, SubmoduleSpec


def build_sonic_adapter_submodules() -> AgentSpec:
    return AgentSpec(
        name="sonic_adapter",
        goal="Thin adapter around official GEAR-SONIC sim2sim and training artifacts.",
        submodules=(
            SubmoduleSpec(
                name="observation_bridge",
                responsibility="Map simulator state to SONIC observation tensors.",
                inputs=("sim_state", "robot_state", "command_state"),
                outputs=("sonic_observation",),
            ),
            SubmoduleSpec(
                name="reference_motion_bridge",
                responsibility="Load and align reference motion for the policy.",
                inputs=("motion_library", "motion_index", "phase"),
                outputs=("reference_motion",),
            ),
            SubmoduleSpec(
                name="policy_runtime",
                responsibility="Call PyTorch, ONNX, or TensorRT policy runtime.",
                inputs=("sonic_observation", "reference_motion"),
                outputs=("raw_policy_action",),
            ),
            SubmoduleSpec(
                name="action_bridge",
                responsibility="Map policy output to simulator or robot actuator command.",
                inputs=("raw_policy_action", "robot_limits"),
                outputs=("actuator_command",),
            ),
        ),
    )

