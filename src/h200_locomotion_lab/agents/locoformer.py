"""LocoFormer-style agent decomposition.

This is not a paper reproduction yet. It is the shape of the implementation we
can fill after a minimal Genesis or MuJoCo loop is running.
"""

from h200_locomotion_lab.agents.base import AgentSpec, SubmoduleSpec


def build_locoformer_submodules() -> AgentSpec:
    return AgentSpec(
        name="locoformer_min",
        goal="Small-scale long-context locomotion policy for RTX 5060 Ti experiments.",
        submodules=(
            SubmoduleSpec(
                name="morphology_encoder",
                responsibility="Encode robot body graph, DOF layout, and actuator metadata.",
                inputs=("robot_description", "joint_layout"),
                outputs=("morphology_tokens",),
            ),
            SubmoduleSpec(
                name="proprio_tokenizer",
                responsibility="Tokenize proprioception and command history.",
                inputs=("joint_positions", "joint_velocities", "imu", "contacts", "commands"),
                outputs=("proprio_tokens",),
            ),
            SubmoduleSpec(
                name="motion_context_encoder",
                responsibility="Encode reference motion and recent rollout context.",
                inputs=("reference_motion", "recent_observations"),
                outputs=("context_tokens",),
            ),
            SubmoduleSpec(
                name="transformer_policy",
                responsibility="Run the long-context policy core.",
                inputs=("morphology_tokens", "proprio_tokens", "context_tokens"),
                outputs=("policy_embedding",),
            ),
            SubmoduleSpec(
                name="actor_critic_heads",
                responsibility="Produce action distribution and value estimate.",
                inputs=("policy_embedding",),
                outputs=("action_distribution", "value"),
            ),
            SubmoduleSpec(
                name="adaptation_buffer",
                responsibility="Keep recent history for online adaptation.",
                inputs=("observations", "actions", "rewards"),
                outputs=("recent_observations",),
            ),
        ),
    )
