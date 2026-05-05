"""PPO loop placeholder.

The first real implementation should target a tiny Genesis or MuJoCo task, then
grow toward the LocoFormer-style policy once observation and action shapes are
stable.
"""


def describe_training_plan() -> tuple[str, ...]:
    return (
        "Validate simulator reset/step API.",
        "Freeze observation and action schema.",
        "Run a small PPO baseline.",
        "Swap in transformer policy after baseline reward improves.",
    )

