import random
import sys

import pytest

from h200_locomotion_lab.tools.genesis_g1_throughput_probe import (
    CapabilityFailure,
    ProbeConfig,
    action_for_step,
    calculate_metrics,
    default_capability_flags,
    format_metrics,
    format_value,
    run_probe,
    step_physics_only,
    validate_probe_config,
)


def test_probe_module_does_not_import_genesis() -> None:
    assert "genesis" not in sys.modules


def test_validate_probe_config_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="n_envs must be positive"):
        validate_probe_config(ProbeConfig(asset_path="asset.xml", n_envs=0))

    with pytest.raises(ValueError, match="measure_policy_steps must be positive"):
        validate_probe_config(ProbeConfig(asset_path="asset.xml", measure_policy_steps=0))

    with pytest.raises(ValueError, match="sim_dt_s must be finite and positive"):
        validate_probe_config(ProbeConfig(asset_path="asset.xml", sim_dt_s=0.0))


def test_action_patterns_have_expected_shape_and_bounds() -> None:
    rng = random.Random(123)

    assert action_for_step("zero", 0, 4, amplitude=0.5, rng=rng) == (0.0, 0.0, 0.0, 0.0)
    random_action = action_for_step("random", 0, 8, amplitude=0.2, rng=rng)
    sine_action = action_for_step("sine", 3, 8, amplitude=0.2, rng=rng)

    assert len(random_action) == 8
    assert len(sine_action) == 8
    assert max(abs(value) for value in random_action) <= 0.2
    assert max(abs(value) for value in sine_action) <= 0.2


def test_calculate_metrics_reports_policy_sim_and_env_rates() -> None:
    metrics = calculate_metrics(
        build_time_s=1.5,
        warmup_time_s=2.0,
        measure_time_s=4.0,
        warmup_policy_steps=50,
        measure_policy_steps=100,
        decimation=4,
        n_envs=8,
        backend="cuda",
        action_pattern="zero",
        capability_flags=default_capability_flags(8, batched_build_supported=True),
    )
    formatted = format_metrics(metrics)

    assert formatted["build_time_s"] == 1.5
    assert formatted["warmup_time_s"] == 2.0
    assert formatted["measure_time_s"] == 4.0
    assert formatted["policy_steps_per_sec"] == 25.0
    assert formatted["sim_steps_per_sec"] == 800.0
    assert formatted["env_steps_per_sec"] == 200.0
    assert formatted["gpu_backend"] is True
    assert formatted["batched_build_supported"] is True
    assert formatted["batched_action_write_supported"] is False
    assert formatted["cpu_readback_per_step"] is True


def test_run_probe_reports_explicit_capability_failure_for_batched_scene_without_genesis() -> None:
    with pytest.raises(CapabilityFailure, match="supports n_envs=1 only"):
        run_probe(
            ProbeConfig(
                asset_path="does-not-need-to-exist-for-n-envs-capability-check.xml",
                n_envs=16,
                warmup_policy_steps=1,
                measure_policy_steps=1,
            )
        )


def test_format_value_uses_parseable_lowercase_booleans() -> None:
    assert format_value(True) == "true"
    assert format_value(False) == "false"
    assert format_value(12.3456789) == "12.3456789"


def test_step_physics_only_avoids_backend_observation_step_path() -> None:
    backend = _FakePhysicsBackend()

    step_physics_only(backend, (0.1, 0.2))

    assert backend.observation_step_called is False
    assert backend.target_actions == [(0.1, 0.2)]
    assert backend.robot.commands == [((0.1, 0.2), (3, 4))]
    assert backend.scene.steps == 3


class _FakeRobot:
    def __init__(self) -> None:
        self.commands = []

    def control_dofs_position(self, target: tuple[float, ...], *, dofs_idx_local: tuple[int, ...]) -> None:
        self.commands.append((target, dofs_idx_local))


class _FakeScene:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _FakeContract:
    decimation = 3


class _FakePhysicsBackend:
    contract = _FakeContract()
    motor_dof_indices = (3, 4)

    def __init__(self) -> None:
        self.robot = _FakeRobot()
        self.scene = _FakeScene()
        self.target_actions = []
        self.observation_step_called = False

    def _motor_targets_from_action(self, action: tuple[float, ...]) -> tuple[float, ...]:
        self.target_actions.append(action)
        return action

    def step(self, action: tuple[float, ...]) -> None:
        self.observation_step_called = True
        raise AssertionError("throughput probe must not call GenesisG1SceneBackend.step")
