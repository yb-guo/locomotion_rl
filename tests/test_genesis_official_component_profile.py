import pytest

from h200_locomotion_lab.tools.genesis_official_component_profile import (
    action_write_rates,
    scene_step_rates,
    state_read_rates,
)


def test_action_write_rates_report_iterations_per_second() -> None:
    rates = action_write_rates(100, 0.5)

    assert rates["action_write_time_s"] == 0.5
    assert rates["action_writes_per_sec"] == pytest.approx(200.0)


def test_state_read_rates_report_iterations_per_second() -> None:
    rates = state_read_rates(80, 0.25)

    assert rates["state_read_time_s"] == 0.25
    assert rates["state_reads_per_sec"] == pytest.approx(320.0)


def test_scene_step_rates_report_scene_and_env_steps() -> None:
    rates = scene_step_rates(scene_steps=400, n_envs=1024, elapsed_s=2.0)

    assert rates["scene_step_time_s"] == 2.0
    assert rates["scene_steps_per_sec"] == pytest.approx(200.0)
    assert rates["env_scene_steps_per_sec"] == pytest.approx(204800.0)


def test_rates_handle_zero_elapsed_time() -> None:
    assert action_write_rates(100, 0.0)["action_writes_per_sec"] == 0.0
    assert state_read_rates(100, 0.0)["state_reads_per_sec"] == 0.0
    assert scene_step_rates(400, 16, 0.0)["env_scene_steps_per_sec"] == 0.0
