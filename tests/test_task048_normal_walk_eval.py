from types import SimpleNamespace

from h200_locomotion_lab.tools.task048_normal_walk_eval import (
    configure_clean_fixed_command,
    evaluate_gate,
    parse_args,
)


def test_task048_eval_defaults_match_written_gate() -> None:
    args = parse_args(["--checkpoint", "model.pt", "--output-json", "eval.json"])

    assert args.task == "Unitree-G1-Flat"
    assert args.num_envs == 256
    assert args.steps == 1000
    assert args.lin_vel_x == 0.5
    assert args.min_zero_fall_ratio == 0.95
    assert args.max_lin_vel_error == 0.35
    assert args.max_yaw_vel_error == 0.35
    assert args.max_gravity_xy == 0.35


def test_task048_clean_fixed_command_disables_randomization() -> None:
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-3.14, 3.14),
    )
    twist = SimpleNamespace(
        heading_command=True,
        rel_heading_envs=1.0,
        rel_standing_envs=1.0,
        ranges=ranges,
    )
    generator = SimpleNamespace(curriculum=True)
    reset_base = SimpleNamespace(params={"pose_range": {"yaw": (-3.14, 3.14)}})
    env_cfg = SimpleNamespace(
        commands={"twist": twist},
        events={"reset_base": reset_base, "push_robot": object(), "foot_friction": object()},
        observations={"actor": SimpleNamespace(enable_corruption=True)},
        curriculum={"command_vel": object()},
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=generator)),
    )

    configure_clean_fixed_command(env_cfg, 0.5)

    assert twist.ranges.lin_vel_x == (0.5, 0.5)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert twist.ranges.heading is None
    assert env_cfg.events == {"reset_base": env_cfg.events["reset_base"]}
    assert reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.observations["actor"].enable_corruption is False
    assert env_cfg.curriculum == {}
    assert generator.curriculum is False


def test_task048_gate_requires_stability_and_tracking() -> None:
    args = parse_args(["--checkpoint", "model.pt", "--output-json", "eval.json"])
    passing = {
        "zero_fall_ratio": 1.0,
        "lin_vel_error_mean": 0.1,
        "yaw_vel_error_mean": 0.05,
        "gravity_xy_mean": 0.03,
    }

    passed, reasons = evaluate_gate(passing, args)
    assert passed is True
    assert reasons == []

    failing = dict(passing, zero_fall_ratio=0.9, lin_vel_error_mean=0.4)
    passed, reasons = evaluate_gate(failing, args)
    assert passed is False
    assert reasons == [
        "zero_fall_ratio_below_threshold",
        "lin_vel_error_above_threshold",
    ]
