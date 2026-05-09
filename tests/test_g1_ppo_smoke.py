import pytest

from h200_locomotion_lab.tools import g1_ppo_smoke


def test_parse_seeds_rejects_duplicates() -> None:
    with pytest.raises(argparse_error(), match="seeds must be unique"):
        g1_ppo_smoke.parse_seeds("0,1,0")


def test_resolve_run_dir_stays_under_project_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    project = g1_ppo_smoke.Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(g1_ppo_smoke, "PROJECT_PREFIX", project)

    run_dir = g1_ppo_smoke.resolve_run_dir(project / "outputs", "run-a")

    assert run_dir == (project / "outputs" / "run-a").resolve()


def test_resolve_run_dir_rejects_non_project_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = g1_ppo_smoke.Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(g1_ppo_smoke, "PROJECT_PREFIX", project)

    with pytest.raises(RuntimeError, match="output path must stay under"):
        g1_ppo_smoke.resolve_run_dir(g1_ppo_smoke.Path.cwd() / "elsewhere", "run-a")


def test_metric_row_rejects_non_finite_value() -> None:
    row = {
        "reward_mean": float("nan"),
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 1.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "collect_env_policy_steps_per_sec": 1.0,
        "update_samples_per_sec": 1.0,
        "tensor_device_ok": True,
    }

    with pytest.raises(ValueError, match="reward_mean"):
        g1_ppo_smoke.assert_metric_row_ok(row)


def argparse_error() -> type[Exception]:
    return Exception
