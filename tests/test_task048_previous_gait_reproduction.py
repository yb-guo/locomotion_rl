import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TASK048_DIR = (
    ROOT / ".agent" / "task" / "task048-local-4090-previous-gait-reproduction"
)


def test_task048_registration_patch_is_idempotent(tmp_path: Path) -> None:
    module = _load_registration_script()
    velocity_path = tmp_path / "velocity_command.py"
    env_cfg_path = tmp_path / "env_cfgs.py"
    init_path = tmp_path / "__init__.py"
    velocity_path.write_text(
        "class UniformVelocityCommand:\n"
        "  def sample(self, env_ids):\n"
        "    r = object()\n"
        + module.VELOCITY_COMMAND_RESAMPLE_OLD
        + "\n@dataclass\n"
        "class UniformVelocityCommandCfg:\n"
        + module.VELOCITY_COMMAND_CFG_OLD,
        encoding="utf-8",
    )
    env_cfg_path.write_text(
        "def unitree_g1_gripper_flat_env_cfg(play=False):\n  return object()\n",
        encoding="utf-8",
    )
    init_path.write_text(
        "from h200_locomotion_lab.training.rsl_history_wrapper import (\n"
        "  Task038TrueTxlMemoryK160Runner,\n"
        ")\n"
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from mjlab.tasks.registry import register_mjlab_task\n",
        encoding="utf-8",
    )

    for _ in range(2):
        module.patch_velocity_command(velocity_path)
        module.patch_env_cfgs(env_cfg_path)
        module.patch_init(init_path)

    velocity = velocity_path.read_text(encoding="utf-8")
    env_cfg = env_cfg_path.read_text(encoding="utf-8")
    init = init_path.read_text(encoding="utf-8")
    assert velocity.count("lin_vel_x_choices = self.cfg.lin_vel_x_choices") == 1
    assert velocity.count("lin_vel_x_choice_weights: tuple[float, ...] | None") == 1
    assert env_cfg.count("def unitree_g1_gripper_flat_task048_clean_bins_env_cfg(") == 1
    assert "_strip_randomization(cfg)" in env_cfg
    assert "TASK048_SPEED_BINS = (0.4, 1.2, 2.0)" in env_cfg
    for task_id in module.REGISTER_BLOCKS:
        assert init.count(task_id) == 1
    for runner_name in module.RUNNER_NAMES:
        assert init.count(runner_name) == 2


def test_task048_local_launch_profile_preserves_ppo_budget(tmp_path: Path) -> None:
    script = TASK048_DIR / "task048_launch_reproduction_stage.sh"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "ROOT": str(tmp_path),
        "PROJECT_ROOT": str(ROOT),
        "PY": sys.executable,
    }

    result = subprocess.run(
        ["bash", str(script), "mlp-prior"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Task048-Mlp-OfficialCurriculum-Train" in result.stdout
    assert "num-envs=4096" in result.stdout
    assert "max-iterations=1000" in result.stdout
    assert "num-learning-epochs=5" in result.stdout
    assert "num-mini-batches=4" in result.stdout


def test_task048_historical_adapt_profile_matches_recorded_run(tmp_path: Path) -> None:
    script = TASK048_DIR / "task048_launch_reproduction_stage.sh"
    checkpoint = tmp_path / "model_5349.pt"
    checkpoint.touch()
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "PROFILE": "historical",
        "SOURCE_CHECKPOINT": str(checkpoint),
        "ROOT": str(tmp_path),
        "PROJECT_ROOT": str(ROOT),
        "PY": sys.executable,
    }

    result = subprocess.run(
        ["bash", str(script), "adaptk4"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Task048-AdaptK4-CleanBins-Train" in result.stdout
    assert "num-envs=8192" in result.stdout
    assert "max-iterations=60" in result.stdout
    assert "seed=3603630" in result.stdout
    assert "learning-rate=0.000003" in result.stdout
    assert "entropy-coef=0.0003" in result.stdout
    assert "agent.resume=True" in result.stdout


def test_task048_eval_matrix_requires_all_historical_speeds_and_zero_falls() -> None:
    script = (TASK048_DIR / "task048_eval_clean_matrix.sh").read_text(encoding="utf-8")

    assert "speeds=(0.4 1.2 2.0)" in script
    assert "max_lin_errors=(0.25 0.55 0.90)" in script
    assert "--max-final-fall-ratio 0.0" in script
    assert 'TASK="Unitree-G1-Gripper-Flat-Task048-TrueTxl-CleanBins-Eval"' in script
    assert "Task038-TrainTrueTxlRunnerSmoke" not in script
    assert 'matrix_pass = len(cases) == 3 and all(case_passes)' in script
    assert '"reproduction_claim": False' in script


def test_task048_docs_distinguish_historical_and_fresh_reproduction() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(TASK048_DIR.glob("*.md"))
    )

    for checkpoint in ("model_5349.pt", "model_5408.pt", "model_5467.pt"):
        assert checkpoint in docs
    assert "3603630" in docs
    assert "3700705" in docs
    assert "3e-6" in docs
    assert "3e-4" in docs
    assert "not bitwise or seed-identical" in docs
    assert "True-TXL scratch training is not an allowed substitute" in docs


def _load_registration_script():
    path = TASK048_DIR / "task048_register_previous_gait_reproduction.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
