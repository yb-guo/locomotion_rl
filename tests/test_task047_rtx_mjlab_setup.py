import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "configs" / "requirements" / "rtx5060ti-mjlab-constraints.txt"
SETUP_SCRIPT = ROOT / "scripts" / "setup_rtx_mjlab.sh"
SMOKE_SCRIPT = ROOT / "scripts" / "run_rtx_mjlab_smoke.sh"
MIGRATION_SCRIPT = ROOT / "scripts" / "check_task044_migration.sh"
TASK028_GENERATOR = (
    ROOT
    / ".agent"
    / "task"
    / "task028-randomized-wholebody-morphology-env"
    / "artifacts"
    / "task028_create_g1_gripper_task.py"
)


def _constraints() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        result[name] = version
    return result


def test_task047_constraints_pin_the_verified_compatibility_boundary() -> None:
    constraints = _constraints()

    assert constraints["torch"] == "2.13.0"
    assert constraints["mjlab"] == "1.2.0"
    assert constraints["mujoco"] == "3.5.0"
    assert constraints["mujoco-warp"] == "3.5.0"
    assert constraints["warp-lang"] == "1.12.0"
    assert constraints["scipy"] == "1.17.1"


def test_task047_shell_entry_points_parse() -> None:
    for script in (SETUP_SCRIPT, SMOKE_SCRIPT, MIGRATION_SCRIPT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_task047_setup_is_explicit_and_revision_pinned() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "1425b15f73bd4095f0df53709d7c389c3eb9e790" in text
    assert "--torch-backend cu130" in text
    assert "rtx5060ti-mjlab-constraints.txt" in text
    assert "uv\" pip check" not in text
    assert '"${UV_BIN}" pip check' in text
    assert "git clone" in text  # Printed instructions only; no automatic fetch.
    assert 'git clone https://github.com' not in text.split("cat >&2 <<EOF", maxsplit=1)[0]


def test_task047_smoke_uses_vram_safe_defaults() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "RTX_MJLAB_SMOKE_TASK:-Unitree-G1-Flat" in text
    assert "RTX_MJLAB_SMOKE_NUM_ENVS:-32" in text
    assert "--agent.max-iterations 1" in text
    assert "--agent.num-steps-per-env" in text
    assert "--gpu-ids" not in text
    assert 'PYTHONPATH="${REPO_ROOT}/src' in text


def test_task047_migration_audit_names_the_missing_cumulative_boundary() -> None:
    text = MIGRATION_SCRIPT.read_text(encoding="utf-8")

    assert "Task029 motor-failure environment base" in text
    assert "Task030 dynamic-failure environment" in text
    assert "Task031 unified-speed/dead-grid environment" in text
    assert "Task044TrueTxlMemoryK160ClearHistoryRunner" in text
    assert "do not treat it as the 31-action custom algorithm" in text


def test_task028_generator_accepts_a_new_machine_checkout_root(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("task028_generator", TASK028_GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    root = tmp_path / "unitree_rl_mjlab"
    args = module.parse_args(["--root", str(root)])

    assert args.root == root
