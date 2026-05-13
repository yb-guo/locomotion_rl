from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from h200_locomotion_lab.tools import g1_base_attitude_height_stabilization as probe


def test_parse_args_exposes_task023_modes_and_metadata() -> None:
    args = probe.parse_args(
        [
            "--mode",
            "attitude_height",
            "--steps",
            "32",
            "--seed",
            "7",
            "--asset-path",
            "outputs/task023/assets/g1.xml",
            "--asset-variant-label",
            "ankle_roll_larger_spheres",
            "--asset-source-path",
            "/source/g1.xml",
            "--output-root",
            "outputs/task023/probe",
            "--run-name",
            "local",
        ]
    )

    assert set(probe.STABILIZER_MODES) == {"none", "attitude", "height", "attitude_height"}
    assert args.runner == "local_toy"
    assert args.mode == "attitude_height"
    assert args.steps == 32
    assert args.seed == 7
    assert args.asset_path == Path("outputs/task023/assets/g1.xml")
    assert args.asset_variant_label == "ankle_roll_larger_spheres"
    assert args.asset_source_path == Path("/source/g1.xml")
    assert args.output_root == Path("outputs/task023/probe")
    assert args.run_name == "local"


def test_module_import_path_does_not_pull_genesis_backend() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    top_level = source.split("def run_genesis_probe", maxsplit=1)[0]

    assert "vectorized_genesis_backend" not in top_level
    assert "GenesisG1SceneBackend" not in source
    assert "genesis" not in inspect.getsource(probe.run_toy_rollout).lower()


def test_controller_mode_and_gain_clipping_are_bounded() -> None:
    args = probe.parse_args(
        [
            "--mode",
            "attitude_height",
            "--attitude-kp",
            "99",
            "--height-kp",
            "21",
            "--max-gain",
            "5",
            "--max-joint-delta",
            "0.03",
        ]
    )
    requested = probe.StabilizerGains(
        attitude_kp=args.attitude_kp,
        attitude_kd=args.attitude_kd,
        height_kp=args.height_kp,
        height_kd=args.height_kd,
        max_joint_delta=args.max_joint_delta,
    )
    gains = probe.clip_gains(requested, max_gain=args.max_gain)
    state = probe.ToyState(
        step=10,
        root_height=0.40,
        root_height_velocity=-0.2,
        roll=0.5,
        pitch=-0.4,
        roll_velocity=0.1,
        pitch_velocity=-0.1,
    )

    output = probe.compute_controller_output(
        mode=args.mode,
        gains=gains,
        state=state,
        target_height=args.target_height,
    )

    assert gains.attitude_kp == pytest.approx(5.0)
    assert gains.height_kp == pytest.approx(5.0)
    assert output.clipped is True
    assert output.max_abs_delta <= 0.03 + 1e-12
    assert output.roll_delta == pytest.approx(-0.03)
    assert output.pitch_delta == pytest.approx(0.03)
    assert output.height_delta == pytest.approx(0.03)


def test_local_none_rollout_reports_first_tilt_reset_and_schema() -> None:
    root = fresh_test_dir("none")
    args = probe.parse_args(
        [
            "--mode",
            "none",
            "--steps",
            "180",
            "--seed",
            "0",
            "--asset-path",
            "source.xml",
            "--asset-variant-label",
            "source",
            "--output-root",
            str(root),
            "--run-name",
            "none",
        ]
    )

    summary = probe.run_probe(args)

    run_dir = root / "none"
    assert summary["status"] == "completed"
    assert summary["effective_asset_path"] == "source.xml"
    assert summary["asset_metadata"]["variant_label"] == "source"
    assert summary["stabilizer"]["mode"] == "none"
    assert summary["improvement_classification"] == "baseline"
    assert summary["first_tilt_step"] is not None
    assert summary["first_reset_step"] == summary["first_tilt_step"]
    assert summary["root_height_timeline_summary"]["min"] < 0.78
    assert summary["upright_timeline_summary"]["final"] < summary["upright_timeline_summary"]["initial"]
    assert summary["top_joint_errors"]
    assert {"ankle_roll", "ankle_pitch"} <= set(summary["contact_trace_summary"])
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 180
    assert {"ankle_roll_contact_force", "ankle_pitch_contact_force", "joint_errors"} <= set(
        rows[0]
    )


def test_local_attitude_height_improves_over_local_baseline() -> None:
    root = fresh_test_dir("attitude_height")
    args = probe.parse_args(
        [
            "--mode",
            "attitude_height",
            "--steps",
            "180",
            "--seed",
            "0",
            "--asset-path",
            "larger.xml",
            "--asset-variant-label",
            "ankle_roll_larger_spheres",
            "--asset-source-path",
            "source.xml",
            "--max-joint-delta",
            "0.05",
            "--output-root",
            str(root),
            "--run-name",
            "attitude-height",
        ]
    )

    summary = probe.run_probe(args)

    assert summary["stabilizer"]["mode"] == "attitude_height"
    assert summary["asset_metadata"]["source_path"] == "source.xml"
    assert summary["baseline_first_reset_step"] is not None
    assert summary["first_reset_step"] is None
    assert summary["improvement_classification"] == "physical_stability"
    assert summary["stabilizer"]["clipping"]["clipped_steps"] > 0
    top_joints = {row["joint"] for row in summary["top_joint_errors"]}
    assert top_joints & {
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
    }
    assert summary["contact_trace_summary"]["ankle_roll"]["max_force"] > 0.0
    assert summary["contact_trace_summary"]["ankle_pitch"]["active_steps"] == 180


def test_summary_json_option_writes_requested_file() -> None:
    root = fresh_test_dir("summary_json")
    summary_json = root / "summary-copy.json"
    args = probe.parse_args(
        [
            "--mode",
            "height",
            "--steps",
            "24",
            "--asset-path",
            "source.xml",
            "--output-root",
            str(root),
            "--run-name",
            "height",
            "--summary-json",
            str(summary_json),
        ]
    )

    summary = probe.run_probe(args)

    copied = json.loads(summary_json.read_text(encoding="utf-8"))
    assert copied["run_dir"] == summary["run_dir"]
    assert copied["stabilizer"]["mode"] == "height"


def test_genesis_command_uses_guarded_wrapper_without_running_h200() -> None:
    args = probe.parse_args(
        [
            "--runner",
            "genesis",
            "--mode",
            "attitude",
            "--steps",
            "96",
            "--asset-path",
            "/root/project/assets/g1.xml",
            "--run-name",
            "candidate",
        ]
    )

    command = probe.build_h200_genesis_command(args)

    assert command.startswith("/root/agent_workspace/safe_agent/run_guarded.sh bash -lc ")
    assert "h200-locomotion-lab-task023-base-attitude-height-stabilization" in command
    assert "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src" in command
    assert "h200_locomotion_lab.tools.g1_base_attitude_height_stabilization" in command
    assert "--runner genesis" in command
    assert "--physical-gpu 1 --logical-cuda-device cuda:0" in command


def test_genesis_runner_uses_fake_vectorized_backend_and_asset_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = fresh_test_dir("genesis_fake")
    instances: list[FakeGenesisBackend] = []

    class RuntimeFakeGenesisBackend(FakeGenesisBackend):
        def __init__(self, config: object, *, profile: object) -> None:
            super().__init__(config=config, profile=profile, with_contact=True)
            instances.append(self)

    monkeypatch.setattr(
        probe,
        "load_genesis_runtime",
        lambda: (RuntimeFakeGenesisBackend, FakeGenesisConfig, FakeTorch()),
    )
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", fake_profile)
    args = probe.parse_args(
        [
            "--runner",
            "genesis",
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "2",
            "--mode",
            "attitude_height",
            "--steps",
            "4",
            "--asset-path",
            "/tmp/override-g1.xml",
            "--asset-variant-label",
            "source",
            "--output-root",
            str(root),
            "--run-name",
            "fake",
        ]
    )

    summary = probe.run_probe(args)

    assert len(instances) == 2
    backend = instances[0]
    assert backend.config.n_envs == 2
    assert backend.config.backend == "cpu"
    assert backend.profile.asset.path == "/tmp/override-g1.xml"
    assert summary["status"] == "completed"
    assert summary["runner"] == "genesis"
    assert summary["effective_asset_path"] == "/tmp/override-g1.xml"
    assert summary["asset_metadata"]["effective_path"] == "/tmp/override-g1.xml"
    assert summary["genesis"]["n_envs"] == 2
    assert summary["genesis"]["profile"]["asset_path"] == "/tmp/override-g1.xml"
    assert summary["hardware_metadata"]["physical_gpu"] == "1"
    assert summary["hardware_metadata"]["logical_cuda_device"] == "cpu"
    assert summary["stabilizer"]["mode"] == "attitude_height"
    assert summary["stabilizer"]["clipping"]["max_abs_delta"] <= args.max_joint_delta
    assert summary["contact_trace_summary"]["ankle_roll"]["available"] is True
    assert summary["contact_trace_summary"]["ankle_pitch"]["available"] is True
    assert summary["contact_trace_summary"]["ankle_roll"]["max_force"] > 0.0
    assert summary["top_joint_errors"]
    assert (root / "fake" / "metrics.jsonl").is_file()
    assert (root / "fake" / "baseline_metrics.jsonl").is_file()
    assert any(
        abs(value) > 0.0
        for action in backend.actions
        for row in action
        for value in row
    )
    assert all(
        abs(value) <= 1.0
        for action in backend.actions
        for row in action
        for value in row
    )


def test_genesis_contact_schema_reports_unavailable_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = fresh_test_dir("genesis_no_contact")

    class RuntimeNoContactBackend(FakeGenesisBackend):
        def __init__(self, config: object, *, profile: object) -> None:
            super().__init__(config=config, profile=profile, with_contact=False)

    monkeypatch.setattr(
        probe,
        "load_genesis_runtime",
        lambda: (RuntimeNoContactBackend, FakeGenesisConfig, FakeTorch()),
    )
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", fake_profile)
    args = probe.parse_args(
        [
            "--runner",
            "genesis",
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--mode",
            "none",
            "--steps",
            "2",
            "--output-root",
            str(root),
            "--run-name",
            "fake",
        ]
    )

    summary = probe.run_probe(args)

    assert summary["status"] == "completed"
    assert summary["contact_trace_summary"]["ankle_roll"]["available"] is False
    assert summary["contact_trace_summary"]["ankle_pitch"]["available"] is False
    assert summary["contact_trace_summary"]["ankle_roll"]["missing"]


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / "outputs" / "task023" / ".test_tmp" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root


@dataclass(frozen=True)
class FakeAsset:
    path: str = "default-g1.xml"
    format: str = "mjcf"
    genesis_morph: str = "MJCF"
    usage: str = "test"


@dataclass(frozen=True)
class FakeControl:
    default_angles_rad: tuple[float, ...]
    action_scales_rad: tuple[float, ...]


@dataclass(frozen=True)
class FakeProfile:
    asset: FakeAsset
    control: FakeControl
    actuator_order: tuple[str, ...]
    action_dim: int = 27
    name: str = "fake-g1"
    family: str = "unitree_g1"
    route: str = "VectorizedGenesisBackend"
    source_path: Path | None = None


def fake_profile() -> FakeProfile:
    return FakeProfile(
        asset=FakeAsset(),
        control=FakeControl(
            default_angles_rad=(0.0,) * 27,
            action_scales_rad=(0.2,) * 27,
        ),
        actuator_order=probe.G1_27DOF_NOHAND_ACTUATOR_ORDER,
    )


@dataclass
class FakeGenesisConfig:
    n_envs: int
    backend: str = "cpu"
    logical_cuda_device: str = "cpu"


class FakeTorch:
    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


class FakeLink:
    def __init__(self, index: int) -> None:
        self.idx_local = index


class FakeContactRobot:
    link_indices = {
        "left_ankle_roll_link": 0,
        "right_ankle_roll_link": 1,
        "left_ankle_pitch_link": 2,
        "right_ankle_pitch_link": 3,
    }

    def __init__(self, *, with_contact: bool) -> None:
        self.with_contact = with_contact

    def get_link(self, name: str) -> FakeLink:
        if not self.with_contact:
            raise KeyError(name)
        return FakeLink(self.link_indices[name])

    def get_links_net_contact_force(
        self,
        links_idx_local: tuple[int, ...] | None = None,
    ) -> list[list[list[float]]]:
        if not self.with_contact:
            raise RuntimeError("contact unavailable")
        indices = links_idx_local or tuple(self.link_indices.values())
        return [
            [[float(index + 1), 0.0, 0.0] for index in indices],
            [[float(index + 2), 0.0, 0.0] for index in indices],
        ]


class FakeGenesisBackend:
    def __init__(self, config: object, *, profile: object, with_contact: bool) -> None:
        self.config = config
        self.profile = profile
        self.n_envs = config.n_envs
        self.action_dim = 27
        self.default_positions_values = tuple(profile.control.default_angles_rad)
        self.robot = FakeContactRobot(with_contact=with_contact)
        self.step_count = 0
        self.actions: list[list[list[float]]] = []
        self.dof_pos = [list(self.default_positions_values) for _ in range(self.n_envs)]

    def reset(self) -> list[list[float]]:
        self.step_count = 0
        self.dof_pos = [list(self.default_positions_values) for _ in range(self.n_envs)]
        return [[0.0] * 90 for _ in range(self.n_envs)]

    def state(self) -> SimpleNamespace:
        height = 0.72 + (0.01 * self.step_count)
        roll = 0.10 if self.step_count == 0 else 0.03
        pitch = -0.08 if self.step_count == 0 else -0.02
        quat = normalized_small_quat(roll=roll, pitch=pitch)
        return SimpleNamespace(
            root_pos=[[0.0, 0.0, height] for _ in range(self.n_envs)],
            root_quat=[list(quat) for _ in range(self.n_envs)],
            root_vel=[[0.0, 0.0, 0.0] for _ in range(self.n_envs)],
            root_ang_vel=[[0.0, 0.0, 0.0] for _ in range(self.n_envs)],
            dof_pos=[row[:] for row in self.dof_pos],
        )

    def step_physics(self, action: list[list[float]]) -> list[list[float]]:
        clipped = [[max(-1.0, min(1.0, value)) for value in row] for row in action]
        self.actions.append(clipped)
        scales = self.profile.control.action_scales_rad
        self.dof_pos = [
            [
                default + normalized * scale
                for default, normalized, scale in zip(
                    self.default_positions_values,
                    row,
                    scales,
                )
            ]
            for row in clipped
        ]
        self.step_count += 1
        return clipped

    def tensor_device_report(self) -> dict[str, str]:
        return {"fake": self.config.logical_cuda_device}

    def tensor_device_ok(self) -> bool:
        return True

    def contact_solver_config_report(self) -> dict[str, object]:
        return {"configured": False}


def normalized_small_quat(*, roll: float, pitch: float) -> tuple[float, float, float, float]:
    w = 1.0
    x = roll / 2.0
    y = pitch / 2.0
    z = 0.0
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    return (w / norm, x / norm, y / norm, z / norm)
