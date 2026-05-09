import sys

import pytest

from h200_locomotion_lab.tools.genesis_official_batched_api_probe import (
    StateRead,
    build_morph,
    classify_selected_reset_change,
    compute_throughput,
    exercise_selected_reset,
    format_key_value,
    state_device_metrics,
    tensor_device_name,
    tensor_devices_ok,
    variant_build_kwargs,
    variant_morph_kwargs,
    verify_single_visible_cuda_device,
)


def test_format_key_value_uses_lowercase_booleans_and_escapes_newlines() -> None:
    assert format_key_value("status", "ok") == "status=ok"
    assert format_key_value("includes_render", False) == "includes_render=false"
    assert format_key_value("blocker", "first\nsecond") == "blocker=first\\nsecond"
    assert format_key_value("measure_time_s", 1.25) == "measure_time_s=1.250000"


def test_compute_throughput_separates_policy_sim_and_env_rates() -> None:
    metrics = compute_throughput(
        policy_steps=100,
        decimation=4,
        n_envs=16,
        elapsed_s=2.0,
    )

    assert metrics["policy_steps_per_sec"] == pytest.approx(50.0)
    assert metrics["sim_steps_per_sec"] == pytest.approx(200.0)
    assert metrics["env_policy_steps_per_sec"] == pytest.approx(800.0)
    assert metrics["env_sim_steps_per_sec"] == pytest.approx(3200.0)


def test_compute_throughput_handles_zero_elapsed_time() -> None:
    metrics = compute_throughput(policy_steps=100, decimation=4, n_envs=16, elapsed_s=0.0)

    assert set(metrics.values()) == {0.0}


def test_tensor_device_detection_accepts_fake_tensor_like_values() -> None:
    tensor = _FakeTensor([[1.0]], device="cuda:0")

    assert tensor_device_name(tensor) == "cuda:0"
    assert tensor_device_name([[1.0]]) == "not_tensor"
    assert tensor_device_name(None) == "unavailable"
    assert tensor_devices_ok(("cuda:0", "cuda:0"), backend="cuda", logical_cuda_device="cuda:0")
    assert not tensor_devices_ok(
        ("cuda:0", "cpu"),
        backend="cuda",
        logical_cuda_device="cuda:0",
    )
    assert not tensor_devices_ok(
        ("cuda:0", "not_available"),
        backend="cuda",
        logical_cuda_device="cuda:0",
    )


def test_state_device_metrics_include_root_velocity() -> None:
    state = StateRead(
        qpos=_FakeTensor([[0.0]], device="cuda:0"),
        dofs_pos=_FakeTensor([[0.0]], device="cuda:0"),
        dofs_vel=_FakeTensor([[0.0]], device="cuda:0"),
        root_pos=_FakeTensor([[0.0, 0.0, 0.0]], device="cuda:0"),
        root_quat=_FakeTensor([[1.0, 0.0, 0.0, 0.0]], device="cuda:0"),
        root_vel=_FakeTensor([[0.0, 0.0, 0.0]], device="cuda:0"),
    )

    assert state_device_metrics(state)["root_vel_device"] == "cuda:0"


def test_exercise_selected_reset_for_floating_base_calls_root_and_joint_setters() -> None:
    robot = _FakeFloatingBaseRobot()

    result = exercise_selected_reset(
        robot,
        asset_kind="go2",
        n_envs=2,
        dof_indices=(0, 1),
        torch_module=None,
        logical_cuda_device="cuda:0",
    )

    assert result["supported"] is True
    assert result["changes_only_target_envs"] is True
    assert robot.set_pos_calls == 1
    assert robot.set_quat_calls == 1
    assert robot.set_dofs_position_calls == 1
    assert robot.root_pos[0][0] == pytest.approx(0.01)
    assert robot.root_pos[1][0] == pytest.approx(1.0)


def test_exercise_selected_reset_prefers_selected_root_qpos_when_available() -> None:
    robot = _FakeQposRootRobot()

    result = exercise_selected_reset(
        robot,
        asset_kind="g1",
        n_envs=2,
        dof_indices=(0, 1),
        torch_module=None,
        logical_cuda_device="cuda:0",
    )

    assert result["supported"] is True
    assert result["changes_only_target_envs"] is True
    assert robot.set_qpos_calls == 1
    assert robot.set_pos_calls == 0
    assert robot.set_quat_calls == 0
    assert robot.root_pos[0][0] == pytest.approx(0.01)
    assert robot.root_pos[1][0] == pytest.approx(1.0)


def test_variant_morph_kwargs_apply_only_asset_morph_switches() -> None:
    assert variant_morph_kwargs("default") == {}
    assert variant_morph_kwargs("performance_mode") == {"performance_mode": True}
    assert variant_morph_kwargs("convexify") == {"convexify": True}
    assert variant_morph_kwargs("decimate") == {"decimate": True}


def test_variant_build_kwargs_do_not_apply_asset_variants() -> None:
    assert variant_build_kwargs("default") == {}
    assert variant_build_kwargs("performance_mode") == {}
    assert variant_build_kwargs("convexify") == {}
    assert variant_build_kwargs("decimate") == {}


def test_build_morph_uses_urdf_for_g1_urdf_asset() -> None:
    gs = _FakeGenesisMorphs()

    morph = build_morph(
        gs,
        asset_kind="g1",
        asset_path="g1_12dof_sausage.urdf",
        asset_variant="default",
    )

    assert morph == ("URDF", {"file": "g1_12dof_sausage.urdf", "fixed": False})


def test_cuda_isolation_blocks_when_visible_device_is_not_single_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    result = verify_single_visible_cuda_device(
        physical_gpu="1",
        logical_cuda_device="cuda:0",
    )

    assert result.ok is False
    assert result.blocker == "cuda_visible_devices_not_single:0,1"


def test_cuda_isolation_accepts_fake_torch_with_one_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(device_count=1, cuda_available=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = verify_single_visible_cuda_device(
        physical_gpu="1",
        logical_cuda_device="cuda:0",
    )

    assert result.ok is True
    assert result.blocker == ""
    assert result.torch_module is fake_torch


def test_cuda_isolation_rejects_fake_torch_with_multiple_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(device_count=2, cuda_available=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = verify_single_visible_cuda_device(
        physical_gpu="1",
        logical_cuda_device="cuda:0",
    )

    assert result.ok is False
    assert result.blocker == "torch_cuda_device_count_expected_1_got_2"


def test_selected_reset_classifies_only_target_env_changed() -> None:
    before = _FakeTensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    after = _FakeTensor(
        [
            [0.0, 0.0],
            [1.1, 1.0],
            [2.0, 2.0],
        ]
    )

    result = classify_selected_reset_change(before, after, target_env_ids=(1,))

    assert result.supported is True
    assert result.changes_only_target_envs is True
    assert result.reason == "only_target_envs_changed"
    assert result.changed_envs == (1,)


def test_selected_reset_classification_flattens_nested_env_rows() -> None:
    before = _FakeTensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    after = _FakeTensor(
        [
            [[0.1, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )

    result = classify_selected_reset_change(before, after, target_env_ids=(0,))

    assert result.supported is True
    assert result.changes_only_target_envs is True
    assert result.changed_envs == (0,)


def test_selected_reset_rejects_non_target_env_change() -> None:
    before = [[0.0], [1.0], [2.0]]
    after = [[0.0], [1.1], [2.1]]

    result = classify_selected_reset_change(before, after, target_env_ids=(1,))

    assert result.supported is True
    assert result.changes_only_target_envs is False
    assert result.reason == "non_target_env_changed"
    assert result.changed_envs == (1, 2)


def test_selected_reset_does_not_claim_verification_for_one_env() -> None:
    result = classify_selected_reset_change([[0.0]], [[0.1]], target_env_ids=(0,))

    assert result.supported is False
    assert result.changes_only_target_envs is False
    assert result.reason == "not_applicable_n_envs_lt_2"


class _FakeTensor:
    def __init__(self, data: object, *, device: str = "cuda:0") -> None:
        self._data = data
        self.device = device

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> object:
        return self._data


class _FakeTorch:
    def __init__(self, *, device_count: int, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(device_count=device_count, cuda_available=cuda_available)

    def empty(self, _shape: object, *, device: str) -> _FakeTensor:
        return _FakeTensor([0.0], device=device)


class _FakeCuda:
    def __init__(self, *, device_count: int, cuda_available: bool) -> None:
        self._device_count = device_count
        self._cuda_available = cuda_available

    def is_available(self) -> bool:
        return self._cuda_available

    def device_count(self) -> int:
        return self._device_count


class _FakeGenesisMorphs:
    class morphs:
        @staticmethod
        def URDF(**kwargs: object) -> tuple[str, dict[str, object]]:
            return ("URDF", kwargs)

        @staticmethod
        def MJCF(**kwargs: object) -> tuple[str, dict[str, object]]:
            return ("MJCF", kwargs)


class _FakeFloatingBaseRobot:
    def __init__(self) -> None:
        self.root_pos = [[0.0, 0.0, 0.6], [1.0, 0.0, 0.6]]
        self.root_quat = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        self.root_vel = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.dofs_pos = [[0.0, 0.0], [1.0, 1.0]]
        self.dofs_vel = [[0.0, 0.0], [0.0, 0.0]]
        self.qpos = [
            self.root_pos[0] + self.root_quat[0] + self.dofs_pos[0],
            self.root_pos[1] + self.root_quat[1] + self.dofs_pos[1],
        ]
        self.set_pos_calls = 0
        self.set_quat_calls = 0
        self.set_dofs_position_calls = 0

    def get_qpos(self) -> list[list[float]]:
        return self.qpos

    def get_dofs_position(self, dofs_idx_local: tuple[int, ...] | None = None) -> list[list[float]]:
        return self.dofs_pos

    def get_dofs_velocity(self, dofs_idx_local: tuple[int, ...] | None = None) -> list[list[float]]:
        return self.dofs_vel

    def get_pos(self) -> list[list[float]]:
        return self.root_pos

    def get_quat(self) -> list[list[float]]:
        return self.root_quat

    def get_vel(self) -> list[list[float]]:
        return self.root_vel

    def set_pos(
        self,
        pos: list[list[float]] | list[float],
        envs_idx: tuple[int, ...] | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        self.set_pos_calls += 1
        envs = envs_idx or tuple(range(len(self.root_pos)))
        rows = pos if isinstance(pos[0], list) else [pos]  # type: ignore[index]
        for row_index, env_index in enumerate(envs):
            self.root_pos[env_index] = list(rows[row_index])  # type: ignore[index]
        self._refresh_qpos()

    def set_quat(
        self,
        quat: list[list[float]] | list[float],
        envs_idx: tuple[int, ...] | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        self.set_quat_calls += 1
        envs = envs_idx or tuple(range(len(self.root_quat)))
        rows = quat if isinstance(quat[0], list) else [quat]  # type: ignore[index]
        for row_index, env_index in enumerate(envs):
            self.root_quat[env_index] = list(rows[row_index])  # type: ignore[index]
        self._refresh_qpos()

    def set_dofs_position(
        self,
        position: list[list[float]] | list[float],
        dofs_idx_local: tuple[int, ...] | None = None,
        envs_idx: tuple[int, ...] | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        self.set_dofs_position_calls += 1
        envs = envs_idx or tuple(range(len(self.dofs_pos)))
        rows = position if isinstance(position[0], list) else [position]  # type: ignore[index]
        for row_index, env_index in enumerate(envs):
            self.dofs_pos[env_index] = list(rows[row_index])  # type: ignore[index]
        self._refresh_qpos()

    def _refresh_qpos(self) -> None:
        self.qpos = [
            self.root_pos[0] + self.root_quat[0] + self.dofs_pos[0],
            self.root_pos[1] + self.root_quat[1] + self.dofs_pos[1],
        ]


class _FakeQposRootRobot(_FakeFloatingBaseRobot):
    def __init__(self) -> None:
        super().__init__()
        self.set_qpos_calls = 0

    def set_qpos(
        self,
        qpos: list[list[float]] | list[float],
        qs_idx_local: tuple[int, ...] | None = None,
        envs_idx: tuple[int, ...] | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        self.set_qpos_calls += 1
        envs = envs_idx or tuple(range(len(self.qpos)))
        rows = qpos if isinstance(qpos[0], list) else [qpos]  # type: ignore[index]
        for row_index, env_index in enumerate(envs):
            row = list(rows[row_index])  # type: ignore[index]
            self.root_pos[env_index] = row[:3]
            self.root_quat[env_index] = row[3:7]
        self._refresh_qpos()
