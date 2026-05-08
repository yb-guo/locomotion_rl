import csv
import io

import pytest

from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM, SONIC_DECODER_OBS_DIM
from h200_locomotion_lab.sonic.g1_policy_bridge import get_default_sonic_g1_action_bridge
from h200_locomotion_lab.tools.sonic_g1_deployment_dry_run import (
    read_numeric_csv,
    run_dry_run,
    write_summary_csv,
)


def test_run_dry_run_builds_obs_and_motor_target_summary() -> None:
    qpos_rows = (
        (0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0) + (0.0,) * SONIC_ACTION_DIM,
        (0.1, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0) + (0.01,) * SONIC_ACTION_DIM,
    )
    rows = run_dry_run(
        qpos_rows,
        raw_action_rows=((0.25,) * SONIC_ACTION_DIM,),
        frames=2,
    )

    assert len(rows) == 2
    assert rows[0]["obs_dim"] == str(SONIC_DECODER_OBS_DIM)
    assert rows[0]["obs_finite"] == "True"
    assert rows[0]["raw_action_max_abs"] == "0.25"
    assert rows[1]["root_z"] == "0.78"
    assert "target_mujoco_28" in rows[0]
    expected_target = get_default_sonic_g1_action_bridge().policy_action_to_command_targets(
        (0.25,) * SONIC_ACTION_DIM
    )
    assert float(rows[0]["target_mujoco_00"]) == pytest.approx(expected_target[0])
    assert float(rows[0]["target_mujoco_28"]) == pytest.approx(expected_target[28])


def test_write_summary_csv_round_trips_header() -> None:
    rows = [
        {"frame": "0", "root_z": "0.79", "obs_finite": "True"},
        {"frame": "1", "root_z": "0.78", "obs_finite": "True"},
    ]
    path = _FakeCsvPath()

    write_summary_csv(path, rows)  # type: ignore[arg-type]

    with io.StringIO(path.content) as stream:
        read_back = list(csv.DictReader(stream))
    assert read_back == rows


def test_read_numeric_csv_skips_one_header_row() -> None:
    path = _FakeCsvPath(content="a,b,c\n1,2,3\n4,5,6\n")

    rows = read_numeric_csv(path, 3)  # type: ignore[arg-type]

    assert rows == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


class _FakeCsvPath:
    def __init__(self, content: str = "") -> None:
        self.content = content

    @property
    def parent(self) -> "_FakeCsvPath":
        return self

    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        assert parents is True
        assert exist_ok is True

    def open(self, mode: str = "r", newline: str = "") -> io.StringIO:
        if "w" in mode:
            return _RecordingStringIO(self)
        return io.StringIO(self.content)

    def __str__(self) -> str:
        return "fake.csv"


class _RecordingStringIO(io.StringIO):
    def __init__(self, owner: _FakeCsvPath) -> None:
        super().__init__()
        self._owner = owner

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._owner.content = self.getvalue()
        super().__exit__(exc_type, exc_value, traceback)
