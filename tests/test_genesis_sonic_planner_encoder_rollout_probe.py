import io

from h200_locomotion_lab.sonic.g1_planner_encoder import SONIC_PLANNER_QPOS_DIM
from h200_locomotion_lab.tools.genesis_sonic_planner_encoder_rollout_probe import (
    parse_num_pred_frames,
    write_planner_context_csv,
)


def test_parse_num_pred_frames_reads_runner_stdout() -> None:
    output = "\n".join(
        [
            "SONIC_PLANNER_ORT_RUNNER_MODE onnxruntime_cpp",
            "PLANNER_NUM_PRED_FRAMES 44",
            "SONIC_PLANNER_ORT_RUNNER_OK",
        ]
    )

    assert parse_num_pred_frames(output) == 44


def test_write_planner_context_csv_writes_four_36d_rows() -> None:
    rows = tuple(
        tuple(float(row * 100 + col) for col in range(SONIC_PLANNER_QPOS_DIM))
        for row in range(4)
    )
    path = _FakeCsvPath()

    write_planner_context_csv(path, rows)  # type: ignore[arg-type]

    lines = path.content.splitlines()
    assert len(lines) == 4
    assert lines[0].split(",")[:3] == ["0", "1", "2"]
    assert lines[-1].split(",")[-1] == "335"


class _FakeCsvPath:
    content = ""

    @property
    def parent(self) -> "_FakeCsvPath":
        return self

    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        assert parents is True
        assert exist_ok is True

    def open(self, mode: str, newline: str) -> "_RecordingStringIO":
        assert mode == "w"
        assert newline == ""
        return _RecordingStringIO(self)


class _RecordingStringIO(io.StringIO):
    def __init__(self, owner: _FakeCsvPath) -> None:
        super().__init__()
        self._owner = owner

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._owner.content = self.getvalue()
        super().__exit__(exc_type, exc_value, traceback)
