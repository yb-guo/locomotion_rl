import importlib.util
import json
from collections import UserDict
from pathlib import Path

TASK038_DIR = (
    Path(__file__).resolve().parents[1]
    / ".agent"
    / "task"
    / "task038-locoformer-min-g1like-reproduction"
)


def test_task038_registration_patcher_is_idempotent() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")
    constants = MemoryPath(
        "from pathlib import Path\n"
        "import mujoco\n"
        "from mjlab.entity import EntityCfg\n"
        "from mjlab.utils.os import update_assets\n"
        "G1_GRIPPER_ARTICULATION = object()\n"
    )
    env_cfgs = MemoryPath(
        "from src.assets.robots.unitree_g1_gripper.g1_gripper_constants import (\n"
        "  G1_GRIPPER_ACTION_SCALE,\n"
        "  G1_GRIPPER_BODY_ACTION_SCALE,\n"
        "  get_g1_gripper_robot_cfg,\n"
        ")\n"
        "from mjlab.envs import ManagerBasedRlEnvCfg\n"
        "def unitree_g1_gripper_flat_env_cfg(play: bool = False):\n"
        "  return object()\n"
    )
    init = MemoryPath(
        "from mjlab.tasks.registry import register_mjlab_task\n"
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from .rl_cfg import unitree_g1_gripper_ppo_runner_cfg\n"
    )

    for _ in range(2):
        module.patch_constants(constants)
        module.patch_env_cfgs(env_cfgs)
        module.patch_init(init)

    constants_text = constants.read_text(encoding="utf-8")
    env_text = env_cfgs.read_text(encoding="utf-8")
    init_text = init.read_text(encoding="utf-8")

    assert constants_text.count("def get_g1_gripper_robot_cfg_for_xml") == 1
    assert constants_text.count("TASK038_TRAIN_XML") == 1
    assert constants_text.count("TASK038_HELDOUT_XML") == 1
    assert env_text.count("def unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg") == 1
    assert env_text.count("def unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg") == 1
    assert init_text.count(module.TRAIN_TASK_ID) == 1
    assert init_text.count(module.HELDOUT_TASK_ID) == 1
    assert init_text.count("unitree_g1_gripper_ppo_runner_cfg()") == 2
    assert "runner_cls=MjlabOnPolicyRunner" in init_text


def test_task038_registration_constants_include_task_ids_and_xml_paths() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")

    assert module.TRAIN_TASK_ID == "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke"
    assert module.HELDOUT_TASK_ID == "Unitree-G1-Gripper-Flat-Task038-HeldoutAssetSmoke"
    assert module.TASK038_TRAIN_XML.endswith("g1like-train-none-e6ba46370d.xml")
    assert module.TASK038_HELDOUT_XML.endswith("g1like-heldout-combined-6ac730c265.xml")
    assert "TASK038_TRAIN_XML" in module.CONSTANTS_BLOCK
    assert module.TRAIN_TASK_ID in module.REGISTER_BLOCKS
    assert module.HELDOUT_TASK_ID in module.REGISTER_BLOCKS


def test_task038_constants_block_updates_assets_for_absolute_and_relative_meshdir() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")
    block = module.CONSTANTS_BLOCK

    assert "asset_dir = meshdir if meshdir.is_absolute() else variant_xml.parent / meshdir" in block
    assert "if asset_dir.exists():" in block
    assert "spec.assets = {}" in block
    assert "update_assets(spec.assets, asset_dir, spec.meshdir)" in block


def test_task038_probe_parse_args_defaults() -> None:
    module = _load_src_tool("task038_mjlab_variant_env_load_probe.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke"
    assert args.output_json == "out.json"
    assert args.expected_action_dim == 31
    assert args.num_envs == 8
    assert args.steps == 2
    assert args.device == "cuda:0"


def test_task038_expected_xml_mismatch_blocks_pass() -> None:
    module = _load_src_tool("task038_mjlab_variant_env_load_probe.py")
    summary = _passing_summary()
    summary["expected_xml_path"] = "/tmp/expected.xml"
    summary["registered_xml_path"] = "/tmp/actual.xml"
    summary["xml_path_matches_expected"] = module.xml_path_match(
        summary["expected_xml_path"], summary["registered_xml_path"]
    )

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "expected_xml_path_mismatch" in reasons


def test_task038_positive_pass_requires_registered_xml_and_non_empty_obs() -> None:
    module = _load_src_tool("task038_mjlab_variant_env_load_probe.py")
    summary = _passing_summary()

    passed, reasons = module.evaluate_probe_pass(summary)
    assert passed is True
    assert reasons == []

    summary["obs"] = {}
    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "obs_summary_empty" in reasons


def test_task038_obs_summary_recurses_fake_tensordict_like() -> None:
    module = _load_src_tool("task038_mjlab_variant_env_load_probe.py")

    obs = FakeTensorDictLike(
        {
            "actor": FakeTensor((2, 104)),
            "critic": UserDict({"state": FakeTensor((2, 119))}),
            "scalar": 1.0,
        }
    )

    summary = module._obs_summary(FakeTorch, obs)

    assert summary == {
        "actor": {"shape": [2, 104], "finite": True},
        "critic.state": {"shape": [2, 119], "finite": True},
        "scalar": {"shape": [], "finite": True},
    }


def test_task038_probe_failure_summary_and_writer(tmp_path: Path) -> None:
    module = _load_src_tool("task038_mjlab_variant_env_load_probe.py")
    args = module.parse_args(
        [
            "--task",
            "Unitree-G1-Gripper-Flat-Task038-HeldoutAssetSmoke",
            "--expected-xml-path",
            "/tmp/heldout.xml",
            "--output-json",
            str(tmp_path / "summary.json"),
            "--device",
            "cpu",
        ]
    )

    summary = module.build_failure_summary(args, RuntimeError("missing mjlab"))
    module.write_json_summary(args.output_json, summary)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert loaded["pass"] is False
    assert loaded["zero_step_ok"] is False
    assert loaded["variant_label"] == "heldout"
    assert loaded["expected_xml_path"] == "/tmp/heldout.xml"
    assert loaded["registered_xml_path"] is None
    assert loaded["xml_path_matches_expected"] is False
    assert "RuntimeError" in loaded["error"]
    assert loaded["expected_action_dim"] == 31


def test_task038_probe_failure_summary_records_resolved_registered_xml(
    monkeypatch,
) -> None:
    module = _load_src_tool("task038_mjlab_variant_env_load_probe.py")
    args = module.parse_args(
        [
            "--task",
            "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke",
            "--expected-xml-path",
            "/tmp/train.xml",
            "--output-json",
            "summary.json",
        ]
    )
    monkeypatch.setattr(
        module,
        "resolve_registered_xml_path",
        lambda task: ("/tmp/train.xml", None),
    )

    summary = module.build_failure_summary(args, RuntimeError("late failure"))

    assert summary["registered_xml_path"] == "/tmp/train.xml"
    assert summary["xml_path_matches_expected"] is True
    assert summary["xml_resolution_error"] is None
    assert summary["pass"] is False


def test_task038_docs_do_not_claim_h200_runner_eval_or_reproduction_pass() -> None:
    task_doc = (TASK038_DIR / "009-mjlab-variant-env-load-smoke.md").read_text(
        encoding="utf-8"
    )
    task_md = (TASK038_DIR / "task.md").read_text(encoding="utf-8")
    combined = task_doc + "\n" + task_md

    forbidden = (
        "runner pass",
        "eval pass",
        "reproduction pass",
        "Status: passed",
        "H200 pass",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "Status: closed for the `009` env-load-only slice." in task_doc
    assert "No runner, eval, video, reproduction, or TXL superiority claim" in task_doc


def _load_task_script(name: str):
    path = TASK038_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_src_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "src/h200_locomotion_lab/tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryPath:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_text(self, encoding: str) -> str:
        assert encoding == "utf-8"
        return self.text

    def write_text(self, text: str, encoding: str) -> None:
        assert encoding == "utf-8"
        self.text = text


class FakeTensorDictLike:
    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._data),)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __getitem__(self, key):
        return self._data[key]


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], finite: bool = True) -> None:
        self.shape = shape
        self.finite = finite

    def detach(self):
        return self

    def float(self):
        return self


class FakeFiniteTensor:
    def __init__(self, finite: bool) -> None:
        self.finite = finite

    def all(self):
        return FakeScalar(self.finite)


class FakeScalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def item(self):
        return self.value


class FakeTorch:
    @staticmethod
    def is_tensor(value) -> bool:
        return isinstance(value, FakeTensor)

    @staticmethod
    def as_tensor(value):
        if isinstance(value, FakeTensor):
            return value
        if isinstance(value, (int, float, bool)):
            return FakeTensor(())
        raise TypeError(f"cannot convert {type(value).__name__}")

    @staticmethod
    def isfinite(value):
        if not isinstance(value, FakeTensor):
            raise TypeError(f"not a tensor: {type(value).__name__}")
        return FakeFiniteTensor(value.finite)


def _passing_summary() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke",
        "expected_xml_path": "/tmp/train.xml",
        "registered_xml_path": "/tmp/train.xml",
        "xml_path_matches_expected": True,
        "expected_action_dim": 31,
        "action_dim": 31,
        "total_action_dim": 31,
        "zero_step_ok": True,
        "obs": {"actor": {"shape": [8, 100], "finite": True}},
        "obs_all_finite": True,
    }
