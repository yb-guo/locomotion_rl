import importlib.util
import json
from pathlib import Path


TASK038_DIR = (
    Path(__file__).resolve().parents[1]
    / ".agent"
    / "task"
    / "task038-locoformer-min-g1like-reproduction"
)


def test_task038_patcher_adds_runner_smoke_tasks_idempotently() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")
    init = MemoryPath(
        "from mjlab.tasks.registry import register_mjlab_task\n"
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from .rl_cfg import unitree_g1_gripper_ppo_runner_cfg\n"
    )

    module.patch_init(init)
    once = init.read_text(encoding="utf-8")
    module.patch_init(init)
    twice = init.read_text(encoding="utf-8")

    assert once == twice
    assert module.TRAIN_RUNNER_SMOKE_TASK_ID in once
    assert module.HELDOUT_RUNNER_SMOKE_TASK_ID in once
    assert once.count(module.TRAIN_RUNNER_SMOKE_TASK_ID) == 1
    assert once.count(module.HELDOUT_RUNNER_SMOKE_TASK_ID) == 1
    assert once.count("Task037TxlMemoryK160DeterministicRunner") == 3
    assert once.count("runner_cls=Task037TxlMemoryK160DeterministicRunner") == 2
    assert once.count("unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg()") == 2
    assert once.count("unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg()") == 2


def test_task038_runner_smoke_task_ids_are_separate_from_env_load_ids() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")

    assert module.TRAIN_TASK_ID == "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke"
    assert module.HELDOUT_TASK_ID == "Unitree-G1-Gripper-Flat-Task038-HeldoutAssetSmoke"
    assert module.TRAIN_RUNNER_SMOKE_TASK_ID == (
        "Unitree-G1-Gripper-Flat-Task038-TrainRunnerSmoke"
    )
    assert module.HELDOUT_RUNNER_SMOKE_TASK_ID == (
        "Unitree-G1-Gripper-Flat-Task038-HeldoutRunnerSmoke"
    )
    assert module.REGISTER_BLOCKS[module.TRAIN_RUNNER_SMOKE_TASK_ID].count(
        "Task037TxlMemoryK160DeterministicRunner"
    ) == 1


def test_task038_runner_probe_parse_args_defaults() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task037TxlMemoryK160DeterministicRunner"
    assert args.num_envs == 8
    assert args.steps == 2
    assert args.device == "cuda:0"
    assert args.require_inner_outer_reset is False


def test_task038_runner_probe_positive_pass_gate() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is True
    assert reasons == []


def test_task038_runner_probe_pass_gate_blocks_overclaims_and_missing_policy() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()
    summary["policy_action_finite"] = False
    summary["quality_claim"] = True

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "policy_action_not_finite" in reasons
    assert "claim_boundary_violation" in reasons


def test_task038_runner_probe_blocks_wrong_policy_action_shape() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()
    summary["policy_action_shape"] = [8, 30]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "policy_action_shape_mismatch" in reasons

    summary = _passing_runner_summary()
    summary["policy_action_shape"] = [7, 31]
    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "policy_action_shape_mismatch" in reasons


def test_task038_runner_probe_blocks_zero_steps_and_no_step_count() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()
    summary["steps"] = 0
    summary["step_count"] = 0
    summary["zero_step_ok"] = True

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "no_steps_executed" in reasons


def test_task038_runner_probe_blocks_reset_only_required_extras() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()
    summary["required_extras_missing"] = []
    summary["step_required_extras_missing"] = ["trial_done", "episode_done"]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "step_required_extras_missing" in reasons


def test_task038_runner_probe_requires_done_consistency_checked_on_step() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()
    summary["done_episode_consistency_checked"] = False
    summary["done_matches_episode_done"] = True

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "done_extras_not_checked_on_step" in reasons


def test_task038_runner_probe_inner_outer_reset_only_required_when_requested() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    summary = _passing_runner_summary()
    summary["saw_inner_reset"] = False
    summary["saw_outer_reset"] = False

    passed, reasons = module.evaluate_probe_pass(summary)
    assert passed is True
    assert reasons == []

    summary["require_inner_outer_reset"] = True
    passed, reasons = module.evaluate_probe_pass(summary)
    assert passed is False
    assert "inner_reset_not_seen" in reasons
    assert "outer_reset_not_seen" in reasons


def test_task038_runner_probe_obs_summary_recurses_actor_critic_history() -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    obs = {
        "actor": FakeTensor((2, 104)),
        "critic": FakeTensor((2, 119)),
        "actor_history": FakeTensor((2, 160, 104)),
    }

    summary = module._obs_summary(FakeTorch, obs)

    assert summary == {
        "actor": {"shape": [2, 104], "finite": True},
        "critic": {"shape": [2, 119], "finite": True},
        "actor_history": {"shape": [2, 160, 104], "finite": True},
    }


def test_task038_runner_probe_failure_summary_and_writer(tmp_path: Path) -> None:
    module = _load_src_tool("task038_mjlab_runner_smoke_probe.py")
    args = module.parse_args(
        [
            "--task",
            "Unitree-G1-Gripper-Flat-Task038-HeldoutRunnerSmoke",
            "--output-json",
            str(tmp_path / "summary.json"),
            "--device",
            "cpu",
            "--require-inner-outer-reset",
        ]
    )

    summary = module.build_failure_summary(args, RuntimeError("missing mjlab"))
    module.write_json_summary(args.output_json, summary)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert loaded["pass"] is False
    assert loaded["variant_label"] == "heldout"
    assert loaded["zero_step_ok"] is False
    assert loaded["require_inner_outer_reset"] is True
    assert loaded["quality_claim"] is False
    assert loaded["training_claim"] is False
    assert loaded["eval_claim"] is False
    assert loaded["failure_reasons"] == ["probe_exception"]
    assert "RuntimeError" in loaded["error"]


def test_task038_runner_smoke_doc_does_not_overclaim() -> None:
    doc = (TASK038_DIR / "010-mjlab-runner-smoke.md").read_text(encoding="utf-8")
    task_md = (TASK038_DIR / "task.md").read_text(encoding="utf-8")
    combined = doc + "\n" + task_md

    forbidden = (
        "true TXL reproduction passed",
        "LocoFormer reproduced",
        "policy quality passed",
        "eval passed",
        "Status: passed",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "quality_claim:false" in doc
    assert "training_claim:false" in doc
    assert "eval_claim:false" in doc
    assert "Status: closed for the `010` runner-smoke-only slice." in doc
    assert "runner construction, one policy forward, and short" in doc


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


def _passing_runner_summary() -> dict:
    return {
        "runner_cls": "Task037TxlMemoryK160DeterministicRunner",
        "expected_runner_cls": "Task037TxlMemoryK160DeterministicRunner",
        "expected_action_dim": 31,
        "actual_num_envs": 8,
        "action_dim": 31,
        "total_action_dim": 31,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "steps": 2,
        "step_count": 2,
        "zero_step_ok": True,
        "required_extras_missing": [],
        "step_required_extras_missing": [],
        "done_episode_consistency_checked": True,
        "done_matches_episode_done": True,
        "obs": {"actor_history": {"shape": [8, 160, 104], "finite": True}},
        "obs_all_finite": True,
        "saw_inner_reset": False,
        "saw_outer_reset": False,
        "require_inner_outer_reset": False,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
    }


class MemoryPath:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_text(self, encoding: str) -> str:
        assert encoding == "utf-8"
        return self.text

    def write_text(self, text: str, encoding: str) -> None:
        assert encoding == "utf-8"
        self.text = text


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
