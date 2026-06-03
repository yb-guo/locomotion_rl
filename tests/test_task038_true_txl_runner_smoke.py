import importlib.util
from pathlib import Path


TASK038_DIR = (
    Path(__file__).resolve().parents[1]
    / ".agent"
    / "task"
    / "task038-locoformer-min-g1like-reproduction"
)


def test_task038_patcher_adds_true_txl_runner_smoke_tasks_idempotently() -> None:
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
    assert module.TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID in once
    assert module.HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID in once
    assert once.count(module.TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID) == 1
    assert once.count(module.HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID) == 1
    assert once.count("Task038TrueTxlMemoryK160Runner") == 3
    assert once.count("runner_cls=Task038TrueTxlMemoryK160Runner") == 2
    assert once.count("Task037TxlMemoryK160DeterministicRunner") == 3


def test_task038_patcher_extends_existing_runner_import_block_idempotently() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")
    init = MemoryPath(
        "from mjlab.tasks.registry import register_mjlab_task\n"
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from h200_locomotion_lab.training.rsl_history_wrapper   import(\n"
        "  Task037TxlMemoryK160DeterministicRunner,\n"
        "  Task037OtherRunner,\n"
        "  Task037AnotherRunner,\n"
        " )\n"
        "from .rl_cfg import unitree_g1_gripper_ppo_runner_cfg\n"
    )

    module.patch_init(init)
    once = init.read_text(encoding="utf-8")
    module.patch_init(init)
    twice = init.read_text(encoding="utf-8")

    assert once == twice
    runner_import = once.split("from h200_locomotion_lab.training.rsl_history_wrapper", 1)[
        1
    ].split("from .rl_cfg", 1)[0]
    assert runner_import.count("Task038TrueTxlMemoryK160Runner") == 1
    assert runner_import.count("Task037TxlMemoryK160DeterministicRunner") == 1
    assert "Task037OtherRunner,\n  Task037AnotherRunner," in runner_import
    assert once.count("runner_cls=Task038TrueTxlMemoryK160Runner") == 2


def test_task038_true_txl_runner_task_ids_are_not_task010_ids() -> None:
    module = _load_task_script("task038_register_mjlab_variant_assets.py")

    assert module.TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID == (
        "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    )
    assert module.HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID == (
        "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke"
    )
    assert module.TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID != module.TRAIN_RUNNER_SMOKE_TASK_ID
    assert module.HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID != module.HELDOUT_RUNNER_SMOKE_TASK_ID
    assert "Task038TrueTxlMemoryK160Runner" in module.REGISTER_BLOCKS[
        module.TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
    ]


def test_task038_true_txl_probe_parse_args_defaults() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.num_envs == 8
    assert args.steps == 2
    assert args.device == "cuda:0"


def test_task038_true_txl_probe_positive_pass_gate() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is True
    assert reasons == []


def test_task038_true_txl_probe_rejects_wrong_runner_or_model() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()
    summary["runner_cls"] = "Task037TxlMemoryK160DeterministicRunner"
    summary["actor_model_class"] = "Task037TxlStyleMemoryModel"

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "runner_cls_mismatch" in reasons
    assert "actor_model_class_mismatch" in reasons


def test_task038_true_txl_probe_rejects_missing_txl_debug() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()
    summary["txl_debug"] = {}

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_invalid" in reasons
    assert "txl_debug_no_incremental_steps" in reasons
    assert "txl_debug_no_previous_memory_exposure" in reasons


def test_task038_true_txl_probe_rejects_no_previous_memory_exposure() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()
    summary["txl_debug"]["envs"][0]["last_attended_previous_memory_lengths"] = [0, 0]
    summary["txl_debug"]["last_attended_previous_memory_lengths"] = [[0, 0]]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_no_previous_memory_exposure" in reasons


def test_task038_true_txl_probe_rejects_wrong_policy_shape() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()
    summary["policy_action_shape"] = [8, 30]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "policy_action_shape_mismatch" in reasons


def test_task038_true_txl_probe_rejects_overclaim_flags() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()
    summary["quality_claim"] = True
    summary["runner_smoke_only"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "claim_boundary_violation" in reasons


def test_task038_true_txl_probe_rejects_missing_step_extras_and_nonfinite_obs() -> None:
    module = _load_src_tool("task038_true_txl_runner_smoke_probe.py")
    summary = _passing_true_txl_summary()
    summary["step_required_extras_missing"] = ["trial_done"]
    summary["obs_all_finite"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "step_required_extras_missing" in reasons
    assert "obs_not_finite" in reasons


def test_task038_true_txl_doc_does_not_claim_training_eval_reproduction_or_superiority() -> None:
    doc = (TASK038_DIR / "011-true-txl-runner-consumer-smoke.md").read_text(encoding="utf-8")
    task_md = (TASK038_DIR / "task.md").read_text(encoding="utf-8")
    combined = doc + "\n" + task_md

    forbidden = (
        "true TXL reproduction passed",
        "LocoFormer reproduced",
        "policy quality passed",
        "eval passed",
        "TXL superiority passed",
        "training passed",
        "Status: passed",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "quality_claim:false" in doc
    assert "training_claim:false" in doc
    assert "eval_claim:false" in doc
    assert "runner_smoke_only:true" in doc
    assert "Status: closed by final reviewer confirmation" in doc
    assert "runner-consumer smoke only" in doc
    assert "locomotion" in doc
    assert "quality, training progress, evaluation success" in doc


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


def _passing_true_txl_summary() -> dict:
    return {
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "expected_runner_cls": "Task038TrueTxlMemoryK160Runner",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_action_dim": 31,
        "actual_num_envs": 8,
        "action_dim": 31,
        "total_action_dim": 31,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "steps": 2,
        "step_count": 2,
        "required_extras_missing": [],
        "step_required_extras_missing": [],
        "obs": {"actor_history": {"shape": [8, 16640], "finite": True}},
        "obs_all_finite": True,
        "txl_debug": {
            "num_layers": 2,
            "memory_len": 64,
            "segment_len": 16,
            "last_attended_previous_memory_lengths": [[16, 16]],
            "segments_appended": 16,
            "tokens_appended": 128,
            "envs": [
                {
                    "env_id": 0,
                    "memory_lengths": [32, 32],
                    "inner_reset_events": 0,
                    "outer_reset_events": 0,
                    "incremental_steps": 2,
                    "segments_appended": 2,
                    "tokens_appended": 32,
                    "last_attended_previous_memory_lengths": [16, 16],
                }
            ],
        },
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "runner_smoke_only": True,
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
