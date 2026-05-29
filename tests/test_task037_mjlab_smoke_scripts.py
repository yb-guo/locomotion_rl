import importlib.util
from pathlib import Path


TASK037_DIR = (
    Path(__file__).resolve().parents[1]
    / ".agent"
    / "task"
    / "task037-locoformer-style-multitrial-long-context-training"
)


def test_task037_registration_patch_is_idempotent() -> None:
    module = _load_script("task037_register_multitrial_stages.py")
    init_path = MemoryPath(
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from mjlab.tasks.registry import register_mjlab_task\n",
    )

    module.patch_init(init_path)
    once = init_path.read_text(encoding="utf-8")
    module.patch_init(init_path)
    twice = init_path.read_text(encoding="utf-8")

    task_id = "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0"
    deterministic_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0"
    )
    adapt_task_id = "Unitree-G1-Gripper-Flat-Task037-AdaptK4-DeterministicInnerReset-Fast2p0"
    txl_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DeterministicInnerReset-Fast2p0"
    )
    assert once == twice
    assert once.count("Task037BufferOnlyK4AutoResetRunner") == 2
    assert once.count("Task037BufferOnlyK4DeterministicInnerResetRunner") == 2
    assert once.count("Task037AdaptK4DeterministicInnerResetRunner") == 2
    assert once.count("Task037TxlMemoryK160DeterministicRunner") == 2
    assert once.count(task_id) == 1
    assert once.count(deterministic_task_id) == 1
    assert once.count(adapt_task_id) == 1
    assert once.count(txl_task_id) == 1


def test_task037_extras_probe_parse_args_defaults_to_task037_id() -> None:
    module = _load_src_tool("task037_mjlab_multitrial_extras_probe.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0"
    assert args.output_json == "out.json"
    assert args.episode_length_s > 0


def test_task037_inner_reset_probe_parse_args_defaults_to_deterministic_task_id() -> None:
    module = _load_src_tool("task037_mjlab_inner_reset_probe.py")

    args = module.parse_args(["--output-json", "inner.json"])

    assert args.task == (
        "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0"
    )
    assert args.output_json == "inner.json"
    assert args.max_command_delta > 0


def test_task037_multitrial_eval_parse_args_defaults_to_adapt_task_id() -> None:
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "eval.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task037-AdaptK4-DeterministicInnerReset-Fast2p0"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "eval.json"
    assert args.trial_length_s == 2.0
    assert args.min_final_completion_ratio > 0


def _load_script(name: str):
    path = TASK037_DIR / name
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
