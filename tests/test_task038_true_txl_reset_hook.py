import importlib.util
from pathlib import Path

from h200_locomotion_lab.training import rsl_history_wrapper

ROOT = Path(__file__).resolve().parents[1]
TASK038_DIR = ROOT / ".agent" / "task" / "task038-locoformer-min-g1like-reproduction"


def test_task038_reset_hook_wrapper_dispatches_inner_and_outer_without_touching_task037() -> None:
    rsl_history_wrapper._require_torch = lambda: _FakeTorch
    env = _FakeEnv(
        [
            {
                "inner_reset": [True, False],
                "task037_outer_reset": [False, True],
            }
        ]
    )
    actor = _FakeActor(num_envs=2, num_layers=2)
    wrapper = rsl_history_wrapper.Task038TrueTxlResetHookVecEnvWrapper(env)
    wrapper.attach_task038_txl_actor(actor)

    wrapper.step([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert actor.inner_reset_calls == [[0]]
    assert actor.outer_reset_calls == [[1]]
    assert actor.memory_lengths[0] == [16, 16]
    assert actor.memory_lengths[1] == [0, 0]
    assert env.step_calls == 1


def test_task038_reset_hook_wrapper_full_reset_clears_existing_actor_cache_without_event() -> None:
    env = _FakeEnv([])
    actor = _FakeActor(num_envs=2, num_layers=2)
    wrapper = rsl_history_wrapper.Task038TrueTxlResetHookVecEnvWrapper(env)
    wrapper.attach_task038_txl_actor(actor)

    wrapper.reset()

    assert actor.clear_calls == [None]
    assert actor.outer_reset_calls == []
    assert actor.memory_lengths == [[0, 0], [0, 0]]


def test_task038_reset_hook_probe_parse_args_defaults() -> None:
    module = _load_src_tool("task038_true_txl_reset_hook_probe.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.num_envs == 8
    assert args.steps == 96
    assert args.episode_length_s == 0.15
    assert args.device == "cuda:0"


def test_task038_reset_hook_probe_positive_pass_gate() -> None:
    module = _load_src_tool("task038_true_txl_reset_hook_probe.py")

    passed, reasons = module.evaluate_probe_pass(_passing_summary())

    assert passed is True
    assert reasons == []


def test_task038_reset_hook_probe_rejects_missing_reset_events() -> None:
    module = _load_src_tool("task038_true_txl_reset_hook_probe.py")
    summary = _passing_summary()
    summary["saw_inner_reset"] = False
    summary["actor_outer_reset_events_total"] = 0

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "inner_reset_not_seen" in reasons
    assert "actor_outer_reset_event_missing" in reasons


def test_task038_reset_hook_probe_rejects_inner_memory_clear_false_pass() -> None:
    module = _load_src_tool("task038_true_txl_reset_hook_probe.py")
    summary = _passing_summary()
    summary["inner_reset_preserved_memory_before_next_policy"] = False
    summary["inner_reset_examples"] = [{"env_id": 0, "before": [16, 16], "after": [0, 0]}]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "inner_reset_memory_not_preserved" in reasons


def test_task038_reset_hook_probe_rejects_outer_memory_not_clear_false_pass() -> None:
    module = _load_src_tool("task038_true_txl_reset_hook_probe.py")
    summary = _passing_summary()
    summary["outer_reset_cleared_memory_before_next_policy"] = False
    summary["outer_reset_examples"] = [{"env_id": 1, "before": [16, 16], "after": [16, 16]}]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "outer_reset_memory_not_cleared" in reasons


def test_task038_reset_hook_probe_rejects_wrong_runner_model_and_claims() -> None:
    module = _load_src_tool("task038_true_txl_reset_hook_probe.py")
    summary = _passing_summary()
    summary["runner_cls"] = "Task037TxlMemoryK160DeterministicRunner"
    summary["actor_model_class"] = "Task037TxlStyleMemoryModel"
    summary["quality_claim"] = True
    summary["reproduction_claim"] = True
    summary["reset_hook_integration_smoke_only"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "runner_cls_mismatch" in reasons
    assert "actor_model_class_mismatch" in reasons
    assert "claim_boundary_violation" in reasons


def test_task038_reset_hook_docs_do_not_overclaim() -> None:
    doc = (TASK038_DIR / "012-true-txl-reset-hook-integration-smoke.md").read_text(
        encoding="utf-8"
    )
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
    assert "reproduction_claim:false" in doc
    assert "superiority_claim:false" in doc
    assert "reset_hook_integration_smoke_only:true" in doc
    assert "makes no training/eval/quality/reproduction/superiority claim" in doc


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_summary() -> dict:
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
        "step_count": 12,
        "required_extras_missing": [],
        "step_required_extras_missing": [],
        "obs": {"actor_history": {"shape": [8, 16640], "finite": True}},
        "obs_all_finite": True,
        "saw_inner_reset": True,
        "saw_outer_reset": True,
        "inner_reset_preserved_memory_before_next_policy": True,
        "outer_reset_cleared_memory_before_next_policy": True,
        "inner_reset_examples": [{"env_id": 0, "before": [16, 16], "after": [16, 16]}],
        "outer_reset_examples": [{"env_id": 1, "before": [16, 16], "after": [0, 0]}],
        "actor_inner_reset_events_total": 1,
        "actor_outer_reset_events_total": 1,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "reset_hook_integration_smoke_only": True,
    }


class _FakeEnv:
    def __init__(self, step_extras: list[dict]) -> None:
        self.num_envs = 2
        self.device = "cpu"
        self.max_episode_length = 5
        self.num_actions = 3
        self.cfg = object()
        self.unwrapped = self
        self.episode_length_buf = [0, 0]
        self._step_extras = step_extras
        self.step_calls = 0

    def reset(self):
        return self.get_observations(), {}

    def get_observations(self):
        return {"actor": [[0.0] * 4, [0.0] * 4], "critic": [[0.0] * 4, [0.0] * 4]}

    def step(self, actions):
        extras = self._step_extras[min(self.step_calls, len(self._step_extras) - 1)]
        self.step_calls += 1
        return self.get_observations(), [0.0, 0.0], [False, False], extras

    def close(self) -> None:
        pass

    def seed(self, seed: int = -1) -> int:
        return seed


class _FakeActor:
    def __init__(self, num_envs: int, num_layers: int) -> None:
        self.memory_lengths = [[16 for _ in range(num_layers)] for _ in range(num_envs)]
        self.inner_reset_calls: list[list[int]] = []
        self.outer_reset_calls: list[list[int]] = []
        self.clear_calls: list[object] = []

    def task038_txl_record_inner_reset(self, env_ids) -> None:
        self.inner_reset_calls.append([int(value) for value in env_ids.detach().cpu().tolist()])

    def task038_txl_outer_reset(self, env_ids) -> None:
        ids = [int(value) for value in env_ids.detach().cpu().tolist()]
        self.outer_reset_calls.append(ids)
        for env_id in ids:
            self.memory_lengths[env_id] = [0 for _ in self.memory_lengths[env_id]]

    def task038_txl_clear_memory(self, env_ids=None) -> None:
        self.clear_calls.append(env_ids)
        ids = range(len(self.memory_lengths)) if env_ids is None else env_ids
        for env_id in ids:
            self.memory_lengths[int(env_id)] = [0 for _ in self.memory_lengths[int(env_id)]]


class _FakeScalar:
    def __init__(self, value) -> None:
        self.value = value

    def item(self):
        return self.value


class _FakeTensor:
    def __init__(self, values, *, dtype=None) -> None:
        if isinstance(values, _FakeTensor):
            values = values.values
        self.values = list(values)
        self.dtype = dtype

    @property
    def shape(self):
        return (len(self.values),)

    def to(self, device=None, dtype=None):
        if dtype == _FakeTorch.bool:
            return _FakeTensor([bool(value) for value in self.values], dtype=dtype)
        if dtype == _FakeTorch.long:
            return _FakeTensor([int(value) for value in self.values], dtype=dtype)
        return _FakeTensor(self.values, dtype=dtype or self.dtype)

    def __or__(self, other):
        return _FakeTensor([bool(a) or bool(b) for a, b in zip(self.values, other.values)])

    def any(self):
        return _FakeScalar(any(bool(value) for value in self.values))

    def nonzero(self, as_tuple=False):
        return _FakeTensor([idx for idx, value in enumerate(self.values) if bool(value)])

    def flatten(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeTorch:
    bool = "bool"
    long = "long"

    @staticmethod
    def as_tensor(values, device=None, dtype=None):
        if dtype == _FakeTorch.bool:
            values = [bool(value) for value in values]
        elif dtype == _FakeTorch.long:
            values = [int(value) for value in values]
        return _FakeTensor(values, dtype=dtype)
