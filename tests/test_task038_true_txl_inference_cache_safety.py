from pathlib import Path

from h200_locomotion_lab.training import rsl_history_wrapper

ROOT = Path(__file__).resolve().parents[1]
TASK038_DIR = ROOT / ".agent" / "task" / "task038-locoformer-min-g1like-reproduction"


def test_task038_clone_inference_tensor_returns_mutable_clone() -> None:
    tensor = _FakeTensor([0, 1], inference=True)

    clone = rsl_history_wrapper._task038_clone_inference_tensor(tensor)
    clone[0] = 9

    assert tensor.is_inference() is True
    assert clone.is_inference() is False
    assert tensor.tolist() == [0, 1]
    assert clone.tolist() == [9, 1]


def test_task038_true_txl_append_clones_inference_cache_before_mutation() -> None:
    original_require_torch = rsl_history_wrapper._require_torch
    rsl_history_wrapper._require_torch = lambda: _FakeTorch
    try:
        model = _fake_model(num_envs=2, num_layers=1)
        model._memory_tensors = [_FakeTensor(_zeros((2, 4, 3)), inference=True)]
        model._memory_lengths = [_FakeTensor([0, 0], inference=True)]
        segment_tokens = _FakeTensor(
            [
                [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
                [[3.0, 3.1, 3.2], [4.0, 4.1, 4.2]],
            ],
            inference=False,
        )

        model._append_layer_memory(0, segment_tokens)
        model._append_layer_memory(0, segment_tokens)

        assert model._memory_tensors[0].is_inference() is False
        assert model._memory_lengths[0].is_inference() is False
        assert model._memory_lengths[0].tolist() == [4, 4]
    finally:
        rsl_history_wrapper._require_torch = original_require_torch


def test_task038_true_txl_reset_hooks_clone_inference_counters_and_lengths() -> None:
    original_require_torch = rsl_history_wrapper._require_torch
    rsl_history_wrapper._require_torch = lambda: _FakeTorch
    try:
        model = _fake_model(num_envs=2, num_layers=2)
        model._memory_lengths = [
            _FakeTensor([16, 16], inference=True),
            _FakeTensor([16, 16], inference=True),
        ]
        model._memory_tensors = [_FakeTensor(_zeros((2, 4, 3)), inference=True) for _ in range(2)]
        model._inner_reset_events = _FakeTensor([0, 0], inference=True)
        model._outer_reset_events = _FakeTensor([0, 0], inference=True)
        model._incremental_steps = _FakeTensor([3, 3], inference=True)
        model._segments_appended = _FakeTensor([3, 3], inference=True)
        model._tokens_appended = _FakeTensor([48, 48], inference=True)

        model.task038_txl_record_inner_reset(0)
        model.task038_txl_outer_reset(1)
        model.task038_txl_clear_memory(0)

        assert model._inner_reset_events.tolist() == [1, 0]
        assert model._outer_reset_events.tolist() == [0, 1]
        assert [lengths.tolist() for lengths in model._memory_lengths] == [[0, 0], [0, 0]]
        assert all(lengths.is_inference() is False for lengths in model._memory_lengths)
        assert model._inner_reset_events.is_inference() is False
        assert model._outer_reset_events.is_inference() is False
    finally:
        rsl_history_wrapper._require_torch = original_require_torch


def test_task038_true_txl_debug_snapshot_exposes_forward_counters() -> None:
    model = _fake_model(num_envs=2, num_layers=2)
    model._memory_lengths = [_FakeTensor([2, 3]), _FakeTensor([4, 5])]
    model._total_actor_forward_batches = 4
    model._total_actor_forward_samples = 40
    model._env_cache_stateful_forward_batches = 3
    model._env_cache_stateful_forward_samples = 24
    model._stateless_forward_batches = 1
    model._stateless_forward_samples = 16

    snapshot = model.txl_debug_snapshot()

    assert snapshot["total_actor_forward_batches"] == 4
    assert snapshot["total_actor_forward_samples"] == 40
    assert snapshot["env_cache_stateful_forward_batches"] == 3
    assert snapshot["env_cache_stateful_forward_samples"] == 24
    assert snapshot["stateless_forward_batches"] == 1
    assert snapshot["stateless_forward_samples"] == 16
    assert snapshot["stateless_fallback_forward_batches"] == 1
    assert snapshot["stateless_fallback_forward_samples"] == 16
    assert snapshot["envs"][0]["memory_lengths"] == [2, 4]


def test_task038_true_txl_inference_cache_safety_doc_does_not_overclaim() -> None:
    doc = (TASK038_DIR / "014-true-txl-inference-cache-safety-smoke.md").read_text(
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
        "heldout training passed",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "quality_claim:false" in doc
    assert "training_claim:false" in doc
    assert "eval_claim:false" in doc
    assert "reproduction_claim:false" in doc
    assert "superiority_claim:false" in doc
    assert "inference_cache_safety_smoke_only:true" in doc


def _fake_model(*, num_envs: int, num_layers: int):
    model = object.__new__(rsl_history_wrapper.Task038TrueTxlMemoryModel)
    model.memory_len = 4
    model.segment_len = 2
    model.num_layers = num_layers
    model.token_dim = 3
    model._memory_tensors = []
    model._memory_lengths = []
    model._inner_reset_events = _FakeTensor([0 for _ in range(num_envs)])
    model._outer_reset_events = _FakeTensor([0 for _ in range(num_envs)])
    model._incremental_steps = _FakeTensor([0 for _ in range(num_envs)])
    model._segments_appended = _FakeTensor([0 for _ in range(num_envs)])
    model._tokens_appended = _FakeTensor([0 for _ in range(num_envs)])
    model._last_attended_previous_memory_lengths = [
        [0 for _ in range(num_layers)] for _ in range(num_envs)
    ]
    model._total_actor_forward_batches = 0
    model._total_actor_forward_samples = 0
    model._env_cache_stateful_forward_batches = 0
    model._env_cache_stateful_forward_samples = 0
    model._stateless_forward_batches = 0
    model._stateless_forward_samples = 0
    return model


def _zeros(shape):
    if len(shape) == 1:
        return [0 for _ in range(shape[0])]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class _FakeScalar:
    def __init__(self, value) -> None:
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value

    def __iadd__(self, value):
        self.value += value
        return self


class _FakeIndexTensor:
    def __init__(self, values) -> None:
        self.values = [int(value) for value in values]

    def to(self, device=None, dtype=None):
        return self

    def flatten(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def __iter__(self):
        return iter(self.values)


class _FakeTensor:
    def __init__(self, data, *, inference: bool = False) -> None:
        self.data = _deepcopy(data)
        self._inference = inference
        self.device = "cpu"
        self.dtype = "float"

    @property
    def shape(self):
        return _shape(self.data)

    def is_inference(self):
        return self._inference

    def clone(self):
        return _FakeTensor(self.data, inference=False)

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.data

    def tolist(self):
        return _deepcopy(self.data)

    def sum(self):
        return _FakeScalar(_sum_data(self.data))

    def __getitem__(self, index):
        if isinstance(index, _FakeIndexTensor):
            return _FakeTensor([self.data[i] for i in index.values], inference=self._inference)
        if isinstance(index, tuple):
            env_id, token_slice = index
            return _FakeTensor(self.data[env_id][token_slice], inference=self._inference)
        value = self.data[index]
        if isinstance(value, list):
            return _FakeTensor(value, inference=self._inference)
        return _FakeScalar(value)

    def __setitem__(self, index, value) -> None:
        if self._inference:
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode")
        if isinstance(index, _FakeIndexTensor):
            values = value.tolist() if isinstance(value, _FakeTensor) else value
            if isinstance(values, list):
                for item, item_value in zip(index.values, values):
                    self.data[item] = item_value
            else:
                for item in index.values:
                    self.data[item] = values
            return
        if isinstance(index, tuple):
            env_id, token_slice = index
            rows = value.tolist() if isinstance(value, _FakeTensor) else value
            start, stop, step = token_slice.indices(len(self.data[env_id]))
            if step != 1:
                raise AssertionError("fake tensor only supports step=1 slices")
            self.data[env_id][start:stop] = rows
            return
        if isinstance(value, _FakeScalar):
            value = value.item()
        self.data[index] = value

    def __iadd__(self, value):
        if self._inference:
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode")
        self.data = _add_scalar(self.data, value)
        return self


class _FakeTorch:
    long = "long"

    @staticmethod
    def as_tensor(values, device=None, dtype=None):
        if isinstance(values, int):
            values = [values]
        return _FakeIndexTensor(values)

    @staticmethod
    def arange(count, device=None, dtype=None):
        return _FakeIndexTensor(range(count))

    @staticmethod
    def zeros_like(tensor):
        return _FakeTensor(_zeros(tensor.shape), inference=False)

    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        combined = []
        for tensor in tensors:
            combined.extend(tensor.tolist())
        return _FakeTensor(combined, inference=False)


def _shape(data):
    if isinstance(data, list):
        if not data:
            return (0,)
        return (len(data),) + _shape(data[0])
    return ()


def _deepcopy(data):
    if isinstance(data, list):
        return [_deepcopy(item) for item in data]
    return data


def _add_scalar(data, value):
    if isinstance(data, list):
        return [_add_scalar(item, value) for item in data]
    return data + value


def _sum_data(data):
    if isinstance(data, list):
        return sum(_sum_data(item) for item in data)
    return data
