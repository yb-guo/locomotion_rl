import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TASK040_DIR = ROOT / ".agent" / "task" / "task040-sequence-aware-txl-ppo-update"


def test_task040_sequence_txl_ppo_update_parse_args_defaults() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.num_envs == 8
    assert args.rollout_steps == 2
    assert args.iterations == 2
    assert args.num_mini_batches == 1
    assert args.device == "cuda:0"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.expected_algorithm_class == "Task040SequenceAwareTrueTxlPPO"


def test_task040_sequence_txl_ppo_update_mutates_agent_cfg_for_sequence_update() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    cfg = {
        "num_steps_per_env": 24,
        "max_iterations": 10001,
        "save_interval": 100,
        "logger": "wandb",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        },
    }

    mutated = module.mutate_agent_cfg_for_sequence_smoke(
        cfg,
        rollout_steps=2,
        iterations=1,
        seed=123,
        num_mini_batches=2,
    )

    assert mutated["num_steps_per_env"] == 2
    assert mutated["max_iterations"] == 1
    assert mutated["save_interval"] == 1000000
    assert mutated["logger"] == "tensorboard"
    assert mutated["upload_model"] is False
    assert mutated["resume"] is False
    assert mutated["algorithm"]["class_name"] == module.TASK040_ALGORITHM_CLASS
    assert mutated["algorithm"]["num_learning_epochs"] == 1
    assert mutated["algorithm"]["num_mini_batches"] == 2


def test_task040_sequence_txl_ppo_update_preflight_rejects_bad_batch_split() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    args = module.parse_args(
        [
            "--output-json",
            "out.json",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "3",
        ]
    )

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["num_envs_not_divisible_by_num_mini_batches"]
    else:
        raise AssertionError("expected bad batch split preflight rejection")


def test_task040_sequence_txl_ppo_update_installs_ipython_display_stub() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("IPython.display", None)

        module._install_ipython_display_stub()

        assert "IPython.display" in sys.modules
        display = sys.modules["IPython.display"]
        assert display.display("ignored") is None
        assert display.HTML("ignored")._repr_html_() == ""


def test_task040_sequence_txl_ppo_update_installs_wandb_stub() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("wandb", None)

        module._install_wandb_stub()

        assert "wandb" in sys.modules
        wandb = sys.modules["wandb"]
        assert wandb.run is None
        assert wandb.save("ignored") is None


def test_task040_sequence_txl_ppo_update_installs_wcwidth_stub() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("wcwidth", None)

        module._install_wcwidth_stub()

        assert "wcwidth" in sys.modules
        wcwidth = sys.modules["wcwidth"]
        assert wcwidth.wcwidth("x") == 1
        assert wcwidth.wcswidth("abc") == 3


def test_task040_sequence_txl_ppo_update_main_rejects_bad_batch_split() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    argv = [
        "task040_sequence_txl_ppo_update_smoke.py",
        "--output-json",
        "out.json",
        "--num-envs",
        "8",
        "--num-mini-batches",
        "3",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module, "run_probe") as run_probe,
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    run_probe.assert_not_called()
    summary = write_json_summary.call_args.args[1]
    assert summary["preflight_rejected"] is True
    assert summary["failure_reasons"] == ["num_envs_not_divisible_by_num_mini_batches"]


def test_task040_sequence_txl_ppo_update_positive_pass_gate() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")

    passed, reasons = module.evaluate_probe_pass(_passing_summary())

    assert passed is True
    assert reasons == []


def test_task040_sequence_txl_ppo_update_rejects_stateless_fallback() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["txl_debug"]["stateless_fallback_forward_batches"] = 1
    summary["txl_debug"]["stateless_fallback_forward_samples"] = 16

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_stateless_fallback_seen" in reasons
    assert "txl_debug_stateless_fallback_samples_seen" in reasons


def test_task040_sequence_txl_ppo_update_requires_sequence_counters_and_loss() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["txl_debug"]["sequence_update_forward_batches"] = 0
    summary["algorithm_debug"]["sequence_update_batches"] = 0
    summary["algorithm_debug"]["last_loss_dict"] = {}

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_no_sequence_update_forward" in reasons
    assert "algorithm_debug_no_sequence_update_batches" in reasons
    assert "algorithm_debug_missing_loss_dict" in reasons


def test_task040_sequence_txl_ppo_update_requires_logprob_parity() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["algorithm_debug"]["last_logprob_parity"]["max_logprob_abs_error"] = 2e-5
    summary["algorithm_debug"]["last_logprob_parity"]["pass"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "algorithm_debug_logprob_parity_failed" in reasons
    assert "algorithm_debug_logprob_error_too_high" in reasons


def test_task040_sequence_txl_ppo_update_requires_nonempty_rollout_start_memory() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["algorithm_debug"]["last_logprob_parity"]["rollout_start_memory_non_empty"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "algorithm_debug_rollout_start_memory_empty" in reasons


def test_task040_sequence_logprob_parity_uses_per_env_rollout_start_memory() -> None:
    torch = _import_torch()
    wrapper = _load_training_module("rsl_history_wrapper.py")
    alg = object.__new__(wrapper.Task040SequenceAwareTrueTxlPPO)
    alg.actor = _ParityActor(torch)
    alg.num_mini_batches = 2
    alg.device = "cpu"
    storage = _parity_storage(torch)
    rollout_start_state = {
        "mode": "rollout_start_snapshot",
        "snapshot_id": 1,
        "non_empty": True,
        "memory_tensors": [torch.zeros((4, 2, 1))],
        "memory_lengths": [torch.tensor([2, 4, 1, 3])],
    }
    storage.actions_log_prob = _expected_parity_log_probs(
        torch,
        initial_lengths=rollout_start_state["memory_lengths"][0],
        dones=storage.dones,
        actions=storage.actions,
    )

    diagnostics = alg._task040_logprob_parity_diagnostics(storage, rollout_start_state, torch)
    zero_state = {
        **rollout_start_state,
        "non_empty": False,
        "memory_lengths": [torch.zeros(4, dtype=torch.long)],
    }
    zero_diagnostics = alg._task040_logprob_parity_diagnostics(storage, zero_state, torch)

    assert diagnostics["pass"] is True
    assert diagnostics["rollout_start_memory_non_empty"] is True
    assert diagnostics["non_empty_slice_count"] == 2
    assert diagnostics["max_logprob_abs_error"] <= 1e-5
    assert diagnostics["max_ratio_abs_error"] <= 1e-5
    assert zero_diagnostics["pass"] is False
    assert zero_diagnostics["max_logprob_abs_error"] > 1.0


def test_task040_sequence_logprob_parity_uses_per_step_actor_normalizer_state() -> None:
    torch = _import_torch()
    wrapper = _load_training_module("rsl_history_wrapper.py")
    alg = object.__new__(wrapper.Task040SequenceAwareTrueTxlPPO)
    alg.actor = _ParityActor(torch)
    alg.num_mini_batches = 2
    alg.device = "cpu"
    storage = _parity_storage(torch)
    rollout_start_state = {
        "mode": "rollout_start_snapshot",
        "snapshot_id": 1,
        "non_empty": True,
        "memory_tensors": [torch.zeros((4, 2, 1))],
        "memory_lengths": [torch.tensor([2, 4, 1, 3])],
    }
    normalizer_biases = [0.0, 3.0, -2.0]
    normalizer_states = [
        {"mean": torch.tensor([[bias]], dtype=torch.float32), "std": torch.ones(1, 1), "eps": 0.0}
        for bias in normalizer_biases
    ]
    storage.actions_log_prob = _expected_parity_log_probs(
        torch,
        initial_lengths=rollout_start_state["memory_lengths"][0],
        dones=storage.dones,
        actions=storage.actions,
        normalizer_biases=normalizer_biases,
    )

    diagnostics = alg._task040_logprob_parity_diagnostics(
        storage,
        rollout_start_state,
        torch,
        normalizer_states=normalizer_states,
    )
    live_normalizer_diagnostics = alg._task040_logprob_parity_diagnostics(
        storage,
        rollout_start_state,
        torch,
    )

    assert diagnostics["pass"] is True
    assert diagnostics["normalizer_replay_mode"] == "per_step_snapshot"
    assert diagnostics["normalizer_snapshot_count"] == 3
    assert diagnostics["max_logprob_abs_error"] <= 1e-5
    assert live_normalizer_diagnostics["pass"] is False
    assert live_normalizer_diagnostics["normalizer_replay_mode"] == "live_actor_normalizer"
    assert live_normalizer_diagnostics["max_logprob_abs_error"] > 1.0


def test_task040_sequence_step_count_accepts_tensor_and_tensordict_shapes() -> None:
    torch = _import_torch()
    wrapper = _load_training_module("rsl_history_wrapper.py")

    assert wrapper._task040_sequence_step_count(torch.zeros((3, 2, 1))) == 3
    assert wrapper._task040_sequence_step_count(SimpleNamespace(batch_size=(4, 2))) == 4


def test_task040_sequence_actor_head_replays_rollout_batch_shape_per_step() -> None:
    torch = _import_torch()
    wrapper = _load_training_module("rsl_history_wrapper.py")
    actor = object.__new__(wrapper.Task038TrueTxlMemoryModel)
    actor.mlp = _ShapeRecordingMlp(torch)
    obs = torch.zeros((3, 2, 1))
    latents = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    output = actor._task040_forward_sequence_mlp_per_rollout_step(obs, latents)

    assert actor.mlp.calls == [(2, 2), (2, 2), (2, 2)]
    expected = torch.cat(
        (
            latents[:2].sum(dim=-1, keepdim=True) + 1.0,
            latents[2:4].sum(dim=-1, keepdim=True) + 2.0,
            latents[4:].sum(dim=-1, keepdim=True) + 3.0,
        ),
        dim=0,
    )
    assert torch.equal(output, expected)


def test_task040_attention_backend_configuration_disables_mismatched_fastpaths() -> None:
    wrapper = _load_training_module("rsl_history_wrapper.py")
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mha=_FakeMhaBackend(),
            cuda=_FakeCudaBackend(),
            cudnn=SimpleNamespace(allow_tf32=True),
        )
    )

    settings = wrapper._task040_configure_attention_replay_determinism(fake_torch)

    assert settings["mha_fastpath_before"] is True
    assert settings["mha_fastpath_after"] is False
    assert settings["flash_sdp_after"] is False
    assert settings["mem_efficient_sdp_after"] is False
    assert settings["cudnn_sdp_after"] is False
    assert settings["math_sdp_after"] is True
    assert settings["cuda_matmul_allow_tf32_after"] is False
    assert settings["cudnn_allow_tf32_after"] is False


def test_task040_sequence_txl_ppo_update_rejects_overclaim_flags() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["quality_claim"] = True
    summary["training_claim"] = True
    summary["eval_claim"] = True
    summary["reproduction_claim"] = True
    summary["superiority_claim"] = True
    summary["sequence_aware_ppo_update_smoke_only"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "claim_boundary_violation" in reasons


def test_task040_sequence_txl_ppo_update_writer_and_failure_summary() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    output = ROOT / ".agent" / "tmp" / "task040_writer_summary.json"
    summary = _passing_summary()

    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "write_text", return_value=None) as write_text,
    ):
        module.write_json_summary(output, summary)

        assert summary["json_path"] == str(output.resolve())
        mkdir.assert_called_once()
        write_text.assert_called_once()

    args = module.parse_args(
        ["--output-json", str(output), "--log-dir", str(ROOT / ".agent" / "tmp" / "logs")]
    )
    failure = module.build_failure_summary(args, RuntimeError("boom"))
    assert failure["pass"] is False
    assert failure["learn_returned"] is False
    assert failure["sequence_aware_ppo_update_smoke_only"] is True
    assert failure["failure_reasons"] == ["probe_exception"]


def test_task040_docs_define_sequence_update_acceptance_without_overclaim() -> None:
    docs = [
        TASK040_DIR / "task.md",
        TASK040_DIR / "001-rsl-storage-and-update-contract.md",
        TASK040_DIR / "002-sequence-aware-txl-actor-forward.md",
        TASK040_DIR / "003-sequence-aware-ppo-update-smoke.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    assert "stateless_fallback_forward_batches == 0" in combined
    assert "sequence_update_forward_batches > 0" in combined
    assert "quality_claim:false" in combined
    assert "training_claim:false" in combined
    assert "eval_claim:false" in combined
    assert "reproduction_claim:false" in combined
    assert "superiority_claim:false" in combined
    assert "Status: passed" not in combined


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_training_module(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "training" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _import_torch():
    import pytest

    return pytest.importorskip("torch")


class _ParityActor:
    def __init__(self, torch) -> None:
        self.torch = torch
        self._log_prob = None

    def task040_forward_sequence(
        self,
        obs,
        *,
        reset_mask,
        initial_memory_state,
        normalizer_states=None,
        stochastic_output,
    ):
        del stochastic_output
        lengths = initial_memory_state["memory_lengths"][0].clone().to(dtype=self.torch.float32)
        rows = []
        for step in range(int(obs.shape[0])):
            reset = reset_mask[step]
            lengths[reset] = 0.0
            normalizer_bias = 0.0
            if normalizer_states is not None:
                state = normalizer_states[step]
                if state is not None:
                    normalizer_bias = state["mean"].reshape(-1)[0].to(dtype=self.torch.float32)
            rows.append(obs[step, :, 0] + lengths + normalizer_bias)
            lengths += 1.0
        self._log_prob = self.torch.stack(rows, dim=0).reshape(-1)
        return obs.reshape(int(obs.shape[0]) * int(obs.shape[1]), -1)

    def get_output_log_prob(self, actions):
        del actions
        assert self._log_prob is not None
        return self._log_prob


class _FakeMhaBackend:
    def __init__(self) -> None:
        self.enabled = True

    def get_fastpath_enabled(self) -> bool:
        return self.enabled

    def set_fastpath_enabled(self, value: bool) -> None:
        self.enabled = bool(value)


class _FakeCudaBackend:
    def __init__(self) -> None:
        self.flash = True
        self.mem_efficient = True
        self.cudnn = True
        self.math = False
        self.matmul = SimpleNamespace(allow_tf32=True)

    def flash_sdp_enabled(self) -> bool:
        return self.flash

    def enable_flash_sdp(self, value: bool) -> None:
        self.flash = bool(value)

    def mem_efficient_sdp_enabled(self) -> bool:
        return self.mem_efficient

    def enable_mem_efficient_sdp(self, value: bool) -> None:
        self.mem_efficient = bool(value)

    def cudnn_sdp_enabled(self) -> bool:
        return self.cudnn

    def enable_cudnn_sdp(self, value: bool) -> None:
        self.cudnn = bool(value)

    def math_sdp_enabled(self) -> bool:
        return self.math

    def enable_math_sdp(self, value: bool) -> None:
        self.math = bool(value)


class _ShapeRecordingMlp:
    def __init__(self, torch) -> None:
        self.torch = torch
        self.calls = []

    def __call__(self, values):
        self.calls.append(tuple(values.shape))
        return values.sum(dim=-1, keepdim=True) + float(len(self.calls))


def _parity_storage(torch):
    observations = torch.tensor(
        [
            [[10.0], [20.0], [30.0], [40.0]],
            [[11.0], [21.0], [31.0], [41.0]],
            [[12.0], [22.0], [32.0], [42.0]],
        ]
    )
    actions = observations.clone()
    dones = torch.tensor(
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ]
    )
    return SimpleNamespace(
        num_envs=4,
        observations=observations,
        actions=actions,
        actions_log_prob=torch.zeros((3, 4)),
        dones=dones,
    )


def _expected_parity_log_probs(torch, *, initial_lengths, dones, actions, normalizer_biases=None):
    reset_mask = torch.zeros_like(dones, dtype=torch.bool)
    reset_mask[1:] = dones[:-1]
    lengths = initial_lengths.clone().to(dtype=torch.float32)
    rows = []
    for step in range(int(actions.shape[0])):
        lengths[reset_mask[step]] = 0.0
        normalizer_bias = 0.0 if normalizer_biases is None else float(normalizer_biases[step])
        rows.append(actions[step, :, 0] + lengths + normalizer_bias)
        lengths += 1.0
    return torch.stack(rows, dim=0)


def _passing_summary() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke",
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "expected_runner_cls": "Task038TrueTxlMemoryK160Runner",
        "algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "expected_algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_action_dim": 31,
        "actual_num_envs": 8,
        "num_envs": 8,
        "rollout_steps": 2,
        "iterations": 2,
        "num_mini_batches": 1,
        "action_dim": 31,
        "total_action_dim": 31,
        "learn_returned": True,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "txl_debug": {
            "stateless_fallback_forward_batches": 0,
            "stateless_fallback_forward_samples": 0,
            "sequence_update_forward_batches": 1,
            "sequence_update_forward_samples": 16,
        },
        "algorithm_debug": {
            "sequence_update_batches": 2,
            "sequence_update_samples": 32,
            "sequence_update_steps": 4,
            "last_loss_dict": {"value": 1.0, "surrogate": 0.0, "entropy": 0.5},
            "last_logprob_parity": {
                "mode": "rollout_start_snapshot",
                "rollout_start_memory_non_empty": True,
                "non_empty_slice_count": 1,
                "slice_count": 1,
                "sample_count": 16,
                "max_logprob_abs_error": 0.0,
                "max_ratio_abs_error": 0.0,
                "threshold": 1e-5,
                "pass": True,
            },
        },
        "log_dir_exists": True,
        "wall_time_s": 1.5,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "sequence_aware_ppo_update_smoke_only": True,
    }
