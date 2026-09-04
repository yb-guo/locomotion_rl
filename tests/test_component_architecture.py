from __future__ import annotations

import ast
from pathlib import Path

import pytest

from h200_locomotion_lab.algorithms.ppo import compute_gae as canonical_compute_gae
from h200_locomotion_lab.core.rl import (
    AlgorithmSpec,
    CompositionError,
    PolicyOutput,
    PolicySpec,
    TaskSpec,
    TaskStep,
    TensorSpace,
    UpdateReport,
    validate_composition,
)
from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingVectorizedEnv as legacy_g1_task,
)
from h200_locomotion_lab.experiments.config import (
    CONFIG_ROOT,
    ComponentConfigError,
    load_algorithm,
    load_experiment,
    load_policy,
    load_task,
)
from h200_locomotion_lab.experiments.loop import run_interaction
from h200_locomotion_lab.policies.tanh_gaussian_actor_critic import (
    build_tanh_gaussian_actor_critic,
)
from h200_locomotion_lab.tasks.g1_velocity_tracking import (
    G1VelocityTrackingVectorizedEnv as canonical_g1_task,
)
from h200_locomotion_lab.training.ppo_loop import (
    build_actor_critic as legacy_actor_critic,
)
from h200_locomotion_lab.training.ppo_loop import compute_gae as legacy_compute_gae

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "h200_locomotion_lab"


def test_default_experiment_composes_independent_components() -> None:
    experiment = load_experiment()

    assert experiment.name == "unitree_g1_flat_ppo"
    assert experiment.task.name == "unitree_g1_flat_walk"
    assert experiment.task.observation("policy").flat_dim == 98
    assert experiment.task.observation("value").flat_dim == 113
    assert experiment.task.action.flat_dim == 29
    assert experiment.policy.family == "gaussian_actor_critic"
    assert experiment.policy.capabilities == {"sample", "log_prob", "value"}
    assert experiment.algorithm.family == "clipped_policy_gradient"
    assert experiment.algorithm.interaction == "on_policy"
    assert experiment.runtime.backend == "mjlab_mujoco_warp"
    assert experiment.runtime.device == "cuda:0"
    assert experiment.runtime.num_envs == 4096


def test_each_component_loads_without_its_peers() -> None:
    task = load_task()
    policy = load_policy()
    algorithm = load_algorithm()

    assert task.action.flat_dim == 29
    assert policy.parameters["actor_hidden_dims"] == [512, 256, 128]
    assert algorithm.parameters["rollout_steps"] == 24

    policy_text = (CONFIG_ROOT / "policies" / "mlp_gaussian_actor_critic.yaml").read_text()
    algorithm_text = (CONFIG_ROOT / "algorithms" / "ppo.yaml").read_text()
    for duplicated_dimension in ("obs_dim", "observation_dim", "action_dim"):
        assert duplicated_dimension not in policy_text
        assert duplicated_dimension not in algorithm_text


def test_component_loader_rejects_cross_owned_structural_keys(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    bad_policy = policy_dir / "bad.yaml"
    bad_policy.write_text(
        """policy:
  name: coupled_policy
  family: gaussian
  capabilities: [sample]
  reward_scale: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ComponentConfigError, match="unknown keys: reward_scale"):
        load_policy("policies/bad.yaml", config_root=tmp_path)


def test_algorithm_policy_compatibility_is_capability_based() -> None:
    task = TaskSpec(
        name="walk",
        observations={"policy": TensorSpace((8,))},
        action=TensorSpace((2,)),
        max_episode_steps=100,
    )
    implicit_policy = PolicySpec(
        name="jit_x0",
        family="manifold_flow",
        capabilities=frozenset({"sample"}),
    )
    ppo = AlgorithmSpec(
        name="ppo",
        family="clipped_policy_gradient",
        interaction="on_policy",
        required_policy_capabilities=frozenset({"sample", "log_prob", "value"}),
    )

    with pytest.raises(CompositionError, match="log_prob, value"):
        validate_composition(task, implicit_policy, ppo)


def test_interaction_loop_does_not_interpret_task_or_algorithm_metrics() -> None:
    task = _FakeTask()
    policy = _FakePolicy()
    algorithm = _FakeAlgorithm()

    summary = run_interaction(task, policy, algorithm=algorithm, steps=3)

    assert summary.steps == 3
    assert summary.environment_steps == 6
    assert summary.episode_ends == 1
    assert summary.updates == 1
    assert summary.final_observation == [3.0, 3.0]
    assert len(algorithm.transitions) == 3
    assert algorithm.transitions[0].task_metrics == {"gait/phase": 1}


def test_component_packages_enforce_one_way_dependencies() -> None:
    forbidden = {
        "core": {
            "agents",
            "algorithms",
            "envs",
            "experiments",
            "policies",
            "robots",
            "runtime",
            "sonic",
            "tasks",
            "tools",
            "training",
        },
        "tasks": {"agents", "algorithms", "experiments", "policies", "training"},
        "policies": {
            "agents",
            "algorithms",
            "envs",
            "experiments",
            "robots",
            "tasks",
            "training",
        },
        "algorithms": {
            "agents",
            "envs",
            "experiments",
            "policies",
            "robots",
            "tasks",
            "training",
        },
    }

    violations: list[str] = []
    for owner, blocked_packages in forbidden.items():
        for path in sorted((PACKAGE_ROOT / owner).rglob("*.py")):
            for imported in _local_imports(path):
                if imported in blocked_packages:
                    violations.append(f"{path.relative_to(REPO_ROOT)} -> {imported}")

    assert violations == []


def test_legacy_imports_delegate_to_new_component_owners() -> None:
    assert legacy_g1_task is canonical_g1_task
    assert legacy_actor_critic is build_tanh_gaussian_actor_critic
    assert legacy_compute_gae is canonical_compute_gae


def test_tensor_space_rejects_ambiguous_contracts() -> None:
    assert TensorSpace((2, 3)).flat_dim == 6

    with pytest.raises(ValueError, match="positive integers"):
        TensorSpace((0,))
    with pytest.raises(ValueError, match="low must be <= high"):
        TensorSpace((1,), low=1.0, high=-1.0)


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    prefix = "h200_locomotion_lab."
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.startswith(prefix):
                imports.add(name.removeprefix(prefix).split(".", maxsplit=1)[0])
    return imports


class _FakeTask:
    spec = TaskSpec(
        name="fake_task",
        observations={"policy": TensorSpace((1,))},
        action=TensorSpace((1,)),
        max_episode_steps=3,
    )
    num_envs = 2

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self) -> list[float]:
        return [0.0, 0.0]

    def step(self, action: object) -> TaskStep:
        del action
        self.step_count += 1
        return TaskStep(
            observation=[float(self.step_count)] * self.num_envs,
            reward=[1.0, 1.0],
            terminated=[self.step_count == 2, False],
            truncated=[False, False],
            metrics={"gait/phase": self.step_count},
        )


class _FakePolicy:
    spec = PolicySpec(
        name="fake_policy",
        family="constant",
        capabilities=frozenset({"sample"}),
    )

    def act(self, observation: object, *, deterministic: bool = False) -> PolicyOutput:
        del observation, deterministic
        return PolicyOutput(action=[0.0, 0.0], info={"latent_norm": 0.0})


class _FakeAlgorithm:
    spec = AlgorithmSpec(
        name="fake_algorithm",
        family="counter",
        interaction="on_policy",
        required_policy_capabilities=frozenset({"sample"}),
    )

    def __init__(self) -> None:
        self.transitions = []

    def observe(self, transition: object) -> None:
        self.transitions.append(transition)

    def update(self, policy: object) -> UpdateReport | None:
        del policy
        if len(self.transitions) % 2:
            return None
        return UpdateReport(updated=True, samples=2, metrics={"loss": 0.5})
