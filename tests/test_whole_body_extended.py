from __future__ import annotations

import math

import pytest

from h200_locomotion_lab.algorithms.flow_matching import (
    AdvantageWeightedFlowMatching,
    FlowMatchingPPOAdapter,
    FlowMatchingRLConfig,
)
from h200_locomotion_lab.algorithms.masked_ppo import (
    mask_action,
    masked_entropy,
    masked_log_prob,
)
from h200_locomotion_lab.core.checkpoint import (
    WholeBodyCheckpointMetadata,
    make_checkpoint_payload,
    manifest_hash,
    validate_checkpoint_payload,
)
from h200_locomotion_lab.evaluation.whole_body_ood import (
    build_whole_body_ood_plan,
    paired_bootstrap_ci,
    run_ood_suite,
    validate_checkpoint_selection_metadata,
)
from h200_locomotion_lab.policies.recurrent_whole_body import (
    WholeBodyGRUConfig,
    WholeBodyGRUPolicy,
    WholeBodyTransformerXLPolicy,
    WholeBodyTXLConfig,
    masked_sequence_mean,
    reset_recurrent_state,
    sequence_padding_mask,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    MorphologyGenerator,
    build_morphology_split_manifest,
    compile_mjcf,
    morphology_instance_key,
)
from h200_locomotion_lab.robots.whole_body_adapter import (
    gather_action_batch,
    mask_unified_batch,
    scatter_joint_values_batch,
)
from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_SCHEMA_HASH,
    build_berkeley_humanoid_mapping,
)
from h200_locomotion_lab.robots.whole_body_stance import StanceSolution, stance_cache_key
from h200_locomotion_lab.tools.whole_body_ppo_smoke import parse_args
from h200_locomotion_lab.training.whole_body_curriculum import (
    CurriculumScheduler,
    WholeBodyShardPlan,
    normalize_whole_body_reward,
)


def _checkpoint_metadata(**overrides: object) -> WholeBodyCheckpointMetadata:
    values = {
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "manifest_hash": "a" * 64,
        **overrides,
    }
    return WholeBodyCheckpointMetadata(**values)  # type: ignore[arg-type]


def test_checkpoint_metadata_separates_tensor_and_embodiment_contracts() -> None:
    metadata = _checkpoint_metadata()
    payload = make_checkpoint_payload({"weight": [1]}, metadata)
    assert (
        validate_checkpoint_payload(
            payload,
            expected_embodiment_contract_version=PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
            expected_embodiment_contract_hash=PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
            expected_manifest_hash="a" * 64,
        )
        == metadata
    )
    assert manifest_hash({"b": 2, "a": 1}) == manifest_hash({"a": 1, "b": 2})
    assert metadata.schema_hash == WHOLE_BODY_SCHEMA_HASH
    with pytest.raises(ValueError):
        _checkpoint_metadata(schema_hash="wrong")
    with pytest.raises(ValueError, match="full SHA-256"):
        _checkpoint_metadata(embodiment_contract_hash="z" * 64)
    with pytest.raises(ValueError, match="embodiment contract hash"):
        validate_checkpoint_payload(payload, expected_embodiment_contract_hash="0" * 64)
    legacy_metadata = metadata.as_dict()
    del legacy_metadata["embodiment_contract_hash"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_checkpoint_payload({"metadata": legacy_metadata, "state_dict": {}})


def test_stance_solution_is_bound_to_exact_physical_instance() -> None:
    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 4)
    first = generator.sample_physical_params(blueprint, 101, range_fraction=0.5)
    second = generator.sample_physical_params(blueprint, 102, range_fraction=0.5)
    first_key = morphology_instance_key(blueprint, first)
    assert first_key == morphology_instance_key(blueprint, first)
    assert first_key != morphology_instance_key(blueprint, second)

    solution = StanceSolution(
        instance_key=first_key,
        base_height=0.9,
        joint_qpos={
            joint.semantic_slot: joint.nominal + first.nominal_offsets[joint.semantic_slot]
            for joint in blueprint.joints
        },
        actuator_ctrl={
            joint.semantic_slot: joint.nominal + first.nominal_offsets[joint.semantic_slot]
            for joint in blueprint.joints
        },
    )
    solution.validate_for(blueprint, first)
    assert solution.cache_key == stance_cache_key(first_key)
    assert solution.manifest()["actuator_ctrl_eq"]
    legacy_positional = StanceSolution(
        first_key,
        0.9,
        dict(solution.joint_qpos),
        dict(solution.actuator_ctrl),
        (0.1, -0.2),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert legacy_positional.root_xy == (0.1, -0.2)
    assert legacy_positional.model_xml_sha256 is None
    with pytest.raises(ValueError, match="different morphology/physical instance"):
        solution.validate_for(blueprint, second)


def test_manifest_has_fixed_family_counts_and_no_hash_leakage() -> None:
    manifest = build_morphology_split_manifest()
    assert len(manifest.train) == 64
    assert len(manifest.validation) == 16
    assert len(manifest.heldout) == 16
    assert {item.family for item in manifest.train} == {"biped", "quadruped"}
    assert not ({item.structural_hash for item in manifest.train} & {item.structural_hash for item in manifest.heldout})


def test_physical_randomization_includes_limits_nominal_and_positive_inertia() -> None:
    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 4)
    physical = generator.sample_physical_params(blueprint, 5)
    assert all(0.75 <= value <= 1.25 for value in physical.joint_limit_scales.values())
    assert all(abs(value) <= 0.15 for value in physical.nominal_offsets.values())
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(compile_mjcf(blueprint, physical))
    assert (model.dof_M0 > 0.0).all()


def test_batched_mapping_and_mask_support_torch() -> None:
    torch = pytest.importorskip("torch")
    mapping = build_berkeley_humanoid_mapping()
    unified = torch.arange(90, dtype=torch.float32).reshape(2, 45)
    robot = gather_action_batch(unified, mapping)
    scattered = scatter_joint_values_batch(robot, mapping)
    assert tuple(robot.shape) == (2, 12)
    assert tuple(scattered.shape) == (2, 45)
    mask = torch.tensor(mapping.mask)
    masked = mask_unified_batch(unified, mapping)
    assert torch.equal(masked[:, mask], scattered[:, mask])
    assert torch.equal(masked[:, ~mask], torch.zeros_like(masked[:, ~mask]))


def test_masked_probability_terms_ignore_inactive_slots() -> None:
    torch = pytest.importorskip("torch")
    mask = torch.tensor([True, False, True])
    values = torch.ones(2, 3)
    assert torch.equal(mask_action(values, mask)[:, 1], torch.zeros(2))
    assert torch.equal(masked_log_prob(values, mask), torch.full((2,), 2.0))
    assert torch.equal(masked_entropy(values, mask), torch.full((2,), 2.0))


def test_recurrent_reset_semantics_and_sequence_padding() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.ones(2, 4)
    trial_done = torch.tensor([True, True])
    assert torch.equal(reset_recurrent_state(hidden, trial_done=trial_done, context_done=None), hidden)
    context_done = torch.tensor([True, False])
    reset = reset_recurrent_state(hidden, trial_done=None, context_done=context_done)
    assert torch.equal(reset[0], torch.zeros(4))
    assert torch.equal(reset[1], torch.ones(4))
    reset_trial = reset_recurrent_state(
        hidden,
        trial_done=trial_done,
        context_done=None,
        reset_on_trial=True,
    )
    assert torch.equal(reset_trial, torch.zeros_like(hidden))
    padding = sequence_padding_mask(torch.tensor([2, 4]), 4)
    assert bool(padding[0, 2]) and not bool(padding[1, 3])
    assert float(masked_sequence_mean(torch.ones(2, 4), padding)) == 1.0


def test_gru_and_txl_default_contracts_run_one_step() -> None:
    torch = pytest.importorskip("torch")
    mask = torch.ones(45, dtype=torch.bool)
    obs = torch.zeros(2, 193)
    gru = WholeBodyGRUPolicy(action_mask=mask, config=WholeBodyGRUConfig(hidden_dim=16))
    gru_out = gru.step(obs, gru.initial_state(2), trial_done=torch.ones(2, dtype=torch.bool))
    assert tuple(gru_out.action.shape) == (2, 45)
    txl = WholeBodyTransformerXLPolicy(
        action_mask=mask,
        config=WholeBodyTXLConfig(layers=2, hidden_dim=16, attention_heads=4),
    )
    txl_out = txl.step(obs, txl.initial_state(2))
    assert tuple(txl_out.action.shape) == (2, 45)
    assert txl_out.state.memory.shape[1:] == (2, 16)


def test_curriculum_and_shard_plan_are_fixed() -> None:
    plan = WholeBodyShardPlan()
    assert plan.num_envs == 2048
    assert plan.expanded_num_envs == 4096
    scheduler = CurriculumScheduler()
    assert scheduler.current.name == "narrow_topology"
    assert scheduler.observe({"zero_fall_ratio": 0.5, "normalized_velocity_error": 0.2}) is False
    assert scheduler.observe({"zero_fall_ratio": 0.95, "normalized_velocity_error": 0.2}) is True
    assert scheduler.current.name == "all_train_topologies"
    assert math.isclose(normalize_whole_body_reward(2.0, robot_mass=10.0, robot_scale=1.0, active_actuators=12), 2.0)


def test_ood_plan_named_mappings_and_bootstrap_ci() -> None:
    cases = build_whole_body_ood_plan()
    assert len(cases) == 8
    assert cases[4].mapping is not None and cases[4].mapping.active_count == 12
    assert cases[7].mapping is not None and cases[7].mapping.active_count == 12
    validate_checkpoint_selection_metadata(_checkpoint_metadata(topology_split="validation"))
    with pytest.raises(ValueError):
        validate_checkpoint_selection_metadata(_checkpoint_metadata(topology_split="heldout"))
    results = run_ood_suite(lambda case: {"zero_fall_ratio": 1.0, "normalized_velocity_error": 0.1})
    assert all(result.passed for result in results)
    lower, upper = paired_bootstrap_ci([1.0, 1.0, 1.0], samples=100, seed=1)
    assert lower > 0.0 and upper > 0.0


def test_flow_matching_requires_true_likelihood_or_weighted_fallback() -> None:
    class Intractable:
        has_tractable_log_prob = False

        def vector_field(self, state, time, condition):
            return state

    policy = Intractable()
    with pytest.raises(ValueError):
        FlowMatchingPPOAdapter(policy)
    trainer = AdvantageWeightedFlowMatching(policy, FlowMatchingRLConfig())
    torch = pytest.importorskip("torch")
    loss = trainer.loss(torch.ones(2, 3), torch.zeros(2, 3), torch.ones(2))
    assert torch.isfinite(loss)
    assert trainer.training_mode() == "advantage_weighted_flow_matching"


def test_whole_body_training_smoke_cli_defaults() -> None:
    args = parse_args([])
    assert args.family == "biped"
    assert args.num_envs == 8
    assert args.rollout_steps == 32
