from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from h200_locomotion_lab.core.whole_body import (
    WholeBodyPolicyOutput,
    WholeBodyRolloutBatch,
    WholeBodyStep,
)
from h200_locomotion_lab.envs.whole_body_mux import WholeBodyRolloutMux
from h200_locomotion_lab.robots.motor_process import MotorProcess, MotorProcessConfig
from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    MorphologyGenerator,
    MorphologyGeneratorConfig,
    compile_mjcf,
    compile_with_mujoco,
    validate_mjcf_text,
)
from h200_locomotion_lab.robots.whole_body_adapter import BoundEmbodiment
from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_ACTION_DIM,
    WHOLE_BODY_ACTOR_OBS_DIM,
    WHOLE_BODY_SCHEMA_VERSION,
    WHOLE_BODY_SLOT_NAMES,
    build_anymal_c_mapping,
    build_berkeley_humanoid_mapping,
    build_g1_whole_body_mapping,
)
from h200_locomotion_lab.tasks.whole_body_contract import make_whole_body_task_spec
from h200_locomotion_lab.tools.whole_body_stance_diagnosis import diagnose_seed


def test_whole_body_schema_and_task_dimensions_are_frozen() -> None:
    task = make_whole_body_task_spec()
    assert WHOLE_BODY_SCHEMA_VERSION == "whole_body_v1_45"
    assert WHOLE_BODY_ACTION_DIM == 45
    assert WHOLE_BODY_ACTOR_OBS_DIM == 193
    assert task.observation("policy").flat_dim == 193
    assert task.observation("value").flat_dim == 193
    assert task.action.flat_dim == 45
    assert task.parameters["schema_version"] == WHOLE_BODY_SCHEMA_VERSION


def test_named_robot_mappings_cover_expected_active_actuators() -> None:
    g1 = build_g1_whole_body_mapping()
    berkeley = build_berkeley_humanoid_mapping()
    anymal = build_anymal_c_mapping()

    assert (g1.robot_action_dim, g1.active_count) == (29, 29)
    assert (berkeley.robot_action_dim, berkeley.active_count) == (12, 12)
    assert (anymal.robot_action_dim, anymal.active_count) == (12, 12)
    assert g1.round_trip(tuple(range(29))) == tuple(float(value) for value in range(29))
    assert berkeley.round_trip(tuple(range(12))) == tuple(float(value) for value in range(12))
    assert anymal.round_trip(tuple(range(12))) == tuple(float(value) for value in range(12))


def test_inactive_slots_are_zeroed_and_do_not_leak() -> None:
    mapping = build_berkeley_humanoid_mapping()
    unified = tuple(float(index + 1) for index in range(45))
    masked = mapping.mask_values(unified)
    assert all(value == 0.0 for value, active in zip(masked, mapping.mask) if not active)
    assert mapping.project_to_robot_order(masked) == (
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        5.0,
        8.0,
        9.0,
        10.0,
        11.0,
        13.0,
        12.0,
    )


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_procedural_generator_is_seeded_and_xml_valid(family: str) -> None:
    generator = MorphologyGenerator()
    first = generator.generate(family, 17)
    second = generator.generate(family, 17)
    assert first.manifest() == second.manifest()
    xml_text = compile_mjcf(first)
    validate_mjcf_text(xml_text)
    model = compile_with_mujoco(xml_text)
    assert model.nu == len(first.actuators)
    assert model.nv > 6


def test_r2_stance_contract_keeps_schema_separate() -> None:
    assert PROCEDURAL_EMBODIMENT_CONTRACT_VERSION == (
        "procedural_whole_body_v2_footpad_actual_stance_feedforward"
    )
    assert len(PROCEDURAL_EMBODIMENT_CONTRACT_HASH) == 64
    task = make_whole_body_task_spec()
    assert task.action.flat_dim == 45
    assert task.observation("policy").flat_dim == 193


def test_terminal_leg_links_compile_to_scaled_footpads() -> None:
    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 17)
    physical = generator.sample_physical_params(blueprint, 117, range_fraction=0.5)
    terminal_feet = [link for link in blueprint.links if link.foot]
    assert terminal_feet == [link for link in blueprint.links if link.end_site]
    assert len(terminal_feet) == 2

    root = ET.fromstring(compile_mjcf(blueprint, physical))
    geoms = {str(geom.get("name")): geom for geom in root.iter("geom")}
    for link in terminal_feet:
        assert link.foot_size is not None
        scale = physical.global_scale * physical.link_scales[link.name]
        shank = geoms[f"{link.name}_geom"]
        footpad = geoms[f"{link.name}_footpad"]

        assert shank.get("type") == "capsule"
        assert shank.get("contype") == "0"
        assert shank.get("conaffinity") == "0"
        assert footpad.get("type") == "box"
        assert footpad.get("contype") == "1"
        assert footpad.get("conaffinity") == "1"

        size = tuple(float(value) for value in str(footpad.get("size")).split())
        pos = tuple(float(value) for value in str(footpad.get("pos")).split())
        assert size == pytest.approx(
            (
                link.foot_size[0] * scale,
                link.foot_size[1] * scale,
                link.size[0] * scale,
            )
        )
        assert pos == pytest.approx((0.0, 0.0, -(link.length + link.size[0]) * scale))


def test_mujoco_shard_counts_only_footpads_as_feet() -> None:
    pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 17)
    physical = generator.sample_physical_params(blueprint, 117, range_fraction=0.0)
    shard = WholeBodyMuJoCoShard(blueprint, physical=physical, num_envs=1)
    assert shard._foot_geoms == {
        f"{link.name}_footpad" for link in blueprint.links if link.foot
    }


def test_mujoco_shard_reset_uses_qpos_eq_and_zero_action_uses_ctrl_eq() -> None:
    pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    generator = MorphologyGenerator()
    blueprint = generator.generate("biped", 0)
    physical = generator.sample_physical_params(blueprint, 10_000_000, range_fraction=0.5)
    shard = WholeBodyMuJoCoShard(blueprint, physical=physical, num_envs=1)
    data = shard.data[0]
    solution = shard.stance_solution
    solution.validate_for(blueprint, physical)

    assert float(data.qpos[2]) == pytest.approx(solution.base_height)
    assert tuple(float(value) for value in data.qpos[0:2]) == pytest.approx(solution.root_xy)
    assert tuple(float(value) for value in data.qpos[3:7]) == pytest.approx(solution.root_quat)
    leg_solution_values = []
    leg_fixed_values = []
    ctrl_position_deltas = []
    for joint, qpos_address in zip(blueprint.joints, shard._joint_qpos):
        expected = solution.joint_qpos[joint.semantic_slot]
        fixed_nominal = joint.nominal + physical.nominal_offsets[joint.semantic_slot]
        assert float(data.qpos[qpos_address]) == pytest.approx(expected)
        if "leg" in joint.child_link:
            leg_solution_values.append(expected)
            leg_fixed_values.append(fixed_nominal)
    assert any(
        abs(solution_value - fixed_value) > 1e-3
        for solution_value, fixed_value in zip(leg_solution_values, leg_fixed_values)
    )

    shard._set_targets(data, tuple(0.0 for _ in blueprint.joints))
    for joint, actuator, actuator_id, qpos_address in zip(
        blueprint.joints,
        blueprint.actuators,
        shard._actuator_ids,
        shard._joint_qpos,
    ):
        assert float(data.ctrl[actuator_id]) == pytest.approx(
            solution.actuator_ctrl[actuator.semantic_slot]
        )
        ctrl_position_deltas.append(float(data.ctrl[actuator_id]) - float(data.qpos[qpos_address]))
        assert actuator.semantic_slot == joint.semantic_slot
    assert max(abs(value) for value in ctrl_position_deltas) > 1e-3


def test_stance_diagnosis_uses_footpad_corners_for_support_hull() -> None:
    pytest.importorskip("mujoco")
    report = diagnose_seed("biped", 0, range_fraction=0.0, horizon_steps=0)
    support = report["support_all_feet"]
    assert support["degenerate"] is False
    assert support["hull_area"] > 0.02
    assert report["feet_near_floor"] == report["num_feet"]
    assert report["foot_height_spread"] <= 0.005


def test_procedural_generator_has_structural_variation_without_invalid_slots() -> None:
    generator = MorphologyGenerator()
    bipeds = [generator.generate("biped", seed) for seed in range(100)]
    quadrupeds = [generator.generate("quadruped", seed) for seed in range(100)]
    assert len({item.structural_hash for item in bipeds}) > 1
    assert len({item.structural_hash for item in quadrupeds}) > 1
    assert all(set(item.active_slots).issubset(set(WHOLE_BODY_SLOT_NAMES)) for item in bipeds)
    assert all(item.family == "quadruped" for item in quadrupeds)


def test_r3_biped_grammar_requires_mirrored_ankled_roll_legs() -> None:
    generator = MorphologyGenerator()
    for seed in range(20):
        blueprint = generator.generate("biped", seed)
        left = {
            joint.semantic_slot.split("_", 1)[1]: joint
            for joint in blueprint.joints
            if joint.semantic_slot.startswith("limb0_")
        }
        right = {
            joint.semantic_slot.split("_", 1)[1]: joint
            for joint in blueprint.joints
            if joint.semantic_slot.startswith("limb1_")
        }
        assert tuple(left) == tuple(right)
        for required in ("hip_pitch", "hip_roll", "knee_pitch", "ankle_pitch"):
            assert required in left
        for suffix, left_joint in left.items():
            right_joint = right[suffix]
            if suffix.endswith(("roll", "yaw")):
                assert right_joint.axis == pytest.approx(tuple(-value for value in left_joint.axis))
            else:
                assert right_joint.axis == left_joint.axis


def test_r3_biped_toggles_do_not_change_quadruped_grammar() -> None:
    default = MorphologyGenerator().generate("quadruped", 11)
    relaxed = MorphologyGenerator(
        MorphologyGeneratorConfig(require_biped_ankle=False, mirror_biped_legs=False)
    ).generate("quadruped", 11)
    assert default.manifest() == relaxed.manifest()


def test_generated_embodiment_encodes_193d_actor_observation() -> None:
    blueprint = MorphologyGenerator().generate("biped", 3)
    embodiment = BoundEmbodiment.from_blueprint(blueprint)
    observation = embodiment.encode_actor_observation(
        base_linear_velocity=(0.0, 0.0, 0.0),
        base_angular_velocity=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        command=(0.5, 0.0, 0.0),
        joint_position=(0.0,) * len(blueprint.joints),
        joint_velocity=(0.0,) * len(blueprint.joints),
        previous_action=(0.0,) * 45,
        trial_start=1.0,
    )
    assert len(observation) == 193
    embodiment.validate_observation(observation)
    assert observation[-1] == 1.0


class _FakeShard:
    def __init__(self, num_envs: int, offset: float) -> None:
        self.num_envs = num_envs
        self.offset = offset

    def reset(self) -> list[float]:
        return [self.offset] * self.num_envs

    def step(self, action: list[list[float]]) -> WholeBodyStep:
        assert len(action) == self.num_envs
        return WholeBodyStep(
            actor_observation=[[self.offset] for _ in action],
            critic_observation=[[self.offset] for _ in action],
            reward=[self.offset] * self.num_envs,
            trial_done=[False] * self.num_envs,
            context_done=[False] * self.num_envs,
            active_action_mask=[[True] for _ in action],
            metrics={"shard": [self.offset] * self.num_envs},
        )


def test_rollout_mux_splits_actions_and_concatenates_shards() -> None:
    mux = WholeBodyRolloutMux((_FakeShard(2, 1.0), _FakeShard(3, 2.0)))
    step = mux.step([[0.0]] * 5)
    assert mux.num_envs == 5
    assert step.reward == [1.0, 1.0, 2.0, 2.0, 2.0]
    assert step.actor_observation == [[1.0], [1.0], [2.0], [2.0], [2.0]]


def test_motor_process_is_deterministic_and_preserves_context_events() -> None:
    active = ("limb0_knee_pitch", "waist_yaw", "left_arm_elbow_pitch")
    config = MotorProcessConfig(no_event_probability=0.0, max_events=1)
    first = MotorProcess(active, config=config)
    second = MotorProcess(active, config=config)
    state_first = first.reset_context(5)
    state_second = second.reset_context(5)
    assert state_first == state_second
    assert first.events == second.events
    old_events = first.events
    assert first.reset_trial().events == first.events
    first.reset_context(6)
    assert first.events != old_events or first.events == ()


def test_motor_process_does_not_expose_fault_labels_as_actor_observation() -> None:
    process = MotorProcess(("limb0_knee_pitch",), config=MotorProcessConfig(no_event_probability=0.0))
    state = process.reset_context(1)
    assert "strength" in state.critic_payload()
    assert "events" not in state.critic_payload()


def test_whole_body_rollout_types_are_framework_neutral() -> None:
    step = WholeBodyStep([], [], [], [], [], [])
    output = WholeBodyPolicyOutput(action=[])
    batch = WholeBodyRolloutBatch([], [], [], [], [], [], [], [], [])
    assert step.context_done == []
    assert output.action == []
    assert batch.active_action_mask == []
