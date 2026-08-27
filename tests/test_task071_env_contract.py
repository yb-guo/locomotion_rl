from __future__ import annotations

import hashlib
import importlib.util
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TASK_DIR = ROOT / ".agent/task/task071-multimorphology-training-readiness"


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHYSICS = _load_module("task071_physics_overlay", TASK_DIR / "task071_physics_overlay.py")
ENV_CONTRACT = _load_module("task071_env_contract", TASK_DIR / "task071_env_contract.py")
LOCAL_INPUTS = PHYSICS.FROZEN.is_dir() and PHYSICS.SOURCE.is_dir()


@pytest.fixture(scope="module")
def bound_inputs():
    pytest.importorskip("mujoco")
    overlay = PHYSICS.generate_overlay(write_artifact=False)
    r1 = PHYSICS.run_bound_r1(overlay, write_artifact=False)
    return overlay, r1


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
def test_r2_bound_g1_go2_environment_contract_is_real_and_complete() -> None:
    pytest.importorskip("mujoco")
    result = ENV_CONTRACT.run_env_contract(write_artifact=False)

    assert result["denominator"] == 2
    assert result["task071_r2_admission_passed"] is True
    assert result["claim_boundary"] == {
        "bounded_environment_contract_only": True,
        "training_performed": False,
        "ppo_started": False,
        "walking_claimed": False,
        "task071_passed": False,
    }
    assert {record["reference_id"] for record in result["records"]} == {
        "unitree_g1",
        "unitree_go2",
    }
    assert all(count == 2 for count in result["summary"].values())
    assert all(record["passed"] for record in result["records"])
    assert {record["schema"]["active_count"] for record in result["records"]} == {12, 29}
    for record in result["records"]:
        assert record["schema"]["version"] == "whole_body_v1_45"
        assert record["schema"]["action_dim"] == 45
        assert record["schema"]["observation_dim"] == 193
        assert record["failure_reasons"] == []
        assert all(record["checks"].values())


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
def test_precompiled_model_rejects_missing_blueprint_joint_before_address_lookup(
    bound_inputs,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    overlay, r1 = bound_inputs
    case = PHYSICS.load_frozen_case("unitree_g1")
    stance = ENV_CONTRACT._stance_solution(PHYSICS, case, overlay, r1)
    _, xml = PHYSICS._bind_case(case, mujoco, write_artifact=False)
    original = case.blueprint.joints[0].name
    renamed = f"{original}_renamed"
    xml = xml.replace(f'name="{original}"', f'name="{renamed}"', 1)
    xml = xml.replace(f'joint="{original}"', f'joint="{renamed}"', 1)

    with pytest.raises(ValueError, match="missing a generated joint"):
        WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            model_xml=xml,
            model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            stance_solution=stance,
        )


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
def test_precompiled_model_requires_canonical_root_site(bound_inputs) -> None:
    mujoco = pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    overlay, r1 = bound_inputs
    case = PHYSICS.load_frozen_case("unitree_go2")
    stance = ENV_CONTRACT._stance_solution(PHYSICS, case, overlay, r1)
    _, xml = PHYSICS._bind_case(case, mujoco, write_artifact=False)
    start = xml.index('<site name="canonical_root"')
    end = xml.index("/>", start) + 2
    xml = xml[:start] + xml[end:]

    with pytest.raises(ValueError, match="canonical root site"):
        WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            model_xml=xml,
            model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            stance_solution=stance,
        )


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
def test_precompiled_model_requires_exact_xml_and_stance_sha_binding(bound_inputs) -> None:
    pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    overlay, r1 = bound_inputs
    case = PHYSICS.load_frozen_case("unitree_go2")
    record = ENV_CONTRACT._record(overlay["records"], case.reference_id)
    stance = ENV_CONTRACT._stance_solution(PHYSICS, case, overlay, r1)
    _, xml = PHYSICS._bind_case(case, pytest.importorskip("mujoco"), write_artifact=False)

    with pytest.raises(ValueError, match="model_xml SHA mismatch"):
        WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            model_xml=xml,
            model_xml_sha256="0" * 64,
            stance_solution=stance,
        )
    wrongly_bound_stance = replace(stance, model_xml_sha256="0" * 64)
    with pytest.raises(ValueError, match="different model XML"):
        WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            model_xml=xml,
            model_xml_sha256=record["output_xml_sha256"],
            stance_solution=wrongly_bound_stance,
        )


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
@pytest.mark.parametrize("tamper", ("mass", "contact"))
def test_precompiled_model_bound_stance_rejects_mass_or_contact_tamper(
    bound_inputs,
    tamper: str,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    overlay, r1 = bound_inputs
    case = PHYSICS.load_frozen_case("unitree_go2")
    stance = ENV_CONTRACT._stance_solution(PHYSICS, case, overlay, r1)
    _, xml = PHYSICS._bind_case(case, mujoco, write_artifact=False)
    root = ET.fromstring(xml)
    if tamper == "mass":
        inertial = root.find(".//inertial")
        assert inertial is not None
        inertial.set("mass", str(float(inertial.get("mass", "0")) * 1.01))
    else:
        contact = root.find(".//geom[@friction]")
        assert contact is not None
        contact.set("friction", "0.1 0.01 0.001")
    tampered_xml = ET.tostring(root, encoding="unicode") + "\n"

    tampered_sha = hashlib.sha256(tampered_xml.encode()).hexdigest()
    with pytest.raises(ValueError, match="different model XML"):
        WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            model_xml=tampered_xml,
            model_xml_sha256=tampered_sha,
            stance_solution=stance,
        )


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("motor_type", "position actuator semantics"),
        ("gear", "actuator gear"),
        ("ctrlrange", "position actuator semantics"),
        ("joint_axis", "joint axis"),
        ("joint_range", "joint range"),
        ("canonical_frame", "canonical root frame"),
    ),
)
def test_precompiled_model_rejects_semantic_tampering(
    bound_inputs,
    tamper: str,
    message: str,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from h200_locomotion_lab.envs.whole_body_mujoco import WholeBodyMuJoCoShard

    overlay, r1 = bound_inputs
    case = PHYSICS.load_frozen_case("unitree_go2")
    stance = ENV_CONTRACT._stance_solution(PHYSICS, case, overlay, r1)
    _, xml = PHYSICS._bind_case(case, mujoco, write_artifact=False)
    root = ET.fromstring(xml)
    joint = root.find(f".//joint[@name='{case.blueprint.joints[0].name}']")
    actuator = root.find(f".//position[@name='{case.blueprint.actuators[0].name}']")
    canonical = root.find(".//site[@name='canonical_root']")
    assert joint is not None and actuator is not None and canonical is not None
    if tamper == "motor_type":
        actuator.tag = "motor"
        actuator.attrib.pop("kp")
        actuator.attrib.pop("kv")
    elif tamper == "gear":
        actuator.set("gear", "2")
    elif tamper == "ctrlrange":
        actuator.set("ctrlrange", "-0.5 0.5")
    elif tamper == "joint_axis":
        joint.set("axis", "0 1 0")
    elif tamper == "joint_range":
        joint.set("range", "-0.5 0.5")
    else:
        canonical.set("pos", "0.01 0.02 0.03")
    tampered_xml = ET.tostring(root, encoding="unicode") + "\n"
    tampered_sha = hashlib.sha256(tampered_xml.encode()).hexdigest()

    with pytest.raises(ValueError, match=message):
        WholeBodyMuJoCoShard(
            case.blueprint,
            physical=case.physical,
            model_xml=tampered_xml,
            model_xml_sha256=tampered_sha,
            stance_solution=stance,
        )
