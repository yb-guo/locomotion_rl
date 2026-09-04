"""Loader for the G1 27DoF no-hand Genesis training asset profile."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_G1_27DOF_ACTION_DIM = 27
DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "robots"
    / "unitree_g1_27dof_nohand_genesis.yaml"
)
REMOVED_FROM_G1_29DOF_COMMAND_ORDER = ("waist_roll_joint", "waist_pitch_joint")
G1_27DOF_NOHAND_ACTUATOR_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_BANNED_BODY_JOINT_TOKENS = ("hand", "finger")


class G1NoHandProfileError(ValueError):
    """Raised when the G1 27DoF no-hand profile is structurally invalid."""


@dataclass(frozen=True, slots=True)
class G1NoHandAssetMetadata:
    format: str
    genesis_morph: str
    path: str
    usage: str


@dataclass(frozen=True, slots=True)
class G1NoHandObservationSegment:
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class G1NoHandTrainingContract:
    sim_dt_s: float
    decimation: int
    policy_rate_hz: int
    action_size: int
    observation_dim: int
    observation_segments: tuple[G1NoHandObservationSegment, ...]


@dataclass(frozen=True, slots=True)
class G1NoHandControlProfile:
    order: str
    default_angles_rad: tuple[float, ...]
    action_scales_rad: tuple[float, ...]
    kp: tuple[float, ...]
    kv: tuple[float, ...]
    force_limits: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class G1NoHandH200Evidence:
    cuda_visible_devices: str
    physical_gpu: int
    logical_cuda_device: str
    output: str
    n_envs: int
    env_policy_steps_per_sec: float
    env_sim_steps_per_sec: float
    build_time_s: float
    measure_time_s: float
    tensor_device_ok: bool
    selected_reset_target_only: bool


@dataclass(frozen=True, slots=True)
class G1NoHandComponentEvidence:
    output: str
    action_write_time_s: float
    state_read_time_s: float
    scene_step_time_s: float
    scene_steps_per_sec: float
    env_scene_steps_per_sec: float
    combined_env_policy_steps_per_sec: float
    combined_env_sim_steps_per_sec: float


@dataclass(frozen=True, slots=True)
class G1NoHandGenesisTrainingProfile:
    name: str
    family: str
    dof_count: int
    body_profile: str
    route: str
    asset: G1NoHandAssetMetadata
    training_contract: G1NoHandTrainingContract
    actuator_order: tuple[str, ...]
    removed_from_29dof_command_order: tuple[str, ...]
    excludes_floating_base_dofs: bool
    control: G1NoHandControlProfile
    h200_evidence: G1NoHandH200Evidence
    component_evidence: G1NoHandComponentEvidence
    source_path: Path | None = None

    @property
    def action_dim(self) -> int:
        return self.dof_count

    def validated(self) -> "G1NoHandGenesisTrainingProfile":
        _validate_asset(self.asset)
        _validate_h200_evidence(self.h200_evidence)
        _validate_component_evidence(self.component_evidence)
        return self


def load_g1_27dof_nohand_profile(
    path: str | Path = DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE,
) -> G1NoHandGenesisTrainingProfile:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return load_g1_27dof_nohand_profile_dict(data, source_path=profile_path)


def load_g1_27dof_nohand_profile_dict(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> G1NoHandGenesisTrainingProfile:
    if not isinstance(data, Mapping):
        raise G1NoHandProfileError("G1 27DoF profile must be a mapping")

    robot = _required_mapping(data, "robot")
    asset_data = _required_mapping(data, "asset")
    training_contract_data = _required_mapping(data, "training_contract")
    joint_order_data = _required_mapping(data, "joint_order")
    control_data = _required_mapping(data, "control")
    h200_evidence_data = _required_mapping(data, "h200_evidence")
    component_evidence_data = _required_mapping(data, "component_evidence")

    dof_count = _required_int(robot, "robot.dof_count")
    if dof_count != EXPECTED_G1_27DOF_ACTION_DIM:
        raise G1NoHandProfileError(f"robot.dof_count must be 27, got {dof_count}")

    route = _required_str(robot, "robot.route")
    if route != "VectorizedGenesisBackend":
        raise G1NoHandProfileError("robot.route must be VectorizedGenesisBackend")

    actuator_order = _required_tuple_of_str(joint_order_data, "joint_order.actuator")
    removed = _required_tuple_of_str(
        joint_order_data, "joint_order.removed_from_29dof_command_order"
    )
    excludes_floating_base_dofs = _required_bool(
        joint_order_data, "joint_order.excludes_floating_base_dofs"
    )
    _validate_joint_order_schema(joint_order_data)
    _validate_actuator_order(actuator_order, removed, excludes_floating_base_dofs)

    training_contract = G1NoHandTrainingContract(
        sim_dt_s=_required_float(training_contract_data, "training_contract.sim_dt_s"),
        decimation=_required_int(training_contract_data, "training_contract.decimation"),
        policy_rate_hz=_required_int(training_contract_data, "training_contract.policy_rate_hz"),
        action_size=_required_int(training_contract_data, "training_contract.action_size"),
        observation_dim=_required_int(training_contract_data, "training_contract.observation_dim"),
        observation_segments=_required_observation_segments(
            training_contract_data, "training_contract.observation_segments"
        ),
    )
    _validate_training_contract(training_contract, dof_count)

    control = G1NoHandControlProfile(
        order=_required_str(control_data, "control.order"),
        default_angles_rad=_required_tuple_of_float(control_data, "control.default_angles_rad"),
        action_scales_rad=_required_tuple_of_float(control_data, "control.action_scales_rad"),
        kp=_required_tuple_of_float(control_data, "control.kp"),
        kv=_required_tuple_of_float(control_data, "control.kv"),
        force_limits=_required_tuple_of_float(control_data, "control.force_limits"),
    )
    _validate_control(control, dof_count)

    return G1NoHandGenesisTrainingProfile(
        name=_required_str(robot, "robot.name"),
        family=_required_str(robot, "robot.family"),
        dof_count=dof_count,
        body_profile=_required_str(robot, "robot.body_profile"),
        route=route,
        asset=G1NoHandAssetMetadata(
            format=_required_str(asset_data, "asset.format"),
            genesis_morph=_required_str(asset_data, "asset.genesis_morph"),
            path=_required_str(asset_data, "asset.path"),
            usage=_required_str(asset_data, "asset.usage"),
        ),
        training_contract=training_contract,
        actuator_order=actuator_order,
        removed_from_29dof_command_order=removed,
        excludes_floating_base_dofs=excludes_floating_base_dofs,
        control=control,
        h200_evidence=G1NoHandH200Evidence(
            cuda_visible_devices=_required_str(
                h200_evidence_data, "h200_evidence.CUDA_VISIBLE_DEVICES"
            ),
            physical_gpu=_required_int(h200_evidence_data, "h200_evidence.physical_gpu"),
            logical_cuda_device=_required_str(
                h200_evidence_data, "h200_evidence.logical_cuda_device"
            ),
            output=_required_str(h200_evidence_data, "h200_evidence.output"),
            n_envs=_required_int(h200_evidence_data, "h200_evidence.n_envs"),
            env_policy_steps_per_sec=_required_float(
                h200_evidence_data, "h200_evidence.env_policy_steps_per_sec"
            ),
            env_sim_steps_per_sec=_required_float(
                h200_evidence_data, "h200_evidence.env_sim_steps_per_sec"
            ),
            build_time_s=_required_float(h200_evidence_data, "h200_evidence.build_time_s"),
            measure_time_s=_required_float(h200_evidence_data, "h200_evidence.measure_time_s"),
            tensor_device_ok=_required_bool(h200_evidence_data, "h200_evidence.tensor_device_ok"),
            selected_reset_target_only=_required_bool(
                _required_mapping(h200_evidence_data, "selected_reset"),
                "h200_evidence.selected_reset.target_only",
            ),
        ),
        component_evidence=G1NoHandComponentEvidence(
            output=_required_str(component_evidence_data, "component_evidence.output"),
            action_write_time_s=_required_float(
                component_evidence_data, "component_evidence.action_write_time_s"
            ),
            state_read_time_s=_required_float(
                component_evidence_data, "component_evidence.state_read_time_s"
            ),
            scene_step_time_s=_required_float(
                component_evidence_data, "component_evidence.scene_step_time_s"
            ),
            scene_steps_per_sec=_required_float(
                component_evidence_data, "component_evidence.scene_steps_per_sec"
            ),
            env_scene_steps_per_sec=_required_float(
                component_evidence_data, "component_evidence.env_scene_steps_per_sec"
            ),
            combined_env_policy_steps_per_sec=_required_float(
                component_evidence_data, "component_evidence.combined_env_policy_steps_per_sec"
            ),
            combined_env_sim_steps_per_sec=_required_float(
                component_evidence_data, "component_evidence.combined_env_sim_steps_per_sec"
            ),
        ),
        source_path=Path(source_path) if source_path is not None else None,
    ).validated()


def _validate_actuator_order(
    actuator_order: tuple[str, ...],
    removed: tuple[str, ...],
    excludes_floating_base_dofs: bool,
) -> None:
    _expect_length(actuator_order, EXPECTED_G1_27DOF_ACTION_DIM, "joint_order.actuator")
    if len(set(actuator_order)) != EXPECTED_G1_27DOF_ACTION_DIM:
        raise G1NoHandProfileError("joint_order.actuator must not contain duplicate joints")
    if removed != REMOVED_FROM_G1_29DOF_COMMAND_ORDER:
        raise G1NoHandProfileError(
            "joint_order.removed_from_29dof_command_order must remove waist_roll_joint "
            "and waist_pitch_joint"
        )
    if not excludes_floating_base_dofs:
        raise G1NoHandProfileError("joint_order.excludes_floating_base_dofs must be true")
    banned = [
        joint_name
        for joint_name in actuator_order
        if any(token in joint_name.lower() for token in _BANNED_BODY_JOINT_TOKENS)
    ]
    if banned:
        raise G1NoHandProfileError(f"profile must not include hand/finger joints: {banned[0]}")
    present_removed = [joint_name for joint_name in removed if joint_name in actuator_order]
    if present_removed:
        raise G1NoHandProfileError(
            f"removed joint still appears in actuator order: {present_removed[0]}"
        )
    if actuator_order != G1_27DOF_NOHAND_ACTUATOR_ORDER:
        raise G1NoHandProfileError(
            "joint_order.actuator must match the canonical 29DoF command order "
            "without waist_roll_joint and waist_pitch_joint"
        )


def _validate_joint_order_schema(joint_order_data: Mapping[str, Any]) -> None:
    if _required_str(joint_order_data, "joint_order.order") != "actuator":
        raise G1NoHandProfileError("joint_order.order must be actuator")
    if (
        _required_str(joint_order_data, "joint_order.derived_from")
        != "unitree_g1_29dof_sonic.command_mujoco"
    ):
        raise G1NoHandProfileError(
            "joint_order.derived_from must be unitree_g1_29dof_sonic.command_mujoco"
        )


def _validate_control(control: G1NoHandControlProfile, dof_count: int) -> None:
    if control.order != "actuator":
        raise G1NoHandProfileError("control.order must be actuator")
    for key, values in (
        ("control.default_angles_rad", control.default_angles_rad),
        ("control.action_scales_rad", control.action_scales_rad),
        ("control.kp", control.kp),
        ("control.kv", control.kv),
        ("control.force_limits", control.force_limits),
    ):
        _expect_length(values, dof_count, key)


def _validate_asset(asset: G1NoHandAssetMetadata) -> None:
    if asset.format != "mjcf":
        raise G1NoHandProfileError("asset.format must be mjcf")
    if asset.genesis_morph != "MJCF":
        raise G1NoHandProfileError("asset.genesis_morph must be MJCF")
    if not asset.path.endswith("/g1_27dof_nohand.xml"):
        raise G1NoHandProfileError("asset.path must point to g1_27dof_nohand.xml")
    if asset.usage != "prepared_path_metadata_only":
        raise G1NoHandProfileError("asset.usage must be prepared_path_metadata_only")


def _validate_training_contract(
    training_contract: G1NoHandTrainingContract,
    dof_count: int,
) -> None:
    if training_contract.sim_dt_s <= 0:
        raise G1NoHandProfileError("training_contract.sim_dt_s must be positive")
    if training_contract.decimation <= 0:
        raise G1NoHandProfileError("training_contract.decimation must be positive")
    if training_contract.policy_rate_hz != round(
        1.0 / (training_contract.sim_dt_s * training_contract.decimation)
    ):
        raise G1NoHandProfileError(
            "training_contract.policy_rate_hz must match sim_dt_s and decimation"
        )
    if training_contract.action_size != dof_count:
        raise G1NoHandProfileError("training_contract.action_size must match robot.dof_count")
    segment_total = sum(segment.size for segment in training_contract.observation_segments)
    if training_contract.observation_dim != segment_total:
        raise G1NoHandProfileError(
            "training_contract.observation_dim must equal observation segment total"
        )
    expected_observation_dim = 9 + (3 * dof_count)
    if training_contract.observation_dim != expected_observation_dim:
        raise G1NoHandProfileError(
            f"training_contract.observation_dim must be {expected_observation_dim}"
        )


def _validate_h200_evidence(evidence: G1NoHandH200Evidence) -> None:
    if evidence.cuda_visible_devices != "1":
        raise G1NoHandProfileError("h200_evidence.CUDA_VISIBLE_DEVICES must be 1")
    if evidence.physical_gpu != 1:
        raise G1NoHandProfileError("h200_evidence.physical_gpu must be 1")
    if evidence.logical_cuda_device != "cuda:0":
        raise G1NoHandProfileError("h200_evidence.logical_cuda_device must be cuda:0")
    if evidence.n_envs < 1024:
        raise G1NoHandProfileError("h200_evidence.n_envs must be at least 1024")
    if evidence.env_policy_steps_per_sec <= 0 or evidence.env_sim_steps_per_sec <= 0:
        raise G1NoHandProfileError("h200_evidence throughput must be positive")
    if evidence.build_time_s <= 0 or evidence.measure_time_s <= 0:
        raise G1NoHandProfileError("h200_evidence timing values must be positive")
    if not evidence.tensor_device_ok:
        raise G1NoHandProfileError("h200_evidence.tensor_device_ok must be true")
    if not evidence.selected_reset_target_only:
        raise G1NoHandProfileError(
            "h200_evidence.selected_reset.target_only must be true"
        )
    if not evidence.output.startswith("/root/agent_workspace/project/"):
        raise G1NoHandProfileError("h200_evidence.output must stay under project workspace")


def _validate_component_evidence(evidence: G1NoHandComponentEvidence) -> None:
    if not evidence.output.startswith("/root/agent_workspace/project/"):
        raise G1NoHandProfileError("component_evidence.output must stay under project workspace")
    for key, value in (
        ("component_evidence.action_write_time_s", evidence.action_write_time_s),
        ("component_evidence.state_read_time_s", evidence.state_read_time_s),
        ("component_evidence.scene_step_time_s", evidence.scene_step_time_s),
        ("component_evidence.scene_steps_per_sec", evidence.scene_steps_per_sec),
        ("component_evidence.env_scene_steps_per_sec", evidence.env_scene_steps_per_sec),
        (
            "component_evidence.combined_env_policy_steps_per_sec",
            evidence.combined_env_policy_steps_per_sec,
        ),
        (
            "component_evidence.combined_env_sim_steps_per_sec",
            evidence.combined_env_sim_steps_per_sec,
        ),
    ):
        if value <= 0:
            raise G1NoHandProfileError(f"{key} must be positive")


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise G1NoHandProfileError(f"{key} must be a mapping")
    return value


def _required_str(data: Mapping[str, Any], key_path: str) -> str:
    value = _get_key_path(data, key_path)
    if not isinstance(value, str) or not value.strip():
        raise G1NoHandProfileError(f"{key_path} must be a non-empty string")
    return value


def _required_int(data: Mapping[str, Any], key_path: str) -> int:
    value = _get_key_path(data, key_path)
    if not isinstance(value, int):
        raise G1NoHandProfileError(f"{key_path} must be an integer")
    return value


def _required_bool(data: Mapping[str, Any], key_path: str) -> bool:
    value = _get_key_path(data, key_path)
    if not isinstance(value, bool):
        raise G1NoHandProfileError(f"{key_path} must be a boolean")
    return value


def _required_float(data: Mapping[str, Any], key_path: str) -> float:
    value = _get_key_path(data, key_path)
    if not isinstance(value, int | float):
        raise G1NoHandProfileError(f"{key_path} must be a number")
    return float(value)


def _required_tuple_of_str(data: Mapping[str, Any], key_path: str) -> tuple[str, ...]:
    value = _get_key_path(data, key_path)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise G1NoHandProfileError(f"{key_path} must be a sequence of strings")
    values = tuple(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise G1NoHandProfileError(f"{key_path} must contain only non-empty strings")
    return values


def _required_tuple_of_float(data: Mapping[str, Any], key_path: str) -> tuple[float, ...]:
    value = _get_key_path(data, key_path)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise G1NoHandProfileError(f"{key_path} must be a sequence of numbers")
    values = tuple(value)
    if any(not isinstance(item, int | float) for item in values):
        raise G1NoHandProfileError(f"{key_path} must contain only numbers")
    return tuple(float(item) for item in values)


def _required_observation_segments(
    data: Mapping[str, Any],
    key_path: str,
) -> tuple[G1NoHandObservationSegment, ...]:
    value = _get_key_path(data, key_path)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise G1NoHandProfileError(f"{key_path} must be a sequence of mappings")
    segments: list[G1NoHandObservationSegment] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise G1NoHandProfileError(f"{key_path} must contain only mappings")
        segments.append(
            G1NoHandObservationSegment(
                name=_required_str(item, f"{key_path}.name"),
                size=_required_int(item, f"{key_path}.size"),
            )
        )
    if not segments:
        raise G1NoHandProfileError(f"{key_path} must not be empty")
    if any(segment.size <= 0 for segment in segments):
        raise G1NoHandProfileError(f"{key_path}.size values must be positive")
    return tuple(segments)


def _get_key_path(data: Mapping[str, Any], key_path: str) -> Any:
    key = key_path.split(".")[-1]
    if key not in data:
        raise G1NoHandProfileError(f"{key_path} is required")
    return data[key]


def _expect_length(values: Sequence[object], expected: int, key_path: str) -> None:
    if len(values) != expected:
        raise G1NoHandProfileError(f"{key_path} length must be {expected}, got {len(values)}")
