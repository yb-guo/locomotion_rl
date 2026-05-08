"""NumPy batched action bridge from policy order to command target order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.robots import CompiledRobotProfile


@dataclass(frozen=True, slots=True)
class NumpyTensorActionBridge:
    """Map batched raw policy actions to command-order motor targets."""

    action_dim: int
    command_to_policy: Any
    default_angles_command: Any
    action_scale_command: Any
    _np: Any

    @classmethod
    def from_profile(cls, profile: CompiledRobotProfile) -> "NumpyTensorActionBridge":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised only without numpy installed
            raise RuntimeError(
                "NumpyTensorActionBridge requires numpy. Install the training extra."
            ) from exc

        return cls(
            action_dim=profile.action_dim,
            command_to_policy=np.asarray(
                profile.mapping.command_mujoco_index_to_policy_isaaclab_index,
                dtype=np.intp,
            ),
            default_angles_command=np.asarray(profile.control.default_angles_rad, dtype=np.float64),
            action_scale_command=np.asarray(profile.control.action_scales_rad, dtype=np.float64),
            _np=np,
        )

    def policy_actions_to_command_targets(self, raw_actions_policy: Any) -> Any:
        """Apply target[:, i] = default[i] + raw[:, map[i]] * scale[i]."""

        raw_actions = self._np.asarray(raw_actions_policy, dtype=self._np.float64)
        if raw_actions.ndim != 2:
            raise ValueError(
                f"Expected raw action rank 2 with shape [N, {self.action_dim}], "
                f"got rank {raw_actions.ndim}"
            )
        if raw_actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected raw action shape [N, {self.action_dim}], got {raw_actions.shape}"
            )
        if not self._np.isfinite(raw_actions).all():
            raise ValueError("raw actions must contain only finite values")

        return (
            self.default_angles_command[None, :]
            + raw_actions[:, self.command_to_policy] * self.action_scale_command[None, :]
        )
