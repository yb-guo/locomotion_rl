"""Versioned checkpoint metadata for whole-body tensor and embodiment contracts.

Weights are framework-specific, but their tensor layout, action/reset meaning,
and concrete training manifest are not. Every checkpoint writer embeds this
small JSON-safe payload and validates it before loading a policy into a task.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Kept here rather than importing the robot package: checkpoint metadata is a
# core boundary and must remain usable before any embodiment is loaded.  The
# slot module asserts that its canonical layout hashes to this value.
WHOLE_BODY_SCHEMA_VERSION = "whole_body_v1_45"
WHOLE_BODY_SCHEMA_HASH = "6a3dfcc5f1ca4b27e2312e8ca402823c2740216a7c42eae1ca1fbe5aed278a58"


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class WholeBodyCheckpointMetadata:
    """Stable metadata stored alongside model parameters."""

    embodiment_contract_version: str
    embodiment_contract_hash: str
    manifest_hash: str
    schema_version: str = WHOLE_BODY_SCHEMA_VERSION
    schema_hash: str = WHOLE_BODY_SCHEMA_HASH
    task_name: str = "procedural_whole_body_velocity"
    policy_family: str = "mlp"
    topology_split: str = "train"

    def __post_init__(self) -> None:
        if not self.embodiment_contract_version:
            raise ValueError("embodiment_contract_version must be non-empty")
        if not _is_sha256(self.embodiment_contract_hash):
            raise ValueError("embodiment_contract_hash must be a full SHA-256 digest")
        if self.schema_version != WHOLE_BODY_SCHEMA_VERSION:
            raise ValueError(f"unsupported whole-body schema {self.schema_version!r}")
        if self.schema_hash != WHOLE_BODY_SCHEMA_HASH:
            raise ValueError("whole-body schema hash does not match the current contract")
        if self.topology_split not in {"train", "validation", "heldout", "ood"}:
            raise ValueError("topology_split must be train, validation, heldout, or ood")
        if not _is_sha256(self.manifest_hash):
            raise ValueError("manifest_hash must be a full SHA-256 digest")

    def as_dict(self) -> dict[str, str]:
        return {
            "embodiment_contract_version": self.embodiment_contract_version,
            "embodiment_contract_hash": self.embodiment_contract_hash,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "task_name": self.task_name,
            "policy_family": self.policy_family,
            "topology_split": self.topology_split,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WholeBodyCheckpointMetadata:
        required = {
            "embodiment_contract_version",
            "embodiment_contract_hash",
            "schema_version",
            "schema_hash",
            "task_name",
            "policy_family",
            "manifest_hash",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"checkpoint metadata missing fields: {sorted(missing)}")
        return cls(
            embodiment_contract_version=str(payload["embodiment_contract_version"]),
            embodiment_contract_hash=str(payload["embodiment_contract_hash"]),
            schema_version=str(payload["schema_version"]),
            schema_hash=str(payload["schema_hash"]),
            task_name=str(payload["task_name"]),
            policy_family=str(payload["policy_family"]),
            topology_split=str(payload.get("topology_split", "train")),
            manifest_hash=str(payload["manifest_hash"]),
        )


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a concrete runtime/training manifest independent of key ordering."""

    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_checkpoint_payload(
    state_dict: Mapping[str, Any],
    metadata: WholeBodyCheckpointMetadata,
) -> dict[str, Any]:
    """Return a portable payload; the caller decides how to serialize weights."""

    return {"metadata": metadata.as_dict(), "state_dict": state_dict}


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_embodiment_contract_version: str | None = None,
    expected_embodiment_contract_hash: str | None = None,
    expected_manifest_hash: str | None = None,
) -> WholeBodyCheckpointMetadata:
    """Fail closed on tensor, embodiment, or concrete-manifest mismatches."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint payload must contain mapping metadata")
    parsed = WholeBodyCheckpointMetadata.from_mapping(metadata)
    if (
        expected_embodiment_contract_version is not None
        and parsed.embodiment_contract_version != expected_embodiment_contract_version
    ):
        raise ValueError("checkpoint embodiment contract version does not match the runtime")
    if (
        expected_embodiment_contract_hash is not None
        and parsed.embodiment_contract_hash != expected_embodiment_contract_hash
    ):
        raise ValueError("checkpoint embodiment contract hash does not match the runtime")
    if expected_manifest_hash is not None and parsed.manifest_hash != expected_manifest_hash:
        raise ValueError("checkpoint manifest hash does not match the runtime")
    return parsed
