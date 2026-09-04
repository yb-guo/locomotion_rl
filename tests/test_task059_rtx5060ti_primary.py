from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    path = ROOT / "configs" / "experiments" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_rtx5060ti_is_the_active_whole_body_profile() -> None:
    experiment = _load("whole_body_rtx5060ti.yaml")["experiment"]
    runtime = experiment["runtime"]

    assert experiment["hardware_profile"] == "rtx5060ti_16gb"
    assert experiment["status"] == "active"
    assert runtime["device"] == "cuda:0"
    assert runtime["topology_shards"] * runtime["envs_per_shard"] == runtime["total_envs"]
    assert runtime["total_envs"] == 256


def test_h200_whole_body_profile_is_explicitly_disabled() -> None:
    experiment = _load("whole_body_h200.yaml")["experiment"]

    assert experiment["hardware_profile"] == "h200_legacy"
    assert experiment["status"] == "disabled"
    assert "whole_body_rtx5060ti.yaml" in experiment["disabled_reason"]
