"""Compile and passively step procedural whole-body models.

This is a small Task051/052 smoke, not a training script.  It intentionally
does not download assets or depend on Isaac Lab.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyGenerator,
    compile_mjcf,
)


def run_smoke(*, seeds: int, steps: int, device: str = "cpu") -> dict[str, object]:
    del device  # MuJoCo CPU smoke; the training path owns CUDA placement.
    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - local optional extra
        raise RuntimeError("install the mujoco extra to run the generator smoke") from exc

    generator = MorphologyGenerator()
    records: list[dict[str, object]] = []
    for family in ("biped", "quadruped"):
        for seed in range(seeds):
            blueprint = generator.generate(family, seed)
            physical = generator.sample_physical_params(blueprint, seed + 100_000)
            xml_text = compile_mjcf(blueprint, physical)
            model = mujoco.MjModel.from_xml_string(xml_text)
            data = mujoco.MjData(model)
            data.qpos[2] = blueprint.nominal_height * physical.global_scale
            for joint in blueprint.joints:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
                if joint_id < 0:
                    raise RuntimeError(f"compiled model lost joint {joint.name}")
                nominal = joint.nominal + physical.nominal_offsets.get(joint.semantic_slot, 0.0)
                lower, upper = model.jnt_range[joint_id]
                if not lower <= nominal <= upper:
                    raise RuntimeError(f"illegal nominal pose for {joint.name}")
                data.qpos[model.jnt_qposadr[joint_id]] = nominal
            mujoco.mj_forward(model, data)
            for _ in range(steps):
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)
            finite = all(math.isfinite(float(value)) for value in data.qpos)
            bounded = all(abs(float(value)) < 100.0 for value in data.qpos)
            # World/root bodies may legitimately be massless; every generated
            # link body must carry positive mass and a positive diagonal.
            mass_positive = all(float(value) > 0.0 for value in model.body_mass[1:])
            inertia_positive = all(float(value) > 0.0 for value in model.body_inertia[1:].flat)
            records.append(
                {
                    "family": family,
                    "seed": seed,
                    "structural_hash": blueprint.structural_hash,
                    "links": len(blueprint.links),
                    "joints": len(blueprint.joints),
                    "actuators": model.nu,
                    "has_arms": blueprint.has_arms,
                    "finite_qpos": finite,
                    "bounded_qpos": bounded,
                    "mass_positive": mass_positive,
                    "inertia_positive": inertia_positive,
                    "qpos_z": float(data.qpos[2]),
                }
            )
            if not (finite and bounded and mass_positive and inertia_positive):
                raise RuntimeError(f"invalid passive state in {family} seed {seed}")
    return {
        "families": ("biped", "quadruped"),
        "seeds_per_family": seeds,
        "passive_steps": steps,
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run_smoke(seeds=args.seeds, steps=args.steps)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
