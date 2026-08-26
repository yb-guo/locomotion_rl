#!/usr/bin/env python3
"""Patch H200 MJLab dynamic training scheduler for Task044 multi-trial memory."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")

OLD_SIGNATURE = """  dynamic_dead_probability: float = 0.70,
  transient_window_s: float = 0.3,
) -> None:
"""

NEW_SIGNATURE = """  dynamic_dead_probability: float = 0.70,
  transient_window_s: float = 0.3,
  preserve_schedule_across_inner_resets: bool = False,
) -> None:
"""

OLD_RESET_MASK = """  reset_mask = (env.episode_length_buf <= 1) | (
    env._task030_dynamic_training_case_id < 0
  )
"""

NEW_RESET_MASK = """  if preserve_schedule_across_inner_resets:
    reset_mask = env._task030_dynamic_training_case_id < 0
  else:
    reset_mask = (env.episode_length_buf <= 1) | (
      env._task030_dynamic_training_case_id < 0
    )
"""

OLD_PARAMS = """      "dynamic_dead_probability": 0.70,
      "transient_window_s": 0.3,
"""

NEW_PARAMS = """      "dynamic_dead_probability": 0.70,
      "transient_window_s": 0.3,
      "preserve_schedule_across_inner_resets": False,
"""

OLD_RESAMPLE_SIGNATURE = """  dead_scale_range: tuple[float, float],
  dynamic_dead_probability: float,
) -> None:
"""

NEW_RESAMPLE_SIGNATURE = """  dead_scale_range: tuple[float, float],
  dynamic_dead_probability: float,
  dynamic_single_onset_range_s: tuple[float, float] = (1.0, 4.0),
  dynamic_single_duration_range_s: tuple[float, float] = (0.6, 2.0),
) -> None:
"""

OLD_SINGLE_TIMING = """    onset = 1.0 + 3.0 * torch.rand(count, device=env.device)
    duration = 0.6 + 1.4 * torch.rand(count, device=env.device)
"""

NEW_SINGLE_TIMING = """    onset = dynamic_single_onset_range_s[0] + (
      dynamic_single_onset_range_s[1] - dynamic_single_onset_range_s[0]
    ) * torch.rand(count, device=env.device)
    duration = dynamic_single_duration_range_s[0] + (
      dynamic_single_duration_range_s[1] - dynamic_single_duration_range_s[0]
    ) * torch.rand(count, device=env.device)
"""

OLD_SCHEDULER_SIGNATURE_TAIL = """  transient_window_s: float = 0.3,
  preserve_schedule_across_inner_resets: bool = False,
) -> None:
"""

NEW_SCHEDULER_SIGNATURE_TAIL = """  transient_window_s: float = 0.3,
  preserve_schedule_across_inner_resets: bool = False,
  dynamic_single_onset_range_s: tuple[float, float] = (1.0, 4.0),
  dynamic_single_duration_range_s: tuple[float, float] = (0.6, 2.0),
) -> None:
"""

OLD_RESAMPLE_CALL_TAIL = """      dynamic_dead_probability,
    )
"""

NEW_RESAMPLE_CALL_TAIL = """      dynamic_dead_probability,
      dynamic_single_onset_range_s,
      dynamic_single_duration_range_s,
    )
"""

OLD_TIMING_PARAMS = """      "preserve_schedule_across_inner_resets": False,
"""

NEW_TIMING_PARAMS = """      "preserve_schedule_across_inner_resets": False,
      "dynamic_single_onset_range_s": (1.0, 4.0),
      "dynamic_single_duration_range_s": (0.6, 2.0),
"""


def patch_env_cfgs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "preserve_schedule_across_inner_resets" not in text:
        if OLD_SIGNATURE not in text:
            raise RuntimeError("dynamic training scheduler signature anchor not found")
        text = text.replace(OLD_SIGNATURE, NEW_SIGNATURE, 1)
        if OLD_RESET_MASK not in text:
            raise RuntimeError("dynamic training reset mask anchor not found")
        text = text.replace(OLD_RESET_MASK, NEW_RESET_MASK, 1)
        if OLD_PARAMS not in text:
            raise RuntimeError("dynamic training params anchor not found")
        text = text.replace(OLD_PARAMS, NEW_PARAMS, 1)
    if "dynamic_single_onset_range_s" not in text:
        if OLD_RESAMPLE_SIGNATURE not in text:
            raise RuntimeError("dynamic training resample signature anchor not found")
        text = text.replace(OLD_RESAMPLE_SIGNATURE, NEW_RESAMPLE_SIGNATURE, 1)
        if OLD_SINGLE_TIMING not in text:
            raise RuntimeError("dynamic single timing anchor not found")
        text = text.replace(OLD_SINGLE_TIMING, NEW_SINGLE_TIMING, 1)
        if OLD_SCHEDULER_SIGNATURE_TAIL not in text:
            raise RuntimeError("dynamic scheduler timing signature anchor not found")
        text = text.replace(OLD_SCHEDULER_SIGNATURE_TAIL, NEW_SCHEDULER_SIGNATURE_TAIL, 1)
        if text.count(OLD_RESAMPLE_CALL_TAIL) < 2:
            raise RuntimeError("dynamic resample call anchors not found")
        text = text.replace(OLD_RESAMPLE_CALL_TAIL, NEW_RESAMPLE_CALL_TAIL, 2)
        if OLD_TIMING_PARAMS not in text:
            raise RuntimeError("dynamic timing params anchor not found")
        text = text.replace(OLD_TIMING_PARAMS, NEW_TIMING_PARAMS, 1)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    path = args.root / "src/tasks/velocity/config/g1_gripper/env_cfgs.py"
    patch_env_cfgs(path)
    print(path)


if __name__ == "__main__":
    main()
