# 025: Official Plant Overlay Probe

## Route

Use the fixed official-motion SONIC decoder replay from `023` as the feedback
loop. Keep SONIC planner/encoder/decoder inputs unchanged and vary only mjlab
plant details.

Ranked hypotheses:

1. If passive joint fields are the main mismatch, applying official-like
   `damping=0.05`, `armature=0.01`, and `frictionloss=0.2` to mjlab joints
   should reduce done count, base pitch, root-height collapse, or contact
   impulses under the same decoder targets.
2. If foot contact is the main mismatch, replacing mjlab's seven foot capsules
   with an official-like box sole contact and floor/friction treatment should
   reduce contact impulses and stabilize base pitch without changing SONIC I/O.
3. If both are coupled, neither single overlay may be sufficient, but the
   combined passive-joint plus contact overlay should improve the replay.
4. If none improve the replay, the next likely mismatch is deeper torque
   realization: official external PD torque clipping versus mjlab built-in
   position actuator dynamics.

## Log

- 2026-05-18 Opened after `024` found that startup randomization was not the
  primary cause and identified plant contract mismatch as the strongest
  remaining explanation.
- 2026-05-18 Added trace-only `--official-plant-overlay` with four values:
  `none`, `passive-joints`, `contact`, and `passive-joints-and-contact`.
  The switch is wired through the replay tool and the online alignment trace
  builder, but defaults to `none`.
- 2026-05-18 Overlay implementation is intentionally reversible:
  - `passive-joints` preserves actuator PD stiffness/damping/effort while
    setting actuator-provided joint `armature/frictionloss` to official-like
    values and wrapping the spec function to set non-free joint damping;
  - `contact` wraps the spec function to add one official-like box sole geom
    under each ankle roll link, disables the original foot capsule collisions
    through collision config, and uses `friction=1.0` on collision geoms.
- 2026-05-18 H200 2-step smoke passed with
  `--official-plant-overlay passive-joints-and-contact`, confirming the
  trace-only MjSpec/contact overlay compiles and the replay loop starts.
- 2026-05-18 Ran H200 400-step fixed official-motion SONIC decoder replay
  ablations with `seed=123`, `--fixed-base-reset`,
  `--disable-startup-randomization`, and `--sonic-hip-pitch-actuator`:

  | Overlay | Done count | abs pitch p95 | root z final | root z min | joint error RMS mean | foot contact max |
  | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | `none` | 14 | 0.9208 | 0.7118 | 0.2514 | 0.7152 | [1218.17, 1212.50] |
  | `passive-joints` | 13 | 0.9656 | 0.7886 | 0.2894 | 0.7290 | [1526.06, 1523.56] |
  | `contact` | 12 | 0.9953 | 0.7011 | 0.2942 | 0.7194 | [1196.78, 1603.82] |
  | `passive-joints-and-contact` | 13 | 0.9787 | 0.7385 | 0.2036 | 0.7231 | [1288.61, 1903.89] |

  Trace files:
  - `outputs/task025/official_plant_overlay_none_400/sonic_decoder_official_plant_none_400.json`
  - `outputs/task025/official_plant_overlay_passive-joints_400/sonic_decoder_official_plant_passive-joints_400.json`
  - `outputs/task025/official_plant_overlay_contact_400/sonic_decoder_official_plant_contact_400.json`
  - `outputs/task025/official_plant_overlay_passive-joints-and-contact_400/sonic_decoder_official_plant_passive-joints-and-contact_400.json`
- 2026-05-18 Local tests passed:
  `PYTHONPATH=src python -m pytest tests/test_mjlab_sonic_alignment_trace.py tests/test_mjlab_official_motion_replay_trace.py -q`
  reported `18 passed` with only the existing local pytest cache permission
  warning.

## Review

Status: passed; simple plant overlay hypotheses falsified.

The overlays changed the failure shape but did not remove the failure. Contact
alone slightly reduced done count from 14 to 12 and raised `root_z_min`, but it
worsened `abs_pitch_p95` and did not materially improve joint tracking. Passive
joint fields raised final/root minimum height but reduced forward velocity and
increased contact peaks. The combined overlay was not additive and produced the
largest right-foot contact peak.

This falsifies the narrow hypothesis that mjlab instability is fixed by copying
official passive joint fields or foot contact geometry/friction alone. The next
diagnostic should target torque realization: official sim receives LowCmd
`q_des/dq_des/kp/kd/tau`, computes external PD torque, and clips torque, while
mjlab currently uses built-in position actuators. A trace-only backend that
drives mjlab `ctrl` with official-style PD torque, or at least an offline
torque-response comparison at matched q/qdot/target, is the next useful probe.
