# 001 — Freeze Failed V2 Preview

## Route

1. Treat the current `unitree_g1_seed000` preview as a negative example, not a
   partial pass.
2. Preserve its manifest/XML/PNG SHA evidence and record concrete failure modes:
   arm chain visually collapsed, hip/knee/ankle offsets not source-tree driven,
   torso/waist/arm attachment too hand-authored, and no stance evidence.
3. Add an explicit artifact or task-log marker so future R4/R5 gates cannot count
   this preview as v2 acceptance evidence.

## Log

- Completed 2026-08-25. The old
  `artifacts/preview_task070_v2/unitree_g1_seed000` preview is frozen as
  negative evidence only.
- Preserved failed artifact identity:
  XML `25ea1fa678627e37e18bcaf005c951414ca139bfb6795ad6154f84bef266701e`,
  sheet PNG `b525d0d0ae87e2fd4fde2b35211f0cf3fcaff8a2bcf7012558652dd6e23ff7be`,
  original manifest
  `fb8799efb1e23cd3713a7ca1886d027d25c6324fa1c9029ee0e49f8ed86c2c09`.
- Updated the manifest to include
  `counts_toward_task070_v2_pass=false` and a `task070_v2_rejection`
  block. Updated manifest SHA:
  `c14793ec983d71f1bc9b494fdfd74549e088e5ab80c28fd331053b69102cef35`.
- Rejection reasons: arm chain visually collapsed or too close for reliable
  audit; hip/knee/ankle offsets were not parsed-source-tree driven;
  torso/waist/arm attachments were hand-authored; no stance evidence was run.

## Review

- Pass. The failed preview is still reproducible and clearly marked
  `counts_toward_task070_v2_pass=false`.
- Fail if any summary can still read the preview as a valid G1-derived linkage
  witness.
