# Manuscript-final results snapshot v2

This is the focused results package generated from the deterministic compact
manuscript manifest. It does not modify the broad provenance registry.

- Selected rows: 156
- Artifact statuses: {'SUPPORTING': 12, 'VERIFIED': 144}
- Pair statuses: {'primary:VALID': 12}
- Primary RQ2 pair rows: 12
- Primary RQ2 metric rows: 24
- RQ3 metric rows: 66

Primary RQ2 includes complete-seed valid matched pairs in CBraMod/faced/channel_patch, CBraMod/isruc/channel_patch, CBraMod/tuev/channel_patch, LaBraM/TUEV/channel_patch.
No row was selected by score.

Key files:

- `method_summary.csv`: overall performance, parameter, memory, and timing table.
- `rq1_effect_summary.csv`: native adaptation minus frozen probe.
- `rq2_effect_summary.csv`: primary native-minus-axis-blind effects only.
- `rq2_all_effect_summary.csv`: primary plus boundary/supporting RQ2 effects.
- `rq3_effect_summary.csv`: separate frozen and full-backbone contrasts.
- `artifact_audit.csv` and `rq2_pair_audit.csv`: provenance and inclusion checks.

RQ2 parameter matching is computed from recorded adaptation-module parameter counts; total trainable parameters remain reported separately.

Historical per-epoch test trajectories and diagnostics remain supplementary.
