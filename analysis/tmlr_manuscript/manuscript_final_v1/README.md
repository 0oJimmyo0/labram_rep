# Manuscript-final results snapshot v1

This is the focused results package generated from the deterministic compact
manuscript manifest. It does not modify the broad provenance registry.

- Selected rows: 192
- Artifact statuses: {'SUPPORTING': 12, 'VERIFIED': 180}
- Pair statuses: {'boundary:VALID': 3, 'primary:VALID': 12, 'supporting_boundary:VALID': 3}
- Primary RQ2 pair rows: 12
- Primary RQ2 metric rows: 24
- RQ3 metric rows: 84

Primary RQ2 includes only valid matched pairs from FACED, ISRUC, TUEV, and
LaBraM TUEV. SEED-V is a geometry/protocol boundary case; PhysioNet-MI is a
bounded supporting extension. No row was selected by score.

Key files:

- `method_summary.csv`: overall performance, parameter, memory, and timing table.
- `rq1_effect_summary.csv`: native adaptation minus frozen probe.
- `rq2_effect_summary.csv`: primary native-minus-axis-blind effects only.
- `rq2_all_effect_summary.csv`: primary plus boundary/supporting RQ2 effects.
- `rq3_effect_summary.csv`: separate frozen and full-backbone contrasts.
- `artifact_audit.csv` and `rq2_pair_audit.csv`: provenance and inclusion checks.

Historical per-epoch test trajectories and diagnostics remain supplementary.
