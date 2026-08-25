# Full evidence audit

This package is the broad provenance and filesystem audit for the TMLR study.
It is deliberately separate from `manuscript_inclusion_manifest.csv`, which
is a claims-focused selection layer.

Run from the LaBraM repository with:

```bash
python analysis/tmlr_manuscript/evidence_audit_20260825/audit_full_evidence.py
```

The normal command rescans every textual log and inventories binary TensorBoard
event files without decoding them. If the log store has not changed and only
the derived tables need regeneration, use:

```bash
python analysis/tmlr_manuscript/evidence_audit_20260825/audit_full_evidence.py \
  --reuse-log-inventory
```

The package distinguishes completed test contracts, incomplete/preflight
directories, audit/smoke runs, legacy log-only records, and supplemental
completed artifacts. The four-cell matched RQ2 scope is unchanged by this
audit.
