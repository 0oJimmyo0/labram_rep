# TMLR final analysis contract

Status: contract specification; the evidence snapshot remains unfreezed until
artifact and pair reviews are complete.

This document supersedes neither the prospective execution plan nor the
historical experiment logs. It defines how completed artifacts become
manuscript evidence.

## Review sequence

```text
artifact discovery
  -> candidate classification
  -> manual artifact adjudication
  -> regenerate surviving primary pairs
  -> manual pair-contract adjudication
  -> frozen_v1 snapshot
  -> manuscript tables and figures
```

Artifact review must happen before pair review. A pair is not reviewed if one
of its artifacts is demoted from `PRIMARY` to `SUPPORTING`, `PILOT`, or
`EXCLUDED`.

## Artifact status

The registry's automatic labels are provisional. Human review uses exactly:

- `PRIMARY`
- `SUPPORTING`
- `PILOT`
- `EXCLUDED`

Every reviewed row records reviewer, date, notes, previous status, change
reason, and an exclusion reason code when applicable.

## RQ2 pair contract

The primary alignment comparison is:

```text
native aligned adaptation - parameter-matched axis-blind adaptation
```

The automatic contract checks the available registry fields and requires:

- same backbone, dataset, seed, and native axis;
- frozen-versus-trainable regime compatibility;
- trainable-parameter mismatch no greater than 5% for RQ2;
- valid completed artifacts.

The pair review must additionally verify split/protocol, pretrained checkpoint,
classifier, initialization, optimizer and schedule, update budget, checkpoint
selector, insertion location, preprocessing, and data manifest. If a nuisance
field cannot be verified for a legacy artifact, the pair remains unverified.

`PAIR_STRUCTURALLY_INVALID` pairs are excluded from RQ2 but may remain useful
for other research questions. `PAIR_ARTIFACT_UNVERIFIED` pairs require manual
review. Only pairs with two manually `PRIMARY` artifacts, a valid contract,
and pair review status `VALID` enter the final RQ2 analysis.

## Metrics and selection

- Primary test metrics: balanced accuracy and macro-F1.
- Secondary metrics: Cohen's kappa and weighted F1.
- Checkpoint selection: validation Cohen's kappa only.
- Seeds: 42, 1024, and 3407.
- Test-at-every-epoch results are post-hoc diagnostics, not selection inputs.

## Finalization

`scripts/finalize_tmlr_snapshot.py` is fail-closed. It refuses to create a
`frozen_v1/` directory while candidate artifacts are unreviewed, retained RQ2
pairs lack valid pair review, or an unknown status appears. Structural failures,
supporting artifacts, pilots, and exclusions do not block finalization once
they are explicitly adjudicated and excluded from the relevant analysis.
