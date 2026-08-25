# Full two-backbone experiment evidence audit

Audit date: 2026-08-25

## Scope

This audit inventories every experiment-like directory under `CBraMod/results` and `LaBraM/checkpoints`, and every `.out`, `.err`, `.vu`, and `log.txt` file under both repositories. It separates completed test contracts from smoke, preflight, incomplete, legacy, and auxiliary artifacts.

The manuscript inclusion manifest remains claims-first. This audit is the broader evidence layer and does not promote legacy or non-homogeneous records into the primary matched RQ2 analysis.

## Artifact inventory

- Experiment-like directories audited: **769**.
- Registry members: **485**; registry-complete test contracts: **485**.
- Supplemental completed artifacts outside the registry: **8**.
- Log files audited: **2515**.
- Textual logs with failure markers: **71**; binary TensorBoard event files were inventoried without text decoding.

### Artifact classifications

| Classification | Count | Interpretation |
|---|---:|---|
| registry_complete | 485 | Completed artifacts represented in the provenance registry |
| supplemental_operator_audit | 6 | Completed branch-local MLP follow-up artifacts outside the older registry |
| supplemental_parity | 2 | Auxiliary parity artifacts; not a manuscript condition |
| unregistered_complete | 0 | Completed artifact requiring explicit manual role assignment |
| audit_or_smoke | 26 | Audit gates or smoke/preflight runs; excluded from performance claims |
| incomplete_training_or_missing_final_test | 132 | Training/log output without a final test contract |
| preflight_or_config_only | 31 | Configuration/checkpoint setup without a final test contract |
| legacy_log_only | 56 | Historical output with no recoverable final artifact |
| empty_or_unresolved | 31 | Empty or unresolved directory |

### Log classifications

| Log status | Count | Interpretation |
|---|---:|---|
| completion_marker | 669 | Contains a normal completion/final-test marker |
| completed_with_error_markers | 0 | Contains completion and a warning/failure marker; artifact contract governs final status |
| failure_marker | 71 | Contains a failure/termination marker without a completion marker |
| no_terminal_marker | 1236 | Log has content but no recognized terminal marker |
| empty | 71 | Empty log, usually a queued/preflight job output |
| binary_event_file | 468 | TensorBoard event files inventoried without text decoding |

## Interpretation

- All **485** registry artifacts are present on disk; the registry contains **271 CBraMod** records and **214 LaBraM** records.
- The registry aggregate contains **89 condition rows**; **88** are complete three-seed groups and one LaBraM--ISRUC dense row is a one-seed legacy group.
- The six branch-local MLP TUEV runs are complete and are retained in `supplemental_condition_results.csv`; they remain secondary operator evidence.
- LaBraM--ISRUC and LaBraM--SEED-V have recoverable supporting records, but the legacy/non-homogeneous records are not treated as strict matched RQ2 evidence.
- The four-cell primary matched RQ2 estimand is unchanged.

## Generated files

- `filesystem_artifact_audit.csv`: one row for every experiment-like directory.
- `log_inventory.csv`: one row for every discovered log/Slurm output file.
- `registry_condition_results.csv`: all 89 registry aggregate condition rows, including incomplete/legacy flags.
- `supplemental_condition_results.csv`: LaBraM FACED/ISRUC context and the TUEV branch-local MLP audit.
- `backbone_dataset_coverage.csv`: the ten-row coverage table requested for manuscript scope reporting.
- `artifact_classification_summary.csv`: classification counts by backbone and dataset.
- `log_failure_summary.csv`: failure-marker counts by log type.
