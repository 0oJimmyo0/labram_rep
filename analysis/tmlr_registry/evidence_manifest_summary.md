# TMLR evidence manifest

This manifest is a conservative manuscript-audit layer over the deterministic artifact registry.
It does not promote legacy or incomplete artifacts into the primary evidence set automatically.

Artifacts: 483
Automatic candidates — primary: 264; supporting: 163; pilot: 34; excluded: 22
Manual artifact status: UNREVIEWED for every row; automatic candidate labels are not manuscript adjudications.
Paired metric effects: 612 rows across 204 summaries
Structurally invalid paired-effect rows: 18; these are excluded from provisional RQ2 eligibility.

## Alignment-control coverage

| Backbone | Dataset | Axis | Control status | Artifacts |
|---|---|---|---|---:|
| CBraMod | faced | channel | control_only | 5 |
| CBraMod | faced | channel | primary_pair_unverified | 3 |
| CBraMod | faced | channel_patch | control_only | 5 |
| CBraMod | faced | channel_patch | primary_pair_unverified | 3 |
| CBraMod | faced | patch | control_only | 4 |
| CBraMod | faced | patch | primary_pair_unverified | 3 |
| CBraMod | isruc | channel | control_only | 1 |
| CBraMod | isruc | channel | primary_pair_unverified | 3 |
| CBraMod | isruc | channel_patch | control_only | 3 |
| CBraMod | isruc | channel_patch | primary_pair_unverified | 3 |
| CBraMod | isruc | patch | control_only | 1 |
| CBraMod | isruc | patch | primary_pair_unverified | 3 |
| CBraMod | physionet_mi | channel | control_only | 1 |
| CBraMod | physionet_mi | channel | primary_pair_unverified | 3 |
| CBraMod | physionet_mi | channel_patch | control_only | 2 |
| CBraMod | physionet_mi | channel_patch | primary_pair_unverified | 3 |
| CBraMod | physionet_mi | patch | control_only | 1 |
| CBraMod | physionet_mi | patch | primary_pair_unverified | 3 |
| CBraMod | seedv | channel | control_only | 4 |
| CBraMod | seedv | channel | primary_pair_unverified | 3 |
| CBraMod | seedv | patch | control_only | 3 |
| CBraMod | seedv | patch | primary_pair_unverified | 3 |
| CBraMod | tuev | channel | control_only | 2 |
| CBraMod | tuev | channel | primary_pair_unverified | 3 |
| CBraMod | tuev | channel_patch | control_only | 6 |
| CBraMod | tuev | channel_patch | primary_pair_unverified | 3 |
| CBraMod | tuev | patch | control_only | 1 |
| CBraMod | tuev | patch | primary_pair_unverified | 3 |
| LaBraM | FACED | patch | no_primary_control | 6 |
| LaBraM | ISRUC | patch | control_only | 3 |
| LaBraM | PhysioNet-MI | channel_patch | no_primary_control | 4 |
| LaBraM | SEED-V | channel | no_primary_control | 4 |
| LaBraM | SEED-V | patch | no_primary_control | 4 |
| LaBraM | TUEV | channel | primary_pair_structurally_invalid | 3 |
| LaBraM | TUEV | channel_patch | primary_pair_unverified | 3 |
| LaBraM | TUEV | patch | primary_pair_structurally_invalid | 3 |

## Interpretation rule

The strongest alignment claim is restricted to manually validated RQ2 pairs. Current paired effects are provisional and require configuration-level nuisance matching before use in the manuscript.

## Main paired-effect files

- `paired_effects.csv`: same-seed metric differences.
- `paired_effects_summary.csv`: mean, standard deviation, seed count, and positive-seed count.
- `manual_review_queue.csv`: candidate rows prepared for human adjudication.
- `pair_review_queue.csv`: pair-level contract review required before final RQ2 analysis.
