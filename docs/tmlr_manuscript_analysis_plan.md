# TMLR manuscript analysis plan

Status: manuscript-focused analysis plan; focused results snapshot completed,
2026-08-18.

The broad training program is frozen. The historical execution plan remains
unchanged as provenance; this document defines the smaller evidence subset used
for the manuscript.

## Paper question

Does interaction-aligned lightweight adaptation provide value beyond matched
generic trainable capacity when adapting EEG foundation models?

## Main conclusion to test

Interaction-aligned adaptation does not provide a consistent advantage over
matched generic capacity across EEG foundation models. Its relative benefit
varies across backbone, downstream task, realized interaction geometry, and
trainability regime.

## Research questions

1. **RQ1 — Adaptation:** Does lightweight native adaptation improve over a
   frozen probe?
2. **RQ2 — Alignment:** Does native interaction alignment improve over a
   parameter-matched axis-blind control?
3. **RQ3 — Backbone freedom:** What changes when the pretrained backbone is
   allowed to update?

SEED-V geometry, serialized-data limitations, and optimization trajectories are
boundary/mechanistic analyses, not independent factorial research questions.

## Inclusion rules

The manuscript manifest is generated before interpreting scores. It includes:

- complete canonical three-seed groups with seeds `{42, 1024, 3407}`;
- one deterministic native variant per backbone/dataset, preferring
  channel-plus-patch, then channel, then patch;
- its frozen probe;
- axis-blind controls only when the automatic parameter contract passes;
- generic, LoRA, upper-layer, and dense/full references when complete;
- SEED-V and PhysioNet-MI boundary rows under explicit boundary roles.

No run is selected because it wins or loses. The full registry remains the
provenance layer for failed, duplicated, exploratory, and legacy artifacts.

## Primary versus supporting use

Rows in `analysis/tmlr_manuscript/manuscript_inclusion_manifest.csv` are
`PRIMARY_CANDIDATE` rows requiring focused audit. The paper's direct RQ2 claim
uses only rows whose native/control pair passes the parameter contract and is
confirmed by the source configuration. LaBraM TUEV channel and patch pairs
remain outside primary RQ2 because they fail the 5% parameter-budget rule;
channel-plus-patch is the currently eligible LaBraM TUEV pair.

Legacy or incomplete results may be reported as supporting context, but they
are not pooled into the direct alignment claim.

## Main tables and figures

The manuscript should use a compact, fixed visual package: five main tables and
up to five main figures. TMLR has no numeric table/figure cap; this budget is a
clarity decision, not a submission rule. Four result tables would be sufficient
for the statistical evidence, but a small adapter/control table improves
technical auditability. A four-figure version is acceptable if the capacity or
training-dynamics figure is moved to the supplement.

Main tables:

1. Dataset and evaluation contract: task, class count, segment and channel
   geometry, split rule, eligible axes, backbone, seed count, and
   primary/supporting qualification.
2. Adapter and control specification: trainable components, residual insertion,
   mixed axes, dropout, residual scaling, parameter budget, and backbone state.
3. Primary validation-locked performance: BA and macro-F1 mean +/- standard
   deviation for the principal methods.
4. Direct native-minus-axis-blind paired effects: BA and macro-F1 effect mean
   +/- standard deviation, seed-level positivity or seed differences, and
   parameter-validity status.
5. Backbone plasticity and resource context: frozen-to-trainable increments for
   aligned and control settings plus trainable parameters; include memory/time
   only when measured under a comparable protocol.

Main figures:

1. End-to-end research workflow from dataset to validation-locked RQ1/RQ2/RQ3
   analyses.
2. Adapter mechanism and native-versus-axis-blind control geometry. The CBraMod
   panel should show its spatial/channel versus temporal/patch feature-half
   split; the LaBraM panel should show channel- and patch-axis mixing on the
   normalized embedding grid without importing that split. Include tensor
   dimensions, residual path, trainable components, dropout, and scale.
3. Prespecified training-dynamics small multiples for validation and per-epoch
   test behavior; full cell-level traces belong in the supplement if crowded.
4. Two-panel paired-effect forest plot with every seed, cell mean +/- standard
   deviation, and a zero-effect line.
5. Frozen-to-trainable plasticity contrast with capacity/performance context, or
   a parameter scatter moved to the supplement if the combined figure is dense.

The main visuals must use a consistent cell order, metric convention, seed
aggregation, and validation-locked checkpoint rule. They should expose
variability and mixed/negative effects rather than emphasize only favorable
cells. The supplement should contain the complete method matrix, raw per-seed
values, per-epoch test diagnostics, checkpoint traces, class-level TUEV results,
regularization trials, failed/corrected runs, and provenance/configuration
audits.

## Experimental freeze

No new training run is justified by a negative or mixed result. A new run may
be considered only if the focused audit identifies a missing or corrupted
canonical comparison required by a prespecified retained claim.

## Focused results finalization

The manuscript-final results package is:

`analysis/tmlr_manuscript/manuscript_final_v1/`

It is generated by `scripts/build_manuscript_final_results.py` from the compact
manifest. The package verifies 180 of 192 selected rows and retains 12 legacy
rows as supporting-only because their frozen-evaluation-mode provenance is
incomplete. It contains 12 primary valid RQ2 seed-pairs, with SEED-V and
PhysioNet-MI retained as boundary/supporting analyses. The broad registry
remains intentionally unfreezed and provenance-only; it is not needed to block
the focused manuscript snapshot.
