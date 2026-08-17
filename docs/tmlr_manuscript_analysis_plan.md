# TMLR manuscript analysis plan

Status: manuscript-focused analysis plan, 2026-08-17.

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

- Table 1: overall performance and trainable parameters.
- Table 2: valid native-minus-axis-blind paired effects for BA and macro-F1.
- Table 3: frozen versus backbone-trainable comparisons.
- Figure 1: common adapter and experimental logic.
- Figure 2: alignment-effect forest plot showing seed-level effects and cell
  means.
- Figure 3: performance-efficiency frontier.

Per-epoch diagnostics, residual/gradient behavior, failed runs, and additional
regularization trials belong in supplementary material unless required to
explain a primary result.

## Experimental freeze

No new training run is justified by a negative or mixed result. A new run may
be considered only if the focused audit identifies a missing or corrupted
canonical comparison required by a prespecified retained claim.
