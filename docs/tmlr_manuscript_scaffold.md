# TMLR manuscript scaffold: Interaction-Aligned Adaptation for EEG Foundation Models

Status: evidence inventory and paired-effect audit, 2026-08-16

This is a working cross-backbone manuscript scaffold. It separates completed
evidence from exploratory artifacts and from results that are not yet
comparable enough for a pooled claim.

## 1. Proposed paper identity

**Primary title:** Interaction-Aligned Adaptation for EEG Foundation Models

**Fallback title:** When Does Encoder-Structure-Aware Adaptation Help EEG
Foundation Models?

### Research question

Does placing a common lightweight residual interaction primitive on the
semantically defined, non-degenerate interaction axes of an EEG foundation
model provide value beyond generic trainable capacity and parameter-matched
axis-blind controls?

### Prospective design rule

1. Identify the backbone's semantic representation axes and native interaction
   sites.
2. Record the realized downstream geometry `[C,S,D]`.
3. Exclude axes with no meaningful interaction sequence (`length <= 1`).
4. Apply the same high-level `Down -> native-axis mixer -> Up` residual family.
5. Match trainable capacity and report the realized parameter count.
6. Freeze the mapping and select checkpoints using validation only.

The backbone-specific part is the axis mapping required by the encoder. The
paper must not present CBraMod and LaBraM as two separately tuned adapters.

### Evidence-freeze artifacts

The deterministic registry is supplemented by a conservative manuscript audit:

- `analysis/tmlr_registry/evidence_manifest.csv` records automatic candidate
  labels, orthogonal RQ eligibility, and human-review fields. Every artifact's
  current `manual_status` remains `UNREVIEWED`.
- `analysis/tmlr_registry/paired_effects.csv` contains same-seed metric
  differences for native-versus-control comparisons, with provisional pair
  validity and parameter-match metadata.
- `analysis/tmlr_registry/paired_effects_summary.csv` aggregates those effects
  without pooling seeds as independent observations.
- `analysis/tmlr_registry/manual_review_queue.csv` is the review template for
  candidate artifacts.
- `analysis/tmlr_registry/pair_review_queue.csv` is the separate pair-level
  contract review template; structurally invalid parameter-mismatched pairs
  are marked ineligible automatically.
- `scripts/finalize_tmlr_snapshot.py` refuses to create `frozen_v1/` until
  candidate artifacts and retained RQ2 pairs have completed manual review.
- `analysis/tmlr_registry/evidence_manifest_summary.md` records control
  coverage and the interpretation rule for the primary claim.

The strongest causal alignment claim is restricted to manually validated RQ2
native/control pairs. In the current inventory, CBraMod has matched axis-blind coverage across
the listed datasets, while LaBraM has strict matched coverage on TUEV. LaBraM
ISRUC has an axis-blind packet but its native comparator remains supporting
legacy evidence; FACED, PhysioNet-MI, and SEED-V do not have the same strict
matched axis-blind pair.

## 2. What the completed experiments currently show

The central result is conditional, not universal:

> Native-axis adapters are technically active and can improve a frozen
> backbone in selected dataset/backbone regimes, but the gain is not stable
> across datasets, seeds, metrics, or generic PEFT controls. When the backbone
> is fully trainable, performance is usually explained primarily by backbone
> updates rather than by the residual adapter.

This is a credible TMLR result if the paper makes the boundary conditions and
negative cases central. It is not yet evidence for the stronger claim that
native alignment generally beats generic PEFT or full fine-tuning.

### Completed three-seed or locked packets

| Dataset / backbone | Main evidence | Current interpretation |
| --- | --- | --- |
| CBraMod / FACED | Corrected frozen probe, native channel/patch/channel+patch, generic, LoRA, upper-2, axis-blind; dense and native-full references | Frozen native channel+patch is modestly above the probe, but LoRA and upper-2 are stronger. Full native results are near dense because the backbone is trainable. |
| CBraMod / SEED-V | Corrected frozen r=64 packet, channel as the only genuine interaction axis, singleton patch control, generic, LoRA, upper-2, axis-blind, native-full channel | Frozen channel does not beat the corrected probe on the three-seed mean; native full channel is competitive with dense. The patch branch is a capacity control, not temporal interaction evidence. |
| CBraMod / ISRUC | 12-condition x 3-seed packet; dense, probe, native axes, generic, axis-blind, LoRA, upper-2, native-full | Frozen channel+patch is active but below the probe mean; upper-2 and native-full controls are stronger. The result supports bounded adaptation, not universal superiority. |
| CBraMod / PhysioNet-MI | 12-condition x 3-seed packet with audit and diagnostics | Frozen native means are approximately probe-level; generic and LoRA are competitive, upper-2 is strongest among constrained controls. Full native patch is near dense but is mostly a full-fine-tuning result. |
| LaBraM / FACED | Three-seed dense and native patch; corrected frozen dense/patch/generic controls | Frozen dense is underfit, generic is the strongest frozen control, and native patch does not provide a robust positive result. Treat FACED as a boundary/negative case. |
| LaBraM / SEED-V | Three-seed dense-versus-channel primary packet; corrected controls exist | Native channel is essentially tied with dense and does not establish a consistent gain. The existing protocol remains within-subject and needs a subject-disjoint qualification. |
| LaBraM / ISRUC | Three-seed dense, low-rank native axes, generic, upper-2, and related controls | Native low-rank axes are slightly below dense and do not beat the strongest simple controls. This is useful negative evidence for universal alignment claims. |
| LaBraM / TUEV | Locked 11-condition x 3-seed packet | Frozen channel improves over the frozen probe conditionally, but native full channel is slightly below dense and channel+patch is weaker. TUEV is a strong boundary case. |
| LaBraM / PhysioNet-MI | Locked three-seed dense, frozen, low-rank native, generic, LoRA, upper-2, and native-full packets | Frozen native adaptation gives a modest conditional gain over the frozen probe; full native patch/channel+patch are near or slightly above dense, but backbone updates dominate the interpretation. |

### Representative locked means already available

These values are from the repository status/memory records and are intended
for manuscript tables after the final registry audit. They use validation-
kappa-selected checkpoints and test evaluation after selection.

| Dataset / backbone / condition | Test BA (mean +/- SD) | Reading |
| --- | ---: | --- |
| CBraMod / ISRUC / dense | 0.7857 +/- 0.0091 | Dense reference |
| CBraMod / ISRUC / frozen channel+patch | 0.7554 +/- 0.0076 | Active but below probe/dense mean |
| CBraMod / PhysioNet-MI / dense | 0.6228 +/- 0.0035 | Dense reference |
| CBraMod / PhysioNet-MI / frozen channel+patch | 0.5432 +/- 0.0034 | Approximately probe-level |
| LaBraM / ISRUC / dense | approximately 0.7936 | Dense reference |
| LaBraM / ISRUC / low-rank native channel | approximately 0.7924 +/- 0.0055 | Slightly below dense |
| LaBraM / TUEV / dense | 0.5961 +/- 0.0125 | Dense reference |
| LaBraM / TUEV / native channel | 0.5934 +/- 0.0108 | Slightly below dense |
| LaBraM / TUEV / frozen probe | 0.3658 | Frozen reference |
| LaBraM / TUEV / frozen channel | 0.4323 +/- 0.0186 | Conditional frozen gain |
| LaBraM / PhysioNet-MI / dense | approximately 0.6093 +/- 0.0147 | Dense reference |
| LaBraM / PhysioNet-MI / frozen channel+patch | approximately 0.283 | Modest gain over frozen probe |

CBraMod TUEV now has a canonical 12-condition multi-seed packet under the
20-epoch operational contract. The earlier dropout, gamma, stability, and
head-dropout runs are separate seed-42 diagnostics and must not be mixed into
this locked packet.

## 3. Mechanistic story to develop

The most interesting cross-backbone result is a regime separation:

1. **Frozen backbone:** the adapter is the only representation-learning path.
   It can be active, receive nonzero native Q/K/V gradients, and occasionally
   improve the frozen probe. However, early classifier/adaptor overfitting and
   seed sensitivity limit the benefit.
2. **Full fine-tuning:** native adapters can coexist with strong results, but
   residual ratios and backbone update norms show that the backbone is doing
   most of the work. These runs should be reported as boundary controls, not
   as evidence that the adapter caused the gain.
3. **Geometry:** eligibility matters. SEED-V has a genuine channel sequence but
   a singleton patch axis; ISRUC and TUEV have both eligible axes. The results
   do not support treating every nominal axis as equally useful.
4. **Optimization:** many constrained runs peak early while train loss keeps
   falling. The classifier head is a major part of the instability, so
   validation trajectories and post-hoc per-epoch test diagnostics help
   distinguish a useful residual from a short-lived checkpoint effect.

This makes the paper more informative than a single leaderboard comparison:
it asks when structure-aware adaptation is useful, what regime it helps, and
when it fails.

## 4. Claim ladder

### Claims currently supportable

- A common native-axis residual family can be implemented independently in
  CBraMod and LaBraM with strict checkpoint, geometry, trainability, and
  validation-selection reporting.
- Native-axis adapters are not dormant: corrected artifacts contain nonzero
  adapter gradients and residual activity while frozen-backbone update norms
  remain zero.
- Frozen-backbone adaptation can produce conditional gains, but the gains are
  dataset- and seed-dependent and often do not exceed generic controls.
- Full native-adapter results should be interpreted as constrained-model
  comparisons only when backbone update norms are reported.
- Early overfitting and checkpoint timing are part of the adapter behavior,
  especially on TUEV.

### Claims not yet supportable

- “Native alignment consistently beats LoRA, generic bottlenecks, or
  upper-layer tuning.”
- “The method is parameter-matched across both backbones.”
- “The same low-rank primitive has been tested under one identical numerical
  contract on every dataset.”
- “The adapter improves subject-independent SEED-V generalization.”
- “The TUEV result is a final CBraMod multi-seed conclusion.”

## 5. Remaining high-value work, in priority order

The goal is evidence closure, not another broad hyperparameter sweep.

1. Manually review the generated evidence manifest and promote only artifacts
   that satisfy the locked protocol into the primary tables.
2. Recompute the final paired-effect tables and figures from the manifest,
   reporting individual seed differences, mean differences, standard
   deviations, and positive-seed counts.
3. Resolve the remaining legacy metadata cells. They may be manually promoted
   to supporting or primary evidence only when their logs/configurations prove
   the required contract; otherwise they remain supporting.
4. Preserve the SEED-V within-subject and singleton-patch qualification and the
   PhysioNet-MI serialized-only qualification.
5. Freeze commits, configuration hashes, checkpoint hashes, split hashes, and
   the final anonymized reproducibility snapshot.

### Runs that are not currently justified

- More TUEV dropout/gamma sweeps after the two active jobs.
- More depth, alpha, rank, or learning-rate sweeps on ISRUC or PhysioNet-MI.
- Additional CBraMod FACED/SEED-V seeds unless the artifact audit finds an
  integrity failure.
- Treating a test-best later epoch as a new primary result.

## 6. Manuscript structure

1. **Introduction:** frozen foundation models are cheap to reuse but may have
   a representation mismatch; motivate structure-aware adaptation as a
   falsifiable design rule.
2. **Method:** define the common residual operator, axis eligibility, zero-init
   parity, parameter accounting, and independent CBraMod/LaBraM mappings.
3. **Experimental protocol:** data audits, splits, seeds, validation-only
   selection, controls, metrics, and efficiency measurements.
4. **Main results:** mean +/- SD tables for dense, frozen probe, generic,
   LoRA, upper-k, axis-blind, and native aligned conditions.
5. **Frozen-versus-full analysis:** show probe-relative gains, backbone update
   norms, residual ratios, and why full native results cannot establish adapter
   causality by themselves.
6. **Trajectory and stability analysis:** train/validation gaps, per-epoch
   test diagnostics, early overfitting, and TUEV per-class metrics/confusions.
7. **Geometry and mechanism:** eligible versus singleton axes and
   native-versus-axis-blind comparisons.
8. **Limitations:** seed count, dataset-specific protocols, subject
   provenance, serialized data, and the boundary-case results.
9. **Conclusion:** native alignment is a conditional tool for limited-backbone
   adaptation, not a universal replacement for full fine-tuning.

## 7. Figures and tables to prepare

- **Figure 1:** prospective construction rule and the CBraMod/LaBraM axis map.
- **Figure 2:** performance versus trainable parameters, with seed error bars.
- **Figure 3:** frozen probe-relative BA/macro-F1 by dataset and backbone.
- **Figure 4:** representative validation trajectories, including residual
  ratio, validation metric, and train loss; move post-hoc per-epoch test
  diagnostics to the supplement unless needed for a specific audit.
- **Figure 5:** TUEV per-class recall/F1 and confusion matrices for probe,
  aligned, generic, LoRA, and dense references.
- **Table 1:** dataset geometry, eligibility, split, and provenance.
- **Table 2:** complete primary multi-seed results.
- **Table 3:** parameter/runtime/memory accounting.
- **Appendix:** all exploratory runs, failed/corrected artifacts, sensitivity
  checks, and the two CBraMod TUEV boundary jobs after completion.

## 8. Publication decision gate

Use the method-paper framing only if the final matched registry shows that the
aligned method improves at least two primary datasets across both backbones,
beats or Pareto-dominates generic/LoRA/upper-layer controls, and beats a
parameter-matched axis-blind control.

Otherwise use the empirical-study framing: structure-aware adaptation is a
useful but conditional intervention whose value depends on backbone geometry,
frozen-versus-full training regime, and optimization stability. That outcome
is still valuable if the negative cases are analyzed deeply and the causal
controls are complete.

## 9. Aggregation artifacts

The current traceable registry is generated by
`LaBraM/scripts/build_tmlr_registry.py` and writes:

- `LaBraM/analysis/tmlr_registry/all_artifacts.csv`: every completed artifact
  with provenance and contract fields;
- `LaBraM/analysis/tmlr_registry/canonical_candidates.csv`: deterministic
  manuscript candidates, excluding known smoke/repair/stability artifacts;
- `LaBraM/analysis/tmlr_registry/aggregate.csv`: mean and sample standard
  deviation by backbone, dataset, method, and axis;
- `LaBraM/analysis/tmlr_registry/isruc_axisblind_trajectory.csv`:
  validation-selected versus test-best epoch diagnostics for the three
  LaBraM ISRUC axis-blind seeds.
- `LaBraM/analysis/tmlr_registry/isruc_axisblind_trajectory_mean.csv`: seed
  mean and standard deviation curves for the trajectory figure.

Candidate rows remain subject to manual contract review, especially legacy
LaBraM SEED-V and ISRUC artifacts whose older run configurations do not record
all current strict-load fields.
