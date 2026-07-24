# Interaction-Aligned Adaptation: Canonical Cross-Backbone Execution Plan

Last updated: 2026-07-23

This is the canonical scientific plan for the cross-backbone study. The
CBraMod implementation lives in the sibling `EEGxPlore/EEGxPlore` repository;
the LaBraM implementation lives in this `LaBraM` repository. The two execution
paths are intentionally separate.

## 1. Non-negotiable repository boundary

| Backbone | Repository | Active branch | Current HEAD | Paper-grade rule |
| --- | --- | --- | --- | --- |
| CBraMod | `/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore` | `SEED-V` | `861c222` | Run CBraMod only here. |
| LaBraM | `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM` | `adaptor` | `973b61f` | Run LaBraM only here. |

The LaBraM import/substitution path under `EEGxPlore/EEGxPlore/models/labram_backbone.py`
is a historical engineering path. It must not be used for the revised
paper's LaBraM experiments. Existing results from that path remain historical
and are not silently deleted or relabeled.

`LaBraM-depth` is an isolated historical/depth worktree, not a third active
backbone repository.

## 2. Paper identity and research question

Provisional paper identity: **Interaction-Aligned Adaptation for EEG Foundation
Models**.

Research question:

> Does placing a common lightweight residual interaction primitive on the
> semantically defined, non-degenerate interaction axes of each EEG backbone
> improve the performance-efficiency trade-off over generic PEFT and simple
> fine-tuning controls?

The paper is not framed as one custom adapter for CBraMod and another custom
adapter for LaBraM. The contribution is the design rule and its controlled
backbone-specific instantiations.

## 3. Hypotheses, claims, and nonclaims

### Primary hypothesis

Under matched trainable-parameter and optimization budgets, an
interaction-aligned adapter has a better performance-efficiency trade-off than
LoRA, a generic bottleneck adapter, an axis-blind parameter-matched adapter,
and upper-layer-only fine-tuning across CBraMod and LaBraM.

### Mechanistic hypotheses

- Adapter utility depends on the encoder's native interaction structure and the
  realized downstream `[C,S,D]` token shape.
- Channel and patch interactions contribute differently when those axes remain
  operational.
- An interaction axis is eligible only when it is semantically defined and has
  a non-degenerate realized sequence length.

### Negative/diagnostic hypothesis

Explicit upper-depth aggregation does not consistently improve a simpler
interaction-aligned adapter under the evaluated settings.

### Nonclaims

We do not claim a universal EEG adapter, architecture-independent
generalization, that channel/patch decomposition is optimal for every backbone,
or that depth aggregation is generally harmful. We do not report a best-seed
headline. Until final multi-seed confirmation, use “we test whether.”

## 4. Formal design rule

For each backbone:

1. Identify its semantic axes and native encoder interactions.
2. Recover the realized downstream token shape `[B,C,S,D]` at a prespecified
   upper-layer location.
3. Mark an axis eligible only if it is semantically defined and its realized
   sequence length is greater than one.
4. Apply the same low-rank residual interaction primitive at eligible axes.
5. Allocate a fixed parameter budget and match controls within a prespecified
   tolerance.
6. Select architecture and hyperparameters from validation only.

“Native interaction structure” means the axis-specific operation already used
by the encoder to combine tokens, such as CBraMod's spatial/channel and
temporal/spectral branches. “Realized token shape” is the actual runtime
`[C,S,D]` grid after preprocessing and backbone patching. “Interaction-aligned”
means the adapter is inserted at an eligible native axis; “axis-blind” means a
parameter-matched residual module that does not use the backbone's semantic
axis structure.

## 5. Common adapter primitive

The paper-grade primitive is:

```text
z     = Down(LayerNorm(h))       # D -> r, r << D
m     = AxisMixer(z, axis)       # channel or patch native interaction
delta = Up(m)                    # r -> D
h_out = h + alpha * delta
```

Requirements:

- same primitive family in both repositories;
- `r` substantially smaller than native `D`;
- separate channel and patch parameters are allowed;
- zero-residual parity is exact within numerical tolerance;
- common dense parameters are not changed by adapter initialization;
- alpha, residual-to-backbone norms, gradients, parameter counts, and optimizer
  groups are logged;
- no depth routing, MoE specialists, compact EEG context, or PSD context in the
  primary method.

The low-rank aligned adapter is not LoRA. It retains native-axis residual
placement and uses a low-dimensional `Down -> AxisMixer -> Up` computation
inside the adapter. LoRA is a separate generic PEFT baseline that updates
low-rank factors associated with backbone linear weights. Primary experiments
must not combine LoRA with the native-axis adapter; the comparison is
LoRA-versus-generic-versus-axis-blind-versus-interaction-aligned.

Implementation status: historical direct LaBraM channel/patch results used
full-width `nn.MultiheadAttention` and remain pilot/mechanistic evidence. The
new low-rank Down/Mixer/Up implementation is now present in
`modeling_finetune.py` and passes construction, geometry, gradient, and
gamma-zero parity checks. Parameter-matched controls and dataset-level
confirmation remain pending.

## 6. Backbone audit

### LaBraM

Verified in `LaBraM/modeling_finetune.py`:

- `LaBraMNativeAxisResidualAdapter` at lines 280 onward operates on a
  `[B,C,S,D]` token grid.
- channel attention is created at line 325 and patch attention at line 332;
  runtime diagnostics are emitted in `forward` around lines 366-424.
- `forward_features` and the adapter depth-summary path are around lines
  748-839.
- direct training flags and LR-group controls are in
  `run_class_finetuning.py:113-158` and `run_class_finetuning.py:629-705`.
- gamma-zero and geometry/parity tests exist under `tests/`.

Eligibility rule for the paper: `channel` is eligible when `C > 1`; `patch` is
eligible when `S > 1`. The current SEED-V pilot has `C=62,S=1`, so channel is
eligible and temporal patch interaction is degenerate. ISRUC has six channels
and 30 temporal patches per epoch, so both axes are eligible. FACED has 32
channels and 10 patches in the validated LaBraM contract.

### CBraMod

Verified in `EEGxPlore/EEGxPlore/models/cbramod.py` and
`models/criss_cross_transformer.py`:

- `PatchEmbedding` constructs a `[B,C,S,D]` grid (`cbramod.py:358-382`).
- the native encoder separates spatial/channel and temporal/spectral processing
  through the criss-cross transformer.
- `models/attn_res.py:17` contains the current `FullAttnRes` component.
- `models/moe.py:190` contains the typed specialist FFN; current route mode is
  `typed_capacity_domain`, and dispatch supports `hard_capacity` and `soft`.
- router input modes include `full`, `delta_only`, `attnres_only`, and
  `baseline_only`; optional EEG, PSD, subject, and depth context are present.

The current CBraMod method is more complex than the revised primary method.
AttnRes/MoE/depth/context options remain available for historical diagnosis and
secondary ablations, but the primary revised implementation must not use the
legacy typed-MoE pathway.

## 7. Dataset and split contracts

Primary shared datasets:

- FACED
- ISRUC
- TUEV

Mechanistic/generalization dataset:

- SEED-V, with the existing within-subject protocol retained only as a
  reproducibility protocol and a separate subject-disjoint/grouped evaluation
  required for the paper-grade claim.

Optional fifth dataset:

- PhysioNet-MI, conditional on a completed data/provenance audit.

For every backbone/dataset pair record raw or serialized location, artifact
hash, sample shape, channel order, scaling, filtering/resampling, labels,
participant/session/trial identifiers, split membership, overlap audits,
checkpoint hash, classifier head, and pooling rule.

Current verified protocol facts:

- LaBraM ISRUC protocol is documented in `docs/isruc_protocol.md`: subjects
  1-80 train, 81-90 validation, 91-100 test; six bipolar channels; 30 temporal
  patches of 200 samples per epoch; stored sequence shape `[20,6,30,200]`;
  no additional `/100` scaling.
- LaBraM SEED-V uses the validated 62-electrode manifest plus CLS slot and the
  provisional status `metadata_verified_row_linkage_pending`.
- LaBraM FACED uses the validated 32-channel manifest and `/100` scaling.
- CBraMod dataset loaders exist for FACED, ISRUC, SEED-V, TUEV, PhysioNet-MI,
  and other legacy datasets in `EEGxPlore/EEGxPlore/datasets/`.

Data/provenance blockers are not waived by existing scores. SEED-V channel-row
linkage and subject-independent evaluation remain pending. ISRUC serialized-only
provenance is weaker than a raw-EDF audit. TUEV class support and split hashes
must be recorded for every final table.

## 8. Baseline registry

Every backbone/dataset pair must register:

1. `frozen_probe`: frozen backbone plus shared classifier/head;
2. `full_finetune`;
3. `upper_k_finetune`;
4. LoRA;
5. generic bottleneck adapter;
6. parameter-matched axis-blind residual adapter;
7. interaction-aligned adapter;
8. interaction-aligned plus one prespecified depth extension;
9. channel-off, patch-off, and both-off ablations where axes are eligible.

Current gaps:

- LaBraM direct repo: the formal low-rank primitive is implemented and parity
  tested. Separate LoRA, generic token-wise residual, and upper-k controls are
  now implemented and smoke-tested; their ISRUC multiseed results remain to be
  collected. Parameter-matched axis-blind controls are still a later budgeted
  refinement, not silently conflated with the generic control.
- EEGxPlore CBraMod: frozen/full/AttnRes/MoE pathways exist; LoRA and generic
  bottleneck baselines are absent or incomplete and must be added without
  importing the LaBraM implementation.
- Neither repository currently has a single cross-backbone result registry,
  parameter-budget checker, or final paired statistical aggregation contract.

A baseline is not called “matched” until trainable parameter counts are within
`±5%` of the aligned adapter target for that backbone.

## 9. Fairness and test governance

Development uses validation only, equal search budgets, fixed method families,
and development seeds `42,1024,3407` where the dataset supports them. After
method and protocol freeze, predeclare two additional final seeds and use the
same five seeds for all primary methods.

The primary checkpoint selector is validation Cohen's kappa for continuity with
the current LaBraM contract; the final paper should confirm a class-balanced
selection metric before freezing. Report balanced accuracy and macro F1 as main
test metrics, with kappa and weighted F1 secondary. Test data is accessed once
per frozen run and never used for architecture, hyperparameter, or checkpoint
selection.

Save commit, clean-tree status, artifact hashes, split hash, manifest hash,
checkpoint hash, complete CLI/config, trainable module names, parameter count,
optimizer groups, seed, best validation epoch, test metrics, memory, timing,
and throughput for every run.

## 10. Dataset-by-dataset and backbone-by-backbone execution

### Reusable run-ladder mindmap

For every backbone–dataset cell, follow this ladder once. The purpose is to
finish a section and promote it to multiseed confirmation, rather than tuning
one adapter family indefinitely.

```text
Backbone × dataset
├── 0. Protocol gate
│   ├── preprocessing, channel order, tensor shape, scaling
│   ├── split/subject audit and artifact hashes
│   └── one dense smoke run
├── 1. Baseline section
│   ├── frozen probe
│   ├── full fine-tune
│   ├── upper-k fine-tune
│   └── record validation trajectory and test checkpoint behavior
├── 2. Adapter section
│   ├── LoRA
│   ├── generic bottleneck
│   ├── parameter-matched axis-blind adapter
│   └── low-rank interaction-aligned adapter (separate from LoRA)
│       ├── channel-off
│       ├── patch-off
│       └── eligible axes together
├── 3. Mechanism section
│   ├── one prespecified depth extension
│   ├── realized [C,S,D] and gradient/ratio diagnostics
│   └── stop if depth or extra routing does not pass the aligned gate
├── 4. Promotion gate
│   ├── stop single-seed tuning when the run ladder is complete
│   ├── lock method/config from validation
│   └── run development seeds 42, 1024, 3407
└── 5. Final section
    ├── freeze protocol and method
    ├── run the five-seed primary block
    └── evaluate test only after validation selection is frozen
```

Validation metrics may be logged throughout development, and test metrics may
be written for every completed run for transparency. Test results must not
select epochs, architectures, or hyperparameters during the development
section.

### Phase 0: audit and contract freeze

Complete cheap repository audits, implement the registry/contract fields, and
resolve data provenance before broad training.

### Phase 1: finish the direct LaBraM repository

1. **SEED-V closeout:** archive the native channel pilot as exploratory; retain
   the conclusion that channel adaptation delays overfitting but is not yet a
   universal performance win. Implement the formal low-rank primitive and mark
   old full-width results as pilot.
2. **ISRUC:** audit the existing CBraMod-matched preprocessing, replicate dense
   LaBraM, then run frozen/full/upper-k and aligned channel+patch. Channel-only,
   patch-only, and depth are mechanism ablations.
3. **FACED:** rerun dense and the frozen common primitive under the final
   protocol; preserve historical patch/alpha pilots as archive evidence.
4. **TUEV:** establish preprocessing, class support, and dense baseline before
   any adapter comparison; then run the same registry.
5. **PhysioNet-MI:** proceed only after its preprocessing and subject split
   contract is verified.

### Phase 2: finish the CBraMod repository

Run CBraMod only in `EEGxPlore/EEGxPlore`:

1. SEED-V trust audit and mechanism decomposition: frozen/full/upper-k, simple
   AttnRes, axis-aligned primitive, axis-blind and generic controls; compare
   dispatch and router settings only as diagnostic ablations.
2. Freeze the simplest stable CBraMod method or invoke the empirical-study
   fallback if generic PEFT matches it.
3. Confirm on FACED, ISRUC, TUEV, and PhysioNet-MI using the frozen method.
4. Run cross-backbone comparisons only after both repositories have completed
   their baseline and aligned-method gates.

### Phase 3: final confirmation and mechanism study

Use five predeclared seeds for dense, LoRA, generic bottleneck,
axis-blind, aligned, and aligned+depth on the primary dataset/backbone cells.
Then run channel-off, patch-off, both-off, eligibility, within-subject versus
subject-disjoint SEED-V, and depth diagnostics.

## 11. Main experiment matrix

| Phase | Backbone | Datasets | Methods | Seeds | Gate |
| --- | --- | --- | --- | --- | --- |
| A | CBraMod, LaBraM | FACED, ISRUC, TUEV | frozen, full | 3 development | protocol replication |
| A | CBraMod, LaBraM | SEED-V | frozen, full | 3 development | geometry/protocol audit |
| A | CBraMod, LaBraM | PhysioNet-MI | frozen, full | 3 development | data audit first |
| B | CBraMod, LaBraM | same primary cells | upper-k, LoRA, generic bottleneck | 3 development | baseline registry complete |
| C | CBraMod, LaBraM | same primary cells | axis-blind, aligned | 3 development | matched parameter budget |
| D | CBraMod, LaBraM | same primary cells | aligned + depth | 3 development | secondary negative hypothesis |
| E | CBraMod, LaBraM | frozen primary cells | all primary methods | 5 final | method/protocol frozen |
| F | CBraMod, LaBraM | SEED-V and ISRUC first | channel/patch/eligibility/depth interventions | 3-5 | mechanism attribution |

## 12. Mechanistic and efficiency analysis

Log realized `[C,S,D]`, eligibility mask, sequence lengths, alpha, residual
ratios, adapter/backbone gradients, Q/K/V/output gradients, adapter-on/off
validation effects, channel/patch interventions, and aligned-versus-axis-blind
deltas. A nonzero gradient is not evidence of usefulness without an
intervention or matched performance difference.

Measure trainable and total parameters, percentage, checkpoint size, peak GPU
memory, wall-clock time, GPU-hours, throughput, latency, and FLOPs/MACs where
practical under identical hardware, precision, batch size, and timing windows.

## 13. Decision gates and stop rules

### Method-paper gate

Require a positive mean aligned effect on both backbones, support on at least
two primary datasets, stable seed behavior, a Pareto advantage over generic
controls, a matched-budget aligned-versus-axis-blind effect, and a measured
efficiency benefit.

### Empirical-study fallback

If generic PEFT matches aligned adaptation, only one backbone benefits, effects
are strongly dataset-specific, or upper-k fine-tuning explains the result,
change the paper identity to:

> When Does Structure-Aware PEFT Help EEG Foundation Models? A Controlled Study
> Across CBraMod and LaBraM.

Stop adding depth, routing, context, or new adapter branches when the simple
aligned primitive has not passed the matched-baseline gate. Do not expand a
failed method across all datasets merely to increase table size.

## 14. Immediate next ten actions

1. Treat this file as the canonical plan in both repositories.
2. Keep all LaBraM runs in `LaBraM` and all CBraMod runs in `EEGxPlore/EEGxPlore`.
3. Finish the LaBraM ISRUC data/protocol audit and dense baseline replication.
4. Implement and parity-test the common low-rank axis residual primitive in
   LaBraM without changing dense common weights.
5. Collect the queued LaBraM ISRUC LoRA, generic, and upper-k control packets;
   then add the parameter-matched axis-blind check if the budget requires it.
6. Freeze the LaBraM ISRUC method-development budget and run validation-only
   development seeds.
7. Complete LaBraM TUEV and formal FACED baseline/adapter cells.
8. Audit the CBraMod paper/code dispatch and result provenance in EEGxPlore.
9. Implement CBraMod generic/LoRA/axis-blind controls without importing LaBraM.
10. Freeze both backbone protocols before any five-seed final test block.

## 15. Human approval questions

- Should PhysioNet-MI be a primary fifth dataset or an optional extension after
  the three shared datasets and SEED-V mechanism study?
- Should final checkpoint selection remain validation kappa, or change to a
  prespecified class-balanced metric such as balanced accuracy?
- What two additional final seeds should be added to the existing development
  packet `42,1024,3407`?
- Is the existing SEED-V within-subject protocol acceptable as a reproducibility
  result if the subject-disjoint evaluation is reported separately?
- Should the current CBraMod AttnRes/MoE results be presented only as historical
  evidence or retained as explicit negative/diagnostic baselines?
