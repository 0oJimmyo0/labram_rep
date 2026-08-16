# Interaction-Aligned Adaptation: Canonical Cross-Backbone Execution Plan

Last updated: 2026-08-02

This is the canonical scientific plan for a two-manuscript program. The TMLR
interaction-aligned study and the ICASSP CBraMod depth study have separate
questions, repositories, methods, result registries, and claims.

## Active execution priority

Only TMLR is active now. The LaBraM/TUEV operational closure is complete under
the existing three-seed registry. The clean original-CBraMod clone has passed
its FACED provenance, checkpoint, geometry, and native-branch gates. FACED is
now in the controlled comparison stage. ICASSP remains deferred and is not an
active workstream.

## 1. Non-negotiable repository boundary

| Backbone | Repository | Active branch | Current HEAD | Paper-grade rule |
| --- | --- | --- | --- | --- |
| CBraMod, TMLR | `/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod` | `main` | `f5a9668` | Build and evaluate the CBraMod-specific instantiation here; do not use EEGxPlore. |
| CBraMod, ICASSP | `/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore` | `SEED-V` | `861c222` | Depth probing/fusion only; CBraMod only. |
| LaBraM | `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM` | `adaptor` | `2e5840c` at this update | Run LaBraM only here. |

The LaBraM import/substitution path under `EEGxPlore/EEGxPlore/models/labram_backbone.py`
is a historical engineering path. It must not be used for either manuscript's
new LaBraM/TMLR evidence. Existing results from that path remain historical
and are not silently deleted or relabeled.

`LaBraM-depth` is an isolated historical/depth worktree, not a third active
backbone repository.

### 1.1 Manuscript separation override

TMLR is **Interaction-Aligned Adaptation for EEG Foundation Models** (fallback:
**When Does Interaction-Aligned Adaptation Help EEG Foundation Models?**). It
tests one common low-rank residual primitive, `Down -> native-axis mixer ->
Up`, on eligible native interaction axes of CBraMod and LaBraM. The realized
axis must be semantically defined and have sequence length greater than one.
LaBraM uses channel/patch branches over `[B,C,S,D]`; CBraMod preserves its
native spatial/channel and temporal/spectral branch separation and
recombination. The TMLR CBraMod implementation must be developed in the new
clean clone of original CBraMod, not in EEGxPlore.

TMLR requires frozen-plus-head, full fine-tuning, upper-block, independent
LoRA, generic bottleneck, parameter-matched axis-blind, native frozen
channel-only/patch-only/channel-plus-patch controls, and native
full-backbone-plus-adapter channel-only/patch-only/channel-plus-patch controls.
It uses exactly three
final seeds `{42,1024,3407}`, validation-selected test evaluation, BA and
macro-F1 as primary metrics, and parameter/efficiency plus residual, alpha,
gradient, and update diagnostics. LoRA is a separate generic PEFT method; it
must not be combined with the native-axis adapter. Depth aggregation, depth
routing, typed specialist MoE, compact EEG/PSD routers, and the historical
depth-aware model are excluded from TMLR.

ICASSP is **Depth Probing and Fusion in Pretrained EEG Encoders** (fallback:
**When Do Earlier CBraMod Layers Improve Downstream EEG Decoding?**). It stays
in EEGxPlore and uses CBraMod only, with frozen or nearly frozen backbone
probes/fusion at prespecified depths. It owns final-layer probes,
individual-layer probes, uniform fusion, learned global depth weights, and an
interpretable compact sample-conditioned probe if used. It owns depth weights,
entropy, earlier-layer mass, seed consistency, class-conditioned weights,
per-class metrics, depth-removal interventions, and efficiency diagnostics.
This design is deferred until the TMLR cross-backbone matrix is locked.

ICASSP must not include LaBraM, the TMLR interaction-aligned adapter, TMLR
channel/patch eligibility, the TMLR low-rank operator, LoRA-vs-aligned or
axis-blind comparisons, or TMLR tables/figures. Complete and lock TMLR first,
then execute ICASSP separately. Loaders/checkpoints may be shared when
necessary, but no numerical results, trained model families, figures, tables,
manuscript text, primary analyses, or central conclusions may be reused across
the two papers. Historical NeurIPS material remains context only.

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

The TMLR alignment test is the native-axis versus parameter-matched
axis-blind comparison. Depth probing and depth aggregation are not TMLR
conditions; they belong exclusively to the separate ICASSP CBraMod study.

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

### CBraMod (historical EEGxPlore implementation; ICASSP audit)

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
8. channel-off, patch-off, and both-off ablations where axes are eligible;
9. eligibility-mask and realized-geometry diagnostics.

Current gaps:

- LaBraM direct repo: the formal low-rank primitive is implemented and parity
  tested. Separate LoRA, generic token-wise residual, and upper-k controls are
  now implemented and smoke-tested; their ISRUC multiseed results remain to be
  collected. Parameter-matched axis-blind controls are still a later budgeted
  refinement, not silently conflated with the generic control.
- CBraMod TMLR clone: the checkpoint, geometry, native-branch, and preprocessing
  audit is verified for FACED. Dense three-seed confirmation and seed-42
  comparison screens are complete. The native full-backbone-plus-adapter mode
  is implemented as `native_full_finetune`; its construction gate and seed-42
  screens remain pending. Native frozen multiseeds remain gated by residual
  scale/trajectory review. Do not transfer the EEGxPlore AttnRes/MoE/depth/router
  path.
- EEGxPlore CBraMod: remains the ICASSP depth-probing/fusion repository; its
  TMLR LoRA, generic, axis-blind, and aligned controls must not be added to the
  ICASSP result registry.
- Neither repository currently has a single cross-backbone result registry,
  parameter-budget checker, or final paired statistical aggregation contract.

A baseline is not called “matched” until trainable parameter counts are within
`±5%` of the aligned adapter target for that backbone.

## 9. Fairness and test governance

Development uses validation only, equal search budgets, fixed method families,
and the fixed three-seed packet `42,1024,3407` where the dataset supports it.
The same three seeds are the multiseed results and the locked final test
results; no five-seed confirmation block is part of this project.

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
│       ├── frozen-backbone regime: channel, patch, channel+patch
│       └── full-backbone-plus-adapter regime: channel, patch, channel+patch
├── 3. Mechanism section
│   ├── realized [C,S,D] and gradient/ratio diagnostics
│   ├── channel-off, patch-off, and both-off interventions
│   └── stop if an axis is ineligible or the matched alignment gate fails
├── 4. Promotion gate
│   ├── stop single-seed tuning when the run ladder is complete
│   ├── lock method/config from validation
│   └── run development seeds 42, 1024, 3407
└── 5. Final section
    ├── freeze protocol and method
    ├── run the locked three-seed primary block
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
4. **TUEV:** operationally closed with the complete 11-condition x three-seed
   registry. Retain its mixed/negative native-adapter result as a boundary
   case and do not launch duplicate tuning.
5. **PhysioNet-MI:** proceed only after its preprocessing and subject split
   contract is verified.

### Phase 2: finish the dedicated CBraMod TMLR repository

Run the CBraMod TMLR matrix only in the new clean clone of the original
CBraMod repository:

1. audit preprocessing, native branch geometry, checkpoint loading, and
   channel/temporal eligibility;
2. implement the same high-level `Down -> native-axis mixer -> Up` family and
   the frozen/full/upper-k, LoRA, generic, axis-blind, frozen-aligned, and
   full-backbone-plus-aligned controls;
3. confirm the locked protocol on the TMLR dataset matrix and collect the
   three-seed summaries;
4. do not import LaBraM code or use the EEGxPlore depth/router implementation.

### CBraMod FACED adapter design gate (2026-07-31)

The dedicated clone is verified at `/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod`,
remote `https://github.com/wjq-learning/CBraMod.git`, base commit `0ff6be9`.
The first implementation uses one common TMLR primitive rather than a second
backbone-specific method:

```text
z     = Down(LayerNorm(h_native))
m     = native-axis attention mixer(z)
delta = Up(m)
h_out = h + alpha * gamma * delta
```

CBraMod's own `criss_cross_transformer.py` defines the native semantics. For
`[B,C,S,D]`, the first `D/2` features are the spatial/channel branch and the
second `D/2` features are the temporal/patch branch. The CBraMod adapter maps
`channel` to the first half and mixes over `C`, maps `patch` to the second
half and mixes over `S`, and enables both independent branches for
`channel_patch`. It is attached once after the pretrained encoder, matching
the LaBraM residual placement while leaving CBraMod's pretrained blocks and
branch recombination untouched. It uses zero-initialized Up projections for
exact dense parity, with separate alpha, residual, update, and Q/K/V
diagnostics.

The implementation is isolated in `CBraMod/models/interaction_adapter.py`,
with checkpoint attachment in `models/cbramod.py`; it does not import LaBraM,
EEGxPlore, depth aggregation, MoE, routing, or ICASSP code. The model-level
tests in `tests/test_interaction_adapter.py` pass for dense parity, native
geometry, gradients, and rejection of a degenerate patch axis.

The next FACED order is deliberately a gate sequence, not an immediate
multiseed launch:

1. Build a clean CBraMod-only FACED runner with strict checkpoint-load and
   artifact reporting. The legacy `finetune_main.py`/`finetune_trainer.py`
   remains a reference implementation until that contract is frozen.
2. Audit the local FACED LMDB at `/data/neurogroup/mingyangjiang/data/FACED`:
   manifest, `[32,10,200]` geometry, `/100` scaling, subject split hashes,
   nine labels, and finite samples.
3. Run one seed-42 dense smoke and one seed-42 zero-init adapter parity/training
   smoke using the same classifier head and optimizer contract.
4. Lock the dense recipe from validation only, then screen native `channel`,
   `patch`, and `channel_patch` on seed 42. Do not select using test results.
5. Once the structure and recipe are fixed, run exactly `{42,1024,3407}` for
   the retained matrix, including dense, frozen, native, generic, LoRA, and
   upper-block controls. Axis-blind and final cross-backbone aggregation stay
   as later TMLR closure gates.

The CBraMod FACED result must therefore be interpreted as an instantiation of
the same interaction-alignment rule, not as “CBraMod gets a different custom
adapter.”

### CBraMod FACED promotion checklist (2026-08-02)

The FACED cell is promoted in this order:

1. Dense baseline: complete with seeds `42,1024,3407`.
2. Seed-42 screens: frozen classifier, frozen native channel/patch/channel+patch,
   generic bottleneck, LoRA QKV-r8, upper-2, and axis-blind; all completed with
   strict checkpoint loads.
3. Promote clean controls (frozen classifier, generic bottleneck, LoRA, upper-2,
   axis-blind) to the three-seed packet after artifact checks.
4. Validate `native_full_finetune` for native channel, patch, and channel+patch.
   This mode must train backbone + native adapter + classifier; frozen native
   mode must train only adapter + classifier.
5. Reassess native residual ratios, alpha growth, Q/K/V gradients, backbone
   update norms, and complete validation trajectories. Only then promote native
   frozen and native full conditions to the three-seed packet.
6. Close FACED only when every required condition has three-seed artifacts or a
   documented negative-result decision.

### Phase 3: deferred ICASSP CBraMod depth study

Only after the TMLR protocol and result registry are locked, return to
`EEGxPlore/EEGxPlore` for the CBraMod-only ICASSP study. Run frozen/nearly
frozen final-layer and prespecified-depth probes, uniform fusion, learned
global depth fusion, and interpretable interventions. Do not add TMLR
channel/patch adapters or generic-PEFT comparison cells to this repository's
ICASSP result registry.

### Phase 4: final TMLR confirmation and mechanism study

Use the three predeclared seeds `42,1024,3407` for dense, LoRA, generic
bottleneck, axis-blind, and aligned conditions on the primary TMLR
dataset/backbone cells. Then run channel-off, patch-off, both-off, eligibility,
and within-subject versus subject-disjoint SEED-V diagnostics. Depth
diagnostics are reserved for ICASSP and must be tracked in its separate
registry.

## 11. Main experiment matrix

| Phase | Backbone | Datasets | Methods | Seeds | Gate |
| --- | --- | --- | --- | --- | --- |
| A | CBraMod, LaBraM | FACED, ISRUC, TUEV | frozen, full | 3 development | protocol replication |
| A | CBraMod, LaBraM | SEED-V | frozen, full | 3 development | geometry/protocol audit |
| A | CBraMod, LaBraM | PhysioNet-MI | frozen, full | 3 development | data audit first |
| B | CBraMod, LaBraM | same primary cells | upper-k, LoRA, generic bottleneck | 3 development | baseline registry complete |
| C | CBraMod, LaBraM | same primary cells | axis-blind, aligned | 3 development | matched parameter budget |
| D | CBraMod, LaBraM | same primary cells | alignment interventions and eligibility | 3 development | mechanism attribution |
| E | CBraMod, LaBraM | frozen primary cells | all primary methods | 3 locked | method/protocol frozen |
| F | CBraMod, LaBraM | SEED-V and ISRUC first | channel/patch/eligibility/depth interventions | 3 | mechanism attribution |

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

Require a matched-budget aligned-versus-axis-blind comparison, stable
three-seed behavior, and evidence across both backbones before making a broad
positive alignment claim. A positive aligned effect on every dataset is not
required, but the manuscript must report mixed or negative cells honestly and
must not promote a dataset-specific win to a universal claim. Efficiency,
parameter, and residual-mechanism evidence are part of the gate.

### Empirical-study fallback

If generic PEFT matches aligned adaptation, only one backbone benefits, or
effects are strongly dataset-specific, use the narrower TMLR identity:

> When Does Interaction-Aligned Adaptation Help EEG Foundation Models?

Stop adding depth, routing, context, or new adapter branches when the simple
aligned primitive has not passed the matched-baseline gate. Do not expand a
failed method across all datasets merely to increase table size.

## 14. Immediate next ten actions

1. Treat this file as the canonical plan in both repositories.
2. Keep LaBraM TMLR runs in `LaBraM`, CBraMod TMLR runs in the dedicated
   original-CBraMod clone, and CBraMod ICASSP depth runs in
   `EEGxPlore/EEGxPlore`.
3. Finish the LaBraM ISRUC data/protocol audit and dense baseline replication.
4. Implement and parity-test the common low-rank axis residual primitive in
   LaBraM without changing dense common weights.
5. Collect the queued LaBraM ISRUC LoRA, generic, and upper-k control packets;
   then add the parameter-matched axis-blind check if the budget requires it.
6. Freeze the LaBraM ISRUC method-development budget and run validation-only
   development seeds.
7. Complete LaBraM TUEV and formal FACED baseline/adapter cells.
8. Establish and audit the dedicated original-CBraMod clone for TMLR.
9. Implement the CBraMod TMLR generic/LoRA/axis-blind/aligned controls without
   importing LaBraM or using EEGxPlore's depth/router path.
10. After TMLR is locked, run the separate CBraMod-only ICASSP depth-probe
    and fusion study in EEGxPlore.

## 15. Human approval questions

- Should PhysioNet-MI be a primary fifth dataset or an optional extension after
  the three shared datasets and SEED-V mechanism study?
- Should final checkpoint selection remain validation kappa, or change to a
  prespecified class-balanced metric such as balanced accuracy?
- Are all TMLR repository paths, commits, and experiment ownership entries
  recorded before the dedicated CBraMod clone is used?
- Is the existing SEED-V within-subject protocol acceptable as a reproducibility
  result if the subject-disjoint evaluation is reported separately?
- Should the current CBraMod AttnRes/MoE results be presented only as historical
  evidence or retained as explicit negative/diagnostic baselines?
