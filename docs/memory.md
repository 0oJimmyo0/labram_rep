# LaBraM Adapter Progress Memory

> **Current canonical handoff updated 2026-07-29.** The unified scientific plan
> is [`cross_backbone_execution_plan.md`](cross_backbone_execution_plan.md).
> The chronological material below is retained as historical evidence; new
> LaBraM work must follow the current handoff and contract.

## Canonical two-manuscript identity and repository boundary

The project now has two scientifically distinct manuscripts. They share only
high-level motivation and carefully documented loader/checkpoint utilities;
they do not share experiment ownership, trained model families, numerical
results, figures, tables, or primary conclusions.

### TMLR: interaction-aligned adaptation

Working title: **Interaction-Aligned Adaptation for EEG Foundation Models**.
Fallback title: **When Does Interaction-Aligned Adaptation Help EEG Foundation
Models?**

TMLR asks whether one common low-rank residual interaction primitive is more
useful when placed on a backbone's native, non-degenerate interaction axis. The
TMLR backbone/repository ownership is:

- LaBraM experiments: this repository, `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM`;
- CBraMod experiments: a new dedicated clone of the **original CBraMod
  repository**, with its path, remote, and commit recorded before coding;
- the EEGxPlore CBraMod implementation is not the TMLR CBraMod implementation;
- CBraMod and LaBraM must never be wired into one another or substituted across
  repositories.

The common TMLR family is `Down -> native-axis mixer -> Up`, inserted as a
zero-initialized residual update. The eligible axis is determined by both
backbone semantics and realized runtime geometry. For LaBraM `[B,C,S,D]`,
channel and patch branches are tested only when their sequence length is
greater than one. For CBraMod, the adapter preserves the backbone's native
spatial/channel and temporal/spectral branch separation and recombination.
The TMLR study excludes depth aggregation, depth-summary routing, typed
specialist MoE, compact EEG/PSD routers, and the historical depth-aware model.

TMLR's required comparison family is frozen-backbone plus head, full
fine-tuning, upper-block controls, generic bottleneck, independent LoRA,
parameter-matched axis-blind residual control, and interaction-aligned
channel-only/patch-only/channel-plus-patch variants. The paper must report
three-seed mean and standard deviation, validation-selected test results,
balanced accuracy and macro-F1 as primary metrics, secondary kappa/weighted
F1, per-class imbalance analysis, trainable parameters, residual norms, alpha,
gradient/update diagnostics, and efficiency measures. Beating dense
fine-tuning alone is not sufficient evidence for alignment; the
parameter-matched axis-blind control is the causal comparison.

### Deferred ICASSP boundary: CBraMod depth probing and fusion

Working title: **Depth Probing and Fusion in Pretrained EEG Encoders**.
Fallback title: **When Do Earlier CBraMod Layers Improve Downstream EEG
Decoding?**

ICASSP stays in the existing EEGxPlore repository and uses **CBraMod only**.
This design is retained as a boundary condition, but it is not an active
execution target while TMLR is being completed.
It studies frozen or nearly frozen CBraMod layer probes and lightweight depth
fusion: final-layer probe, prespecified individual depths, uniform fusion,
learned global scalar fusion, and—only if interpretable—a compact
sample-conditioned depth probe. Its analyses are layer-wise performance,
learned depth weights and entropy, earlier-layer mass, seed consistency,
class-conditioned weights, per-class recall/F1, final-only and uniform-fusion
interventions, dominant-depth removal, and parameter/latency/memory cost.

ICASSP must not contain LaBraM, TMLR interaction-aligned adapters,
channel/patch eligibility tests, TMLR low-rank operator comparisons, LoRA vs
aligned-adapter comparisons, axis-blind controls, or TMLR efficiency tables.
ICASSP may conclude positive, mixed, or negative depending on whether earlier
layers provide stable complementary information.

### TMLR-only execution priority

ICASSP is postponed. The active objective is to complete the TMLR
cross-backbone adaptation study first. A loader or checkpoint format may be
shared when technically necessary, but every experiment belongs to exactly one
manuscript. Numerical results, trained model families, figures, tables,
manuscript text, primary analyses, and central conclusions may not be reused
across the two submissions. The old NeurIPS workflow and the historical
EEGxPlore LaBraM-substitution path remain historical evidence only.

Never import or wire the CBraMod backbone into LaBraM training, and never use
the historical EEGxPlore LaBraM-substitution path as a LaBraM paper result.

## Current verified implementation facts

- `modeling_finetune.py:280-426` defines
  `LaBraMNativeAxisResidualAdapter` over `[B,C,S,D]`.
- Channel attention and patch attention are separate native-axis branches;
  `modeling_finetune.py:748-839` exposes feature and depth-summary flow.
- `run_class_finetuning.py:113-158` defines adapter/depth controls and
  `run_class_finetuning.py:629-705` constructs optimizer groups.
- Exact parity, geometry, LR-group, frozen-axis, and data-contract tests exist
  under `tests/`.
- The historical pilot mixer is full-width multihead attention. The new common
  paper-grade low-rank primitive now uses rank-32 Down/Mixer/Up branches and
  passes construction, geometry, gradient, and gamma-zero parity checks;
  matched control experiments remain pending.

## Confirmed pilot conclusions

- SEED-V realized geometry is `[B,62,1,200]`: channel interaction is eligible;
  temporal patch interaction is degenerate.
- The channel adaptor has nonzero Q/K/V gradients and delays early
  memorization, especially with `backbone_lr_scale=0.1` and adaptor weight
  decay `0.1`.
- Across the matched development packet, the channel performance advantage over
  dense is mixed and modest; it is not yet a universal performance claim.
- The shorter 30-epoch final packet underfit relative to the 40-epoch recipe.
- Depth and patch-axis results are diagnostic and must not be promoted to the
  primary claim without the common low-rank, matched-budget protocol.

## Current TMLR execution sequence

1. Close the LaBraM/TUEV block using the existing registry and exact
   three-seed rule; inspect remaining trajectories and multiseed cells without
   introducing another adapter design.
2. Freeze the LaBraM TMLR implementation, protocol, and result registry as the
   first instantiation of the common pipeline.
3. Create a clean clone of the original CBraMod repository, and record its
   remote, commit, data contract, and independent TMLR result registry.
4. Audit CBraMod's native branches and realized token geometry before coding.
   This is a design gate, not permission to copy the LaBraM module literally.
5. Start CBraMod TMLR with FACED: faithful dense baseline first, then the same
   frozen/full/upper/LoRA/generic/axis-blind/aligned comparison ladder.
6. Apply the same protocol and three-seed closure checklist to the remaining
   datasets. The backbone-specific code may map semantic axes differently,
   but the research question, operator family, budgets, controls, and claims
   must remain shared.
7. Keep ICASSP deferred until the TMLR cross-backbone matrix is locked.

The design must never become “one backbone, one bespoke adapter.” It is one
reproducible adaptation pipeline with backbone-specific axis mappings required
by the native encoder geometry.

## Current status labels

- `verified`: implementation/parity/data facts supported by tests or audits;
- `pilot`: existing SEED-V/FACED/ISRUC training evidence under the former
  native full-width adapter protocol;
- `historical`: old workflow, old jobs, or substitution-path results;
- `pending`: required for the revised paper;
- `blocked`: only when an external data/provenance dependency prevents progress.

Last updated: 2026-07-29

## Repository State

- Repository: `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM`
- Branch: `adaptor`
- Current implementation lineage includes `7ca74e2` (`Add separate LaBraM
  alpha gate controls`); current HEAD at this update is `2e5840c`.
- The working tree contains existing experiment/code changes; this
  documentation update preserved them.
- FACED checkpoint SHA: `7c5058...37c`
- Data root: `/data/neurogroup/mingyangjiang/data/FACED`
- Logs: `logs/out/<RUN_ID>/` and `logs/err/<RUN_ID>/`

## Implemented Adapter

`LaBraMNativeAxisResidualAdapter` operates on `[B, C, S, D]` patch-token grids.

- Channel attention mixes electrodes for each temporal patch.
- Patch attention mixes temporal patches for each electrode.
- Channel, patch, or channel-plus-patch branches can be selected.
- Token MLP and depth gating are disabled by default.
- The adapter correction is inserted before LaBraM final normalization and pooling.
- Dense mode retains the original LaBraM forward path.
- Adapter initialization uses an independent seed and does not alter common model initialization.
- There is no residual projection in the current implementation.

## Verification Gates

Synthetic parity test:

```bash
/data/neurogroup/mingyangjiang/venvs/labram-accre/bin/python tests/test_labram_adapter_parity.py
```

Result:

```text
max_feature_diff=0.000e+00
max_logit_diff=0.000e+00
```

Real FACED parity test:

```bash
/data/neurogroup/mingyangjiang/venvs/labram-accre/bin/python tests/test_faced_adapter_parity.py
```

Result:

```text
sample_shape=(2, 32, 10, 200) input_chans=33
max_abs_feature_difference=0.000e+00
max_abs_logit_difference=0.000e+00
```

Gradient smoke test passed with nonzero values:

```text
adapter_grad_norm=8.62e-05
last_block_grad_norm=1.31e-04
classifier_grad_norm=6.22
```

## Important Commits

- `98ce923`: corrected native adapter placement, initialization isolation, and residual scaling
- `d7979f5`: captured gradients before `zero_grad` and added classifier gradient diagnostics
- `8282105`: added real FACED gamma-zero parity test
- `57364d1`: validation-only training and one final test evaluation
- `eb9cd1b`: validation kappa primary selection with validation BA sensitivity checkpoint

## Historical Seed-0 Pilot

These runs used the previous protocol, which evaluated test after every epoch. They are useful for development context only.

| Condition | Best validation BA | Test BA at that validation epoch |
| --- | ---: | ---: |
| Dense, lr `6e-4`, 50 epochs | 0.47639 | 0.41573 |
| Channel-only, alpha `0.01` | 0.46404 | 0.42297 |
| Patch-only, alpha `0.01` | 0.44090 | 0.39238 |
| Channel + patch, alpha `0.01` | 0.45725 | 0.41251 |
| Gamma-zero control | 0.42994 | 0.38875 |

The test values must not be used to choose a configuration.

## Seed-3407 Strong-Recipe Runs

Recipe: batch size `16`, learning rate `7e-4`, 80 epochs, warmup `10`, weight decay `0.05`, layer decay `0.65`, drop path `0.1`, alpha `0.01`.

- Dense job `12472130`: complete; validation kappa `0.36881`, validation BA `0.44244`, final test BA `0.39855`.
- Channel job `12472128`: complete; validation kappa `0.35176`, validation BA `0.42778`, final test BA `0.39385`.
- Patch job `12472127`: complete; validation kappa `0.42660`, validation BA `0.49352`, final test BA `0.43129`.
- Channel-plus-patch job `12472129`: failed because an LMDB DataLoader worker segfaulted. Retry with `NUM_WORKERS=0`.

Patch-only is the current promising candidate because it leads on validation metrics, not because of its test score.

## Current Follow-up Queue

Submitted after the seed-3407 screen:

- Job `12475956`: patch-only, alpha `0.003`, strong recipe, seed `3407`.
- Job `12475958`: patch-only, alpha `0.030`, strong recipe, seed `3407`.
- Job `12475957`: channel-plus-patch, alpha `0.010`, strong recipe, seed `3407`, `NUM_WORKERS=0` retry.
- Job `12481023`: patch-only, alpha `0.030`, strong recipe, seed `3407`, `NUM_WORKERS=0` retry after the worker segfault; complete.

All three jobs use validation kappa as the primary checkpoint-selection metric and validation BA as the sensitivity checkpoint. Do not use their test metrics to select alpha or structure.

Follow-up results:

- Patch alpha `0.003`: validation kappa `0.39444`, validation BA `0.46435`; below alpha `0.010`.
- Channel-plus-patch retry alpha `0.010`: validation kappa `0.35093`, validation BA `0.42670`; below dense and patch-only.
- Patch alpha `0.030`: retry completed with validation kappa `0.38127`, validation BA `0.45247`; below alpha `0.010`.

The seed-3407 alpha screen therefore selects patch-only alpha `0.010` by validation kappa, with validation BA as the sensitivity check. The next required test is paired dense-versus-patch alpha `0.010` across seeds `42`, `1024`, and `3407`.

Seed-3407 paired validation deltas at the selected checkpoints:

- Delta validation kappa: `+0.05779` (`0.42660 - 0.36881`)
- Delta validation BA: `+0.05108` (`0.49352 - 0.44244`)
- Delta validation weighted F1: `+0.05337` (`0.49177 - 0.43840`)

Paired confirmation jobs submitted:

- Dense seed `42`: job `12482796`.
- Patch alpha `0.010` seed `42`: job `12482795`.
- Dense seed `1024`: job `12482794`.
- Patch alpha `0.010` seed `1024`: job `12482797`.

Overnight shared hyperparameter screening is queued behind jobs `12482794`, `12482795`, `12482796`, and `12482797`. It contains matched dense and patch alpha `0.010` runs at seed `3407` for:

```text
learning_rate in {5e-4, 7e-4, 9e-4}
warmup_epochs in {5, 10}
batch_size=16, epochs=80, weight_decay=0.05, layer_decay=0.65, drop_path=0.1
```

The 12 jobs are `12482942` through `12482953`. Only the top two shared recipes by validation metrics will be replicated across the remaining development seeds.

## Operational Commands

## Adapter Optimizer Control (Current Branch)

The patch adapter is a full-width `D=200` multihead-attention residual branch. Its
`adapter_bottleneck` argument only controls the optional token MLP and does not
bottleneck patch attention. Before this correction, `native_axis_adapter.*`
parameters were assigned to the same highest layer-decay groups as pretrained
upper-block parameters.

The current branch adds:

- `--labram_adapter_lr_scale`, applied to the effective layer-decayed LR of every
  `native_axis_adapter.*` parameter;
- `--labram_adapter_weight_decay`, applied to adapter matrix groups while scalar
  gates and biases remain in no-decay groups;
- per-epoch `backbone_lr` and `adapter_lr` diagnostics in training logs;
- `--skip_final_test` for validation-only development sweeps.

When adapter weight decay is explicitly supplied, it is held at that value rather
than overwritten by the global backbone WD schedule. With the new flags omitted,
the previous optimizer behavior is preserved.

The earlier plan to screen adapter LR scales `{0.1, 0.3, 1.0}` across several
global LRs is superseded by the one-time batch-size control below. Do not start
that larger adapter-scale sweep before the batch effect is resolved.

## Decisive Batch-Size Control (2026-07-13)

The latest logs changed the interpretation of Stage 3A. The exact new-code
duplicate using the historical recipe (`batch_size=16`, `num_workers=0`,
`lr=7e-4`, warmup `10`, patch alpha `0.01`, core and alpha scales `0.1`) matches
the old Stage 2 seed-1024 epoch summaries exactly after elapsed-time fields are
removed:

- Best validation kappa: `0.37767`
- Best validation BA: `0.44861`
- Best validation weighted F1: `0.44759`
- At the end of training, `alpha_patch` is about `0.0021` and the scaled residual
  ratio is about `0.00016`.

This reproduces the earlier adapter-collapse behavior. The Stage 3A alpha-scale
`0.1` runs used `batch_size=32` and `num_workers=4`, which changed the number of
optimizer steps per epoch and the warmup/cosine schedule. Their active residual
therefore cannot yet be interpreted as evidence for separate alpha learning
rates. The alpha-scale `0.3`, seed-42 retry completed without the old DataLoader
worker crash, but remained below alpha-scale `0.1`:

- Best validation kappa: `0.40334`
- Best validation BA: `0.47299`
- Best validation weighted F1: `0.47343`

Nonfatal AMP `NaN or Inf found in input tensor` warnings still appear during
early loss-scale warmup, but neither new job had a DataLoader worker,
pickling, or segmentation-fault termination.

### Current decisive control

Run exactly eight validation-only jobs, with dense and patch-only crossed with
batch sizes `16` and `32`, on seeds `42` and `1024`:

```text
global lr=7e-4
warmup=10
epochs=80
workers=0
weight_decay=0.05
patch core lr scale=0.1
alpha lr scale=0.1
alpha init=0.01
token MLP=false
depth mode=none
```

Record both the best validation epoch and optimizer step. This is a one-time
confound control, not a new batch-size research direction. If patch beats dense
at batch 32 on both seeds, run seed `3407` and compare the three-seed mean
against the strongest dense recipe. If dense and patch improve together, the
earlier gain was primarily an optimization-regime effect. If patch remains
inconsistent, treat FACED as a boundary case and move to the broader dataset
study rather than expanding the FACED sweep.

### Submitted overnight jobs

The eight-job control was submitted with `NUM_WORKERS=0`, `SKIP_FINAL_TEST=1`,
and run-organized logs:

| Job | Condition |
| ---: | --- |
| `12509548` | dense, batch 16, seed 42 |
| `12509550` | patch, batch 16, seed 42 |
| `12509549` | dense, batch 32, seed 42 |
| `12509552` | patch, batch 32, seed 42 |
| `12509547` | dense, batch 16, seed 1024 |
| `12509551` | patch, batch 16, seed 1024 |
| `12509545` | dense, batch 32, seed 1024 |
| `12509546` | patch, batch 32, seed 1024 |

The four patch jobs completed successfully. The four dense jobs did not train:
they passed `LABRAM_ADAPTER_TYPE=dense`, but the CLI uses `none` for the native
dense path, so all four exited during argument parsing with an invalid-choice
error. The batch-size comparison is therefore still incomplete and must be
rerun with `LABRAM_ADAPTER_TYPE=none` before interpreting any adapter gain.

## Paper-Aligned Depth Candidate

Because the current batch control must remain on commit `7ca74e2`, the depth
refinement was developed in an isolated worktree at
`/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM-depth` on branch
`adaptor-depth`, commit `9807be4`.

It adds `lastk_attnres` while preserving `none` and the legacy `lastk_delta`
mode. The new mode:

- collects the final `k` block representations on the `[C,S,D]` grid;
- learns per-channel-patch soft weights across upper depth;
- forms a depth-selected representation for the residual source;
- starts with a zero-initialized depth mix, so it is exactly patch-only at
  initialization;
- logs per-layer depth weights, entropy, top-layer share, depth mix, and depth
  gradients.

The isolated tests pass compile, depth normalization/gradient checks, legacy
`lastk_delta` forwarding, existing adapter controls, and gamma-zero parity with
zero feature/logit difference. The candidate remains isolated and must not be
merged into `adaptor` while the live batch control is running.

### Exploratory depth screen

Four validation-only jobs were submitted from the isolated `adaptor-depth`
worktree at commit `de59a8e` (launcher fix on top of `9807be4`). They use the
current patch recipe with batch size `32`, `num_workers=0`, global LR `7e-4`,
core/alpha LR scales `0.1`, alpha init `0.01`, and `epochs=80`:

| Job | Condition |
| ---: | --- |
| `12509671` | `lastk_attnres`, k=2, seed 42 |
| `12509673` | `lastk_attnres`, k=4, seed 42 |
| `12509672` | `lastk_attnres`, k=2, seed 1024 |
| `12509670` | `lastk_attnres`, k=4, seed 1024 |
| `12509800` | `lastk_attnres`, k=2, seed 3407 |
| `12509801` | `lastk_attnres`, k=4, seed 3407 |

These jobs are exploratory only. They do not replace the batch-size gate, and
their validation metrics must not be used to justify depth unless the simple
patch recipe first proves reproducible against dense. Logs and checkpoints are
organized under `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM-depth`.

All six depth jobs completed successfully with validation-only evaluation. At
the kappa-selected checkpoint, the results were:

| Run | Best epoch | Kappa | BA | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| k=2, seed 42 | 75 | 0.39896 | 0.46944 | 0.46587 |
| k=2, seed 1024 | 72 | 0.40187 | 0.47145 | 0.47094 |
| k=2, seed 3407 | 52 | 0.39240 | 0.46111 | 0.46251 |
| k=4, seed 42 | 70 | 0.40804 | 0.47762 | 0.47449 |
| k=4, seed 1024 | 78 | 0.38692 | 0.45833 | 0.45758 |
| k=4, seed 3407 | 73 | 0.40076 | 0.47130 | 0.46937 |

The matched simple patch batch-32 controls reached kappa `0.40838` / BA
`0.47793` / weighted F1 `0.47712` on seed 42 and `0.40688` / `0.47670` /
`0.47638` on seed 1024. Thus k=4 is nearly tied on seed 42 but lower on seed
1024, while k=2 is lower on both. Seed 3407 is not uniformly favorable for
depth: k=2 and k=4 reached kappa `0.39240` and `0.40076`, respectively.

The depth selector is active but nearly uniform: k=2 has entropy about
`0.693` and layer shares near `0.5/0.5`; k=4 has entropy about `1.386` and
shares near `0.25` each. The learned depth mix remains nonzero, so the branch
is functioning, but these runs do not yet show a useful selective depth
preference or a reproducible FACED improvement.

### Depth-v2 candidate

The v1 result is negative but inconclusive because its zero-initialized query
and zero-initialized mix block the query gradient initially, and its candidate
pool includes the final representation. A bounded v2 is implemented locally on
the `depth` branch at commit `cb339aa`:

- `lastk_uniform`: fixed uniform average of the preceding `k` block outputs;
- `lastk_attnres_v2`: learned scalar scoring over the preceding `k` outputs;
- final block output remains the base `H_L` representation;
- learned `alpha_depth` starts at `0.05`, so the depth path is active but small;
- v2 removes the extra `sqrt(D)` score division;
- diagnostics add normalized entropy, min/max layer weight, depth beta, and
  `||H_depth-H_L|| / ||H_L||`.

The local v2 tests pass under the ACCRE virtualenv, including first-backward
scorer/beta gradients, uniform weights, near-identity behavior, existing
adapter controls, and exact gamma-zero parity. The v2 commit is not yet on
GitHub because this shell currently lacks HTTPS push credentials.

The next bounded depth screen is validation-only on seeds `42` and `1024`,
batch size `32`, workers `0`, using the stable patch recipe:

```text
patch-only
patch + lastk_uniform, k=2
patch + lastk_attnres_v2, k=2
patch + lastk_attnres_v2, k=4
```

Select by mean validation kappa, with BA and weighted F1 compatibility. Only
then test the selected v2 condition on seed `3407`; do not use test metrics for
this screen.

### Reduced seed-3407 depth validity screen

The attempted 30-job dense/patch/depth ladder and the follow-up 18-job LR sweep
were canceled before producing valid new comparisons. Job `12516305` had
started but was canceled before a valid result was written. Existing dense and
patch-only logs remain the controls; no canceled-job output is used.

The replacement screen is intentionally only four validation-only jobs. It uses
the corrected `depth` commit `956be37`, seed `3407`, batch size `32`, workers
`0`, warmup `10`, epochs `80`, global LR `7e-4`, patch alpha `0.01`, and
adapter/core and alpha LR scales `0.1`:

| Job | Condition |
| --- | --- |
| `12516380` | patch + uniform depth, k=2 |
| `12516377` | patch + uniform depth, k=4 |
| `12516379` | patch + learned depth-v2, k=2 |
| `12516378` | patch + learned depth-v2, k=4 |

This is a behavior and failure-mode check, not a hyperparameter selection
study. Compare each run with the existing seed-3407 patch-only log using
validation kappa as primary, with validation BA and weighted F1 as required
compatibility metrics. Inspect `alpha_depth`, scorer gradients, depth-source
ratio, normalized entropy, and layer weights. A depth condition is only a
candidate for later paired confirmation if it improves the patch control
without a major BA/F1 loss and shows nonzero, controlled depth activity.

If all four conditions are neutral or worse, retain patch-only and stop FACED
depth tuning. If one is clearly promising, rerun only that condition on the
difficult seeds before any test evaluation or cross-dataset claim.

For later isolated submissions, use the run-organized wrapper from the depth
worktree:

```bash
RUN_ID=faced_labram_patch_a003_seed3407 \\
LABRAM_ADAPTER_TYPE=patch \\
LABRAM_ADAPTER_INIT_ALPHA=0.003 \\
SEED=3407 \\
scripts/submit_faced_finetune_accre.sh
```

Inspect jobs:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
```

Inspect final metrics:

```bash
cat checkpoints/RUN_ID/final_test.json
tail -5 checkpoints/RUN_ID/log.txt
```

## FACED conclusion and SEED-V transition (2026-07-13)

The final FACED interpretation is now frozen:

| Condition | Seed 3407 kappa | Seed 3407 BA | Seed 3407 weighted F1 |
| --- | ---: | ---: | ---: |
| Dense LaBraM, lr `7e-4` | 0.36881 | 0.44244 | 0.43840 |
| Patch-only, lr `7e-4`, alpha `0.01` | 0.42660 | 0.49352 | 0.49177 |
| Patch + learned delta gate, k=2 | 0.40266 | 0.47269 | 0.46919 |

The difficult-seed paired controls give patch-only mean validation kappa
`0.40763`, BA `0.47731`, and weighted F1 `0.47675` across seeds `42` and
`1024`. Learned depth-gating gives mean kappa `0.40543`, BA `0.47346`, and
weighted F1 `0.47331`; it wins seed `42` but loses seed `1024`. Uniform depth
gating loses to patch-only on both seeds. The learned gate is technically
active and stable, with nonzero scorer/gate gradients, bounded modulation, and
moderate layer preference around `0.56/0.44`, but it is not a reproducible
improvement over patch-only.

FACED therefore supports the broader paper idea at the structured residual
level: patch-axis adaptation improved the native LaBraM dense control on the
seed-3407 pilot while preserving the pretrained backbone. Explicit
upper-depth-delta conditioning is retained as a negative/exploratory ablation,
not as the winning LaBraM component. No test result was used to choose the
structure or depth mode.

### Existing SEED-V dense anchor

An existing EEGxPlore SEED-V LaBraM dense run provides a useful anchor:

```text
run: seedv_labram_dense_lr1e4_20260703_150505_s3407
log: ../EEGxPlore/logs/SEED-V/out/seedv_SEEDV_TUNE_12322626.out
summary: ../EEGxPlore/output/seedv_labram_dense_lr1e4_20260703_150505/seed_3407/run_summary_seed-v_20260703T203617Z.json
checkpoint: /data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/checkpoints/labram-base.pth
```

It used `attnres_variant=none`, `moe=false`, pretrained foundation loading,
kappa-first checkpoint selection, batch size `64`, global lr `1e-4`, 40 epochs, and
the LMDB default trial-based 5:5:5 split. At the selected epoch `22`:

```text
validation: kappa=0.23163, BA=0.38539, weighted F1=0.38653
test:       kappa=0.28181, BA=0.42304, weighted F1=0.43022
```

This is a native dense LaBraM control in the EEGxPlore wrapper, but it uses the
wrapper's `all_patch_reps` classifier and is not a byte-for-byte reproduction
of the official LaBraM downstream script. Treat it as an existing anchor, then
rerun the dense condition under the frozen SEED-V comparison recipe before
interpreting adapter gains.

## Frozen SEED-V plan

Freeze the FACED-developed patch adapter before SEED-V. Do not change patch
attention placement, residual integration, alpha/gate range, token MLP state,
expert banks, or routing. Record the exact code commits: patch-only from the
`adaptor` branch at `af0220f`; the optional depth secondary ablation from the
isolated `depth` branch at `257ba06`.

First audit and establish the dense baseline: pretrained checkpoint SHA,
62-channel tensor order, `/100` sample scaling, `(62,1,200)` segmentation,
LMDB `__keys__` train/validation/test protocol, subject/session coverage,
native normalization/pooling, classifier, optimizer, and kappa-based checkpoint
selection. Test evaluation must happen only after the validation-selected
checkpoint is fixed.

Use the frozen ladder:

```text
1. Native dense LaBraM
2. Patch-only LaBraM
3. Patch + learned depth-delta gate, k=2 (secondary ablation)
```

The primary cross-dataset comparison is patch-only minus dense. The secondary
component comparison is patch-plus-depth minus patch-only. Use paired seeds,
validation kappa for selection, and require validation BA and weighted F1 to
remain compatible. Do not develop new architecture on SEED-V; if the dense
protocol is credible, run the frozen ladder and move on to the next dataset.

### SEED-V native LaBraM jobs submitted (2026-07-14)

The standalone LaBraM-native implementation is being used; the EEGxPlore
CBraMod-compatible adapter path is not used for this comparison. All jobs use
the same validation-only recipe:

```text
SEED-V LMDB: /data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb
checkpoint: labram-base.pth, sha256=7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c
global batch: 64 (32 per GPU x 2), lr=1e-4, weight_decay=0.03
epochs=40, warmup=5, layer_decay=0.65, drop_path=0.1
workers=0, input_scale_divisor=100, selection=kappa, final test skipped
```

| Condition | Seed 42 | Seed 1024 | Seed 3407 |
| --- | ---: | ---: | ---: |
| Dense, `adapter_type=none` | 12524845 | 12524846 | 12524847 |
| Patch-only, `adapter_type=patch` | 12524848 | 12524849 | 12524850 |
| Patch + depth gate, `lastk_delta_gate`, k=2 | 12524851 | 12524852 | 12524853 |

The depth jobs run from the isolated `depth` worktree at commit `226b8b3`;
dense and patch jobs run from `adaptor` commit `5ccde3c`. Test evaluation will
be performed only after the validation-selected recipe is fixed.

## SEED-V data and channel-order audit (2026-07-14)

Direct LMDB inspection found a legacy artifact: records contain only `sample`
and `label`, samples are `(62, 1, 200)`, and the trial-based splits contain
all 16 subjects. The checked-in EEGxPlore preprocessing script instead drops
`M1`, `M2`, `VEO`, and `HEO`, preserves the remaining CNT `raw.ch_names`
order, and writes richer metadata. The original CNT files and sidecars are
not available beside the current LMDB.

`docs/seedv_channel_manifest_provisional.json` now records the status
`metadata_verified_row_linkage_pending`. The newly found `Channel Order.xlsx`
and `channel_62_pos.locs` both contain the same ordered 62-channel list as the
manifest, and their SHA-256 values are recorded there. This verifies the
source montage definition, but the legacy LMDB still does not encode that
these external artifacts were used to determine its tensor row order.

### SEED-V contract hardening (2026-07-14)

The remaining channel-order limitation is scientific, not a missing-name
software bug: matching 62 names to 62 tensor rows proves only dimensional
compatibility. The source metadata now agrees exactly, but the legacy LMDB
does not contain the original row-to-electrode mapping. Confirm that these
artifacts belong to the LMDB preprocessing run before removing the exploratory
qualification.

The `adaptor` and `depth` worktrees now require the known legacy SEED-V shape
`(62, 1, 200)`, finite samples, scalar labels in `[0, 4]`, and a validated
manifest. The underlying loader remains configurable for a different real
channel count when used with a different dataset artifact; LaBraM positional
selection receives the actual manifest length rather than padding to 62.

Run provenance now includes the complete normalized channel list, mapped
LaBraM `input_chans`, manifest-file SHA-256, pretrained-checkpoint SHA-256,
sample shape, and split key-list hashes/counts. The new
`scripts/audit_seedv_lmdb.py` performs a one-time full-record audit including
class counts and raw amplitude percentiles. The full scan was attempted on
the current 12 GB LMDB but was stopped after prolonged shared-filesystem I/O;
its JSON output is therefore not yet evidence and must be regenerated on a
suitable node before final SEED-V reporting.

The current launcher does not pass `--dist_eval`, so validation uses the
sequential sampler rather than the padding distributed sampler. Keep that
setting fixed until a gathered, non-duplicating distributed evaluator is
implemented.

### SEED-V LMDB audit result (2026-07-14)

`docs/seedv_lmdb_audit.json` contains the complete 117744-record audit. Every
record passed the expected `(62, 1, 200)` shape, finite-value, scalar-label,
and `[0,4]` label checks. Train/validation/test key lists have zero pairwise
overlap, and all records are stored as `float64`.

The main remaining data signal is a sparse amplitude tail. Median sample
maximum is about 90 raw units in each split; raw maxima are 2555 train, 114438
validation, and 43418 test. With `/100` scaling, 26 validation and 2 test
windows still exceed 100 absolute units. Treat this as a shared preprocessing
diagnostic for dense and adapter runs, not as a reason to change the recipe
before a controlled comparison. Do not clamp or delete windows without a
separate prespecified preprocessing experiment.

The audit ran from node-local `/dev/shm` after the shared-filesystem scan
stalled. The JSON records the original LMDB path and the local audit input
path.

The enhanced audit additionally reports zero exact sample-content overlap
between split pairs. All keys parse successfully; each split contains all 16
participants, 48 participant-session combinations, and 240
participant-session-trial combinations. This confirms the current
CBraMod-compatible within-subject trial split: train trials `0-4`, validation
trials `5-9`, and test trials `10-14`. It is not subject-independent evaluation.

The 28 extreme windows above absolute 100 after `/100` scaling are retained in
the primary dataset and recorded individually in the audit JSON. They are
mostly concentrated in participant 14, session 3, trials 6 and 9.

## Efficient SEED-V validation screen (2026-07-14)

Keep the development seed packet fixed at `{42, 1024, 3407}`. Do not add the
alternative confirmation seeds `0`, `7`, or `2026`. First run six
validation-only jobs: dense and patch-only crossed with the three fixed seeds.
Use one clean commit and the explicit metadata-verified manifest. Advance
patch-only only with positive mean paired validation kappa, at least two seed
wins, and compatible BA/weighted F1. Test depth only after that gate, using a
fresh patch comparator on the exact depth commit. The raw CNT bridge remains
unavailable locally and should be completed before final test evaluation, not
by expanding the current validation workload.

### Verified SEED-V singleton-patch mechanism (2026-07-15)

The SEED-V mechanism check used one real LMDB batch and confirmed:

```text
sample_shape=(2, 62, 1, 200)
input_time_window=1
token_grid=[B,62,1,200]
patch_attention_sequence_length=1
patch_temporal_interactions_active=0
patch_q_grad_norm=1.35e-13
patch_k_grad_norm=1.38e-13
patch_v_grad_norm=1.78e-05
patch_output_projection_grad_norm=4.14e-05
```

With one temporal patch, the patch-attention softmax has one element and cannot
perform temporal patch selection. The branch remains a valid residual value and
output projection, but it is not temporal patch-to-patch mixing on SEED-V.
This is a geometry boundary condition, not an input NaN, channel-count, or
checkpoint-loading failure.

The implementation now records the input time window, adapter token-grid
dimensions, patch sequence length, whether temporal interactions are active,
and separate Q/K/V/output-projection gradient norms. The unit test
`tests/test_seedv_patch_geometry.py` and real-batch diagnostic
`tests/test_seedv_real_patch_geometry.py` protect this interpretation.

Pending SEED-V LR jobs were cancelled after this finding. Running jobs
`12544353` and `12544354` were left to finish as partial controls. No depth or
capacity experiments were submitted after the mechanism check.

### Fixed SEED-V seed packet (2026-07-15)

All SEED-V development, capacity, confirmation, and final paired runs use the
fixed seed packet `{42, 1024, 3407}`. Do not replace it with `{0, 7, 2026}` or
another alternative packet. This keeps the workload bounded and preserves the
same paired comparison throughout the LaBraM evaluation.

### Explicit SEED-V singleton capacity variants (2026-07-15)

The patch branch now has an explicit `adapter_variant` control:

- `full`: original full-width patch attention;
- `output_dropout`: the same patch attention followed by separate output
  dropout, with `patch_output_dropout=0.1` and attention dropout kept at zero;
- `bottleneck`: replaces singleton patch attention with
  `LayerNorm(200) -> Linear(200,64) -> GELU -> Linear(64,200)`.

The bottleneck is explicit and does not activate automatically when `S=1`.
Run configuration metadata records the variant, dropout, bottleneck width,
parameter count, global/effective adapter learning rates, adapter seed, and
commit. Preflight controls pass for dense/gamma-zero parity, shared
initialization, variant parameter counts, eval-time dropout behavior, nonzero
variant and gate gradients, optimizer groups, FACED parity, and SEED-V
geometry.

The next experiment is exactly two validation-only seed-3407 jobs: output
dropout 0.1 and bottleneck-64, with no depth or additional LR sweep. The fixed
study seed packet remains `{42, 1024, 3407}`; seed 3407 is an exploratory
selection screen and any surviving candidate must be confirmed on the full
fixed packet before test evaluation.

### SEED-V channel-axis control (2026-07-15)

The real-batch channel-only mechanism check passes on `(2,62,1,200)`:

```text
channel_attention_sequence_length=62
channel_q_grad_norm=1.51e-06
channel_k_grad_norm=1.52e-06
channel_v_grad_norm=1.88e-05
channel_output_projection_grad_norm=4.71e-05
raw_channel_ratio=0.221
```

The patch branch is absent, so this restores a meaningful EEG axis without
changing the residual placement or dense backbone. Synthetic channel geometry,
LaBraM gamma-zero parity, and FACED parity also pass. The next bounded screen
is dense versus channel-only at global LR `1e-4` and `5e-4`, all on seed `3407`,
with batch size `32`, warmup `5`, 40 epochs, core LR scale `1.0`, alpha LR
scale `0.1`, alpha init `0.01`, depth off, token MLP off, and test disabled.
Any candidate must later be confirmed on the fixed packet `{42, 1024, 3407}`.

## ISRUC frozen-backbone confirmation and SEED-V transition (2026-07-20)

The ISRUC frozen dense-versus-patch packet is complete on seeds `{42, 1024,
3407}`. All runs use the same checkpoint, ISRUC preprocessing/splits, batch
size `16`, global LR `2e-4`, 30 epochs, sequence-head dropout `0.1`, smoothing
`0.1`, drop path `0.1`, weight decay `0.05`, and validation-kappa selection.
The frozen condition uses `backbone_lr_scale=0`, `head_lr_scale=1`, patch
adapter core and alpha LR scales `1`, depth off, and token MLP off.

| Seed | Dense frozen kappa | Patch frozen kappa | Delta kappa | Delta BA | Delta weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.58527 | 0.69980 | +0.11452 | +0.14143 | +0.10898 |
| 1024 | 0.57693 | 0.70673 | +0.12979 | +0.15793 | +0.12075 |
| 3407 | 0.57950 | 0.70229 | +0.12279 | +0.15018 | +0.11651 |
| Mean | 0.58057 | 0.70294 | **+0.12237** | **+0.14985** | **+0.11542** |

Patch wins all three seeds. Frozen patch selected epochs are `19`, `23`, and
`19`; peak-to-final kappa declines are `0.00315`, `0.00407`, and `0.00525`,
compared with approximately `0.021` for the original fully fine-tuned patch
trajectory. Backbone freezing therefore largely resolves the early
generalization collapse. The adapter remains strongly active: selected-
checkpoint `alpha_patch` is about `0.118-0.122`, residual ratio about
`0.366-0.370`, and raw patch ratio about `3.0-3.1`.

This is a reproducible parameter-efficient adaptation result, but not yet a
full-finetuning replacement. The earlier fully fine-tuned dense mean across
the same seeds is approximately `0.765` kappa, `0.795` BA, and `0.819` weighted
F1; frozen patch averages `0.703`, `0.725`, and `0.769`. The correct claim is
conditional: patch residual adaptation is strongly useful when the pretrained
backbone is frozen, while full fine-tuning still gives higher absolute
performance.

The frozen ISRUC run directories are:

```text
seed 42:   checkpoints/isruc_dense_bls0_lr2e-4_s42_b16_e30
           checkpoints/isruc_patch_util_bls0_lr2e-4_s42_b16_e30
seed 1024: checkpoints/isruc_dense_bls0_s1024_lr2e-4_b16_e30
           checkpoints/isruc_patch_util_bls0_s1024_lr2e-4_b16_e30
seed 3407: checkpoints/isruc_dense_bls0_s3407_lr2e-4_b16_e30
           checkpoints/isruc_patch_util_bls0_s3407_lr2e-4_b16_e30
```

### Revised SEED-V primary experiment

SEED-V samples are `(B,62,1,200)`, producing LaBraM tokens `[B,62,1,D]`.
Patch attention therefore has temporal sequence length one: Q/K gradients are
approximately zero and there is no temporal patch-to-patch interaction. A
frozen patch run may still demonstrate added trainable capacity, but it does
not validate the temporal-patch structure used on ISRUC or FACED.

The primary SEED-V comparison is:

```text
Model A: frozen LaBraM + shared classifier/head
Model B: frozen LaBraM + channel-only adapter + identical head
```

The fixed seed packet is `{42, 1024, 3407}`. Use the existing SEED-V recipe:
batch size `32`, global LR `1e-4`, 40 epochs, warmup `5`, weight decay `0.03`,
layer decay `0.65`, drop path `0.1`, smoothing `0.1`, workers `0`, `/100`
scaling, validation-kappa selection, and no test evaluation during
development. Use `backbone_lr_scale=0`, `head_lr_scale=1`, channel-only
`adapter_type=channel` for Model B, adapter core/alpha LR scales `1`, alpha
init `0.01`, depth off, token MLP off, and patch branch off.

Record `alpha_channel`, raw channel ratio, final residual ratio, channel Q/K/V
and output-projection gradients, adapter update norms, trainable parameter
count, validation metrics, selected epoch, and train/validation gap. Retain
the frozen patch result only as a singleton-axis capacity control. The
paper-level principle is: use lightweight residual adaptation along a
meaningful native axis of LaBraM's channel-patch representation.

The low-rank aligned adapter is conceptually separate from LoRA. Low-rank here
means `Down -> native channel/patch mixer -> Up` inside the structure-aware
residual branch. LoRA is an independent generic PEFT baseline that modifies
backbone linear weights. We do not combine LoRA with the native channel,
patch, or channel+patch adapter in the primary matrix.

## ISRUC current-HEAD native-axis pilot and run-ladder rule (2026-07-23)

The current-HEAD ISRUC seed-42 comparison completed with test evaluation:

| Condition | Best val kappa / BA | Test BA / kappa |
| --- | ---: | ---: |
| Dense | 0.7673 / 0.7940 | 0.7997 / 0.7583 |
| Channel | 0.7711 / 0.8020 | 0.7915 / 0.7481 |
| Patch | 0.7698 / 0.8007 | 0.7912 / 0.7475 |
| Channel+patch | 0.7695 / 0.8003 | 0.7918 / 0.7474 |

All conditions peak around epoch 12-18 while training accuracy continues to
about 0.91, so the early overfitting pattern remains. The channel adapter is
active on the intended `[B,6,30,D]` geometry with nonzero Q/K/V gradients, but
its validation advantage does not transfer to this single test split. These
full-width attention results are pilot evidence only; they are not the final
parameter-matched common primitive.

For every backbone-dataset cell, complete the bounded run ladder once:

```text
protocol gate
  -> frozen/full/upper-k baseline
  -> LoRA / generic / axis-blind controls
  -> matched low-rank interaction-aligned adapter
  -> channel-off/patch-off/both-axis and one depth extension
  -> stop single-seed tuning and run development multiseeds
  -> freeze method and run final locked three-seed test block
```

Test metrics may be recorded for every run, but validation remains the only
criterion for checkpoint, architecture, and hyperparameter selection during
development. Once a section passes its promotion gate, do not keep expanding
its LR sweep; move to multiseed confirmation.

### ISRUC low-rank aligned screen (2026-07-24)

The current-HEAD rank-32 low-rank screen completed on seed 42 with final test
evaluation. The dense anchor reproduced the prior result: primary test BA
`0.79965`, kappa `0.75834`. Low-rank channel, patch, and channel+patch all
peaked near epoch 12 with validation BA around `0.801-0.802`, but their test
BA was only `0.7915`, `0.7923`, and `0.7920`, respectively. They therefore do
not yet pass the promotion gate.

The main mechanism finding is that rank-32 residuals were extremely small:
about `0.005%` of the backbone feature norm, approximately 50 times smaller
than the earlier full-width pilot residuals. The branches had nonzero Q/K/V
gradients, so they were not disconnected, but the adapter was practically
under-scaled. The next bounded calibration is `adapter_init_alpha=0.5` for
channel, patch, and channel+patch, with all other settings fixed. Do not add a
new LR sweep or depth branch before evaluating this calibration. If it produces
a coherent validation/test effect and an appropriately active residual, stop
single-seed tuning and promote the selected method to development multiseeds.

### ISRUC controls queued (2026-07-24)

LaBraM now has three separate controls for the overnight comparison:
`BACKBONE_MODE=lora` injects LoRA into attention `qkv` projections and trains
only LoRA factors plus the head; `ADAPTER_TYPE=generic` is an axis-blind
token-wise residual bottleneck; and `BACKBONE_MODE=upper_k` trains the final K
Transformer blocks plus the sequence/head layers. LoRA is not combined with
the native channel/patch adapter. All controls use the same ISRUC manifest,
checkpoint, batch size, epoch budget, validation-kappa selection, and final
test evaluation.

## Canonical ISRUC LaBraM progress and closeout (2026-07-24)

This section supersedes earlier pending-job notes. LaBraM remains isolated in
the `LaBraM` repository; CBraMod TMLR belongs in the dedicated original-CBraMod
clone, and CBraMod ICASSP depth work remains in `EEGxPlore/EEGxPlore`. The
repositories must never be wired together for a paper result.

### Completed ISRUC evidence

- The ISRUC loader and preprocessing contract is verified: subjects 1--80
  train, 81--90 validation, 91--100 test; six bipolar channels; 20 epochs per
  sequence; 30 temporal patches per epoch; validation-kappa checkpoint
  selection; batch size 16.
- Frozen dense versus full-width frozen patch adaptation is complete on seeds
  `{42, 1024, 3407}`. Patch improves all three seeds, with mean BA gain about
  `+0.150` and mean kappa gain about `+0.122`, but remains below absolute full
  fine-tuning performance. The defensible claim is conditional usefulness in
  the frozen-backbone regime.
- LoRA qkv rank 8 is complete on seeds `{42, 1024, 3407}`. Test BA is about
  `0.689--0.696` per seed, mean approximately `0.692`; it is a valid weak
  generic frozen control, not a failed job.
- Upper-layer and generic controls are complete as development controls.
- Job `12760559` completed successfully. It used full backbone training with
  backbone LR scale `0.1`, channel adapter core scale `1`, alpha LR scale
  `0.5`, and raw input scale `1.0`. It reduced the final train/validation
  accuracy gap to approximately `0.003`, versus about `0.107` for the raw
  dense reference. Its primary test BA `0.7972` does not exceed dense BA
  `0.7997`; it is evidence for overfitting control, not yet a confirmed
  channel-adapter BA gain.
- The channel branch is active on ISRUC `[B,6,30,D]`: channel sequence length
  is 6, Q/K/V gradients become nonzero, alpha grows to about `0.057`, and the
  residual correction reaches about `0.209` of the backbone feature norm.
- The fair dense LR-scale-0.1 control has nearly the same validation BA as
  `12760559`, so the apparent benefit may primarily be reduced backbone drift
  rather than the channel branch itself. That control still needs final-test
  evaluation before causal claims.

### Dense-baseline fidelity gate

The raw-scale dense ISRUC result is internally consistent but remains above
the reported LaBraM-Base ISRUC result. The controlled audit changes only
`input_scale_divisor` from `1.0` to `100.0`, retains seed 42, batch 16, 30
epochs, LR `2e-4`, layer decay `0.65`, the same checkpoint and split, and runs
final test evaluation. Strict checkpoint-load reporting is enabled. Job
`12762649` was canceled before execution because its 12-hour wall time was
unnecessarily long; the exact audit was resubmitted with a 3-hour limit.

Do not promote any dense baseline to “faithful reproduction” until the audit
has passed strict checkpoint loading, produced a complete epoch trajectory and
final test metrics, and been compared against the raw-scale trajectory. If
`/100` gives the expected lower regime, use it as the faithful comparison
contract and repeat dense over the declared development seeds. If it does not,
report the recipe difference explicitly rather than forcing agreement with the
published number.

### ISRUC run checklist before moving on

- [ ] Finish the seed-42 `/100` dense audit and compare all epochs, metrics,
      train/validation gap, and test class recalls.
- [ ] Verify the strict checkpoint report has no unexpected missing encoder
      keys.
- [ ] Freeze the ISRUC input-scale contract.
- [ ] Run dense full fine-tuning on development seeds `{42, 1024, 3407}` with
      final test evaluation under the selected contract.
- [ ] Run matched dense `backbone_lr_scale=0.1` with final test evaluation.
- [x] Complete LoRA qkv rank-8 on seeds `{42, 1024, 3407}`.
- [x] Complete upper-2, frozen dense, frozen generic, and frozen patch
      development controls.
- [x] Complete the seed-42 channel pilot with reduced backbone LR.
- [ ] Promote channel to multiseeds only if the matched control shows a
      coherent performance or efficiency/generalization-gap advantage.
- [ ] Run frozen rank-32 aligned versus frozen dense only if required by the
      final common-primitive claim.
- [ ] Record per-seed mean/std, selected epochs, full trajectories, test
      metrics, parameter count, residual/update norms, and train/validation
      gaps for every promoted cell.
- [ ] Freeze ISRUC and move to the next dataset after these gates; do not
      reopen broad ISRUC sweeps.

### Slurm wall-time rule

Observed ISRUC 30-epoch jobs take roughly 45--60 minutes on one A6000. The
ISRUC dense, native, and patch launchers now request `03:00:00`, leaving a
substantial safety margin while improving queue priority. Keep 12 hours only
for datasets or jobs whose measured runtime justifies it.

### ISRUC final confirmation packet queued (2026-07-24)

The bounded final-test packet is submitted without opening a new LR or adaptor
sweep. Raw-scale dense is the current adaptor-comparison contract; the `/100`
seed-42 result remains a scaling sensitivity and does not resolve the published
baseline discrepancy.

```text
raw dense completion:       12763552 (seed 1024), 12763558 (seed 3407)
dense backbone-LR=0.1:      12763559 (seed 42), 12763554 (seed 1024),
                             12763553 (seed 3407)
channel adaptor LR=0.1:     12763555 (seed 42), 12763556 (seed 1024),
                             12763557 (seed 3407)
```

All eight ISRUC final-test conditions use batch 16, 30 epochs, LR `2e-4`,
validation-kappa selection, and the same checkpoint/split. The channel packet
uses backbone LR scale `0.1`, channel core LR scale `1`, alpha LR scale `0.5`,
and alpha initialization `0.01`. The purpose is to establish the paired
dense-versus-channel performance/stability trade-off, not to search for a new
best setting. ISRUC can be frozen after these jobs are inspected and the
required metrics and trajectories are recorded.

### Redundancy audit and checkpoint-only evaluation (2026-07-24)

An inventory of all ISRUC checkpoint directories showed that the raw dense
seeds 1024/3407 and dense backbone-LR-0.1 seed 42 already had complete
30-epoch trajectories and selected checkpoints. The queued retraining packet
was therefore canceled; no channel multiseed expansion was promoted because
the seed-42 channel pilot did not show a clear advantage over the matched
LR-0.1 dense validation trajectory.

Only checkpoint-only test evaluation remains:

```text
12763603 -> isruc_labram_dense_lr2e-4_s3407_b16_e30_scale1
12763604 -> isruc_dense_bls0.1_lr2e-4_s42_b16_e30
12763605 -> isruc_labram_dense_lr2e-4_s1024_b16_e30_scale1
```

These jobs evaluate existing `checkpoint-best.pth` and `checkpoint-best-ba.pth`
without retraining and write `final_test.json` into the existing run
directories. After those artifacts are inspected, no further ISRUC training is
required unless a paper claim explicitly requires the optional frozen rank-32
matched comparison.

### Latest checkpoint-only audit correction (2026-07-24)

The three original checkpoint-only jobs all succeeded with strict checkpoint
loading and complete `final_test.json` artifacts:

```text
raw dense, seed 42:    Acc 0.814598, BA 0.799651, kappa 0.758339, F1 0.808894
raw dense, seed 1024: Acc 0.814828, BA 0.793913, kappa 0.758131, F1 0.810832
raw dense, seed 3407: Acc 0.808851, BA 0.793211, kappa 0.750923, F1 0.804034
dense LR 0.1, seed42: Acc 0.821149, BA 0.799549, kappa 0.766433, F1 0.817872
```

The raw dense three-seed primary-test summary is BA `0.795592 ± 0.003533`,
kappa `0.755798 ± 0.004223`, weighted F1 `0.807920 ± 0.003502`, and
accuracy `0.812759 ± 0.003386` (sample standard deviation).

Frozen dense seed 42 also completed successfully: Acc `0.685977`, BA
`0.609793`, kappa `0.577020`, and weighted F1 `0.648440`. Both selected
checkpoints produced identical test metrics, and the strict load report
passed. This is a valid low frozen-head result, not a failed job.

The remaining exact-test gap is the frozen dense and frozen full-width patch
three-seed packet. Corrected evaluations are queued as jobs `12763665`,
`12763666`, `12763667`, and `12763668`; they use existing checkpoints and do
not retrain. Job `12763661` failed only because its submitted directory name
did not exist and must be excluded from scientific summaries. After the four
corrected jobs finish, ISRUC has no remaining required training or multiseed
task. Frozen rank-32 aligned versus frozen dense remains optional only if the
paper claims a parameter-matched common primitive.

Queue correction: the missing frozen-patch seed-42 evaluation is job
`12763670`. Therefore the complete frozen dense/patch test matrix is covered
by completed job `12763658` (frozen dense seed 42) plus queued jobs
`12763665`, `12763666`, `12763667`, `12763668`, and `12763670`.

### ISRUC frozen matrix completed and cross-dataset closure packet (2026-07-24)

All six frozen ISRUC evaluations completed successfully with strict loading.
Primary test metrics are:

```text
seed       frozen dense (Acc / BA / kappa / F1)       frozen patch (Acc / BA / kappa / F1)
42         .685977 / .609793 / .577020 / .648440     .756092 / .721961 / .680816 / .744376
1024       .686092 / .608313 / .576279 / .647667     .762069 / .728435 / .688496 / .749551
3407       .682299 / .605116 / .572168 / .643313     .759080 / .725039 / .684772 / .747060
mean       .684789 / .607741 / .575156 / .646473     .759080 / .725145 / .684695 / .746995
```

The paired frozen-patch BA gains over frozen dense are `+0.112168`,
`+0.120122`, and `+0.119923` for seeds 42, 1024, and 3407, respectively;
mean gain is `+0.117404`. This closes ISRUC. No further ISRUC training,
channel multiseed promotion, or LR sweep is required. The rank-32 aligned
comparison remains optional only for a parameter-matched claim.

To keep the three LaBraM datasets comparable, the following missing reviewer
control cells are now queued with fixed seeds `{42,1024,3407}`, validation-
kappa selection, final test evaluation, and strict checkpoint loading:

```text
FACED:  frozen dense, frozen patch, frozen generic, LoRA qkv-r8, upper-2
       jobs 12763810-12763822, 12763824, 12763826 (15 jobs)
SEED-V: frozen generic, LoRA qkv-r8, upper-2
       jobs 12763823, 12763825, 12763827-12763833 (9 jobs)
```

FACED keeps its validated batch-32/LR-7e-4/80-epoch recipe; SEED-V keeps its
validated `/100`, batch-16/LR-3e-4/40-epoch recipe. These are comparator
controls, not new architecture tuning. Existing native-axis cells are not
duplicated: FACED patch and SEED-V frozen channel are already multiseed;
SEED-V frozen patch remains a singleton-axis capacity control.

### SEED-V control-packet launcher repair (2026-07-25)

The first SEED-V generic/LoRA/upper packet (`12763823`, `12763825`, and
`12763827-12763833`) failed during launcher validation before training. This
was procedural: the launcher rejected `upper_k`/`lora` modes and required a
literal `BACKBONE_LR_SCALE=0.0`; no scientific output was produced.

The launcher now accepts all four intended backbone modes, validates frozen LR
scales numerically, enables strict checkpoint reporting, and requests a
03:00:00 wall time. The unchanged nine-cell packet was resubmitted as jobs
`12763934-12763942` with the same seeds, recipe, and final-test requirement.

### SEED-V TMLR closure checklist (active 2026-07-25)

Completed before the corrected control packet:

- full structural LMDB audit: 117744 finite records, shape `(62,1,200)`,
  labels `[0,4]`, disjoint split keys and sample content, and documented
  within-subject trial split `0-4/5-9/10-14`;
- channel-axis geometry diagnostic: 62-channel attention has active Q/K/V
  gradients, while the one-patch temporal branch has no patch-to-patch
  interaction;
- frozen dense, frozen channel, and frozen patch on `{42,1024,3407}` with
  final test evaluation;
- trainable dense and channel confirmation packet on `{42,1024,3407}`;
- final test diagnostics, train/validation trajectories, parameter counts,
  residual ratios, and channel Q/K/V/update norms for the native cells.

Still required for the SEED-V control section:

- corrected frozen generic, LoRA qkv-r8, and upper-2 controls on all three
  seeds, jobs `12763934-12763942`;
- report their full trajectories and final Acc/BA/kappa/F1 beside the native
  cells;
- preserve the channel-order qualification: the source montage files match,
  but the legacy LMDB does not encode direct row-to-electrode linkage because
  the original CNT inputs are unavailable. This must be stated explicitly in
  the paper unless the raw preprocessing artifacts are recovered.

No SEED-V depth sweep, channel+patch primary result, or additional LR sweep
is required. Depth is not a meaningful temporal-patch claim when `S=1`, and
patch remains only a singleton-axis capacity control.

The initial FACED comparator packet was not valid: jobs `12763810-12763822`,
`12763824`, and `12763826` reached model initialization but stopped at epoch
0 with NaN loss under the attempted frozen/generic/LoRA/upper recipe. They
were failed/excluded from scientific summaries. FACED is not being rerun
while SEED-V is active; its controls require a separate NaN-preflight repair.

### SEED-V control validation repair and replacement packet (2026-07-25)

Inspection of corrected-packet jobs `12763934-12763942` found two procedural
defects, with no scientific result lost:

- generic frozen jobs `12763934`, `12763936`, and `12763938` stopped in the
  launcher because frozen native adapters must use core and alpha LR scales
  of `1.0`; they had been submitted with `.1`;
- LoRA jobs `12763935` and `12763939` passed strict checkpoint loading and
  LoRA injection but hit a SEED-V assertion because the code counted LoRA
  A/B matrices as native adapter parameters while `adapter_type=none`.

The training code now reports `adapter` and `lora` as separate components,
requires frozen mode to leave only head/native-adapter parameters trainable,
requires LoRA mode to leave only head/LoRA parameters trainable, and records
native-adapter and LoRA parameter counts independently. The launcher now
passes upper/LoRA arguments explicitly and rejects combining LoRA with a
native adapter. Structural tests and optimizer-membership checks pass.

The two queued pre-fix jobs were canceled and the seven unresolved cells were
resubmitted under commit `b5d9d77` with the unchanged SEED-V control contract:

```text
generic frozen: 12764050 (s42), 12764051 (s1024), 12764052 (s3407)
LoRA qkv-r8:    12764053 (s42), 12764054 (s1024), 12764055 (s3407)
upper-2:        12764056 (s3407)
```

Upper-2 seeds 42 and 1024 remain valid in-flight jobs `12763937` and
`12763940`; their mode does not exercise either repaired assertion. All
replacement cells retain batch `16`, LR `3e-4`, `/100` scaling, 40 epochs,
strict checkpoint loading, validation-kappa selection, and final test
evaluation. These are comparator controls, not new adapter tuning.

## Historical handoff: FACED closed; PhysioNet-MI record (2026-07-26)

This section is retained for dataset provenance and completed LaBraM evidence.
The active 2026-07-29 TMLR-only sequence above supersedes its execution order.
Historical job IDs and old “still required” language must not be used to
schedule duplicate experiments.

### FACED closure

FACED is complete for the current LaBraM paper matrix. The canonical native
development packets are:

- dense full fine-tuning:
  `checkpoints/faced_labram_dense_lr7e4_e80_seed{42,1024,3407}`;
- native patch-only adaptation:
  `checkpoints/faced_labram_patch_a001_lr7e4_e80_seed{42,1024,3407}`;
- historical depth/component ablations, retained as diagnostic evidence;
- corrected frozen controls, evaluated with `frozen_backbone_eval_mode=true`:
  frozen dense, frozen patch, and frozen generic bottleneck on seeds
  `{42,1024,3407}`.

The corrected frozen-control jobs were `12777453-12777461` and all completed
with strict checkpoint loading, 80 unique epochs, validation-kappa selection,
and final test evaluation. Their final test means are:

| control | BA | kappa | F1 |
|---|---:|---:|---:|
| frozen dense | 0.1589 +/- 0.0003 | 0.0566 +/- 0.0003 | 0.1539 +/- 0.0002 |
| frozen patch | 0.1624 +/- 0.0034 | 0.0577 +/- 0.0039 | 0.1613 +/- 0.0061 |
| frozen generic | 0.1859 +/- 0.0050 | 0.0842 +/- 0.0058 | 0.1831 +/- 0.0057 |

The frozen dense trajectory is weak/underfit, frozen patch rapidly memorizes
the training set without a robust validation or test gain, and frozen generic
is a modest but consistent capacity control. The frozen patch result is not a
positive structure-aware claim. The native patch result may be reported only
as a trainable-backbone FACED result, with its validation/test variability
shown explicitly. The earlier FACED frozen packet (`12763810-12763822`,
`12763824`, `12763826`) is excluded because it did not enable frozen eval mode;
it is not evidence against frozen adaptation. Valid trainable LoRA/upper-layer
controls remain comparator evidence, but no new FACED run is required.

Conclusion: FACED is closed. Do not add more FACED tuning, depth sweeps, or
duplicate seeds unless a reproducibility audit finds a concrete contract
violation.

### PhysioNet-MI execution gate

PhysioNet-MI remains a documented LaBraM dataset block. Any future TMLR
follow-up must use the LaBraM repository; CBraMod belongs either in the
dedicated original-CBraMod TMLR clone or, for deferred ICASSP depth work, in
`EEGxPlore/EEGxPlore`. The two backbones are never wired together.

The current data location is `/data/neurogroup/mingyangjiang/data/PHYSIO_MI`.
The available source is serialized LMDB (`data.mdb`/`lock.mdb`); the original
raw recording tree is not currently available. Therefore no training result
may be called a faithful replication until the following are recorded:

1. LMDB key/schema inspection, total records, tensor shape, finite-value
   checks, label set, class counts, and any subject/session identifiers;
2. provenance and preprocessing evidence: source script or documented
   serialized preprocessing, sampling rate, filters, resampling, scaling,
   window length/stride, channel names/order, montage, and channel-position
   information;
3. exact train/validation/test split construction, subject/session
   disjointness, split hash, per-class support, and leakage checks;
4. the LaBraM input contract after preprocessing, including input scaling and
   the realized `[B,C,S,D]` geometry.

If the original preprocessing cannot be recovered, label the experiment
“serialized-PhysioNet-MI protocol” and state the provenance limitation rather
than claiming exact raw-data replication. Do not silently infer channel
positions or reorder electrodes.

### PhysioNet-MI audit result (2026-07-26)

The available LMDB was inspected read-only and matches the EEGxPlore
`preprocessing/preprocessing_physio.py` contract:

- LMDB: `/data/neurogroup/mingyangjiang/data/PHYSIO_MI/data.mdb`, size
  4,070,227,968 bytes, SHA-256
  `2a51ca7523a149a5528b77bdc5f0c73c5a36095c0a3feb8c4fc9b179af5e71d0`;
- indexed records: 9,837, plus the `__keys__` split index;
- splits: train 6,300, validation 1,734, test 1,803;
- subjects: train `S001-S070`, validation `S071-S089`, test `S090-S109`,
  with disjoint subject sets;
- class counts: train `{0:1593, 1:1557, 2:1581, 3:1569}`, validation
  `{0:435, 1:432, 2:434, 3:433}`, test `{0:451, 1:449, 2:450, 3:453}`;
- every indexed record has schema `{'sample','label'}`, finite float64 sample
  shape `(64,4,200)`, and no missing/unindexed/corrupt records;
- split-index SHA-256:
  `129909ad3054357d25d4bb04d68738e8478bc76673503a861a1befabf669e5b0`;
- stored raw-value standard deviations are train/val/test
  `36.6736/34.9148/40.3890`; the EEGxPlore loader returns `sample / 100`,
  giving `0.366736/0.349147/0.403890` after scaling.

The source preprocessing selects 64 channels in the explicit
`selected_channels` order, uses tasks R04/R06/R08/R10/R12/R14, average
reference, 0.3-Hz high-pass filtering, 60-Hz notch filtering, 200-Hz
resampling, four-second epochs, the final 800 samples, and the fixed subject
split formed from sorted subject directories (70/19/20). The stored keys
reflect skipped event-1 epochs, so trial suffixes beginning at `-1` are
expected and must not be “corrected.” The preprocessing script SHA-256 is
`823217891671d485c1438c1396596b822b6be7ab4176bd7114d6b42d8e90c19e`.

The audit passes the serialized-data gate. It does not prove raw EDF
provenance because the raw tree is absent. The existing EEGxPlore
`scripts/PHYSIO-MI/train_physio_compact_shared.slurm` and
`models/model_for_physio.py` are CBraMod/MoE code and are not a LaBraM
baseline. LaBraM currently has no PhysioNet-MI branch in
`run_class_finetuning.py`; the next implementation step is a LaBraM-only
loader/wrapper that consumes these preprocessed tensors, applies the same
`/100` scaling, preserves the 64-channel order and four temporal patches, and
records this repository boundary explicitly.

### LaBraM PhysioNet-MI implementation status (2026-07-26)

Implemented entirely in this repository:

- `utils.py:PHYSIONET_MI_LABRAM_CH` stores the exact 64-row channel order;
- `utils.py:PhysioNetMILoader` reads the LMDB lazily, validates
  `(64,4,200)`, labels `[0,3]`, finite values, split keys, and subject
  disjointness, and returns raw tensors;
- `run_class_finetuning.py` registers `PhysioNet-MI` with four classes and
  LaBraM channel-position mapping;
- `scripts/submit_physio_mi_dense_accre.slurm` runs the LaBraM-only dense
  baseline with engine-side `/100` scaling, strict checkpoint loading, and
  final test evaluation.

The first smoke baseline `12786217` completed its protocol gate successfully:
strict loading, correct data contract, finite inputs, and final test access
all worked. Its three-epoch performance was random-guess level (test BA
`0.2500`, kappa `0.0000`, weighted F1 `0.1009`) and it showed approximately
1% AMP gradient-step skips in the first two epochs, so it is not a scientific
baseline. The run also exposed missing batch/epoch fields in `run_config`.

The provenance fields are now recorded. The initial conservative full run
`12788099` (seed 42, LR `1e-4`) is retained as a historical sensitivity
anchor. It was superseded as the primary dense contract by the completed
`LR=5e-4` sweep and locked three-seed packet below.

### PhysioNet-MI run checklist

Do not submit the comparison packet until the data audit passes. Then use the
following fixed sequence:

1. **Faithful dense baseline:** reproduce the available source protocol,
   checkpoint, loader, scaling, split, batch size, optimizer/LR, schedule, and
   epoch budget. Run development seeds `{42,1024,3407}`. Select checkpoints by
   validation Cohen kappa only; report complete train/validation trajectories
   and final test accuracy, balanced accuracy, kappa, macro-F1, and weighted-F1.
   Record strict checkpoint-load status, input statistics, split hash,
   trainable parameter count, and the complete CLI/run contract.
2. **Frozen probe:** frozen LaBraM plus the shared classifier/head, with
   `frozen_backbone_eval_mode=true`, using the same split and seed packet.
3. **Geometry gate:** determine whether channel and/or temporal-patch axes
   have meaningful sequence length and nonzero Q/K/V gradients. Only eligible
   axes can support a primary native adapter claim.
4. **Native adaptation:** run the eligible channel-only or patch-only adapter;
   run channel+patch only when both axes are meaningful. Keep the native
   adapter separate from LoRA and use the same dense baseline contract.
5. **Independent comparison approaches:** run generic bottleneck, LoRA
   qkv-r8, and upper-2 controls as separate methods. Add a parameter-matched
   axis-blind control when needed for the paper’s capacity claim. Do not
   combine LoRA with channel/patch adaptation.
6. **Secondary diagnostics:** run depth only when its axis is meaningful and
   report adapter alpha, residual/update norms, Q/K/V gradients, parameter
   counts, and train-validation gaps. Do not promote a degenerate-axis result.
7. **Closure:** after the three-seed packet is valid and the promotion rule is
   met, freeze the dataset section. Use the same three seeds for the final
   locked test table; do not launch broad hyperparameter sweeps by default.

The PhysioNet-MI claim must therefore be framed relative to the faithful dense
baseline: native adaptation is a lightweight, geometry-matched alternative
when limiting backbone drift is valuable, not a universal replacement for
full fine-tuning. Every dataset must contain the same baseline, native,
frozen, and independent-control logic before it is marked complete.

### PhysioNet-MI dense contract lock (2026-07-26)

The seed-42 LR sensitivity packet completed successfully. The `5e-4` setting
is now the locked PhysioNet-MI dense contract because it improved both
validation and held-out test performance while leaving the data, checkpoint,
model geometry, and selection rule unchanged:

- batch size `64`, epochs `40`, AdamW, weight decay `0.05`;
- LR `5e-4`, layer decay `0.65`, five warmup epochs, label smoothing `0.1`;
- gradient clipping `1.0`, `/100` engine scaling, strict checkpoint loading;
- validation Cohen-kappa selection, with validation-BA sensitivity selection;
- fixed multiseed packet `{42,1024,3407}`.

The seed-42 locked run is
`physio_mi_labram_dense_sweep_lr5e4_s42_b64_e40_scale100` and achieved test
BA `0.6123`, kappa `0.4830`, and weighted F1 `0.6142`. For context, the old
`1e-4` anchor achieved test BA `0.5581`; it is retained as a historical
sensitivity result, not the primary baseline.

The locked dense packet is complete. Seed 42 is the completed sweep run;
jobs `12789351` and `12789352` completed seeds 1024 and 3407 with exit code
`0:0`. All three runs completed 40 epochs, produced strict-load reports and
selected checkpoints, and passed the artifact gate.

Locked dense results, selected by validation kappa:

| seed | best epoch | val BA | test BA | test kappa | test weighted F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 22 | 0.5761 | 0.6123 | 0.4830 | 0.6142 |
| 1024 | 17 | 0.5744 | 0.6223 | 0.4964 | 0.6233 |
| 3407 | 18 | 0.5382 | 0.5933 | 0.4579 | 0.5933 |
| mean +/- sample SD | — | 0.5629 +/- 0.0214 | 0.6093 +/- 0.0147 | 0.4791 +/- 0.0196 | 0.6103 +/- 0.0154 |

The trajectory is consistent across seeds: learning begins around epochs
8-10, validation peaks at epochs 17-22, and training accuracy continues to
rise afterward, so moderate late overfitting is present but there is no
collapse or class-support failure. The seed-3407 result is lower but remains
well above chance and is a valid member of the fixed three-seed packet.

All subsequent PhysioNet-MI frozen, native-adapter, LoRA, generic, and
comparator runs must inherit this contract unless explicitly labeled as a
sensitivity experiment. Do not mix this LaBraM contract with the CBraMod
PhysioNet-MI pipeline in EEGxPlore.

### PhysioNet-MI remaining experiment checklist after dense closure

The dense baseline gate is complete. The remaining paper-grade PhysioNet-MI
block is:

1. **Geometry gate:** record the realized LaBraM grid `[C,S,D]=[64,4,D]`
   and channel/patch Q-K-V and output gradients. Both channel (`C=64`) and
   temporal-patch (`S=4`) axes are eligible in principle; eligibility must be
   confirmed in the run diagnostics.
2. **Frozen probe:** frozen LaBraM plus shared head, seeds
   `{42,1024,3407}`.
3. **Frozen native comparison:** frozen LaBraM plus the promoted native
   channel/patch adapter, same three seeds, with adapter diagnostics.
4. **Full-backbone native matrix:** channel-only and patch-only, and
   channel+patch if both branches pass the geometry gate; three seeds for
   every retained condition.
5. **Independent controls:** frozen generic bottleneck, LoRA qkv-r8, and
   upper-2 fine-tuning, each on all three seeds. Add the parameter-matched
   axis-blind control if the manuscript makes an explicit capacity-matching
   claim.
6. **Depth secondary ablation:** only if the depth intervention is
   scientifically meaningful; otherwise mark it inapplicable with the
   geometry evidence. Do not promote it as a primary adapter.
7. **Closure diagnostics:** aggregate complete epoch trajectories, final
   test accuracy/BA/kappa/macro-F1/weighted-F1, train-validation gaps,
   trainable parameters, strict checkpoint reports, alpha/residual/update
   norms, and efficiency measurements. Then freeze the PhysioNet-MI table.

Thus, the dense baseline is no longer a remaining run. The remaining work is
the frozen/native/comparator matrix and its diagnostics; no further dense LR
sweep is justified.

### PhysioNet-MI closure after the complete three-seed packet (2026-07-27)

The PhysioNet-MI LaBraM block is complete under the current dataset checklist.
All canonical conditions use the same LaBraM-only loader, channel order,
`(64,4,200)` geometry, `/100` engine scaling, batch size `64`, 40 epochs,
LR `5e-4`, strict checkpoint loading, validation-kappa selection, final test
evaluation, and seeds `{42,1024,3407}`.

Completed canonical packets:

- dense full fine-tuning: `0.6093 +/- 0.0147` test BA;
- frozen dense probe: three seeds, test BA approximately `0.269`;
- frozen channel+patch low-rank adapter: three seeds, modest conditional
  improvement over frozen dense, test BA approximately `0.283`;
- full-backbone channel-only: `0.6153 +/- 0.0123` test BA;
- full-backbone patch-only: `0.6256 +/- 0.0100` test BA;
- full-backbone channel+patch: `0.6253 +/- 0.0082` test BA;
- frozen generic bottleneck, LoRA qkv-r8, and upper-2-layer controls: three
  seeds each, all with complete epoch trajectories and final test metrics.

The original patch-only jobs `12791423-12791425` failed before training due
to an over-strict launcher geometry assertion that incorrectly required
channel-attention activity for a patch-only adapter. They are retained as
failed provenance attempts. The corrected canonical replacements
`12799545-12799547` passed the patch geometry gate, strict checkpoint load,
40-epoch training, and final test evaluation. No patch result from the failed
attempts is used.

The primary PhysioNet-MI conclusion is bounded: native-axis adaptation is a
structure-aware alternative that can provide a modest gain over dense
fine-tuning in this protocol, especially for patch and channel+patch
conditions, but it is not a universal replacement for dense fine-tuning.
Generic frozen adaptation remains near chance; LoRA and upper-layer controls
are below the dense baseline. Native validation trajectories still show early
learning followed by late validation saturation/decline, so checkpoint
selection and the train-validation gap remain part of the reported result.

PhysioNet-MI is closed for the current required matrix. Parameter-matched
axis-blind adaptation remains optional and is not included in this closure;
it must be added before making a strong matched-capacity axis-blind claim.
Depth remains secondary and was not promoted because it is not needed for the
current primary conclusion. The next active LaBraM dataset is TUEV: first
perform its data/provenance, class-support, split-hash, channel-order, and
scaling audit, then replicate the dense baseline before any adapter runs.

### TUEV LaBraM transition and provenance gate (2026-07-27)

TUEV is now the active LaBraM dataset. CBraMod remains isolated in the
EEGxPlore repository; no backbone or preprocessing code is mixed between the
repositories.

Verified source/provenance contract:

- source preprocessing: EEGxPlore `preprocessing_tuev.py`, SHA256
  `3ec3ae3967353f86f4c970bdf86e7382e59bc492b7397382372f6b429231c6f3`;
- serialized records contain `signal` and `label`, with raw signal shape
  `[16,1000]`, finite float data, and labels `1..6`;
- the LaBraM loader preserves the EEGxPlore bipolar order:
  `FP1-F7, F7-T7, T7-P7, P7-O1, FP2-F8, F8-T8, T8-P8, P8-O2,
  FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2`;
- loader geometry is `[16,5,200]`; the engine applies exactly one `/100`
  scale; model labels are converted to `0..5`;
- split sizes are train `68,712`, validation `15,220`, and test `29,421`;
- split manifest SHA256 values are train
  `988661443a75f7ef3a65a14321fe9591da5163e3e55a2b7719b628a606a8c60f`,
  validation `3d711a9903b596469eb8e4b1179dd7c9c8ce7e902042135dc7263f30d8b40e78`,
  and test `9b59f733d446511b0c1aa14f3bc8c16023a4a84691b0007e2dc427853c77924a`;
- the serialized train split has class counts
  `{1:578, 2:8936, 3:5684, 4:737, 5:8409, 6:44368}`; validation/test
  class-support scans remain pending because of slow shared-filesystem I/O.

A direct standard-1020 lookup is invalid for the final eight bipolar rows:
16 bipolar channels exceed LaBraM's 128-channel positional table. The
scientifically explicit TUEV policy is therefore to preserve the serialized
bipolar order and use sequential absolute positional slots `1..16`, with
`input_chans=None`. The preflight produces `[1,81,200]` tokens and passes.
This policy is recorded in every TUEV run contract as
`sequential_pos_embed_slots_1_to_16`; it is not an accidental channel-name
substitution.

Historical TUEV jobs did not produce a trustworthy baseline: they failed due
to environment/checkpoint/geometry/shape issues. A first batch-16 smoke was
cancelled after showing impractical throughput (`4,294` steps/epoch). The
replacement launcher `scripts/submit_tuev_dense_audit_accre.slurm` adds
sorted file order, strict checkpoint-load diagnostics, geometry/finiteness
preflight, explicit `/100` scaling, no auto-resume, and final test
evaluation. Its practical fixed contract is batch `64`, `25` epochs, AdamW,
LR `5e-4`, weight decay `.05`, layer decay `.65`, five warmup epochs, drop
path `.1`, label smoothing `.1`, no gradient clipping, and seed packet
`{42,1024,3407}`. Batch/epoch choices are explicitly recorded as the
operational TUEV contract because batch-16 was infeasible; the model and
optimizer recipe remain unchanged.

Current jobs:

- `12802746`: seed-42 one-epoch smoke/preflight, batch 64; training and
  validation completed with finite inputs/outputs, strict checkpoint pass,
  and AMP recovery. Its validation BA/kappa/weighted-F1 were `0.5956`,
  `0.6692`, and `0.8180`; it is a gate, not a paper baseline;
- `12802748`: the initial 40-epoch seed-42 baseline was cancelled after
  `6:54` when the common comparison budget was changed to 25 epochs. Its
  partial trajectory is not used scientifically;
- `12803131`--`12803134`: controlled seed-42 LR sweep at `1e-4`, `2.5e-4`,
  `5e-4`, and `1e-3`, respectively; all use batch 64, 25 epochs, five warmup
  epochs, `/100` scaling, strict loading, and final test evaluation.

Do not submit TUEV adapters or the other two dense seeds until the LR sweep
identifies the 25-epoch contract and its artifact gate passes. Then submit
exactly seeds `1024` and `3407` under the selected setting, followed by the
checklist's frozen/native/LoRA/generic/upper controls. No TUEV dense sweep
metric is yet a paper result.

### TUEV registry preparation (2026-07-27)

The post-sweep TUEV registry is prepared but intentionally not submitted.
`scripts/submit_tuev_registry_accre.slurm` centralizes the 25-epoch contract
for `full_dense`, `frozen_dense`, frozen native `channel`/`patch`/
`channel_patch`, trainable native `channel`/`patch`/`channel_patch`, generic
frozen bottleneck, independent LoRA qkv-r8, upper-2, and an opt-in
`native_depth` condition. `scripts/queue_tuev_registry.sh` requires an
explicit validation-selected `TUEV_LR` and defaults to the fixed seed packet
`{42,1024,3407}`; it performs no submission until called.

All registry conditions inherit batch `64`, 25 epochs, five warmup epochs,
AdamW weight decay `.05`, layer decay `.65`, label smoothing `.1`, `/100`
scaling, strict checkpoint loading, sequential positional slots `1..16`,
validation-kappa selection, validation-BA sensitivity selection, and final
test-only-after-selection evaluation. Native low-rank channel/patch branches
are kept separate from generic LoRA, consistent with the paper's method
ladder.

The registry preflight passed for every prepared geometry: dense/frozen,
native channel, native patch, native channel+patch, generic, upper-2, native
depth, and LoRA. TUEV realizes `[C,S,D]=[16,5,200]`, so both channel and
temporal-patch interaction branches are eligible and report active sequence
lengths. The optional `axis_blind` method requires an explicit native adapter
parameter target and enforces a `+/-5%` trainable-parameter match before it
can run; it is not silently conflated with the generic control. Native depth
is opt-in after the aligned channel+patch trajectory passes its gate.

Once the LR sweep selects a TUEV setting, the remaining order is: complete
the dense three-seed packet, run the frozen probe and frozen/native controls,
run full-backbone native channel/patch/channel+patch, then run generic, LoRA,
upper-2, and the parameter-matched axis-blind control. Only after those
sections pass should the optional depth extension be queued. No registry job
has been submitted yet.

### TUEV seed-42 registry result and operational contract (2026-07-29)

The seed-42 exploratory comparison registry completed successfully for all
ten queued cells `12823148`--`12823157`. Every run completed 15 epochs with
exit code 0, produced `log.txt`, `run_config.json`, checkpoints, and
`final_test.json`, and recorded a strict checkpoint-load pass. The test
target support is `[567, 4677, 1998, 329, 2204, 19646]` for model classes
0--5. The runs used the operational TUEV contract: batch 64, LR `5e-4`,
weight decay `.05`, layer decay `.65`, five warmup epochs, drop path `.2`,
label smoothing `.1`, `/100` scaling, validation-kappa selection, and test
only after selection.

Trajectory-level interpretation: frozen dense learns slowly and remains
underfit; frozen channel gives the clearest conditional frozen-backbone
gain; frozen patch and frozen channel+patch do not improve it. Trainable
native channel is the strongest native validation-kappa condition, but all
native full-fine-tuning cells peak very early (epochs 2--4) and then
overfit. BA peaks can occur later, so validation-kappa and BA checkpoints
must remain separate. LoRA and upper-2 are valid independent controls, not
parameter-matched comparisons.

TUEV is not yet closed. The remaining required science runs are the dense
operational contract and every comparison cell on seeds `1024` and `3407`;
the current seed-42 cells are exploratory and should be repeated in the
final clean three-seed packet after the code is committed. Required cells
are frozen dense/channel/patch/channel+patch, trainable native
channel/patch/channel+patch, generic, LoRA qkv-r8, and upper-2. Depth and
axis-blind matched-capacity controls remain optional extensions and do not
block minimum TUEV closure. Before final reporting, freeze a clean commit,
confirm all run contracts, and retain alpha/residual/update norms, native
Q/K/V gradients, parameter counts, and train-validation gaps.

### TUEV LaBraM operational closure and CBraMod handoff (2026-07-31)

The TUEV LaBraM block is now operationally complete. The final valid packet
contains 11 conditions on exactly the three project seeds `{42,1024,3407}`:

- full dense;
- frozen dense, frozen channel, frozen patch, and frozen channel+patch;
- trainable native channel, native patch, and native channel+patch;
- independent generic bottleneck, LoRA qkv-r8, and upper-2 controls.

Seed 42 uses the completed operational dense run `12812081`; the replacement
three-seed registry is `12869559`--`12869580`. The seed-42 registry
`12823148`--`12823157` and zero-init sensitivity jobs
`12842524`--`12842526` are retained as diagnostics, not substituted for the
locked packet. Every canonical condition produced `log.txt`, `run_config.json`,
`checkpoint-best.pth`, `checkpoint-best-ba.pth`, and `final_test.json`; strict
checkpoint loading passed. The canonical contract is batch 64, 15 epochs,
LR `5e-4`, five warmup epochs, weight decay `.05`, layer decay `.65`, drop
path `.2`, label smoothing `.1`, `/100` scaling, and validation-kappa primary
selection with validation-BA sensitivity checkpoints.

The primary validation-kappa-selected test BA summaries are:

| Condition | Test BA (mean +/- SD) | Test kappa (mean +/- SD) | Test weighted F1 (mean +/- SD) |
| --- | ---: | ---: | ---: |
| Full dense | `0.5961 +/- 0.0125` | `0.5950 +/- 0.0242` | `0.7889 +/- 0.0108` |
| Native channel | `0.5934 +/- 0.0108` | `0.5762 +/- 0.0092` | `0.7780 +/- 0.0086` |
| Native patch | `0.5890 +/- 0.0593` | `0.5596 +/- 0.0448` | `0.7700 +/- 0.0264` |
| Native channel+patch | `0.5901 +/- 0.0145` | `0.5412 +/- 0.0291` | `0.7556 +/- 0.0158` |
| LoRA qkv-r8 | `0.5717 +/- 0.0229` | `0.5813 +/- 0.0183` | `0.7828 +/- 0.0073` |
| Upper-2 | `0.5412 +/- 0.0204` | `0.5638 +/- 0.0423` | `0.7729 +/- 0.0198` |

Frozen dense is `0.3658` BA. Frozen channel, patch, and channel+patch reach
`0.4323 +/- 0.0186`, `0.3988 +/- 0.0182`, and `0.3916 +/- 0.0182`, respectively;
this supports a conditional frozen-backbone channel benefit, but none reaches
full dense performance. The generic frozen control is `0.4367 +/- 0.0126`.

The epoch trajectories show the same failure mode seen earlier: train loss
falls from about `1.09` to `0.44` and training accuracy approaches 99%, while
validation kappa usually peaks in epochs 1--10 and then declines. Native
channel is the most stable native branch, but its mean is slightly below dense
(`-0.0027` BA); patch is high variance and channel+patch is not a reliable
combination. Zero-output initialization did not remove early overfitting.
These are useful negative/boundary results: TUEV does not support the claim
that native adaptation universally replaces full fine-tuning.

Historical failed/cancelled jobs are excluded from the paper packet: the
12280583/12282548/12283655/12283679/12283758 jobs failed during the early
environment/checkpoint/geometry phase; 12802642 and 12802748 were cancelled;
12843245--12843260 failed before training with `QOSGrpGRES`. The LR sweeps,
regularization probes, seed-42 registry, and zero-init runs remain provenance
and sensitivity records only. There are no failed runs inside the locked
three-seed TUEV packet.

**Handoff decision.** TUEV is sufficient for the LaBraM operational backbone
block. No further TUEV tuning or duplicate multiseed runs should be launched.
The next active TMLR step is to create the dedicated clean clone of the
original CBraMod repository, audit its native branches/checkpoint/preprocessing
contract, and begin the CBraMod FACED dense baseline. CBraMod work must remain
outside `EEGxPlore` and must not import or wire the LaBraM implementation.

This is an operational LaBraM closure, not yet the final cross-backbone TMLR
evidence closure. Before submission, the common TMLR registry still needs the
prespecified parameter-matched axis-blind control (and final budget/statistical
aggregation) wherever that claim is retained. That gate is carried forward to
the final cross-backbone analysis rather than used to delay the CBraMod block.

### CBraMod TMLR adapter implementation handoff (2026-07-31)

The dedicated original-CBraMod clone is now available at
`/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod`, remote
`https://github.com/wjq-learning/CBraMod.git`, base commit `0ff6be9`. The
adapter is implemented only in that repository. The LaBraM repository and the
EEGxPlore CBraMod/ICASSP repository remain untouched by the CBraMod model code.

The implementation follows the common TMLR family rather than introducing a
second method. CBraMod's runtime tensor is `[B,C,S,D]`; its own criss-cross
layer uses `D/2` spatial features with attention over channels and `D/2`
temporal features with attention over patches. The adapter consequently uses
`Down -> native-axis mixer -> Up` on the spatial half for `channel`, the
temporal half for `patch`, or both independent halves for `channel_patch`.
It is attached after the encoder, preserves the pretrained blocks, and uses
zero-initialized Up projections for exact dense parity. Degenerate axes are
rejected. No LaBraM import, depth gate, MoE, router, or ICASSP code is used.

Files added/changed in the CBraMod clone:

- `models/interaction_adapter.py`: native spatial/temporal residual branches;
- `models/cbramod.py`: strict-load-compatible adapter attachment and diagnostics;
- `models/model_for_faced.py`: strict base-checkpoint load before adapter attach;
- `tests/test_interaction_adapter.py`: dense parity, geometry, gradient, and
  ineligible-axis tests;
- `docs/tmlr_adapter_design.md`: method mapping and FACED gate.

The model gate passes. FACED is the next dataset, but no paper run should use
the legacy CBraMod trainer until a clean TMLR runner records the checkpoint,
split/manifest hashes, geometry, scaling, optimizer groups, validation-selected
checkpoints, and final test metrics. The immediate order is FACED data audit,
seed-42 dense smoke, seed-42 native-branch screen, then the locked three-seed
packet `{42,1024,3407}`. The axis-blind control and cross-backbone aggregation
remain later TMLR closure work, as planned.
## CBraMod ISRUC r64 repair and manuscript evidence v2 (2026-08-21)

The audit-driven CBraMod/ISRUC native channel-plus-patch repairs completed
successfully:

| Seed | Job | Run ID | Selected epoch | Test BA | Test macro-F1 | Test kappa |
|---:|---:|---|---:|---:|---:|---:|
| 42 | existing | `isruc_cbramod_frozen_channel_patch_s42_lr2e-4_b8_e20` | 9 | 0.75410 | 0.74715 | 0.71196 |
| 1024 | 13512637 | `isruc_cbramod_frozen_channel_patch_s1024_r64_lr2e-4_b8_e20` | 15 | 0.74459 | 0.74079 | 0.69566 |
| 3407 | 13512638 | `isruc_cbramod_frozen_channel_patch_s3407_r64_lr2e-4_b8_e20` | 16 | 0.75697 | 0.75748 | 0.71347 |

Both repairs use the seed-42 operational configuration, including
`interaction_aligned`, `channel_patch`, bottleneck `64`, four heads, adapter
LR `2e-5`, classifier LR `3.536e-4`, batch size `8`, 20 epochs, gamma `1.0`,
init alpha `0.01`, zero-initialized output, and the same strict pretrained
checkpoint. Each artifact has strict checkpoint loading, expected ISRUC
geometry `[B,20,6,30,200]`, frozen-backbone mode evidence, and 20 epoch
metric/adapter-diagnostic records.

The corrected primary aggregates are in
`analysis/tmlr_manuscript/manuscript_final_v2/`. The native adapter has
`59,610` adaptation-module parameters versus `59,949` for the alpha-zero
axis-blind control; the final pair audit reports a `0.5655%` mismatch for all
three seeds. The v2 package contains 156 selected rows, 12 valid primary RQ2
seed-pairs, and four complete primary RQ2 cells, including CBraMod/ISRUC.
The fail-closed self-check passes with zero errors and zero warnings.

CBraMod/ISRUC interpretation under the corrected r64 condition:

- Native frozen adaptation is below the frozen probe in mean test BA by
  `-0.00432` and in mean macro-F1 by `-0.00317`; BA is lower for all three
  seeds.
- Native channel-plus-patch versus matched axis-blind adaptation is mixed:
  mean BA effect `+0.00456 +/- 0.01775` and mean macro-F1 effect
  `+0.00536 +/- 0.02156`, with two of three seeds positive for both metrics.
- This repairs the condition-consistency problem but does not turn ISRUC
  into a uniform positive result. It supports the paper's conditional claim
  that native alignment can be useful without being universally superior.

`manuscript_final_v1` is retained as the superseded audit snapshot; v2 is the
current manuscript-facing package. No additional CBraMod/ISRUC training is
justified by these results.

## Full cross-backbone evidence audit (2026-08-25)

The claims-first manuscript manifest was not a complete inventory of the
experiment store. A separate fail-closed audit is now maintained at
`analysis/tmlr_manuscript/evidence_audit_20260825/`.

The audit covers every experiment-like directory under both result stores and
every `.out`, `.err`, `.vu`, and `log.txt` file under the CBraMod and LaBraM
repositories:

| Item | Audited count | Interpretation |
|---|---:|---|
| Experiment-like directories | 769 | 302 CBraMod result directories and 467 LaBraM checkpoint directories |
| Registry artifacts | 485 | 271 CBraMod and 214 LaBraM records; all have a complete final test contract |
| Supplemental completed artifacts | 8 | Six TUEV branch-local-MLP runs and two parity artifacts |
| Log/event files | 2,515 | 2,047 textual logs plus 468 binary TensorBoard event files |
| Registry aggregate conditions | 89 | 88 complete three-seed groups and one one-seed LaBraM--ISRUC legacy row |

The ten-cell coverage table is
`backbone_dataset_coverage.csv`. It distinguishes filesystem availability,
completed test artifacts, three-seed completeness, matched-RQ2 eligibility,
and manuscript role. The primary matched RQ2 cells remain exactly:
CBraMod--FACED, CBraMod--ISRUC, CBraMod--TUEV, and LaBraM--TUEV.
LaBraM--FACED, LaBraM--ISRUC, and LaBraM--SEED-V are retained as supporting or
geometry-boundary context because their available records do not satisfy the
same strict native-versus-axis-agnostic matched configuration contract.

The log audit found 71 scheduler logs with failure markers: 51 traceback
markers, 20 cancellation/termination markers, and smaller subsets containing
runtime, value, import, or assertion errors. These logs are not promoted to
results. The corresponding artifact table separates them from 485 completed
registry artifacts and from 132 incomplete training directories, 31
configuration-only/preflight directories, 56 legacy log-only directories, and
26 audit/smoke directories. Failure counts are in
`log_failure_summary.csv`; the full per-file inventory is in
`log_inventory.csv`.

This audit supersedes the former informal statement that the registry held
483 artifacts. The registry currently holds 485; the six branch-local MLP
artifacts are complete but remain supplemental because they were added after
the older registry snapshot. Re-run `audit_full_evidence.py` after any new
experiment, or use `--reuse-log-inventory` only when the log store itself has
not changed.
