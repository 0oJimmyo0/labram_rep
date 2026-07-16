# LaBraM Structured Adapter Research Plan

## Objective

Test whether a LaBraM-native, structure-aware residual adapter improves FACED emotion classification over the original LaBraM backbone under a fair and reproducible protocol.

The paper claim is narrow: minimal channel- or patch-structured residual capacity should improve the representation without replacing LaBraM or adding a collection of unrelated mechanisms.

## Fixed Protocol

- Dataset: FACED at `/data/neurogroup/mingyangjiang/data/FACED`
- Input contract: `(32, 10, 200)` and the validated 32-channel manifest
- Checkpoint: `checkpoints/labram-base.pth`
- Preprocessing: divide samples by `100`
- Model flags: absolute positional embedding enabled, relative positional bias disabled, qkv bias disabled
- Selection metric: validation Cohen's kappa
- Sensitivity selection: validation balanced accuracy
- Final headline: test balanced accuracy from the validation-kappa checkpoint
- Secondary final metrics: test weighted F1 and test Cohen's kappa
- Test evaluation: after training only, from `checkpoint-best.pth` and `checkpoint-best-ba.pth`
- Adapter defaults: token MLP off, depth mode none, gamma `1.0`, alpha `0.01`
- Development seeds: `42`, `1024`, `3407`

Never select a structure, alpha, learning rate, or checkpoint using test performance.

## Strong Dense Recipe

The strongest historical local LaBraM pilot used:

```text
batch_size=16
lr=7e-4
epochs=80
warmup_epochs=10
weight_decay=0.05
layer_decay=0.65
drop_path=0.1
```

The historical seed-0 pilot reached validation BA `0.48904` at epoch 63, but test was evaluated every epoch, so that result is exploratory rather than a final paper result. Rerun this recipe under the current protocol before treating it as the formal baseline.

## Completed Seed-3407 Screen

The first clean strong-recipe screen used seed `3407` and the recipe above.

| Condition | Validation kappa | Validation BA | Test BA from kappa checkpoint | Status |
| --- | ---: | ---: | ---: | --- |
| Dense | 0.36881 | 0.44244 | 0.39855 | Complete |
| Channel-only | 0.35176 | 0.42778 | 0.39385 | Complete |
| Patch-only | 0.42660 | 0.49352 | 0.43129 | Complete, promising |
| Channel + patch | unavailable | unavailable | unavailable | DataLoader worker segfault |

Patch-only is currently the leading active structure by validation metrics. Its test result is recorded for diagnostics only and was not used for selection.

## Next Sequence

### 1. Complete the structure ladder

Retry channel-plus-patch with `NUM_WORKERS=0` using the same seed-3407 recipe. This isolates the LMDB worker failure from the model result.

### 2. Re-establish the dense baseline

Run the strong dense recipe under the current validation-kappa protocol. Keep the output as the formal seed-3407 dense reference.

### 3. Calibrate the leading structure

If patch-only remains strongest by validation kappa/BA, run patch-only with:

```text
alpha in {0.003, 0.010, 0.030}
gamma=1
token_mlp=false
depth_mode=none
```

Use seed `3407` first as a screening pass. If the improvement remains promising, run dense and the selected adapter across seeds `42`, `1024`, and `3407`, then choose alpha by mean paired validation performance.

The seed-3407 screening jobs for alpha `0.003` and `0.030` were submitted. The alpha `0.010` result is already available from the completed patch-only run. A channel-plus-patch alpha `0.010` retry was also submitted with `NUM_WORKERS=0` to remove the LMDB worker failure.

The alpha `0.003` run completed below alpha `0.010` on validation metrics. The alpha `0.030` run is being retried with `NUM_WORKERS=0` after the original LMDB worker failure. The channel-plus-patch retry completed below both dense and patch-only.

### 4. Confirm the paper claim

After selecting one structure and alpha, run a final multi-seed dense-versus-adapter confirmation. Evaluate test only from checkpoints selected by validation kappa. Report test BA as primary, with test weighted F1 and test kappa as secondary outcomes.

### 5. Escalate only if needed

The paired confirmation showed that the original patch recipe is not yet
reproducibly better: it wins seed `3407` but loses seeds `42` and `1024`.
Therefore the seed-3407 improvement is a useful pilot signal, not a confirmed
paper result.

The next bounded optimization phase is adapter-specific optimization. Keep the
architecture fixed and test one factor at a time:

1. Adapter LR scale `{0.1, 0.3, 1.0}` across global LR `{5e-4, 7e-4, 9e-4}`,
   warmup `10`, on seeds `42` and `1024`.
2. If needed, test identity-preserving alpha initialization `{0, 0.001, 0.01}`.
3. If needed, test adapter weight decay `{0, 0.01, 0.05}`, then dropout
   `{0, 0.1}` without combining both grids initially.

Use validation-only jobs for development and do not inspect test results while
choosing settings. If this bounded protocol does not produce a positive mean
paired validation-kappa effect, report FACED as a boundary case and move on.

Historical fallback options, only after the optimizer and initialization checks,
are:

1. A bottlenecked patch mixer with rank `{32, 64}`.
2. Dense warm-start followed by near-identity adapter continuation.

Do not combine learning-rate changes, depth gating, token MLP, larger bottlenecks, routing, and warm starts in one experiment.

## Explicitly Deferred

- Token-wise MLP branch
- Depth-aware gating
- Routing mechanisms
- Larger adapter bottlenecks
- Bottleneck and warm-start fallbacks before the optimizer-control phase is complete

## Revised Decision Gate (2026-07-13)

The prior Stage 3A interpretation is superseded by a decisive batch-size
control. The exact alpha-scale `0.1` duplicate at `batch_size=16` and
`num_workers=0` reproduced the historical seed-1024 trajectory and ended with
an almost inactive adapter (`alpha_patch` about `0.0021`, scaled residual ratio
about `0.00016`). The earlier Stage 3A active residual was obtained with
`batch_size=32` and `num_workers=4`, so it cannot yet be attributed to separate
alpha learning rates.

The alpha-scale `0.3` seed-42 retry completed successfully but was weaker than
alpha-scale `0.1` (best validation kappa `0.40334`, BA `0.47299`, weighted F1
`0.47343`). It is removed from the candidate set.

### Overnight batch control

Run only this eight-run matrix, validation-only:

| Model | Batch size | Seeds |
| --- | ---: | --- |
| Dense | 16, 32 | 42, 1024 |
| Patch-only | 16, 32 | 42, 1024 |

Keep fixed:

```text
global_lr=7e-4
warmup_epochs=10
epochs=80
num_workers=0
weight_decay=0.05
patch core lr scale=0.1
alpha lr scale=0.1
alpha init=0.01
token_mlp=false
depth_mode=none
```

For every run record the best validation-kappa epoch, optimizer step, BA,
weighted F1, alpha, raw patch ratio, scaled residual ratio, and gradient
diagnostics. Do not inspect or evaluate test performance during this gate.

Interpretation:

1. Patch beats dense at batch 32 on both seeds: treat batch 32 as a candidate
   LaBraM recipe, run seed `3407`, then compare three-seed means against the
   strongest dense recipe.
2. Dense and patch improve together: attribute the earlier gain primarily to
   optimization rather than the adapter and move to cross-dataset validation.
3. Patch remains inconsistent: stop detailed FACED optimization and report
   FACED as a boundary case rather than adding depth or more mechanisms.

Only if batch 32 is adapter-favorable may we optionally compare
`batch=32, update_freq=1` with `batch=16, update_freq=2` to separate literal
microbatch effects from effective-batch and update-schedule effects. Do not run
that follow-up otherwise.

Submitted jobs:

```text
12509548 dense batch16 seed42
12509550 patch batch16 seed42
12509549 dense batch32 seed42
12509552 patch batch32 seed42
12509547 dense batch16 seed1024
12509551 patch batch16 seed1024
12509545 dense batch32 seed1024
12509546 patch batch32 seed1024
```

The patch jobs completed, but all four dense jobs failed before training because
the launcher used `LABRAM_ADAPTER_TYPE=dense`; the valid native dense value is
`none`. Rerun the dense half with the corrected value before making any
batch-size or adapter-effect claim.

### Broader paper sequence

After this one-time control, freeze the simple LaBraM-native patch structure and
move to SEED-V and TUEV, followed by ISRUC and PhysioNet-MI. Use the same dense
and adapted preprocessing, splits, checkpoint-selection rule, and paired seeds.
Use a small shared dense/adapter search (`global lr={5e-4,7e-4}`, batch size
`{16,32}`) and a small adapter-scale search only when justified. Select by
validation kappa, require BA and weighted F1 to remain compatible, and evaluate
test only after the recipe is frozen.

Do not add depth, routing, token MLP, or a larger bottleneck because of FACED
alone. Once the simple adapter has reproducible support on multiple datasets,
test `patch + lastk_delta` with `k={2,4}` as the paper-specific depth extension.

The paper-aligned depth implementation is prepared but isolated in branch
`adaptor-depth` at commit `9807be4`. It adds `lastk_attnres`, a LaBraM-native
upper-depth soft aggregation over the `[C,S,D]` grid with a zero-initialized
depth mix. It is ready for the later component ladder:

```text
Dense
Patch-only
Patch + lastk_attnres, k=2
Patch + lastk_attnres, k=4
```

The depth conditions are currently exploratory rather than a decision gate. Do
not merge them into `adaptor`, and do not use their results to select the final
LaBraM recipe, until the current batch-size control establishes a stable simple
patch recipe.

An initial validation-only screen was submitted from the isolated
`adaptor-depth` worktree at launcher commit `de59a8e`:

```text
12509671 patch + lastk_attnres(k=2), seed 42
12509673 patch + lastk_attnres(k=4), seed 42
12509672 patch + lastk_attnres(k=2), seed 1024
12509670 patch + lastk_attnres(k=4), seed 1024
12509800 patch + lastk_attnres(k=2), seed 3407
12509801 patch + lastk_attnres(k=4), seed 3407
```

This screen is useful for detecting gross instability and for checking whether
the depth weights and residual diagnostics behave as designed overnight. It is
not evidence of an adapter improvement by itself; the simple patch-versus-dense
batch control remains the primary decision sequence.

### Depth screen result

All six depth jobs completed validation-only. The kappa-selected results were:

```text
k=2: seed42   kappa=0.39896 BA=0.46944 wF1=0.46587 epoch=75
k=2: seed1024 kappa=0.40187 BA=0.47145 wF1=0.47094 epoch=72
k=2: seed3407 kappa=0.39240 BA=0.46111 wF1=0.46251 epoch=52
k=4: seed42   kappa=0.40804 BA=0.47762 wF1=0.47449 epoch=70
k=4: seed1024 kappa=0.38692 BA=0.45833 wF1=0.45758 epoch=78
k=4: seed3407 kappa=0.40076 BA=0.47130 wF1=0.46937 epoch=73
```

The matched simple patch batch-32 controls were kappa `0.40838` / BA `0.47793`
/ weighted F1 `0.47712` for seed 42 and `0.40688` / `0.47670` / `0.47638`
for seed 1024. Depth therefore has no consistent gain yet. Its depth weights
remain close to uniform, although the depth mix and residual are nonzero. Do
not select k or claim a paper-specific depth benefit from this screen. First
correct and complete the dense control; then either freeze simple patch and
move to cross-dataset validation, or run only a prespecified depth refinement
if the corrected baseline comparison justifies it.

### Depth-v2 bounded follow-up

The v1 screen is negative but inconclusive. Its query and mixing gate both
started at zero, so the query gradient was initially blocked; it also averaged
over a pool containing the final representation and divided scores by
`sqrt(embed_dim)`. These choices can explain the near-uniform weights without
refuting the depth hypothesis.

The isolated `depth` branch now contains commit `cb339aa`, which adds:

```text
lastk_uniform       fixed average of preceding upper-layer states
lastk_attnres_v2    learned scalar attention over preceding upper-layer states
```

Both use `H_L + beta * (H_depth - H_L)` with `beta=0.05` at initialization;
v2 uses direct normalized scalar scores and excludes `H_L` from the candidate
pool. The local test suite passes, including first-backward gradients to the
depth scorer and beta.

Run only this validation-only ladder on seeds `42` and `1024`:

```text
Patch-only
Patch + lastk_uniform, k=2
Patch + lastk_attnres_v2, k=2
Patch + lastk_attnres_v2, k=4
```

Advance learned depth only if it improves mean validation kappa over patch-only,
keeps BA and weighted F1 compatible, and differs meaningfully from the uniform
control. Then test the selected condition on seed `3407`. If v2 still fails,
stop FACED depth tuning and move to the next dataset; interpret the result as
dataset/backbone dependence rather than invalidating the paper's framework.

### Canceled depth-only LR sweep record

The matrix below is a historical record of the canceled 18-job proposal only.
It produced no valid comparisons and must not be used. Dense and patch-only
controls come from existing logs; the active replacement is recorded below.

```text
5e-4: uniform 12516312/12516306, v2-k2 12516307/12516308, v2-k4 12516305/12516311
7e-4: uniform 12516309/12516310, v2-k2 12516313/12516315, v2-k4 12516322/12516319
9e-4: uniform 12516317/12516316, v2-k2 12516321/12516320, v2-k4 12516314/12516318
```

These IDs are retained for auditability only. Do not select a recipe from this
matrix or use its partial output.

Correction: the 18-job sweep described above was canceled before producing
valid results, so its matrix must not be used. Job `12516305` had started but
was canceled before a valid result was written. The replacement is a four-job,
validation-only seed-3407 behavior screen on depth commit `956be37`:

```text
12516380  patch + lastk_uniform, k=2
12516377  patch + lastk_uniform, k=4
12516379  patch + lastk_attnres_v2, k=2
12516378  patch + lastk_attnres_v2, k=4
```

All use batch size `32`, workers `0`, warmup `10`, epochs `80`, global LR
`7e-4`, patch alpha `0.01`, adapter/core LR scale `0.1`, alpha LR scale `0.1`,
and skip final test evaluation. Reuse the existing seed-3407 patch-only log as
the control. Compare validation kappa first, with BA and weighted F1 required
to remain compatible, and inspect depth gradients, residual ratio, normalized
entropy, and layer weights. If no condition improves the patch control without
a metric or stability failure, stop FACED depth tuning.

## FACED decision and SEED-V transition (2026-07-13)

FACED is now treated as the LaBraM interface-development dataset. The main
positive result is the simple patch-only residual adapter: on seed `3407`, it
reached validation kappa `0.42660`, BA `0.49352`, and weighted F1 `0.49177`,
versus dense kappa `0.36881`, BA `0.44244`, and weighted F1 `0.43840` under the
matched `7e-4` recipe. This is a pilot result for structured residual
adaptation, not yet a multi-seed final claim.

The explicit depth-delta gate was functional but did not improve patch-only
reproducibly. Across seeds `42` and `1024`, learned depth had mean validation
kappa `0.40543` versus patch-only `0.40763`; it improved seed `42` but lost
seed `1024`. Uniform depth also lost on both seeds. The depth extension is
therefore a negative FACED component result and must not be allowed to redefine
the frozen adapter.

The paper-level interpretation is precise: preserve the native pretrained
backbone, add lightweight residual capacity aligned with its EEG channel-patch
geometry, and test depth conditioning as a later optional extension. FACED
supports the first invariant and does not support an added depth benefit here.

## SEED-V frozen evaluation

Do not submit adapter jobs until the native dense protocol is audited and, if
necessary, rerun under the same wrapper and recipe. An existing seed-3407 dense
anchor is:

```text
test BA=0.42304, test kappa=0.28181, test weighted F1=0.43022
validation kappa=0.23163, validation BA=0.38539, validation weighted F1=0.38653
best epoch=22, global lr=1e-4, batch size=64, epochs=40
```

The run used `attnres_variant=none`, `moe=false`, the LaBraM foundation
checkpoint, kappa-first selection, and the LMDB default trial-based 5:5:5 split.
It is a credible native dense anchor for the current EEGxPlore pipeline, but
it uses the wrapper's `all_patch_reps` classifier rather than the official
LaBraM script's exact downstream head. That distinction must be documented and
held fixed in all comparisons.

Before the ladder, record and verify:

- foundation checkpoint SHA and successful loaded-tensor count;
- SEED-V channel order and the fact that the 62-channel input uses stored tensor
  slot order because no channel manifest is currently supplied;
- `/100` sample scaling and `(62,1,200)` input schema;
- LMDB `__keys__` split, sessions `1,2,3`, trials `0-4` train, `5-9` validation,
  and `10-14` test;
- native LaBraM normalization/pooling, classifier, optimizer, and scheduler;
- validation-only kappa checkpoint selection and one final test evaluation.

Freeze the FACED-developed architecture with no new tuning:

```text
patch attention: unchanged
placement: final LaBraM [B,C,S,D] token grid before pooling/normalization
token MLP: off
depth: none for the primary adapter; k=2 learned delta gate as secondary
expert banks/routing: off
```

Use this comparison ladder with paired seeds:

```text
Native dense LaBraM
Patch-only LaBraM
Patch + learned depth-delta gate, k=2
```

Select checkpoints by validation kappa. Report test BA as the primary outcome,
test weighted F1 and test kappa as secondary outcomes, and validation BA as a
diagnostic. The main generalization claim is patch-only minus dense; the depth
claim is patch-plus-depth minus patch-only. If patch-only reproduces a positive
paired effect on SEED-V, retain the frozen adapter and proceed to the next
dataset. If it does not, report the FACED success as dataset-specific and
investigate preprocessing or optimization parity before adding capacity.

### SEED-V queue

The frozen native LaBraM ladder was submitted as nine validation-only jobs.
The standalone LaBraM implementation is used for all three conditions; the
EEGxPlore CBraMod-compatible adapter is intentionally excluded.

```text
recipe: global batch 64 (32 per GPU x 2), lr 1e-4, weight decay 0.03
epochs 40, warmup 5, layer decay 0.65, drop path 0.1, workers 0
input_scale_divisor=100, kappa selection, skip final test
```

```text
dense:       12524845 (42), 12524846 (1024), 12524847 (3407)
patch-only:  12524848 (42), 12524849 (1024), 12524850 (3407)
depth k=2:   12524851 (42), 12524852 (1024), 12524853 (3407)
```

Dense and patch jobs use adaptor commit `5ccde3c`; depth jobs use depth commit
`226b8b3`. Do not inspect test results for selection. Compare mean paired
validation kappa first, with BA and weighted F1 as required checks; evaluate
test only after the primary recipe is frozen.

### SEED-V provenance correction

Direct inspection found that the current LMDB is a legacy artifact: its
records contain only `sample` and `label`, its samples are `(62, 1, 200)`, and
its trial-based splits contain all 16 subjects. The current EEGxPlore
preprocessing script preserves CNT channel order after dropping `M1`, `M2`,
`VEO`, and `HEO`, but the original CNT files and generated sidecars are not
present beside this LMDB.

Therefore, previous standalone SEED-V jobs used an unverified channel-order
manifest. `Channel Order.xlsx` and `channel_62_pos.locs` have now been found;
they independently match the checked-in 62-channel sequence exactly. The
manifest records this as `metadata_verified_row_linkage_pending` and includes
both source hashes. Final SEED-V claims may use this order once it is confirmed
that these artifacts were the inputs used to create the legacy LMDB; otherwise
the old LMDB still cannot prove its tensor-row linkage by itself.

### SEED-V contract hardening

The loader and the isolated `depth` worktree now fail closed on the known
legacy contract `(62, 1, 200)`, non-finite samples, non-scalar labels, and
labels outside `[0, 4]`. The loader still supports another explicit channel
count when constructed for a different, provenance-rich artifact; the model
itself is not hard-coded to 62 channels. Both worktrees require an explicit
validated manifest for SEED-V and record the normalized channel list, mapped
`input_chans`, manifest file hash, pretrained checkpoint hash, and split-key
metadata in `run_config.json`.

`scripts/verify_seedv_channel_metadata.py` verifies the spreadsheet, `.locs`,
and manifest order without requiring `openpyxl`. `scripts/audit_seedv_lmdb.py`
provides a one-time full-record audit for shape,
finite values, labels, class counts, key-list hashes, and raw amplitude
percentiles. It should be run before any paper-grade result is accepted. A
62-channel provisional manifest can establish an exploratory mapped baseline,
but it cannot prove that LMDB row 0 is `FP1`, row 1 is `FPZ`, and so on.

The current two-GPU launchers do not enable `--dist_eval`; validation and test
therefore use the sequential sampler on each rank. Do not enable distributed
evaluation for final selection until prediction gathering or a non-padding
evaluation sampler is implemented and tested.

### SEED-V LMDB audit result

The complete audit is saved at `docs/seedv_lmdb_audit.json`. All `117744`
records passed the structural checks:

- every sample is `(62, 1, 200)` and finite;
- every label is scalar and lies in `[0, 4]`;
- train/validation/test key lists have zero pairwise overlap;
- stored dtype is consistently `float64`.

The audit also found a sparse amplitude tail. The median per-sample maximum
is approximately `90` raw units in all splits, while the raw maximum reaches
`2555` in train, `114438` in validation, and `43418` in test. After the current
`/100` scaling, only `26` validation and `2` test samples exceed `100` in
absolute value. This is not a structural loader failure, but it is a
preprocessing risk that should be retained as a diagnostic. Do not clamp,
drop, or retune around these windows before comparing the frozen dense and
patch recipes under identical input scaling.

The audit was executed from a node-local copy because a direct shared-
filesystem scan stalled. The resulting statistics describe the original LMDB
path recorded in the JSON; the local copy was used only to make the read
efficient.

The enhanced audit also found zero exact sample-content overlap across split
pairs. Every key parsed successfully into subject, session, trial, and window
metadata. Each split contains all 16 participants, 48 participant-session
combinations, and 240 participant-session-trial combinations. The protocol is
therefore a CBraMod-compatible within-subject trial split: train uses trials
`0-4`, validation uses `5-9`, and test uses `10-14`. It must not be described as
subject-independent generalization.

All 28 windows exceeding absolute amplitude `100` after `/100` scaling are
saved with key, label, source metadata, maximum channel/sample location, and
amplitude statistics. Most are concentrated in participant `14`, session `3`,
especially trials `6` and `9`. They remain in the primary analysis.

### Efficient SEED-V screen

The development seed packet is fixed to `{42, 1024, 3407}`. Do not add seeds
`0`, `7`, or `2026` for this study. Run six validation-only jobs first:

```text
dense:       seeds 42, 1024, 3407
patch-only:  seeds 42, 1024, 3407
```

Use one clean commit, the explicit metadata-verified manifest, and the frozen
recipe already listed above. Advance patch-only only when mean paired
validation kappa is positive, at least two seeds improve, and BA/weighted F1
remain compatible. If patch-only advances, compare depth only with a fresh
patch control on the exact depth implementation commit; do not compare depth
against a patch run from another commit. Depth is a secondary two-seed screen
first and does not receive new tuning unless it improves over that same-commit
patch control.

The raw CNT reconstruction bridge remains unavailable locally because
`EEG_raw`/CNT files were not found. It is a pre-final-test provenance task, not
a reason to expand the validation workload now. No broad LR sweep is justified
before this mapped dense-versus-patch comparison.

### SEED-V geometry decision (2026-07-15)

The singleton-patch mechanism check is complete. One real batch produces
`(62, 1, 200)`, `input_time_window=1`, and adapter tokens `[B,62,1,200]`.
Patch attention therefore has sequence length one: Q/K gradients are
approximately zero while V/output gradients are nonzero. The current patch
branch cannot perform temporal patch-to-patch interaction on SEED-V.

The mixed LR screen is exploratory only. Do not add depth, more LR values, or
test evaluation until the validation protocol is reconsidered.

Use the fixed seed packet `{42, 1024, 3407}` for every SEED-V development,
capacity, and confirmation run. Do not introduce an alternative packet such as
`{0, 7, 2026}`. If a selected LR gives a positive mean paired validation effect,
freeze the current core-fast recipe and confirm it across the same fixed packet
before final test evaluation. If no LR is positive, retain the FACED patch
result as the direct transfer result and run one bounded seed-3407 capacity
screen:

```text
current core-fast patch reference
patch output dropout=0.1
singleton-patch bottleneck residual: 200 -> 64 -> 200
```

The bottleneck condition replaces the full-width singleton patch-attention
branch; it does not enable the existing optional token MLP. Carry at most one
candidate to seeds `42` and `1024`, selected by validation kappa with BA and
weighted F1 compatibility. Require a smaller patch-versus-dense generalization
gap, bounded residuals, and positive residual-on versus residual-off validation
effects. Do not tune depth in this stage. A channel-only adapter remains a
separate exploratory geometry control because SEED-V has 62 channels but one
temporal patch.

### Singleton capacity implementation gate (2026-07-15)

The two capacity controls are explicit model variants rather than implicit
behavior based on patch count:

```text
adapter_variant=output_dropout
patch_output_dropout=0.1
```

keeps the full-width patch attention and applies dropout to its output after
attention, while:

```text
adapter_variant=bottleneck
adapter_bottleneck=64
```

replaces the patch attention branch with `200 -> 64 -> 200` after LayerNorm.
The optional token MLP remains disabled. Both branches preserve adapter RNG
isolation, gamma-zero parity, separate core/alpha optimizer groups, and
validation-time dropout disabling. Run metadata records the exact variant and
parameter count.

Submit only two exploratory jobs on seed `3407`, using the current core-fast
recipe and validation-only selection. A candidate must improve validation
kappa by approximately `0.003` over the full-width reference, with compatible
BA and weighted F1, a non-worse train-validation gap, controlled residuals,
and positive residual-on versus residual-off validation effect. If a candidate
survives, confirm it using the fixed packet `{42, 1024, 3407}`. Do not use
`{0, 7, 2026}` or add depth, channel mixing, another bottleneck width, or more
LR values during this stage.

### Geometry-matched SEED-V channel control (2026-07-15)

The singleton-patch result is a valid frozen-transfer result, but it is not an
equivalent structural experiment to FACED: FACED supplies ten temporal patches,
whereas SEED-V supplies one. A channel-only control is therefore justified as
one bounded geometry-matched rework:

```text
FACED:  adapter_type=patch,   sequence axis=S=10
SEED-V: adapter_type=channel, sequence axis=C=62
```

The real SEED-V check confirms channel sequence length `62` and nonzero Q/K/V
and output-projection gradients, with no patch branch. It preserves the
pre-norm residual placement, dense path, alpha initialization, optimizer-group
separation, and validation-kappa selection protocol.

Run only four seed-3407 validation-only conditions on one clean commit:

```text
dense        LR=1e-4
channel-only LR=1e-4
dense        LR=5e-4
channel-only LR=5e-4
```

Use batch size `32`, 40 epochs, warmup `5`, weight decay `0.03`, layer decay
`0.65`, drop path `0.1`, alpha init `0.01`, core LR scale `1.0`, alpha LR scale
`0.1`, token MLP off, patch off, depth off, and test off. Advance only with
approximately `+0.005` validation kappa over matched dense, compatible BA and
weighted F1, bounded residual activity, positive channel-on versus channel-off
validation effect, and no larger train-validation gap. If it passes, confirm
the frozen recipe on seeds `42` and `1024`, preserving the fixed packet
`{42, 1024, 3407}`. Do not combine channel and patch branches or add depth in
this screen.
