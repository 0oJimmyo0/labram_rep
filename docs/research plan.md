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
