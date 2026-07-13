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
- Legacy direct hidden-state depth fusion (retained as a negative ablation)
- Routing mechanisms
- Larger adapter bottlenecks
- Bottleneck and warm-start fallbacks before the optimizer-control phase is complete

## Depth-Conditioned Residual Gate

The direct-fusion depth screen was negative on seed `3407`: uniform k=2 was
closest to patch-only, while learned v2 and k=4 were worse. This rejects direct
replacement of the final LaBraM token grid as the first depth interface, but it
does not reject compact depth information as a routing signal.

The isolated `depth` branch now implements:

```text
lastk_delta_gate_uniform  uniform sample-level gate from upper-block deltas
lastk_delta_gate          learned softmax delta summary and sample-level gate
```

The patch adapter still consumes the final `H_L` grid. The depth summary only
modulates the patch residual with `1 + 0.1 * tanh(depth_gate(summary))`; the
gate projection is zero-initialized, so the initial function is exactly
patch-only. The gate has a separate optimizer group with zero weight decay.

Before submission, require the depth tests, adapter controls, and exact parity
tests to pass. The bounded experiment is patch-only versus the two new modes at
`k=2`, on seeds `42` and `1024`, with batch size `32`, global LR `7e-4`, core,
alpha, and gate LR scales `0.1`, warmup `10`, workers `0`, epochs `80`, and no
test evaluation. Select by mean validation kappa, require compatible BA and
weighted F1, and only then evaluate seed `3407`. Do not run k=4 or another LR
sweep unless this sample-level gate is clearly competitive.
