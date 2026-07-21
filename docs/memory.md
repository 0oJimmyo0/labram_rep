# LaBraM Adapter Progress Memory

Last updated: 2026-07-20

## Repository State

- Repository: `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM`
- Branch: `adaptor`
- Current implementation commit: `7ca74e2` (`Add separate LaBraM alpha gate controls`)
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
