# LaBraM Adapter Progress Memory

Last updated: 2026-07-10

## Repository State

- Repository: `/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM`
- Branch: `adaptor`
- Current implementation commit: `eb9cd1b`
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

Submit through the run-organized wrapper:

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
