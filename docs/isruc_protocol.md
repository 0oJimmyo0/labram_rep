# ISRUC LaBraM Protocol

This document freezes the ISRUC contract before LaBraM model runs.

## Dataset and split

- ISRUC-Sleep Subgroup I, subjects 1--100.
- Train: subjects 1--80.
- Validation: subjects 81--90.
- Test: subjects 91--100.
- The split is subject-wise; no sequence may cross a subject boundary.

## Signal contract

- Sampling rate: 200 Hz.
- Six bipolar EEG channels, in this order:
  `F3-A2`, `C3-A2`, `O1-A2`, `F4-A1`, `C4-A1`, `O2-A1`.
- LaBraM positional names: `F3`, `C3`, `O1`, `F4`, `C4`, `O2`.
- Each sleep epoch is 30 seconds, or 6000 samples per channel.
- Each epoch is represented as 30 temporal patches of 200 samples.
- Each stored item contains 20 consecutive epochs and 20 labels.
- Loader tensor shape: `[20, 6, 30, 200]`.

The bipolar suffix is retained in the protocol metadata. The LaBraM standard-1020
position mapping uses the EEG electrode prefix only; this is a positional bridge,
not a claim that the mastoid reference is represented by LaBraM's token geometry.

## Labels and preprocessing

- Use the first expert annotation (`*_1.txt`), matching the existing CBraMod pipeline.
- Raw labels `{0, 1, 2, 3, 5}` map to classes `{0, 1, 2, 3, 4}`.
- Existing serialized arrays are already filtered, notch-filtered, segmented, and
  grouped. Do not filter them again.
- The loader returns stored values without scaling; the training/evaluation engine
  applies the single configured `input_scale_divisor`.

The serialized data follows `EEGxPlore/preprocessing/ISRUC/prepare_ISRUC_1.py`:

- MNE filtering: 0.3--35 Hz FIR, followed by a 50 Hz notch filter.
- No resampling is performed by the script; the source recording is expected at
  200 Hz.
- `raw.to_data_frame().values[:, 1:]` removes the timestamp column, then
  `[:, 2:8]` selects the six stored signal columns.
- Incomplete 30-second epochs are dropped, then incomplete 20-epoch sequences
  are dropped independently for each subject.
- Labels come from the first expert file and use the mapping `{0: 0, 1: 1,
  2: 2, 3: 3, 5: 4}`.

This script is useful provenance when raw EDF files are unavailable, but it does
not independently prove the source channel names or annotation contents. The
auditor therefore supports a serialized-only report, while a full audit with
`--edf-root` remains the stronger gate for paper-facing results.

## Model comparison

All variants share the same sequence encoder, classifier, initialization, optimizer,
loss, split, and selection rule:

1. Dense LaBraM.
2. LaBraM + `channel_patch` native structured residual adapter.
3. LaBraM + `channel_patch` + frozen depth configuration.

`channel` and `patch` are component diagnostics only. Depth is a secondary
modulation of the structured residual and does not replace the final LaBraM grid.

## Selection and reporting

- Select checkpoints using validation Cohen's kappa.
- Report balanced accuracy, Cohen's kappa, and weighted F1.
- During development, use validation-only mode and do not inspect test metrics.
- Confirmatory runs use paired seeds `42`, `3407`, and `2024` unless a formally
  frozen project seed packet supersedes this choice.

## Required audit gates

Run `scripts/audit_isruc.py` before model training. It must verify subject splits,
numeric signal/label pairing, shapes, labels, finite values, channel metadata when
the raw EDF root is supplied, and discarded-epoch accounting.
