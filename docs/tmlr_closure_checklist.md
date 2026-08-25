# TMLR manuscript closure checklist

Status: audited 2026-08-21; manuscript evidence v2 generated after the
CBraMod/ISRUC condition-consistency repair.

This checklist supersedes older “remaining runs” notes when they conflict with
completed artifacts. It is deliberately separated from the historical
execution log.

## A. Current artifact status

| Cell | Status | Training action |
| --- | --- | --- |
| CBraMod FACED | Corrected three-seed packet complete | None |
| CBraMod SEED-V | Corrected three-seed packet complete | None; retain within-subject limitation |
| CBraMod ISRUC | 12-condition three-seed packet complete | None |
| CBraMod PhysioNet-MI | 12-condition three-seed packet complete | None |
| CBraMod TUEV | 12-condition x 3-seed packet complete; seed-3407 LoRA/full-channel replacements completed successfully | Verify replacement artifacts and aggregate; no new training |
| LaBraM FACED | Dense/native and corrected frozen controls complete; LoRA and upper-2 are valid three-seed `reviewer_r2` artifacts | None |
| LaBraM SEED-V | Dense/channel primary packet and generic/LoRA/upper controls available | None unless subject-disjoint generalization is claimed |
| LaBraM ISRUC | Dense, low-rank native, generic, LoRA, upper controls, and repaired sequence-fix axis-blind packet complete | Verify three final artifacts and aggregate; no new training |
| LaBraM TUEV | Locked 11-condition three-seed packet complete; axis-blind packet complete for 3 seeds | No duplicate submission |
| LaBraM PhysioNet-MI | Locked three-seed dense/frozen/native/control packet complete | No new run until axis-blind design is implemented |

## B. CBraMod TUEV completion gate

The two previously missing seed-3407 cells completed successfully:

- `13420213`: frozen LoRA QKV-r8, seed `3407`;
- `13420214`: full native channel, seed `3407`.

The 12-condition x 3-seed packet is therefore complete. Verify for each
replacement artifact:

- 20/20 epoch records and per-epoch test diagnostics;
- strict checkpoint load with zero missing/unexpected keys;
- correct frozen-mode report for LoRA and correct trainable-backbone report for
  full native channel;
- validation-kappa checkpoint selection;
- test metrics evaluated only after validation selection;
- adapter and backbone parameter/update diagnostics;
- matching TUEV contract: batch `8`, 20 epochs, backbone LR `1e-4`, head LR
  `2e-4`, native adapter LR `2e-4` for full channel, LoRA LR `5e-5`, `/100`
  scaling, seed-specific head/loader/adapter seeds.

Then aggregate the complete 12-condition x 3-seed packet, preferring the
replacement artifact where duplicate seed-3407 directories exist. The older dropout,
gamma, stability15, and head-dropout runs remain seed-42 diagnostics and are
not mixed into the locked matrix.

## C. What is genuinely missing

### C1. Parameter-matched axis-blind control

This is the only training addition needed for the strong causal claim that
native alignment contributes beyond generic capacity.

Current finding: CBraMod has axis-blind results, and LaBraM now has completed
three-seed axis-blind packets on both TUEV and ISRUC. The ISRUC packet required
a shared sequence-diagnostic repair, which passed. The LaBraM launcher has an explicit
`METHOD=axis_blind` path: it uses the generic token-wise residual branch,
freezes the backbone, and checks a target parameter count in preflight. The
remaining work is therefore artifact hardening and fair capacity matching,
not inventing a new channel/patch module.

Before treating the queued jobs as manuscript evidence, verify:

1. Keep the `METHOD=axis_blind` label in the run ID and record it alongside
   `adapter_type=generic` in the run contract.
2. Assert that the axis-blind instance has no channel or patch attention
   modules and uses only the token MLP residual.
3. Match the total trainable adapter-plus-head count to the selected native
   frozen comparator within `+/-5%`.
4. Record the target count, realized count, parameter names, and optimizer
   membership in the run artifact.
5. Add a CPU/preflight assertion that the native-axis eligibility mask is not
   used by the axis-blind path.
6. Run one seed-42 smoke for each dataset before launching the three-seed
   packet.

### C2. Recommended axis-blind production scope

Use the smallest scope that supports the intended claim:

| Dataset | Native comparator actually available | Axis-blind target | Seeds | Contract |
| --- | --- | --- | --- |
| ISRUC | frozen patch, full-width, 161,201 adapter parameters | generic bottleneck `400` (161,001 adapter parameters) | `42,1024,3407` | existing ISRUC final contract |
| TUEV | frozen channel+patch, low-rank, 85,810 adapter parameters | generic bottleneck `213` (86,014 adapter parameters) | `42,1024,3407` | existing LaBraM TUEV final contract |

This was a six-run closure packet after the smoke tests. TUEV is the clean
two-axis comparison; its three runs are complete. ISRUC is complete as a
narrower patch-alignment comparison because its matched frozen native
channel+patch comparator was not part of the locked packet.
ISRUC is a narrower patch-alignment comparison because no frozen native
channel+patch packet exists there; do not describe it as a channel+patch
causal test. The ISRUC target is closely matched: the current generic token
MLP with bottleneck 400 has `401*400+601 = 161,001` adapter parameters
versus the native 161,201. For TUEV, bottleneck 213 gives 86,014 versus the
native 85,810, a 0.24% difference.

PhysioNet-MI axis-blind is optional because it is a serialized-only dataset
and is already best treated as a bounded extension. SEED-V axis-blind is not a
priority because its patch axis is singleton and its channel-row/generalization
protocol remains qualified.

### C3. LaBraM FACED harmonization

No missing FACED LoRA or upper-2 training runs were found: valid three-seed
artifacts are under `faced_reviewer_r2_*`. The older native FACED patch packet
uses the earlier full-width adapter, while later LaBraM ISRUC/TUEV packets use
the low-rank variant.

This is a manuscript labeling issue, not an automatic rerun requirement. Either:

- label FACED native results as historical/full-width pilot evidence and keep
  the main common-primitive claim on ISRUC/TUEV; or
- rerun LaBraM FACED with the locked low-rank primitive before presenting FACED
  as a direct cross-backbone aligned-adapter comparison.

The first option is the efficient default and is consistent with the existing
FACED closure decision. Do not launch a FACED rerun unless the manuscript makes
FACED part of a direct common-primitive cross-backbone causal comparison.

## D. Settings to verify before any axis-blind submission

### Shared settings

- seeds exactly `{42,1024,3407}`;
- frozen backbone remains in `.eval()` mode;
- classifier and axis-blind adapter remain in `.train()` mode;
- validation Cohen's kappa selects the primary checkpoint;
- balanced accuracy and macro-F1 are primary reported test metrics;
- weighted F1 and kappa are secondary;
- no test metric is used for method, hyperparameter, or epoch selection;
- strict checkpoint loading and complete artifact reports are mandatory;
- no native channel/patch mixer is instantiated in the axis-blind path;
- no LoRA is combined with the axis-blind or native adapter.

### ISRUC settings

- batch `16`, 30 epochs, LR `2e-4`;
- weight decay `.05`, layer decay `.65`, warmup `1`;
- input scale divisor `1.0`;
- six channels and 30 temporal patches;
- frozen backbone, head plus axis-blind adapter only;
- use the same classifier, split, loader, seed derivation, and sequence-head
  dropout `.1` as the existing frozen patch comparator;
- use full generic/token-MLP bottleneck `400`, not bottleneck `64`;
- target adapter parameters `161,201`; realized axis-blind adapter parameters
  should be `161,001`, and the adapter-plus-head total should be `162,006`
  with the existing ISRUC head.

### TUEV settings

- batch `64`, 15 epochs, LR `5e-4`;
- weight decay `.05`, layer decay `.65`, warmup `5`;
- drop path `.2`, label smoothing `.1`, `/100` scaling;
- realized geometry `[C,S,D]=[16,5,200]`;
- frozen backbone, head plus axis-blind adapter only;
- native comparator is frozen channel+patch low-rank;
- use full generic/token-MLP bottleneck `213`, not bottleneck `64`;
- target adapter parameters `85,810`; realized axis-blind adapter parameters
  should be `86,014`, and the adapter-plus-head total should be within 5% of
  the native `87,016` total.

## E. No-run manuscript work

These are required before submission but do not require new training:

- build one cross-backbone result registry;
- recompute all means, standard deviations, and paired seed differences;
- report trainable parameters, residual ratios, adapter gradients, backbone
  update norms, runtime, and peak memory;
- evaluate saved epochs offline where checkpoints exist, clearly marked as
  post-hoc diagnostics;
- include TUEV per-class recall/F1 and confusion matrices;
- state SEED-V within-subject and ISRUC serialized-only limitations;
- freeze repository commits, checkpoint hashes, split hashes, and manifests.

## F. Submission decision

### If the axis-blind packet is not run

Use the empirical-study framing. Claim that native interaction-aligned
adaptation is active and conditionally useful, but do not claim that alignment
causes an advantage over capacity-matched generic adaptation.

### If the axis-blind packet is completed

Use the stronger method framing only if native adaptation beats axis-blind in a
consistent paired analysis on at least one dataset per backbone and does not
lose systematically to generic/LoRA/upper controls. Otherwise retain the
empirical-study framing even with the new control.

## Current submission decision

The six-run packet was submitted with two-wave GPU gating. TUEV and the repaired
ISRUC retry both completed:

- CBraMod jobs `13420213:13420214` completed successfully;
- original LaBraM jobs `13421416:13421421` completed, with only the three TUEV
  jobs valid;
- retry ISRUC jobs `13422949:13422951` completed with valid artifacts;
- no replacement copy is needed.

The launch settings were reviewed and smoke-tested before submission. Do not
submit another copy. After completion, verify the runtime assertions, run
contracts, final-test files, and per-seed results before deciding whether any
additional experiment is justified.

## Latest closure-run audit

Audited after jobs `13421416:13421421` completed:

- TUEV jobs `13421417`, `13421419`, and `13421421` completed successfully.
  Each has 15 epoch records, five saved checkpoints, strict-load success, and
  `final_test.json`. The logs contain repeated nonfatal `NaN or Inf found in
  input tensor` messages, but all recorded input/output finite flags are valid
  and nonfinite counts are zero.
- ISRUC jobs `13421416`, `13421418`, and `13421420` all failed identically
  before the first optimizer update with `ValueError: too many values to
  unpack (expected 4)` at `modeling_finetune.py:924`. The frozen-repeat check
  called `forward_features()` on the 5-D ISRUC sequence tensor
  `[B,20,6,30,200]`; sequence-aware handling exists in `forward()` instead.
  No ISRUC checkpoint or final-test result was produced.
- This was a shared sequence-path infrastructure failure, not evidence against
  the ISRUC axis-blind adapter. Do not interpret the three original ISRUC
  failures as experimental results; the diagnostic is now repaired and the
  seed-42 smoke passed before retry submission.
- The repair adds a sequence-aware frozen-repeat helper in
  `engine_for_finetuning.py`; it flattens ISRUC epochs for feature extraction
  and applies the frozen sequence encoder before comparing repeated passes.
  Dummy and actual-model smoke tests passed. Retry jobs are `13422949` (seed
  42), `13422950` (seed 1024), and `13422951` (seed 3407).

## Latest CBraMod/ISRUC repair audit

The two prespecified native channel-plus-patch repairs completed successfully:

- job `13512637`: seed `1024`, bottleneck `64`;
- job `13512638`: seed `3407`, bottleneck `64`.

Both artifacts passed strict checkpoint loading, ISRUC geometry checks,
frozen-backbone mode checks, and 20/20 epoch-record checks. The native
adaptation-module count is `59,610` for all three seeds. The selected
alpha-zero axis-blind controls contain `59,949` adaptation-module parameters,
giving a final pair-audit mismatch of `0.5655%`, below the 5% contract.

The regenerated package is
`analysis/tmlr_manuscript/manuscript_final_v2/`; it contains 156 selected
rows, 12 valid primary RQ2 seed-pairs, and complete primary RQ2 cells for
CBraMod/FACED, CBraMod/ISRUC, CBraMod/TUEV, and LaBraM/TUEV. The self-check
passes with zero errors and zero warnings. `manuscript_final_v1` remains the
superseded pre-repair snapshot and is not overwritten.
