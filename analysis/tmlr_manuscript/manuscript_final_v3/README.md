# Manuscript-final v3: targeted operator audit

This package extends `manuscript_final_v2` with six completed TUEV follow-up
runs. The added condition is the branch-local MLP (registry name
`axis_decomposed_mlp`): it preserves the native channel/patch branch layout,
but applies a token-wise bottleneck MLP within each branch and introduces no
cross-channel or cross-patch mixing.

The follow-up uses the same bottleneck settings as the native conditions. The
realized CBraMod adapter count is 59,696; the realized LaBraM branch-local
MLP count is 86,214 versus 85,810 for the LaBraM native adapter. The LaBraM
follow-up is therefore branch-preserving and same-bottleneck, but not exactly
parameter-matched.

The audit is secondary evidence. It does not replace or change the primary
native-versus-axis-agnostic RQ2 estimand, which remains the four-cell,
validation-kappa-selected analysis in `manuscript_final_v2`.

## Reproduction

From this directory, run:

```bash
python aggregate_operator_audit.py
```

The script reads the selected-checkpoint artifacts directly and regenerates:

- `operator_audit_seed_results.csv`: seed-level native, branch-local MLP, and
  existing axis-agnostic values;
- `operator_audit_summary.csv`: mean, sample standard deviation, and positive
  seed counts for the two TUEV backbone cells;
- `operator_audit_contract.json`: source and selection metadata.

Test metrics are read only from validation-Cohen-kappa-selected checkpoints;
they are not used for selection.
