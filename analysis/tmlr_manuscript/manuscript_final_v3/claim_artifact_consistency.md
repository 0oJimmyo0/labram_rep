# Manuscript v3 claim-to-artifact consistency audit

Status: **PASS**

The audit checks the v3 operator follow-up, preserves the v2 primary contract, and verifies the integrated manuscript values.

| Check | Status | Detail |
|---|---|---|
| v3 contract complete | PASS | complete |
| v3 seeds | PASS | [42, 1024, 3407] |
| v3 source artifact coverage | PASS | 18 artifacts |
| test metrics not used for selection | PASS | False |
| primary estimand unchanged | PASS | True |
| six seed cells | PASS | 6 rows |
| supporting context contract complete | PASS | complete |
| supporting context scope | PASS | ['FACED', 'ISRUC'] |
| supporting context aggregate coverage | PASS | 11 three-seed rows |
| full evidence audit contract | PASS | {'status': 'complete', 'audit_date': '2026-08-25', 'seeds': [42, 1024, 3407], 'artifact_directories_audited': 769, 'registry_artifacts': 485, 'registry_complete_artifacts': 485, 'supplemental_complete_artifacts': 8, 'log_files_audited': 2515, 'registry_condition_rows': 89, 'complete_three_seed_registry_condition_rows': 88, 'supplemental_condition_rows': 13, 'primary_rq2_cells': ['CBraMod-FACED', 'CBraMod-ISRUC', 'CBraMod-TUEV', 'LaBraM-TUEV'], 'test_metrics_used_for_selection': False, 'notes': ['A completed checkpoint/test contract is required for complete status.', 'Supporting legacy records are retained but not promoted to matched RQ2 evidence.', 'Macro-F1 values are complemented only from the existing manuscript-v2/v3 audited tables when the raw registry aggregate is missing a scalar; the source column records this explicitly.']} |
| full evidence ten-cell coverage | PASS | 10 coverage rows |
| full evidence primary cells | PASS | four matched cells |
| CBraMod seed 42 source paths | PASS | native, branch-local, and axis-agnostic directories exist |
| CBraMod seed 42 branch parameter count | PASS | 59696 |
| CBraMod seed 42 completed artifact contract | PASS | completed summary, validation-selected test metrics, no attention mixing, realized params |
| CBraMod seed 1024 source paths | PASS | native, branch-local, and axis-agnostic directories exist |
| CBraMod seed 1024 branch parameter count | PASS | 59696 |
| CBraMod seed 1024 completed artifact contract | PASS | completed summary, validation-selected test metrics, no attention mixing, realized params |
| CBraMod seed 3407 source paths | PASS | native, branch-local, and axis-agnostic directories exist |
| CBraMod seed 3407 branch parameter count | PASS | 59696 |
| CBraMod seed 3407 completed artifact contract | PASS | completed summary, validation-selected test metrics, no attention mixing, realized params |
| LaBraM seed 42 source paths | PASS | native, branch-local, and axis-agnostic directories exist |
| LaBraM seed 42 branch parameter count | PASS | 86214 |
| LaBraM seed 42 completed artifact contract | PASS | MLP config, realized params, validation-selected final test, checkpoints |
| LaBraM seed 1024 source paths | PASS | native, branch-local, and axis-agnostic directories exist |
| LaBraM seed 1024 branch parameter count | PASS | 86214 |
| LaBraM seed 1024 completed artifact contract | PASS | MLP config, realized params, validation-selected final test, checkpoints |
| LaBraM seed 3407 source paths | PASS | native, branch-local, and axis-agnostic directories exist |
| LaBraM seed 3407 branch parameter count | PASS | 86214 |
| LaBraM seed 3407 completed artifact contract | PASS | MLP config, realized params, validation-selected final test, checkpoints |
| CBraMod aggregate native_ba | PASS | mean=0.454793701, sd=0.022266852 |
| CBraMod aggregate branch_local_mlp_ba | PASS | mean=0.463663063, sd=0.016804595 |
| CBraMod aggregate native_minus_branch_local_mlp_ba | PASS | mean=-0.008869362, sd=0.031534753 |
| CBraMod aggregate native_macro_f1 | PASS | mean=0.471962541, sd=0.021564315 |
| CBraMod aggregate branch_local_mlp_macro_f1 | PASS | mean=0.478109862, sd=0.006317520 |
| CBraMod aggregate native_minus_branch_local_mlp_macro_f1 | PASS | mean=-0.006147321, sd=0.023487619 |
| LaBraM aggregate native_ba | PASS | mean=0.391609119, sd=0.018243723 |
| LaBraM aggregate branch_local_mlp_ba | PASS | mean=0.431076131, sd=0.003331353 |
| LaBraM aggregate native_minus_branch_local_mlp_ba | PASS | mean=-0.039467012, sd=0.020806910 |
| LaBraM aggregate native_macro_f1 | PASS | mean=0.367625191, sd=0.025962776 |
| LaBraM aggregate branch_local_mlp_macro_f1 | PASS | mean=0.433639915, sd=0.003036337 |
| LaBraM aggregate native_minus_branch_local_mlp_macro_f1 | PASS | mean=-0.066014724, sd=0.028791147 |
| v2 primary contract present | PASS | manuscript_final_v2 |
| v2 primary RQ2 scope unchanged | PASS | rows=12, exclusions=[] |
| v2 three-seed contract | PASS | [42, 1024, 3407] |
| primary table values | PASS | primary RQ1/RQ2 effects are present |
| primary three-seed convention | PASS | primary results use the prespecified three-seed convention |
| absolute primary context | PASS | salient LaBraM--TUEV absolute scores are visible in the main text |
| capacity qualification | PASS | matched-capacity criterion and realized counts are explicit |
| operator audit integrated | PASS | secondary native-branch MLP evidence and interpretation are integrated |
| LaBraM preprocessing qualification | PASS | paired validity and preprocessing scope are distinguished |
| ten-cell scope sentence | PASS | availability and primary inclusion are separated in Section 4 |
| ten-cell appendix table | PASS | coverage table is present in Appendix B |
| compute scope qualification | PASS | parameter matching is not presented as compute matching |
| within-pair initialization matching | PASS | within-pair initialization and cross-backbone scope are explicit |
| reproducibility release scope | PASS | release contents and raw-data constraints are stated |
| related-work expansion | PASS | reviewer-requested benchmark and structural-adaptation context is included |
| LaBraM FACED/ISRUC context integrated | PASS | supporting table and scope qualification are present |
| PDF exists | PASS | 2960030 bytes |
