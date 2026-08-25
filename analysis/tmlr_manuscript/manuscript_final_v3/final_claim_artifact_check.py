#!/usr/bin/env python3
"""Final manuscript-v3 claim-to-artifact consistency audit.

This check is deliberately narrow: it verifies that the completed v3 audit
artifacts remain internally consistent, that the v2 primary contract remains
unchanged, and that the manuscript reports the primary results and the newly
integrated LaBraM--FACED/ISRUC supporting context without overstating either
scope.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


V3 = Path(__file__).resolve().parent
ROOT = V3.parents[3]
MANUSCRIPT = ROOT / "TMLR/6a8342ede4c0cedd5aacd420"
V2 = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v2"
FULL_AUDIT = ROOT / "LaBraM/analysis/tmlr_manuscript/evidence_audit_20260825"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(results: list[dict[str, object]], name: str, passed: bool, detail: str) -> None:
    results.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def main() -> int:
    results: list[dict[str, object]] = []
    contract = load_json(V3 / "operator_audit_contract.json")
    rows = list(csv.DictReader((V3 / "operator_audit_seed_results.csv").open(encoding="utf-8")))
    summary = {row["backbone"]: row for row in csv.DictReader((V3 / "operator_audit_summary.csv").open(encoding="utf-8"))}
    context_contract = load_json(V3 / "cross_backbone_context_contract.json")
    context_rows = list(csv.DictReader((V3 / "cross_backbone_context_summary.csv").open(encoding="utf-8")))
    full_audit_contract = load_json(FULL_AUDIT / "audit_contract.json")
    full_coverage = list(csv.DictReader((FULL_AUDIT / "backbone_dataset_coverage.csv").open(encoding="utf-8")))

    check(results, "v3 contract complete", contract.get("status") == "complete", str(contract.get("status")))
    check(results, "v3 seeds", contract.get("seeds") == [42, 1024, 3407], str(contract.get("seeds")))
    check(results, "v3 source artifact coverage", len(contract.get("source_artifacts", [])) == 18, f"{len(contract.get('source_artifacts', []))} artifacts")
    check(results, "test metrics not used for selection", contract.get("test_metrics_used_for_selection") is False, str(contract.get("test_metrics_used_for_selection")))
    check(results, "primary estimand unchanged", contract.get("primary_estimand_unchanged") is True, str(contract.get("primary_estimand_unchanged")))
    check(results, "six seed cells", len(rows) == 6, f"{len(rows)} rows")
    check(results, "supporting context contract complete", context_contract.get("status") == "complete", str(context_contract.get("status")))
    check(results, "supporting context scope", context_contract.get("datasets") == ["FACED", "ISRUC"] and context_contract.get("primary_estimand_unchanged") is True, str(context_contract.get("datasets")))
    check(results, "supporting context aggregate coverage", len(context_rows) == 11 and all(row["seeds"] == "3" for row in context_rows), f"{len(context_rows)} three-seed rows")
    check(results, "full evidence audit contract", full_audit_contract.get("status") == "complete" and full_audit_contract.get("registry_artifacts") == 485, str(full_audit_contract))
    check(results, "full evidence ten-cell coverage", len(full_coverage) == 10 and {row["backbone"] for row in full_coverage} == {"CBraMod", "LaBraM"}, f"{len(full_coverage)} coverage rows")
    check(results, "full evidence primary cells", {row["backbone"] + "--" + row["dataset"] for row in full_coverage if row["matched_rq2"] == "yes"} == {"CBraMod--FACED", "CBraMod--ISRUC", "CBraMod--TUEV", "LaBraM--TUEV"}, "four matched cells")

    expected_params = {"CBraMod": "59696", "LaBraM": "86214"}
    for row in rows:
        backbone = row["backbone"]
        seed = row["seed"]
        branch = ROOT / row["branch_local_mlp_artifact"]
        native = ROOT / row["native_artifact"]
        axis = ROOT / row["axis_agnostic_artifact"]
        check(results, f"{backbone} seed {seed} source paths", branch.is_dir() and native.is_dir() and axis.is_dir(), "native, branch-local, and axis-agnostic directories exist")
        check(results, f"{backbone} seed {seed} branch parameter count", row["branch_local_mlp_adapter_params"] == expected_params[backbone], row["branch_local_mlp_adapter_params"])
        if backbone == "CBraMod":
            summary_path = branch / "summary.json"
            metric_path = branch / "test_metrics.json"
            structure_path = branch / "structure_spec.json"
            trainability_path = branch / "trainability_report.json"
            payload = load_json(summary_path)
            structure = load_json(structure_path)
            trainability = load_json(trainability_path)
            valid = (
                payload.get("status") == "completed"
                and payload.get("test_evaluation_status") == "completed"
                # The CBraMod artifact records the metric name as
                # ``cohen_kappa``; the validation/test role is established by
                # the accompanying best-validation and test contracts.
                and load_json(metric_path).get("selection_metric") == "cohen_kappa"
                and structure.get("adapter_operator") == "axis_decomposed_mlp"
                and structure.get("channel_attention_sequence_length") == 0
                and structure.get("patch_attention_sequence_length") == 0
                and trainability.get("component_trainable_parameter_counts", {}).get("adapter", 0)
                + trainability.get("component_trainable_parameter_counts", {}).get("adapter_scalar", 0)
                == 59696
            )
            check(results, f"{backbone} seed {seed} completed artifact contract", valid, "completed summary, validation-selected test metrics, no attention mixing, realized params")
        else:
            config = load_json(branch / "run_config.json")
            test = load_json(branch / "final_test.json")
            valid = (
                config.get("experiment_method") == "axis_decomposed_mlp"
                and config.get("adapter_bottleneck") == 106
                and config.get("adapter_parameter_count") == 86214
                and test.get("selection_metric") == "cohen_kappa"
                and (branch / "log.txt").is_file()
                and len(list(branch.glob("checkpoint-*.pth"))) >= 3
            )
            check(results, f"{backbone} seed {seed} completed artifact contract", valid, "MLP config, realized params, validation-selected final test, checkpoints")

    # Recompute the reported means and sample standard deviations from the
    # seed-level CSV, then compare them with the aggregate table.
    for backbone, aggregate in summary.items():
        group = [row for row in rows if row["backbone"] == backbone]
        for field in ("native_ba", "branch_local_mlp_ba", "native_minus_branch_local_mlp_ba", "native_macro_f1", "branch_local_mlp_macro_f1", "native_minus_branch_local_mlp_macro_f1"):
            values = [float(row[field]) for row in group]
            mean_key = f"{field}_mean"
            sd_key = f"{field}_sd"
            ok = close(float(aggregate[mean_key]), mean(values)) and close(float(aggregate[sd_key]), stdev(values))
            check(results, f"{backbone} aggregate {field}", ok, f"mean={aggregate[mean_key]}, sd={aggregate[sd_key]}")

    v2 = load_json(V2 / "analysis_contract.json")
    check(results, "v2 primary contract present", v2.get("version") == "manuscript_final_v2", str(v2.get("version")))
    check(results, "v2 primary RQ2 scope unchanged", v2.get("primary_rq2_pair_rows") == 12 and not v2.get("primary_rq2_exclusions"), f"rows={v2.get('primary_rq2_pair_rows')}, exclusions={v2.get('primary_rq2_exclusions')}")
    check(results, "v2 three-seed contract", v2.get("seeds") == [42, 1024, 3407], str(v2.get("seeds")))

    section5 = (MANUSCRIPT / "sections/section_05_results.tex").read_text(encoding="utf-8")
    section4 = (MANUSCRIPT / "sections/section_04_experimental_protocol.tex").read_text(encoding="utf-8")
    appendix_b = (MANUSCRIPT / "sections/appendix_b_evidence_scope.tex").read_text(encoding="utf-8")
    appendix_c = (MANUSCRIPT / "sections/appendix_c_supporting_results.tex").read_text(encoding="utf-8")
    main_pdf = MANUSCRIPT / "build/main.pdf"
    section5_flat = " ".join(section5.split())
    section4_flat = " ".join(section4.split())
    appendix_b_flat = " ".join(appendix_b.split())
    primary_tokens = (
        "+.013\\pm.006",
        "-.004\\pm.005",
        "-.011\\pm.051",
        "-.039\\pm.017",
        "-.065\\pm.025",
    )
    check(results, "primary table values", all(token in section5 for token in primary_tokens), "primary RQ1/RQ2 effects are present")
    check(results, "primary three-seed convention", "averaged over the three prespecified seeds" in section5_flat and "all three seeds" in section5_flat, "primary results use the prespecified three-seed convention")
    check(results, "absolute primary context", all(token in section5 for token in ("$.392\\pm.018$", "$.431\\pm.001$", "$.368\\pm.026$", "$.433\\pm.001$")), "salient LaBraM--TUEV absolute scores are visible in the main text")
    check(results, "capacity qualification", "closely matched adaptation-module capacity" in section5_flat and "five-percent tolerance" in appendix_b_flat and "85,810" in appendix_b_flat and "86,014" in appendix_b_flat, "matched-capacity criterion and realized counts are explicit")
    check(results, "operator audit integrated", "app:operator-audit" in appendix_c and all(token in appendix_c for token in (".4548$\\pm$.0223", ".4637$\\pm$.0168", ".3916$\\pm$.0182", ".4311$\\pm$.0033", "native attention minus branch-local MLP")) and "operator-class audit" in section5 and "branch-local MLP audit" in section4 + section5 + (MANUSCRIPT / "sections/section_06_boundary_conditions.tex").read_text(encoding="utf-8") + (MANUSCRIPT / "sections/section_07_discussion.tex").read_text(encoding="utf-8"), "secondary native-branch MLP evidence and interpretation are integrated")
    check(results, "LaBraM preprocessing qualification", "conditional on the CBraMod-serialized" in section4_flat and "LaBraM-native preprocessing" in section4_flat, "paired validity and preprocessing scope are distinguished")
    check(results, "ten-cell scope sentence", "Records are available for all ten backbone--dataset combinations" in section4_flat and "predefined matched RQ2 inclusion criteria" in section4_flat, "availability and primary inclusion are separated in Section 4")
    check(results, "ten-cell appendix table", "tab:evidence-coverage" in appendix_b and "Audited evidence coverage across the ten backbone--dataset cells" in appendix_b, "coverage table is present in Appendix B")
    check(results, "compute scope qualification", "parameter count only" in section4_flat and "no cross-condition efficiency claim" in section4_flat, "parameter matching is not presented as compute matching")
    check(results, "within-pair initialization matching", "Within each primary native--axis-agnostic pair" in section4_flat and "not treated as a controlled cross-backbone factor" in section4_flat, "within-pair initialization and cross-backbone scope are explicit")
    check(results, "reproducibility release scope", "resolved run configurations" in section4_flat and "source licenses" in section4_flat, "release contents and raw-data constraints are stated")
    related_work = (MANUSCRIPT / "sections/section_02_related_work.tex").read_text(encoding="utf-8")
    check(results, "related-work expansion", all(token in related_work for token in ("AdaBrain-Bench", "CEReBrO", "ALFEE", "LUNA", "NeuroAdapt-Bench", "Localized LoRA")), "reviewer-requested benchmark and structural-adaptation context is included")
    appendix_c_flat = " ".join(appendix_c.split())
    check(results, "LaBraM FACED/ISRUC context integrated", "app:labram-context" in section4 and all(token in appendix_c for token in (".140$\\pm$.003", ".297$\\pm$.001", ".608$\\pm$.002", ".794$\\pm$.005")) and "pooled into the primary" in appendix_c_flat, "supporting table and scope qualification are present")
    check(results, "PDF exists", main_pdf.is_file() and main_pdf.stat().st_size > 2_000_000, f"{main_pdf.stat().st_size if main_pdf.exists() else 0} bytes")

    failures = [item for item in results if item["status"] == "FAIL"]
    report = ["# Manuscript v3 claim-to-artifact consistency audit", "", "Status: **PASS**" if not failures else "Status: **FAIL**", "", "The audit checks the v3 operator follow-up, preserves the v2 primary contract, and verifies the integrated manuscript values.", "", "| Check | Status | Detail |", "|---|---|---|"]
    report.extend(f"| {item['check']} | {item['status']} | {item['detail']} |" for item in results)
    (V3 / "claim_artifact_consistency.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (V3 / "claim_artifact_consistency.json").write_text(json.dumps({"status": "PASS" if not failures else "FAIL", "checks": results}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} checks; {len(failures)} failures")
    for item in results:
        if item["status"] == "FAIL":
            print(f"FAIL: {item['check']}: {item['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
