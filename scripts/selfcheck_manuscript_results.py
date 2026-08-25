#!/usr/bin/env python3
"""Fail-closed consistency audit for the manuscript evidence package.

This audit is independent of score selection.  It verifies that complete
three-seed groups also use a common training configuration, that paired
comparisons satisfy the recorded capacity rule, and that the final package is
consistent with its source manifest and artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript" / "manuscript_inclusion_manifest.csv"
REGISTRY = ROOT / "LaBraM" / "analysis" / "tmlr_registry"
FINAL_VERSION = os.environ.get("TMLR_MANUSCRIPT_VERSION", "v1")
FINAL = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript" / f"manuscript_final_{FINAL_VERSION}"
OUT_JSON = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript" / "selfcheck_latest.json"
OUT_MD = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript" / "selfcheck_latest.md"
SEEDS = {"42", "1024", "3407"}

# These fields define the experimental condition.  Seed-specific random seeds,
# output paths, and provenance timestamps are intentionally excluded.
CONFIG_FIELDS = (
    "dataset", "method", "experiment_method", "model",
    "adapter_type", "adapter_bottleneck", "adapter_gamma", "adapter_dropout",
    "adapter_heads", "adapter_lr", "adapter_weight_decay",
    "adapter_alpha_weight_decay", "adapter_alpha_lr_scale",
    "adapter_zero_init_output", "lr", "head_lr", "backbone_lr",
    "backbone_lr_scale", "lora_lr", "lora_rank", "lora_alpha", "upper_lr",
    "batch_size", "epochs", "weight_decay", "weight_decay_end", "smoothing",
    "drop", "drop_path", "sequence_head_dropout", "warmup_epochs",
    "selection_metric", "optimizer", "input_scale_divisor",
    "pretrained_checkpoint_sha256", "channel_manifest_sha256",
    "input_position_policy", "backbone_frozen", "frozen_backbone_eval_mode",
)

DEFAULTS = {
    "adapter_alpha_weight_decay": 0.0,
    "adapter_alpha_lr_scale": 1.0,
    "adapter_zero_init_output": False,
    "warmup_epochs": 0,
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_path(row):
    source = Path(row.get("source_artifact", row.get("artifact_dir", "")))
    return source / ("resolved_config.json" if row["backbone"] == "CBraMod" else "run_config.json")


def canonical_config(row):
    config = load_json(config_path(row))
    defaults = dict(DEFAULTS)
    # Older CBraMod artifacts omitted this field from resolved_config.json.
    # The CBraMod ISRUC runner's effective fallback is adapter_weight_decay
    # (0.05), whereas LaBraM's historical contract uses the generic default.
    if row["backbone"] == "CBraMod":
        defaults["adapter_alpha_weight_decay"] = 0.05
    result = {}
    for field in CONFIG_FIELDS:
        value = config.get(field, defaults.get(field))
        if field == "method":
            value = config.get("method", config.get("experiment_method"))
        if field == "experiment_method":
            value = config.get("experiment_method", config.get("method"))
        if field == "adapter_alpha_weight_decay" and value is None:
            value = defaults[field]
        if value is None and field in {"adapter_type", "sequence_head_dropout"}:
            value = "none" if field == "adapter_type" else None
        result[field] = value
    # Adapter hyperparameters are not an experimental condition for a
    # frozen-probe or dense run because no adapter module is instantiated.
    # Do not let historical defaults for inactive modules split an otherwise
    # identical three-seed group.
    method = str(config.get("method", config.get("experiment_method", ""))).strip().lower()
    if method in {"frozen_probe", "full_finetune", "dense"}:
        for field in (
            "adapter_type", "adapter_bottleneck", "adapter_gamma",
            "adapter_dropout", "adapter_heads", "adapter_lr",
            "adapter_weight_decay", "adapter_alpha_weight_decay",
            "adapter_alpha_lr_scale", "adapter_zero_init_output",
        ):
            result[field] = "not_applicable"
    return result


def group_key(row):
    return row["backbone"], row["dataset"], row["method"], row["variant"]


def add_issue(issues, severity, code, message, **context):
    issues.append({"severity": severity, "code": code, "message": message, **context})


def check_groups(rows, issues):
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        roles = {role for row in group for role in row.get("manuscript_role", "").split(";") if role}
        seeds = {row["seed"] for row in group}
        signatures = {}
        for row in group:
            try:
                signatures[row["seed"]] = canonical_config(row)
            except Exception as exc:
                add_issue(issues, "ERROR", "config_unreadable", str(exc), run_id=row["run_id"])
        homogeneous = len({json.dumps(value, sort_keys=True, default=str) for value in signatures.values()}) <= 1
        summaries.append({
            "backbone": key[0], "dataset": key[1], "method": key[2], "variant": key[3],
            "seeds": sorted(seeds, key=lambda x: int(x) if str(x).isdigit() else -1),
            "roles": sorted(roles), "complete_three_seed": seeds == SEEDS,
            "configuration_homogeneous": homogeneous,
        })
        if "RQ1" in roles or "RQ2" in roles or "RQ3" in roles:
            if seeds != SEEDS:
                add_issue(issues, "ERROR", "incomplete_seed_group",
                          "primary-role group does not contain exactly the canonical three seeds",
                          group=key, seeds=sorted(seeds))
            if not homogeneous:
                add_issue(issues, "ERROR", "mixed_configuration_group",
                          "primary-role group contains materially different resolved configurations",
                          group=key, seeds=sorted(seeds))
    return summaries


def check_artifacts(rows, issues):
    required_common = {
        "CBraMod": ("summary.json", "test_metrics.json", "resolved_config.json", "trainability_report.json"),
        "LaBraM": ("run_config.json", "final_test.json", "checkpoint-best.pth"),
    }
    for row in rows:
        source = Path(row["source_artifact"])
        for name in required_common[row["backbone"]]:
            if not (source / name).exists():
                add_issue(issues, "ERROR", "missing_artifact_file", name, run_id=row["run_id"])
        try:
            config = canonical_config(row)
            raw_config = load_json(config_path(row))
            source_seed = str(raw_config.get("seed"))
            if source_seed != row["seed"]:
                add_issue(issues, "ERROR", "seed_mismatch", "manifest seed differs from artifact configuration", run_id=row["run_id"])
            source = Path(row["source_artifact"])
            if row["backbone"] == "CBraMod":
                summary = load_json(source / "summary.json")
                metrics = load_json(source / "test_metrics.json")
                if summary.get("status") != "completed" or summary.get("test_evaluation_status") != "completed":
                    add_issue(issues, "ERROR", "incomplete_cb_reports", "CBraMod report is not completed", run_id=row["run_id"])
                expected_metrics = {
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "cohen_kappa": metrics.get("cohen_kappa"),
                    "weighted_f1": metrics.get("weighted_f1"),
                }
            else:
                final = load_json(source / "final_test.json")
                primary = final.get("primary_test", {})
                expected_metrics = {
                    "balanced_accuracy": primary.get("balanced_accuracy"),
                    "macro_f1": sum(primary.get("per_class_f1", [])) / len(primary.get("per_class_f1", [])) if primary.get("per_class_f1") else None,
                    "cohen_kappa": primary.get("cohen_kappa"),
                    "weighted_f1": primary.get("f1_weighted"),
                }
            for field, value in expected_metrics.items():
                if not finite(value):
                    add_issue(issues, "ERROR", "nonfinite_metric", field, run_id=row["run_id"])
                manifest_value = row.get(field if field != "cohen_kappa" else "kappa")
                if finite(value) and finite(manifest_value) and abs(float(value) - float(manifest_value)) > 1e-10:
                    add_issue(issues, "ERROR", "metric_mismatch", field, run_id=row["run_id"], artifact=value, manifest=manifest_value)
        except Exception as exc:
            add_issue(issues, "ERROR", "artifact_read_error", str(exc), run_id=row["run_id"])


def check_final_package(rows, issues):
    contract_path = FINAL / "analysis_contract.json"
    if not contract_path.exists():
        add_issue(issues, "ERROR", "missing_final_contract", str(contract_path))
        return
    contract = load_json(contract_path)
    if contract.get("source_manifest_sha256") != sha256(MANIFEST):
        add_issue(issues, "ERROR", "stale_final_package", "final package does not match current inclusion manifest")
    final_rows = read_csv(FINAL / "manuscript_final_manifest.csv")
    if len(final_rows) != len(rows):
        add_issue(issues, "ERROR", "row_count_mismatch", "final manifest row count differs from source manifest", source=len(rows), final=len(final_rows))
    if len({row["run_id"] for row in final_rows}) != len(final_rows):
        add_issue(issues, "ERROR", "duplicate_final_run_id", "final manifest contains duplicate run IDs")
    for name in ("rq1_effect_summary.csv", "rq2_effect_summary.csv", "rq3_effect_summary.csv", "rq2_pair_audit.csv"):
        if not (FINAL / name).exists():
            add_issue(issues, "ERROR", "missing_final_output", name)
    checksum = FINAL / "SHA256SUMS"
    if checksum.exists():
        for line in checksum.read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = FINAL / name
            if not path.exists() or sha256(path) != expected:
                add_issue(issues, "ERROR", "checksum_failure", name)
    return contract


def check_pairs(issues):
    path = FINAL / "rq2_pair_audit.csv"
    if not path.exists():
        return
    for row in read_csv(path):
        if row["pair_status"] == "VALID":
            match = float(row["parameter_match_pct"])
            if match > 5.0:
                add_issue(issues, "ERROR", "invalid_valid_pair", "valid pair exceeds capacity tolerance", row=row)
            if row.get("analysis_scope") == "primary" and row["seed"] not in SEEDS:
                add_issue(issues, "ERROR", "invalid_seed_pair", "primary pair uses noncanonical seed", row=row)


def check_registry(issues):
    """Ensure the broad registry cannot promote mixed groups to primary."""
    manifest_path = REGISTRY / "evidence_manifest.csv"
    if not manifest_path.exists():
        add_issue(issues, "ERROR", "missing_registry_manifest", str(manifest_path))
        return 0
    rows = read_csv(manifest_path)
    for row in rows:
        if row.get("candidate", "").lower() != "true":
            continue
        if row.get("configuration_homogeneous") != "true":
            if row.get("evidence_status") == "PRIMARY" or any(
                row.get(field) == "yes" for field in ("rq1_eligible", "rq2_eligible", "rq3_eligible")
            ):
                add_issue(issues, "ERROR", "mixed_group_promoted",
                          "mixed configuration was promoted to primary or eligibility",
                          run_id=row["run_id"])
        source = Path(row["artifact_dir"])
        required = (
            ("summary.json", "test_metrics.json", "resolved_config.json", "trainability_report.json")
            if row["backbone"] == "CBraMod" else
            ("run_config.json", "final_test.json", "checkpoint-best.pth")
        )
        missing = [name for name in required if not (source / name).exists()]
        if missing:
            add_issue(issues, "ERROR", "candidate_missing_artifact", ",".join(missing), run_id=row["run_id"])
    pair_path = REGISTRY / "paired_effects.csv"
    if pair_path.exists():
        for row in read_csv(pair_path):
            if row.get("parameter_matching_basis") != "adapter_module_parameters":
                add_issue(issues, "ERROR", "inconsistent_pair_basis", "paired registry row uses a different capacity basis", row=row)
            if row.get("pair_validity") != "PAIR_STRUCTURALLY_INVALID" and row.get("parameter_match_pct"):
                if float(row["parameter_match_pct"]) > 5.0:
                    add_issue(issues, "ERROR", "invalid_registry_pair", "non-invalid registry pair exceeds capacity tolerance", row=row)
    return len(rows)


def main():
    rows = read_csv(MANIFEST)
    issues = []
    summaries = check_groups(rows, issues)
    check_artifacts(rows, issues)
    contract = check_final_package(rows, issues)
    check_pairs(issues)
    registry_rows = check_registry(issues)
    result = {
        "status": "PASS" if not any(item["severity"] == "ERROR" for item in issues) else "FAIL",
        "source_manifest": str(MANIFEST),
        "source_manifest_sha256": sha256(MANIFEST),
        "selected_rows": len(rows),
        "registry_rows": registry_rows,
        "group_count": len(summaries),
        "error_count": sum(item["severity"] == "ERROR" for item in issues),
        "warning_count": sum(item["severity"] == "WARNING" for item in issues),
        "issues": issues,
        "groups": summaries,
        "final_contract_version": contract.get("version") if contract else None,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Manuscript results self-check", "", f"Status: **{result['status']}**", "",
        f"Selected rows: {len(rows)}; groups: {len(summaries)}; errors: {result['error_count']}; warnings: {result['warning_count']}", "",
    ]
    if issues:
        lines += ["## Issues", "", "| Severity | Code | Message | Context |", "|---|---|---|---|"]
        for item in issues:
            context = ", ".join(f"{key}={value}" for key, value in item.items() if key not in {"severity", "code", "message"})
            lines.append(f"| {item['severity']} | `{item['code']}` | {item['message']} | {context} |")
    else:
        lines += ["No consistency issues detected."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "selected_rows", "group_count", "error_count", "warning_count")}, indent=2))
    raise SystemExit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
