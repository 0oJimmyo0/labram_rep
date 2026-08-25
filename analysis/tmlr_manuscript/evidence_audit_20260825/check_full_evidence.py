#!/usr/bin/env python3
"""Fail-closed consistency check for the full evidence audit package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


AUDIT = Path(__file__).resolve().parent
LABRAM = AUDIT.parents[2]
ROOT = LABRAM.parent
CBRAMOD = ROOT / "CBraMod"


def actual_artifact_paths() -> set[str]:
    paths = set()
    for dataset in (CBRAMOD / "results").iterdir():
        if not dataset.is_dir():
            continue
        paths.update(str(p.resolve()) for p in dataset.iterdir() if p.is_dir() and p.name != "audits")
    paths.update(str(p.resolve()) for p in (LABRAM / "checkpoints").iterdir() if p.is_dir())
    return paths


def actual_log_paths() -> set[str]:
    paths = set()
    for base in (CBRAMOD, LABRAM):
        for p in base.rglob("*"):
            if p.is_file() and (p.name == "log.txt" or p.suffix in {".out", ".err", ".vu"}):
                paths.add(str(p.resolve()))
    return paths


def main() -> int:
    artifacts = pd.read_csv(AUDIT / "filesystem_artifact_audit.csv")
    logs = pd.read_csv(AUDIT / "log_inventory.csv")
    conditions = pd.read_csv(AUDIT / "registry_condition_results.csv")
    coverage = pd.read_csv(AUDIT / "backbone_dataset_coverage.csv")
    registry = pd.read_csv(LABRAM / "analysis/tmlr_registry/all_artifacts.csv")
    checks: dict[str, bool] = {}

    checks["artifact_inventory_matches_filesystem"] = set(artifacts.artifact_dir) == actual_artifact_paths()
    checks["artifact_paths_unique"] = artifacts.artifact_dir.nunique() == len(artifacts)
    checks["registry_paths_are_in_inventory"] = set(registry.artifact_dir).issubset(set(artifacts.artifact_dir))
    checks["registry_count"] = len(registry) == 485
    checks["registry_complete_count"] = int(((artifacts.registry_member) & (artifacts.complete_test_contract)).sum()) == 485
    checks["log_inventory_matches_filesystem"] = set(logs.log_path) == actual_log_paths()
    checks["log_count"] = len(logs) == 2515
    checks["text_log_count"] = int((logs.read_status == "ok").sum()) == 2047
    checks["binary_event_count"] = int((logs.log_status == "binary_event_file").sum()) == 468
    checks["coverage_has_ten_cells"] = len(coverage) == 10 and coverage.groupby("backbone").size().to_dict() == {"CBraMod": 5, "LaBraM": 5}
    checks["condition_count"] = len(conditions) == 89
    checks["complete_three_seed_condition_count"] = int(conditions.complete_three_seed.sum()) == 88
    expected_primary = {
        ("CBraMod", "FACED", "axis_blind", "generic_token_control"),
        ("CBraMod", "FACED", "interaction_aligned", "channel_patch"),
        ("CBraMod", "ISRUC", "axis_blind", "generic_token_control"),
        ("CBraMod", "ISRUC", "interaction_aligned", "channel_patch"),
        ("CBraMod", "TUEV", "axis_blind", "generic_token_control"),
        ("CBraMod", "TUEV", "interaction_aligned", "channel_patch"),
        ("LaBraM", "TUEV", "axis_blind", "generic_token_control"),
        ("LaBraM", "TUEV", "frozen_native_channel_patch", "channel_patch"),
    }
    actual_primary = set(
        conditions[conditions.manuscript_role == "primary_RQ2_candidate"]
        .set_index(["backbone", "dataset", "method", "axis"]).index
    )
    checks["primary_scope_exact"] = actual_primary == expected_primary
    checks["all_condition_source_paths_exist"] = all(
        Path(source).is_dir()
        for value in conditions.source_artifacts.fillna("")
        for source in str(value).split("|")
        if source
    )
    payload = json.loads((AUDIT / "audit_contract.json").read_text(encoding="utf-8"))
    checks["contract_complete"] = payload.get("status") == "complete"

    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"status": "PASS" if not failed else "FAIL", "checks": checks}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
