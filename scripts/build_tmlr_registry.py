#!/usr/bin/env python3
"""Build a traceable cross-backbone TMLR artifact registry.

The registry keeps all valid artifacts visible, then marks one deterministic
candidate per dataset/backbone/method/axis/seed for the manuscript aggregate.
Candidate selection is intentionally conservative and is reported in the
output files; it is not a substitute for final human review.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CB_ROOT = ROOT / "CBraMod" / "results"
LB_ROOT = ROOT / "LaBraM" / "checkpoints"
OUT = ROOT / "LaBraM" / "analysis" / "tmlr_registry"
SEEDS = {42, 1024, 3407}


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def scalar(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def last_jsonl(path: Path):
    last = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def read_labram_best_epoch(path: Path):
    rows = []
    log = path / "log.txt"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in row and "val_selection_score" in row:
                rows.append(row)
    if not rows:
        return None, None
    best = max(float(row["val_selection_score"]) for row in rows)
    selected = next(row for row in rows if float(row["val_selection_score"]) == best)
    return int(selected["epoch"]), best


def cb_axis(resolved, run_id):
    axis = resolved.get("adapter_type") or ""
    if axis in {"channel", "patch", "channel_patch"}:
        return axis
    if "axisblind" in run_id or "axis_blind" in run_id:
        return "generic_token_control"
    return "none"


def cb_candidate_score(dataset, run_id, method, seed):
    noisy = ("smoke", "audit", "boundary", "stability15", "headdropout",
             "corrected_gate", "repair", "source_fidelity", "e25")
    if any(token in run_id for token in noisy):
        return -1000
    # The locked CBraMod ISRUC packet intentionally uses alpha=0 initialization;
    # it is not an exploratory artifact there.
    if "alpha0" in run_id and dataset != "isruc":
        return -1000
    score = 0
    if dataset in {"faced", "seedv"} and "evalmode_frozen" in run_id:
        score += 100
    if dataset == "tuev" and "_b8_e20" in run_id:
        score += 100
    if dataset == "isruc" and "_b8_e20" in run_id:
        score += 100
    # The ISRUC axis-blind reference is the locked alpha=0 packet.  Prefer
    # that complete, homogeneous packet over the earlier seed-42 exploratory
    # control, whose omitted alpha-decay field makes the group mixed.
    if dataset == "isruc" and method == "axis_blind":
        score += 101 if "alpha0" in run_id else -100
    if dataset == "physionet_mi" and "lr2e-4_b64_e40" in run_id:
        score += 100
    if "_lr1e-4_" in run_id:
        score += 5
    if seed in SEEDS:
        score += 2
    return score


def labram_method(config):
    experiment = config.get("experiment_method")
    adapter = config.get("adapter_type") or "none"
    mode = config.get("backbone_mode")
    if mode is None:
        mode = "frozen" if config.get("backbone_frozen") or config.get("backbone_lr_scale") == 0 else "trainable"
    if experiment == "axis_blind":
        return "axis_blind", "generic_token_control"
    if mode == "lora" or config.get("lora_parameter_count", 0):
        return "lora", "qkv"
    if mode == "upper_k":
        return "upper2", "upper"
    if adapter == "none":
        return ("frozen_probe" if mode == "frozen" else "dense"), "none"
    prefix = "frozen_native_" if mode == "frozen" else "native_"
    return prefix + adapter, adapter


def labram_candidate_score(dataset, run_id, method):
    if any(token in run_id for token in ("smoke", "dev", "pilot")):
        return -1000
    score = 0
    if dataset == "FACED" and "reviewer_r2" in run_id:
        score += 100
    if dataset == "SEED-V" and ("matched" in run_id or "final" in run_id):
        score += 100
    if dataset == "TUEV" and run_id.startswith("tuev_labram_"):
        score += 100
    if dataset == "ISRUC" and run_id.startswith("isruc_labram_"):
        score += 100
    if dataset == "PhysioNet-MI" and "physio" in run_id.lower():
        score += 100
    if "retry_seqfix" in run_id:
        score += 20
    if "_test" in run_id:
        score += 5
    return score


def cb_rows():
    rows = []
    for dataset_dir in sorted(CB_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for summary_path in dataset_dir.glob("*/summary.json"):
            run_dir = summary_path.parent
            summary = load_json(summary_path)
            if summary.get("status") != "completed":
                continue
            test = load_json(run_dir / "test_metrics.json")
            resolved = load_json(run_dir / "resolved_config.json")
            best = load_json(run_dir / "best_validation_metrics.json")
            trainability = load_json(run_dir / "trainability_report.json")
            diagnostics = last_jsonl(run_dir / "adapter_diagnostics_by_epoch.jsonl")
            if not test:
                continue
            method = str(summary.get("method", resolved.get("method", "unknown")))
            axis = cb_axis(resolved, run_dir.name)
            seed = summary.get("seed")
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
            rows.append({
                "backbone": "CBraMod",
                "dataset": dataset,
                "run_id": run_dir.name,
                "artifact_dir": str(run_dir),
                "method": method,
                "axis": axis,
                "seed": seed,
                "contract_valid": True,
                "status": summary.get("status"),
                "selected_epoch": test.get("selected_epoch", summary.get("best_epoch")),
                "val_selection_value": best.get("selection_value", summary.get("best_validation_selection_value")),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "cohen_kappa": test.get("cohen_kappa"),
                "macro_f1": test.get("macro_f1"),
                "weighted_f1": test.get("weighted_f1"),
                "trainable_parameters": summary.get("trainable_parameter_count", trainability.get("trainable_parameter_count")),
                "adapter_delta_ratio": diagnostics.get("adapter_delta_ratio"),
                "backbone_update_norm": diagnostics.get("backbone_update_norm"),
                "candidate_score": cb_candidate_score(dataset, run_dir.name, method, seed),
                "source": "summary.json + test_metrics.json",
            })
    return rows


def labram_rows():
    rows = []
    for final_path in sorted(LB_ROOT.glob("*/final_test.json")):
        run_dir = final_path.parent
        final = load_json(final_path)
        if not final.get("primary_test"):
            continue
        config = load_json(run_dir / "run_config.json")
        contract_valid = bool(config) and bool(config.get("strict_checkpoint_load", False))
        dataset = config.get("dataset", "unknown")
        method, axis = labram_method(config) if config else ("legacy_unknown", "unknown")
        seed = config.get("seed")
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
        primary = final["primary_test"]
        best_epoch, best_value = read_labram_best_epoch(run_dir)
        diagnostics = last_jsonl(run_dir / "log.txt")
        trainability = config.get("trainability_summary", {})
        trainable = config.get("trainable_parameter_count")
        if trainable is None:
            trainable = sum(int(v.get("trainable", 0)) for v in trainability.values() if isinstance(v, dict))
        rows.append({
            "backbone": "LaBraM",
            "dataset": dataset,
            "run_id": run_dir.name,
            "artifact_dir": str(run_dir),
            "method": method,
            "axis": axis,
            "seed": seed,
            "contract_valid": contract_valid,
            "status": "completed",
            "selected_epoch": best_epoch,
            "val_selection_value": best_value,
            "balanced_accuracy": primary.get("balanced_accuracy"),
            "cohen_kappa": primary.get("cohen_kappa"),
            "macro_f1": primary.get("macro_f1"),
            "weighted_f1": primary.get("f1_weighted", primary.get("weighted_f1")),
            "trainable_parameters": trainable,
            "adapter_delta_ratio": diagnostics.get("train_adapter_delta_ratio"),
            "backbone_update_norm": diagnostics.get("train_last_block_update_norm"),
            "candidate_score": labram_candidate_score(dataset, run_dir.name, method),
            "source": "run_config.json + final_test.json",
        })
    return rows


def choose_candidates(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["seed"] not in SEEDS or not row["contract_valid"]:
            continue
        key = (row["backbone"], row["dataset"], row["method"], row["axis"], row["seed"])
        groups[key].append(row)
    selected = set()
    for candidates in groups.values():
        best = max(candidates, key=lambda row: (row["candidate_score"], row["run_id"]))
        if best["candidate_score"] >= 0:
            selected.add(best["run_id"])
    return selected


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def write_outputs(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "backbone", "dataset", "run_id", "artifact_dir", "method", "axis", "seed",
        "candidate", "candidate_score", "contract_valid", "status", "selected_epoch",
        "val_selection_value", "balanced_accuracy", "cohen_kappa", "macro_f1",
        "weighted_f1", "trainable_parameters", "adapter_delta_ratio",
        "backbone_update_norm", "source",
    ]
    selected = choose_candidates(rows)
    for row in rows:
        row["candidate"] = row["run_id"] in selected

    with (OUT / "all_artifacts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)

    canonical = [row for row in rows if row["candidate"]]
    with (OUT / "canonical_candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in canonical)

    groups = defaultdict(list)
    for row in canonical:
        key = (row["backbone"], row["dataset"], row["method"], row["axis"])
        groups[key].append(row)
    aggregate = []
    for key, group in sorted(groups.items()):
        out = {"backbone": key[0], "dataset": key[1], "method": key[2], "axis": key[3], "n": len(group),
               "seeds": ",".join(str(x["seed"]) for x in sorted(group, key=lambda r: (r["seed"] is None, r["seed"])))}
        for metric in ("balanced_accuracy", "cohen_kappa", "macro_f1", "weighted_f1", "trainable_parameters", "selected_epoch"):
            values = [numeric(row.get(metric)) for row in group]
            values = [x for x in values if x is not None]
            out[metric + "_mean"] = statistics.mean(values) if values else None
            out[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate.append(out)

    agg_fields = list(aggregate[0].keys()) if aggregate else ["backbone", "dataset", "method", "axis", "n", "seeds"]
    with (OUT / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(aggregate)

    with (OUT / "registry_summary.md").open("w") as f:
        f.write("# TMLR artifact registry\n\n")
        f.write(f"Generated from {len(rows)} completed artifacts; {len(canonical)} canonical candidates.\n\n")
        f.write("Canonical candidates are deterministic suggestions and must be reviewed before final tables.\n\n")
        f.write("| Backbone | Dataset | Method | Axis | n | BA | Kappa | Weighted F1 | Parameters |\n|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate:
            def fmt(metric):
                mean, sd = row.get(metric + "_mean"), row.get(metric + "_sd")
                return "—" if mean is None else f"{mean:.4f} ± {sd:.4f}"
            f.write(f"| {row['backbone']} | {row['dataset']} | {row['method']} | {row['axis']} | {row['n']} | {fmt('balanced_accuracy')} | {fmt('cohen_kappa')} | {fmt('weighted_f1')} | {fmt('trainable_parameters')} |\n")

    trajectory_rows = []
    trajectory_by_epoch = defaultdict(list)
    eval_root = LB_ROOT
    for run_dir in sorted(eval_root.glob("isruc_labram_axis_blind_b400_s*_retry_seqfix")):
        path = run_dir / "all_epoch_test_metrics.jsonl"
        if not path.exists():
            continue
        records = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            continue
        for record in records:
            trajectory_by_epoch[int(record["epoch"])].append(record)
        selected = max(records, key=lambda row: float(row["validation"].get("cohen_kappa", float("-inf"))))
        peak_ba = max(records, key=lambda row: float(row["test"].get("balanced_accuracy", float("-inf"))))
        peak_kappa = max(records, key=lambda row: float(row["test"].get("cohen_kappa", float("-inf"))))
        trajectory_rows.append({
            "run_id": run_dir.name,
            "seed": load_json(run_dir / "run_config.json").get("seed"),
            "n_epochs": len(records),
            "validation_selected_epoch": selected["epoch"],
            "validation_selected_val_kappa": selected["validation"].get("cohen_kappa"),
            "test_ba_at_validation_selected": selected["test"].get("balanced_accuracy"),
            "test_kappa_at_validation_selected": selected["test"].get("cohen_kappa"),
            "test_f1_at_validation_selected": selected["test"].get("f1_weighted"),
            "test_ba_peak": peak_ba["test"].get("balanced_accuracy"),
            "test_ba_peak_epoch": peak_ba["epoch"],
            "test_kappa_peak": peak_kappa["test"].get("cohen_kappa"),
            "test_kappa_peak_epoch": peak_kappa["epoch"],
            "test_peak_is_post_selection": peak_ba["epoch"] != selected["epoch"],
            "artifact_dir": str(run_dir),
        })
    if trajectory_rows:
        with (OUT / "isruc_axisblind_trajectory.csv").open("w", newline="") as f:
            fields = list(trajectory_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trajectory_rows)
    mean_rows = []
    for epoch, records in sorted(trajectory_by_epoch.items()):
        row = {"epoch": epoch, "n_seeds": len(records)}
        for split, metric in (("validation", "cohen_kappa"), ("test", "balanced_accuracy"),
                              ("test", "cohen_kappa"), ("test", "f1_weighted")):
            values = [numeric(record[split].get(metric)) for record in records]
            values = [value for value in values if value is not None]
            label = f"{split}_{metric}"
            row[label + "_mean"] = statistics.mean(values) if values else None
            row[label + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        mean_rows.append(row)
    if mean_rows:
        with (OUT / "isruc_axisblind_trajectory_mean.csv").open("w", newline="") as f:
            fields = list(mean_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(mean_rows)

    (OUT / "registry.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"Wrote {len(rows)} artifacts, {len(canonical)} canonical candidates, {len(aggregate)} aggregate groups to {OUT}")


if __name__ == "__main__":
    write_outputs(cb_rows() + labram_rows())
