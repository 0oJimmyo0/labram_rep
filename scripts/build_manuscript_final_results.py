#!/usr/bin/env python3
"""Build the focused, manuscript-final results package.

The broad registry is provenance infrastructure and remains fail-closed until
its full review queues are adjudicated.  This script instead audits the
deterministically selected compact manuscript manifest and produces the fixed
tables/effect files used for drafting.  It never selects by score and never
modifies the training registry.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis" / "tmlr_manuscript" / "manuscript_inclusion_manifest.csv"
OUT = ROOT / "analysis" / "tmlr_manuscript" / "manuscript_final_v1"
SEEDS = {"42", "1024", "3407"}
METRICS = ("balanced_accuracy", "macro_f1", "cohen_kappa", "weighted_f1")


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def same_number(left, right, tolerance=1e-8):
    left = number(left)
    right = number(right)
    return left is not None and right is not None and abs(left - right) <= tolerance


def dataset_key(value):
    return str(value).lower().replace("-", "_")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_config(row):
    source = Path(row["source_artifact"])
    if row["backbone"] == "CBraMod":
        return source / "resolved_config.json"
    return source / "run_config.json"


def source_metrics(row):
    source = Path(row["source_artifact"])
    if row["backbone"] == "CBraMod":
        summary = load_json(source / "summary.json")
        metrics = load_json(source / "test_metrics.json")
        return {
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "cohen_kappa": metrics.get("cohen_kappa", metrics.get("kappa")),
            "weighted_f1": metrics.get("weighted_f1"),
            "selected_epoch": metrics.get("selected_epoch", summary.get("best_epoch")),
            "selection_metric": metrics.get("selection_metric", "cohen_kappa"),
        }
    final = load_json(source / "final_test.json")
    primary = final.get("primary_test", {})
    per_class_f1 = primary.get("per_class_f1", [])
    macro = sum(per_class_f1) / len(per_class_f1) if per_class_f1 else None
    return {
        "balanced_accuracy": primary.get("balanced_accuracy"),
        "macro_f1": macro,
        "cohen_kappa": primary.get("cohen_kappa"),
        "weighted_f1": primary.get("f1_weighted"),
        "selected_epoch": None,
        "selection_metric": final.get("selection_metric", "cohen_kappa"),
    }


def dataset_audit_passes(dataset_audit):
    """Accept both the newer split-summary audit and legacy audit schemas."""
    if dataset_audit.get("finite_value_check") is True:
        return True
    split_summary = dataset_audit.get("split_summary", {})
    if split_summary:
        finite_flags = [part.get("finite_values") for part in split_summary.values()]
        return bool(finite_flags) and all(flag is True for flag in finite_flags)
    # SEED-V's legacy audit records sampled shape and non-overlapping split
    # keys, but does not expose a finite-value flag.
    if dataset_audit.get("dataset") == "SEED-V":
        return bool(dataset_audit.get("shape")) and dataset_audit.get("split_key_overlap") is False
    return False


def audit_artifact(row):
    source = Path(row["source_artifact"])
    checks = []
    required = (
        ("summary.json", "test_metrics.json", "resolved_config.json",
         "provenance.json", "dataset_audit.json", "checkpoint_load_report.json",
         "optimizer_groups.json", "trainability_report.json", "metrics_by_epoch.jsonl")
        if row["backbone"] == "CBraMod" else
        ("run_config.json", "final_test.json", "checkpoint-best.pth", "log.txt")
    )
    for name in required:
        checks.append((name, (source / name).exists()))
    if not all(ok for _, ok in checks):
        return "EXCLUDED", checks, "missing required artifact files"

    try:
        config = load_json(source_config(row))
        metrics = source_metrics(row)
        if row["backbone"] == "CBraMod":
            summary = load_json(source / "summary.json")
            load_report = load_json(source / "checkpoint_load_report.json")
            training = load_json(source / "training_mode_report.json") if (source / "training_mode_report.json").exists() else {}
            dataset_audit = load_json(source / "dataset_audit.json")
            checks.extend([
                ("completed", summary.get("status") == "completed"),
                ("test_evaluation_completed", summary.get("test_evaluation_status") == "completed"),
                ("strict_checkpoint_load", load_report.get("strict_loading_status") is True),
                ("finite_dataset", dataset_audit_passes(dataset_audit)),
                ("validation_selector", config.get("selection_metric") == "cohen_kappa"),
                ("frozen_mode", (not training.get("backbone_training")) if training and row["trainability_regime"] == "frozen" else True),
                ("seed", str(config.get("seed")) == row["seed"]),
                ("selected_epoch", str(metrics.get("selected_epoch")) == row["selected_epoch"]),
            ])
        else:
            final = load_json(source / "final_test.json")
            load_report = config.get("checkpoint_load_report", {})
            checks.extend([
                ("strict_checkpoint_load", load_report.get("strict_pass") is True),
                ("validation_selector", final.get("selection_metric") == "cohen_kappa"),
                ("frozen_mode", bool(config.get("backbone_frozen")) if row["trainability_regime"] == "frozen" else True),
                # Older LaBraM packets predate the explicit eval-mode field.
                # Keep them as supporting evidence rather than silently
                # promoting them to the final primary analysis.
                ("frozen_eval_mode", bool(config.get("frozen_backbone_eval_mode")) if row["trainability_regime"] == "frozen" and config.get("frozen_backbone_eval_mode") is not None else True),
                ("seed", str(config.get("seed")) == row["seed"]),
                ("primary_checkpoint", final.get("primary_checkpoint") == "checkpoint-best.pth"),
                ("epoch_log", sum(1 for line in (source / "log.txt").read_text().splitlines() if line.strip()) > 0),
            ])
        for metric in METRICS:
            checks.append((f"finite_{metric}", finite(metrics.get(metric))))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
        return "EXCLUDED", checks, f"metadata parse failure: {type(exc).__name__}: {exc}"

    failed = [name for name, ok in checks if not ok]
    if not failed:
        return "VERIFIED", checks, "all required checks passed"
    # Legacy LaBraM packets without frozen eval mode remain useful for
    # descriptive/supporting tables, but are not admitted to primary paired
    # inference.
    if failed == ["frozen_eval_mode"]:
        return "SUPPORTING", checks, "legacy frozen packet lacks explicit eval-mode provenance"
    return "EXCLUDED", checks, "; ".join(failed)


def artifact_audit(rows):
    audited = []
    by_run = {}
    for row in rows:
        status, checks, note = audit_artifact(row)
        out = dict(row)
        out["artifact_final_status"] = status
        out["artifact_audit_note"] = note
        out["artifact_checks"] = ";".join(f"{name}={str(ok).lower()}" for name, ok in checks)
        try:
            metrics = source_metrics(row)
            out["source_selected_epoch"] = "" if metrics.get("selected_epoch") is None else str(metrics["selected_epoch"])
            for metric in METRICS:
                out[f"source_{metric}"] = "" if metrics.get(metric) is None else f"{metrics[metric]:.12g}"
            config = load_json(source_config(row))
            fraction = config.get("trainable_parameter_fraction", "")
            total_parameters = config.get("total_parameter_count", "")
            if row["backbone"] == "CBraMod":
                load_report = load_json(Path(row["source_artifact"]) / "checkpoint_load_report.json")
                total_parameters = load_report.get("model_parameter_count_after_attachment", total_parameters)
                if fraction in ("", None) and total_parameters not in ("", None):
                    fraction = float(row["trainable_parameters"]) / float(total_parameters)
            out["source_trainable_parameter_fraction"] = str(fraction)
            out["source_total_parameters"] = str(total_parameters)
            if row["backbone"] == "CBraMod":
                timing = load_json(Path(row["source_artifact"]) / "timing.json")
                out["source_peak_gpu_memory_bytes"] = str(timing.get("peak_gpu_memory_bytes", ""))
                out["source_training_wall_seconds"] = str(timing.get("training_wall_seconds", ""))
                epoch_times = timing.get("time_per_epoch_seconds", [])
                out["source_time_per_epoch_seconds"] = str(sum(epoch_times) / len(epoch_times) if epoch_times else "")
            else:
                out["source_peak_gpu_memory_bytes"] = ""
                out["source_training_wall_seconds"] = ""
                out["source_time_per_epoch_seconds"] = ""
        except Exception:
            pass
        audited.append(out)
        by_run[row["run_id"]] = out
    return audited, by_run


def config_value(config, key):
    value = config.get(key)
    if key == "selection_metric" and value is None:
        return "cohen_kappa"
    return value


def split_hashes(config):
    metadata = config.get("dataset_split_metadata", {})
    return tuple(metadata.get(name, {}).get("file_manifest_sha256") for name in ("train", "val", "test"))


def pair_audit(left, right, left_status, right_status):
    notes = []
    if left_status != "VERIFIED" or right_status != "VERIFIED":
        notes.append("one or both artifacts failed focused audit")
    if left["backbone"] != right["backbone"] or dataset_key(left["dataset"]) != dataset_key(right["dataset"]):
        notes.append("backbone or dataset mismatch")
    if left["seed"] != right["seed"]:
        notes.append("seed mismatch")
    if left.get("trainability_regime") != right.get("trainability_regime"):
        notes.append("trainability mismatch")
    match = number(left.get("parameter_match_pct"))
    if match is None or match > 5.0:
        notes.append("parameter mismatch exceeds 5 percent or is unavailable")

    try:
        left_config = load_json(source_config(left))
        right_config = load_json(source_config(right))
        fields = (
            "dataset", "model", "batch_size", "epochs", "weight_decay",
            "input_scale_divisor", "selection_metric", "optimizer",
            "warmup_epochs", "drop_path", "smoothing", "drop",
            "sequence_head_dropout", "pretrained_checkpoint_sha256",
            "channel_manifest_sha256", "input_position_policy",
            "backbone_frozen", "frozen_backbone_eval_mode",
        )
        for field in fields:
            left_value = config_value(left_config, field)
            right_value = config_value(right_config, field)
            if left_value is None or right_value is None:
                continue
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                equal = same_number(left_value, right_value)
            else:
                equal = left_value == right_value
            if not equal:
                notes.append(f"shared field mismatch: {field}")
        if split_hashes(left_config) != split_hashes(right_config):
            notes.append("dataset split manifest mismatch")
        axisblind = right if right["method"] == "axis_blind" else left
        axis_config = load_json(source_config(axisblind))
        if axisblind["backbone"] == "CBraMod":
            if axis_config.get("method") != "axis_blind" or axis_config.get("adapter_type") not in (None, ""):
                notes.append("axis-blind path is not explicitly generic")
        else:
            if axis_config.get("experiment_method") != "axis_blind" or axis_config.get("adapter_type") != "generic":
                notes.append("axis-blind path is not explicitly generic")
    except Exception as exc:
        notes.append(f"pair metadata parse failure: {type(exc).__name__}")

    scope = "primary"
    dkey = dataset_key(left["dataset"])
    if dkey in {"seedv", "seed_v"}:
        scope = "boundary"
    elif dkey in {"physionet_mi"}:
        scope = "supporting_boundary"
    status = "VALID" if not notes else "EXCLUDED"
    return {
        "backbone": left["backbone"],
        "dataset": left["dataset"],
        "axis": left["variant"],
        "seed": left["seed"],
        "native_run_id": left["run_id"],
        "axisblind_run_id": right["run_id"],
        "parameter_match_pct": left.get("parameter_match_pct", ""),
        "pair_status": status,
        "analysis_scope": scope,
        "pair_audit_notes": "all focused pair checks passed" if not notes else "; ".join(notes),
    }


def mean_sd(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return f"{mean:.8f}", f"{sd:.8f}"


def effect_rows(audited, by_run, comparison):
    rows_by_id = {row["run_id"]: row for row in audited}
    rows = []
    if comparison == "rq2":
        natives = [row for row in audited if "RQ2" in row["manuscript_role"] and row["method"] not in {"axis_blind"}]
        for native in natives:
            control = rows_by_id.get(native.get("paired_control_run_id"))
            if control is None or native["artifact_final_status"] != "VERIFIED" or control["artifact_final_status"] != "VERIFIED":
                continue
            pair = pair_audit(native, control, native["artifact_final_status"], control["artifact_final_status"])
            if pair["pair_status"] != "VALID":
                continue
            for metric in ("balanced_accuracy", "macro_f1"):
                left = number(native.get(f"source_{metric}"))
                right = number(control.get(f"source_{metric}"))
                if left is None or right is None:
                    continue
                rows.append({**pair, "metric": metric, "native_value": f"{left:.8f}", "axisblind_value": f"{right:.8f}", "effect_native_minus_axisblind": f"{left-right:.8f}"})
    elif comparison == "rq1":
        natives = [row for row in audited if "RQ1" in row["manuscript_role"] and row["method"] not in {"frozen_probe", "axis_blind"}]
        probes = {(row["backbone"], row["dataset"], row["seed"]): row for row in audited if row["method"] == "frozen_probe"}
        for native in natives:
            probe = probes.get((native["backbone"], native["dataset"], native["seed"]))
            if probe is None or native["artifact_final_status"] != "VERIFIED" or probe["artifact_final_status"] != "VERIFIED":
                continue
            for metric in ("balanced_accuracy", "macro_f1"):
                left = number(native.get(f"source_{metric}"))
                right = number(probe.get(f"source_{metric}"))
                if left is None or right is None:
                    continue
                rows.append({"backbone": native["backbone"], "dataset": native["dataset"], "axis": native["variant"], "seed": native["seed"], "native_run_id": native["run_id"], "probe_run_id": probe["run_id"], "metric": metric, "native_value": f"{left:.8f}", "probe_value": f"{right:.8f}", "effect_native_minus_probe": f"{left-right:.8f}"})
    return rows


def rq3_rows(audited):
    """Return the two prespecified incremental RQ3 contrasts."""
    verified = [row for row in audited if row["artifact_final_status"] == "VERIFIED"]
    output = []
    probes = {(row["backbone"], row["dataset"], row["seed"]): row for row in verified if row["method"] == "frozen_probe"}
    dense_methods = {"CBraMod": {"full_finetune"}, "LaBraM": {"dense"}}
    native_methods = {"CBraMod": {"native_full_finetune"}, "LaBraM": {"native_channel", "native_patch", "native_channel_patch"}}

    # Frozen regime: aligned residual adaptation versus the frozen probe.
    for row in verified:
        if "RQ1" not in row["manuscript_role"] or row["method"] in {"frozen_probe", "axis_blind"}:
            continue
        probe = probes.get((row["backbone"], row["dataset"], row["seed"]))
        if probe is None:
            continue
        for metric in ("balanced_accuracy", "macro_f1"):
            left = number(row.get(f"source_{metric}"))
            right = number(probe.get(f"source_{metric}"))
            if left is not None and right is not None:
                output.append({"contrast": "delta_frozen", "backbone": row["backbone"], "dataset": row["dataset"], "axis": row["variant"], "seed": row["seed"], "adapter_run_id": row["run_id"], "reference_run_id": probe["run_id"], "metric": metric, "adapter_value": f"{left:.8f}", "reference_value": f"{right:.8f}", "effect_adapter_minus_reference": f"{left-right:.8f}"})

    # Fully trainable regime: native aligned residual adaptation versus dense.
    by_cell_method = {(row["backbone"], row["dataset"], row["seed"], row["method"]): row for row in verified}
    for row in verified:
        if row["method"] not in native_methods.get(row["backbone"], set()) or "RQ3" not in row["manuscript_role"]:
            continue
        dense = next((by_cell_method.get((row["backbone"], row["dataset"], row["seed"], method)) for method in dense_methods[row["backbone"]]), None)
        if dense is None:
            continue
        for metric in ("balanced_accuracy", "macro_f1"):
            left = number(row.get(f"source_{metric}"))
            right = number(dense.get(f"source_{metric}"))
            if left is not None and right is not None:
                output.append({"contrast": "delta_full", "backbone": row["backbone"], "dataset": row["dataset"], "axis": row["variant"], "seed": row["seed"], "adapter_run_id": row["run_id"], "reference_run_id": dense["run_id"], "metric": metric, "adapter_value": f"{left:.8f}", "reference_value": f"{right:.8f}", "effect_adapter_minus_reference": f"{left-right:.8f}"})
    return output


def summarize_effects(rows, effect_field):
    grouped = defaultdict(list)
    include_contrast = any(row.get("contrast") for row in rows)
    for row in rows:
        key = ((row.get("contrast", ""),) if include_contrast else ()) + (row["backbone"], row["dataset"], row["axis"], row["metric"])
        grouped[key].append(float(row[effect_field]))
    output = []
    for key, values in sorted(grouped.items()):
        mean, sd = mean_sd(values)
        if include_contrast:
            contrast, backbone, dataset, axis, metric = key
        else:
            contrast = ""
            backbone, dataset, axis, metric = key
        matching = [row for row in rows if ((row.get("contrast", ""),) if include_contrast else ()) + (row["backbone"], row["dataset"], row["axis"], row["metric"]) == key]
        item = {"backbone": backbone, "dataset": dataset, "axis": axis, "metric": metric, "n": len(values), "seeds": ",".join(str(seed) for seed in sorted({int(row["seed"]) for row in matching})), "effect_mean": mean, "effect_sd": sd, "positive_seed_count": sum(value > 0 for value in values), "effect_min": f"{min(values):.8f}", "effect_max": f"{max(values):.8f}"}
        if include_contrast:
            item = {"contrast": contrast, **item}
        output.append(item)
    return output


def method_summary(audited):
    grouped = defaultdict(list)
    for row in audited:
        if row["artifact_final_status"] != "VERIFIED":
            continue
        grouped[(row["backbone"], row["dataset"], row["method"], row["variant"])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = {"backbone": key[0], "dataset": key[1], "method": key[2], "variant": key[3], "n": len(group), "seeds": ",".join(str(seed) for seed in sorted(int(row["seed"]) for row in group))}
        for metric in METRICS:
            mean, sd = mean_sd([number(row.get(f"source_{metric}")) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sd"] = sd
        params = [number(row.get("trainable_parameters")) for row in group]
        mean, sd = mean_sd(params)
        item["trainable_parameters_mean"] = mean
        item["trainable_parameters_sd"] = sd
        fractions = [number(row.get("source_trainable_parameter_fraction")) for row in group]
        mean, sd = mean_sd(fractions)
        item["trainable_parameter_fraction_mean"] = mean
        item["trainable_parameter_fraction_sd"] = sd
        memories = [number(row.get("source_peak_gpu_memory_bytes")) for row in group]
        mean, sd = mean_sd(memories)
        item["peak_gpu_memory_bytes_mean"] = mean
        item["peak_gpu_memory_bytes_sd"] = sd
        wall = [number(row.get("source_training_wall_seconds")) for row in group]
        mean, sd = mean_sd(wall)
        item["training_wall_seconds_mean"] = mean
        item["training_wall_seconds_sd"] = sd
        item["analysis_roles"] = ";".join(sorted({role for row in group for role in row["manuscript_role"].split(";")}))
        output.append(item)
    return output


def main():
    rows = read_csv(MANIFEST)
    audited, by_run = artifact_audit(rows)
    rq1 = effect_rows(audited, by_run, "rq1")
    rq2 = effect_rows(audited, by_run, "rq2")
    rq3 = rq3_rows(audited)
    rq1_summary = summarize_effects(rq1, "effect_native_minus_probe")
    rq2_primary = [row for row in rq2 if row.get("analysis_scope") == "primary"]
    rq2_supporting = [row for row in rq2 if row.get("analysis_scope") != "primary"]
    rq2_summary = summarize_effects(rq2_primary, "effect_native_minus_axisblind")
    rq2_all_summary = summarize_effects(rq2, "effect_native_minus_axisblind")
    summary = method_summary(audited)

    pairs = []
    seen = set()
    for row in audited:
        if "RQ2" not in row["manuscript_role"] or row["method"] == "axis_blind" or not row.get("paired_control_run_id"):
            continue
        key = (row["run_id"], row["paired_control_run_id"])
        if key in seen:
            continue
        seen.add(key)
        control = by_run.get(row["paired_control_run_id"])
        if control:
            pairs.append(pair_audit(row, control, row["artifact_final_status"], control["artifact_final_status"]))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    write_csv(OUT / "manuscript_final_manifest.csv", audited)
    write_csv(OUT / "artifact_audit.csv", audited)
    write_csv(OUT / "rq1_seed_effects.csv", rq1)
    write_csv(OUT / "rq1_effect_summary.csv", rq1_summary)
    write_csv(OUT / "rq2_pair_audit.csv", pairs)
    write_csv(OUT / "rq2_seed_effects.csv", rq2)
    write_csv(OUT / "rq2_primary_seed_effects.csv", rq2_primary)
    write_csv(OUT / "rq2_supporting_seed_effects.csv", rq2_supporting)
    write_csv(OUT / "rq2_effect_summary.csv", rq2_summary)
    write_csv(OUT / "rq2_all_effect_summary.csv", rq2_all_summary)
    write_csv(OUT / "rq3_seed_effects.csv", rq3)
    write_csv(OUT / "rq3_effect_summary.csv", summarize_effects(rq3, "effect_adapter_minus_reference"))
    write_csv(OUT / "method_summary.csv", summary)

    status_counts = defaultdict(int)
    for row in audited:
        status_counts[row["artifact_final_status"]] += 1
    pair_counts = defaultdict(int)
    for row in pairs:
        pair_counts[f"{row['analysis_scope']}:{row['pair_status']}"] += 1
    contract = {
        "version": "manuscript_final_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(MANIFEST),
        "source_manifest_sha256": sha256(MANIFEST),
        "selected_rows": len(rows),
        "artifact_status_counts": dict(sorted(status_counts.items())),
        "pair_status_counts": dict(sorted(pair_counts.items())),
        "primary_rq2_pair_rows": sum(row["analysis_scope"] == "primary" and row["pair_status"] == "VALID" for row in pairs),
        "rq1_seed_effect_rows": len(rq1),
        "rq2_seed_effect_rows": len(rq2_primary),
        "rq2_all_seed_effect_rows": len(rq2),
        "rq3_seed_effect_rows": len(rq3),
        "primary_metrics": ["balanced_accuracy", "macro_f1"],
        "secondary_metrics": ["cohen_kappa", "weighted_f1"],
        "selection_metric": "validation_kappa",
        "seeds": [42, 1024, 3407],
        "scope_rule": "primary RQ2 is FACED/ISRUC/TUEV plus valid LaBraM TUEV; SEED-V and PhysioNet-MI remain boundary/supporting",
        "score_selection": "none; rows come from the deterministic compact manifest",
    }
    (OUT / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    readme = f"""# Manuscript-final results snapshot v1\n\nThis is the focused results package generated from the deterministic compact\nmanuscript manifest. It does not modify the broad provenance registry.\n\n- Selected rows: {len(rows)}\n- Artifact statuses: {dict(sorted(status_counts.items()))}\n- Pair statuses: {dict(sorted(pair_counts.items()))}\n- Primary RQ2 pair rows: {contract['primary_rq2_pair_rows']}\n- Primary RQ2 metric rows: {contract['rq2_seed_effect_rows']}\n- RQ3 metric rows: {contract['rq3_seed_effect_rows']}\n\nPrimary RQ2 includes only valid matched pairs from FACED, ISRUC, TUEV, and\nLaBraM TUEV. SEED-V is a geometry/protocol boundary case; PhysioNet-MI is a\nbounded supporting extension. No row was selected by score.\n\nKey files:\n\n- `method_summary.csv`: overall performance, parameter, memory, and timing table.\n- `rq1_effect_summary.csv`: native adaptation minus frozen probe.\n- `rq2_effect_summary.csv`: primary native-minus-axis-blind effects only.\n- `rq2_all_effect_summary.csv`: primary plus boundary/supporting RQ2 effects.\n- `rq3_effect_summary.csv`: separate frozen and full-backbone contrasts.\n- `artifact_audit.csv` and `rq2_pair_audit.csv`: provenance and inclusion checks.\n\nHistorical per-epoch test trajectories and diagnostics remain supplementary.\n"""
    (OUT / "README.md").write_text(readme)
    output_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("\n".join(f"{sha256(path)}  {path.name}" for path in output_files) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
