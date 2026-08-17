#!/usr/bin/env python3
"""Build a conservative manuscript evidence manifest and paired effects.

This consumes the deterministic artifact registry but does not silently treat
every registry candidate as manuscript-ready.  It assigns each artifact a
protocol status and computes paired, same-seed differences for the comparisons
that answer the manuscript research questions.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "LaBraM" / "analysis" / "tmlr_registry"
OUT = REGISTRY
SEEDS = {42, 1024, 3407}
METRICS = ("balanced_accuracy", "cohen_kappa", "macro_f1", "weighted_f1")


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge_manual_review_state(rows):
    path = OUT / "manual_review_queue.csv"
    if not path.exists():
        return
    prior = {row.get("run_id"): row for row in read_csv(path)}
    fields = (
        "artifact_status", "manual_status", "reviewed_by", "review_date",
        "review_notes", "previous_status", "status_change_reason",
        "status_change_date", "exclusion_reason_code",
    )
    for row in rows:
        old = prior.get(row.get("run_id"))
        if old is None:
            continue
        for field in fields:
            if old.get(field, "") != "":
                row[field] = old[field]


def merge_pair_review_state(rows):
    path = OUT / "pair_review_queue.csv"
    if not path.exists():
        return
    keys = (
        "comparison", "backbone", "dataset", "axis", "seed",
        "left_artifact", "right_artifact",
    )
    prior = {tuple(row.get(key, "") for key in keys): row for row in read_csv(path)}
    fields = (
        "pair_artifacts_verified", "pair_review_status", "pair_reviewed_by",
        "pair_review_date", "pair_review_notes",
    )
    for row in rows:
        old = prior.get(tuple(row.get(key, "") for key in keys))
        if old is None:
            continue
        for field in fields:
            if old.get(field, "") != "":
                row[field] = old[field]


def as_bool(value):
    return str(value).lower() == "true"


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def group_key(row):
    return row["backbone"], row["dataset"], row["method"], row["axis"]


def scope_note(row):
    dataset = row["dataset"].lower()
    if dataset == "seedv":
        return "within-subject protocol; patch axis is singleton/degenerate"
    if dataset == "seED-v".lower():
        return "within-subject protocol; patch axis is singleton/degenerate"
    if dataset == "physionet-mi":
        return "serialized-only extension"
    if dataset == "isruc" and row["backbone"] == "LaBraM":
        return "axis-blind packet is a narrower patch-alignment comparator"
    return ""


def classify(rows):
    candidate_groups = defaultdict(list)
    for row in rows:
        if as_bool(row["candidate"]):
            candidate_groups[group_key(row)].append(row)

    for row in rows:
        seed = as_int(row["seed"])
        candidate = as_bool(row["candidate"])
        strict = as_bool(row["contract_valid"])
        group = candidate_groups[group_key(row)]
        group_seeds = {as_int(item["seed"]) for item in group}
        complete = group_seeds == SEEDS
        run_id = row["run_id"].lower()
        noisy = any(token in run_id for token in (
            "smoke", "audit", "boundary", "stability15", "headdropout",
            "corrected_gate", "repair", "source_fidelity", "pilot", "dev",
        ))

        if candidate and strict and complete:
            status = "PRIMARY"
            reason = "strict canonical three-seed group"
        elif candidate and strict:
            status = "SUPPORTING"
            reason = "strict artifact but canonical group is incomplete"
        elif seed in SEEDS and not noisy:
            status = "SUPPORTING"
            reason = "completed legacy or non-strict artifact; manual audit required"
        elif seed in SEEDS:
            status = "PILOT"
            reason = "completed exploratory artifact outside locked packet"
        else:
            status = "EXCLUDED"
            reason = "seed is outside the locked manuscript seed set"

        # This is an automatic, provisional classification only.  It is kept
        # separately from the human adjudication fields below.
        row["evidence_status"] = status
        row["candidate_status"] = {
            "PRIMARY": "PRIMARY_CANDIDATE",
            "SUPPORTING": "SUPPORTING_CANDIDATE",
            "PILOT": "PILOT_CANDIDATE",
            "EXCLUDED": "EXCLUDED_CANDIDATE",
        }[status]
        row["artifact_status"] = "UNREVIEWED"
        row["manual_status"] = "UNREVIEWED"
        row["reviewed_by"] = ""
        row["review_date"] = ""
        row["review_notes"] = ""
        row["exclusion_reason_code"] = "UNREVIEWED"
        row["source_log_path"] = row["artifact_dir"]
        row["previous_status"] = ""
        row["status_change_reason"] = ""
        row["status_change_date"] = ""
        row["evidence_reason"] = reason
        row["scope_note"] = scope_note(row)
        row["seed_int"] = seed
        row["canonical_group_n"] = len(group)
        valid_group_seeds = sorted(x for x in group_seeds if x is not None)
        row["canonical_group_seeds"] = ",".join(str(x) for x in valid_group_seeds)
        row["alignment_control_available"] = "unknown"
        row["probe_available"] = "unknown"
        row["rq1_eligible"] = "no"
        row["rq2_eligible"] = "no"
        row["rq3_eligible"] = "no"
        row["rq4_eligible"] = "yes" if status != "EXCLUDED" else "no"


def pair_artifact_candidate(row):
    """Use manual PRIMARY decisions once available; otherwise use provisional candidates."""
    if row.get("manual_status") != "UNREVIEWED":
        return row.get("manual_status") == "PRIMARY"
    return row.get("evidence_status") == "PRIMARY"


def primary_index(rows):
    exact = {}
    broad = defaultdict(list)
    for row in rows:
        if not pair_artifact_candidate(row):
            continue
        exact[(row["backbone"], row["dataset"], row["axis"], row["seed_int"], row["method"])] = row
        broad[(row["backbone"], row["dataset"], row["seed_int"], row["method"])].append(row)
    return exact, broad


def native_control_keys(row):
    """Return the alignment-control method and probe method for a native row."""
    if row["backbone"] == "CBraMod" and row["method"] == "interaction_aligned":
        return "axis_blind", "frozen_probe"
    if row["backbone"] == "LaBraM" and row["method"].startswith("frozen_native_"):
        return "axis_blind", "frozen_probe"
    return None, None


def is_frozen_regime(row):
    method = row["method"]
    return method in {
        "interaction_aligned", "frozen_probe", "axis_blind",
        "frozen_native_channel", "frozen_native_patch",
        "frozen_native_channel_patch", "frozen_native_generic",
    }


def parameter_match_pct(left, right):
    left_value = as_float(left.get("trainable_parameters"))
    right_value = as_float(right.get("trainable_parameters"))
    if left_value is None or right_value is None or right_value == 0:
        return None
    return abs(left_value - right_value) / abs(right_value) * 100.0


def annotate_alignment_coverage(rows):
    exact, broad = primary_index(rows)
    for row in rows:
        control, probe = native_control_keys(row)
        if control is None:
            continue
        exact_key = (row["backbone"], row["dataset"], row["axis"], row["seed_int"])
        broad_key = (row["backbone"], row["dataset"], row["seed_int"])
        has_control = ((exact_key + (control,)) in exact or
                       bool(broad.get(broad_key + (control,))))
        has_probe = ((exact_key + (probe,)) in exact or
                     bool(broad.get(broad_key + (probe,))))
        if has_control:
            row["alignment_control_available"] = "primary_pair" if pair_artifact_candidate(row) else "control_only"
        else:
            row["alignment_control_available"] = "no_primary_control"
        row["probe_available"] = "yes" if has_probe else "no_primary_probe"


def pair_rows(rows, left_method, right_method, comparison):
    exact, broad = primary_index(rows)

    pairs = []
    left_rows = [row for row in rows if pair_artifact_candidate(row) and row["method"] in left_method]
    for left in left_rows:
        right_name = right_method(left) if callable(right_method) else right_method
        exact_key = (left["backbone"], left["dataset"], left["axis"], left["seed_int"], right_name)
        broad_key = (left["backbone"], left["dataset"], left["seed_int"], right_name)
        right = exact.get(exact_key)
        if right is None:
            controls = broad.get(broad_key, [])
            right = controls[0] if controls else None
        if right is None:
            continue
        match_pct = parameter_match_pct(left, right)
        same_regime = is_frozen_regime(left) == is_frozen_regime(right)
        if comparison == "aligned_vs_axis_blind" and not same_regime:
            pair_validity = "PAIR_STRUCTURALLY_INVALID"
            invalid_reason = "frozen/trainable regime mismatch"
            pair_contract_valid = "False"
        elif match_pct is not None and comparison == "aligned_vs_axis_blind" and match_pct > 5.0:
            pair_validity = "PAIR_STRUCTURALLY_INVALID"
            invalid_reason = "parameter mismatch exceeds 5 percent"
            pair_contract_valid = "False"
        else:
            pair_validity = "PAIR_ARTIFACT_UNVERIFIED"
            invalid_reason = "full configuration-level nuisance matching not yet manually confirmed"
            pair_contract_valid = "True"
        for metric in METRICS:
            left_value = as_float(left.get(metric))
            right_value = as_float(right.get(metric))
            if left_value is None or right_value is None:
                continue
            pairs.append({
                "comparison": comparison,
                "backbone": left["backbone"],
                "dataset": left["dataset"],
                "axis": left["axis"],
                "seed": left["seed_int"],
                "left_method": left["method"],
                "right_method": right["method"],
                "metric": metric,
                "left_value": left_value,
                "right_value": right_value,
                "effect_left_minus_right": left_value - right_value,
                "left_better": int(left_value > right_value),
                "parameter_match_pct": match_pct,
                "pair_validity": pair_validity,
                "pair_contract_valid": pair_contract_valid,
                "pair_artifacts_verified": "False",
                "pair_invalid_reason": invalid_reason,
                "left_artifact": left["run_id"],
                "right_artifact": right["run_id"],
                "scope_note": left["scope_note"] or right["scope_note"],
            })
    return pairs


def make_pairs(rows):
    pairs = []
    pairs.extend(pair_rows(rows, {"interaction_aligned"}, "axis_blind", "aligned_vs_axis_blind"))
    pairs.extend(pair_rows(rows, {"frozen_native_channel", "frozen_native_patch", "frozen_native_channel_patch"}, "axis_blind", "aligned_vs_axis_blind"))
    pairs.extend(pair_rows(rows, {"interaction_aligned"}, "frozen_probe", "aligned_vs_frozen_probe"))
    pairs.extend(pair_rows(rows, {"frozen_native_channel", "frozen_native_patch", "frozen_native_channel_patch"}, "frozen_probe", "aligned_vs_frozen_probe"))

    def full_method(left):
        if left["backbone"] == "CBraMod" and left["method"] == "interaction_aligned":
            return "native_full_finetune"
        if left["backbone"] == "LaBraM" and left["method"].startswith("frozen_native_"):
            return left["method"].replace("frozen_", "", 1)
        return "__none__"

    pairs.extend(pair_rows(rows, {"interaction_aligned", "frozen_native_channel", "frozen_native_patch", "frozen_native_channel_patch"}, full_method, "frozen_aligned_vs_full_native"))
    return pairs


def summarize_pairs(pairs):
    groups = defaultdict(list)
    for row in pairs:
        key = (row["comparison"], row["backbone"], row["dataset"], row["axis"], row["metric"])
        groups[key].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        values = [row["effect_left_minus_right"] for row in group]
        parameter_values = [row["parameter_match_pct"] for row in group if row["parameter_match_pct"] is not None]
        summaries.append({
            "comparison": key[0],
            "backbone": key[1],
            "dataset": key[2],
            "axis": key[3],
            "metric": key[4],
            "n": len(values),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda item: item["seed"])),
            "effect_mean": statistics.mean(values),
            "effect_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "positive_seeds": sum(row["left_better"] for row in group),
            "effect_min": min(values),
            "effect_max": max(values),
            "scope_note": next((row["scope_note"] for row in group if row["scope_note"]), ""),
            "pair_validity": ";".join(sorted({row["pair_validity"] for row in group})),
            "pair_invalid_reason": ";".join(sorted({row["pair_invalid_reason"] for row in group})),
            "parameter_match_pct_mean": statistics.mean(parameter_values) if parameter_values else None,
        })
    return summaries


def annotate_rq_eligibility(rows, pairs):
    by_artifact = defaultdict(list)
    for pair in pairs:
        item = (pair["comparison"], pair["pair_validity"])
        by_artifact[pair["left_artifact"]].append(item)
        by_artifact[pair["right_artifact"]].append(item)
    for row in rows:
        comparisons = defaultdict(list)
        for comparison, validity in by_artifact.get(row["run_id"], []):
            comparisons[comparison].append(validity)
        if "aligned_vs_frozen_probe" in comparisons:
            row["rq1_eligible"] = (
                "PROVISIONAL" if any(v != "PAIR_STRUCTURALLY_INVALID" for v in comparisons["aligned_vs_frozen_probe"])
                else "no"
            )
        if "aligned_vs_axis_blind" in comparisons:
            row["rq2_eligible"] = (
                "PROVISIONAL" if any(v != "PAIR_STRUCTURALLY_INVALID" for v in comparisons["aligned_vs_axis_blind"])
                else "no"
            )
        if "frozen_aligned_vs_full_native" in comparisons:
            row["rq3_eligible"] = (
                "PROVISIONAL" if any(v != "PAIR_STRUCTURALLY_INVALID" for v in comparisons["frozen_aligned_vs_full_native"])
                else "no"
            )


def annotate_pair_status(rows, pairs):
    by_artifact = defaultdict(list)
    for pair in pairs:
        if pair["comparison"] != "aligned_vs_axis_blind":
            continue
        by_artifact[pair["left_artifact"]].append(pair["pair_validity"])
        by_artifact[pair["right_artifact"]].append(pair["pair_validity"])
    for row in rows:
        statuses = by_artifact.get(row["run_id"], [])
        if not statuses:
            continue
        if any(status == "PAIR_ARTIFACT_UNVERIFIED" for status in statuses):
            row["alignment_control_available"] = "primary_pair_unverified"
        else:
            row["alignment_control_available"] = "primary_pair_structurally_invalid"


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_pair_review_queue(pairs):
    unique = {}
    for pair in pairs:
        key = (
            pair["comparison"], pair["backbone"], pair["dataset"], pair["axis"],
            pair["seed"], pair["left_artifact"], pair["right_artifact"],
        )
        if key in unique:
            continue
        unique[key] = {
            "comparison": pair["comparison"],
            "backbone": pair["backbone"],
            "dataset": pair["dataset"],
            "axis": pair["axis"],
            "seed": pair["seed"],
            "left_method": pair["left_method"],
            "right_method": pair["right_method"],
            "left_artifact": pair["left_artifact"],
            "right_artifact": pair["right_artifact"],
            "parameter_match_pct": pair["parameter_match_pct"],
            "pair_contract_valid": pair["pair_contract_valid"],
            "pair_validity": pair["pair_validity"],
            "pair_artifacts_verified": "False",
            "pair_review_status": (
                "NOT_ELIGIBLE" if pair["pair_validity"] == "PAIR_STRUCTURALLY_INVALID" else "UNREVIEWED"
            ),
            "pair_reviewed_by": "",
            "pair_review_date": "",
            "pair_review_notes": "",
        }
    return list(unique.values())


def write_summary(rows, pairs, summaries):
    counts = Counter(row["evidence_status"] for row in rows)
    coverage = Counter()
    for row in rows:
        if row["method"] in {"interaction_aligned", "frozen_native_channel", "frozen_native_patch", "frozen_native_channel_patch"}:
            coverage[(row["backbone"], row["dataset"], row["axis"], row["alignment_control_available"])] += 1

    lines = [
        "# TMLR evidence manifest",
        "",
        "This manifest is a conservative manuscript-audit layer over the deterministic artifact registry.",
        "It does not promote legacy or incomplete artifacts into the primary evidence set automatically.",
        "",
        f"Artifacts: {len(rows)}",
        f"Automatic candidates — primary: {counts['PRIMARY']}; supporting: {counts['SUPPORTING']}; pilot: {counts['PILOT']}; excluded: {counts['EXCLUDED']}",
        "Manual artifact status: UNREVIEWED for every row; automatic candidate labels are not manuscript adjudications.",
        f"Paired metric effects: {len(pairs)} rows across {len(summaries)} summaries",
        f"Structurally invalid paired-effect rows: {sum(row['pair_validity'] == 'PAIR_STRUCTURALLY_INVALID' for row in pairs)}; these are excluded from provisional RQ2 eligibility.",
        "",
        "## Alignment-control coverage",
        "",
        "| Backbone | Dataset | Axis | Control status | Artifacts |",
        "|---|---|---|---|---:|",
    ]
    for key, count in sorted(coverage.items()):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {count} |")
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "The strongest alignment claim is restricted to manually validated RQ2 pairs. Current paired effects are provisional and require configuration-level nuisance matching before use in the manuscript.",
        "",
        "## Main paired-effect files",
        "",
        "- `paired_effects.csv`: same-seed metric differences.",
        "- `paired_effects_summary.csv`: mean, standard deviation, seed count, and positive-seed count.",
        "- `manual_review_queue.csv`: candidate rows prepared for human adjudication.",
        "- `pair_review_queue.csv`: pair-level contract review required before final RQ2 analysis.",
    ])
    (OUT / "evidence_manifest_summary.md").write_text("\n".join(lines) + "\n")


def main():
    rows = read_csv(REGISTRY / "all_artifacts.csv")
    classify(rows)
    merge_manual_review_state(rows)
    annotate_alignment_coverage(rows)
    pairs = make_pairs(rows)
    annotate_pair_status(rows, pairs)
    annotate_rq_eligibility(rows, pairs)
    summaries = summarize_pairs(pairs)

    manifest_fields = [
        "backbone", "dataset", "run_id", "artifact_dir", "method", "axis", "seed",
        "candidate", "candidate_score", "contract_valid", "status", "selected_epoch",
        "val_selection_value", *METRICS, "trainable_parameters", "adapter_delta_ratio",
        "backbone_update_norm", "evidence_status", "candidate_status", "artifact_status",
        "manual_status", "evidence_reason", "scope_note",
        "canonical_group_n", "canonical_group_seeds", "alignment_control_available",
        "probe_available", "rq1_eligible", "rq2_eligible", "rq3_eligible", "rq4_eligible",
        "reviewed_by", "review_date", "review_notes", "source_log_path",
        "previous_status", "status_change_reason", "status_change_date", "exclusion_reason_code", "source",
    ]
    # Keep the original seed column for traceability and remove only internal helpers.
    write_csv(OUT / "evidence_manifest.csv", rows, manifest_fields)
    pair_fields = list(pairs[0].keys()) if pairs else ["comparison"]
    write_csv(OUT / "paired_effects.csv", pairs, pair_fields)
    summary_fields = list(summaries[0].keys()) if summaries else ["comparison"]
    write_csv(OUT / "paired_effects_summary.csv", summaries, summary_fields)
    review_fields = [
        "backbone", "dataset", "run_id", "artifact_dir", "method", "axis", "seed",
        "candidate_status", "artifact_status", "manual_status", "evidence_reason",
        "contract_valid", "selected_epoch", "val_selection_value", *METRICS,
        "trainable_parameters", "source_log_path", "reviewed_by", "review_date",
        "review_notes", "previous_status", "status_change_reason", "status_change_date",
        "exclusion_reason_code",
    ]
    review_rows = [row for row in rows if as_bool(row["candidate"])]
    write_csv(OUT / "manual_review_queue.csv", review_rows, review_fields)
    pair_review_rows = make_pair_review_queue(pairs)
    merge_pair_review_state(pair_review_rows)
    pair_review_fields = list(pair_review_rows[0].keys()) if pair_review_rows else ["comparison"]
    write_csv(OUT / "pair_review_queue.csv", pair_review_rows, pair_review_fields)
    write_summary(rows, pairs, summaries)
    print(f"Wrote {len(rows)} manifest rows, {len(review_rows)} review rows, {len(pairs)} paired effects, and {len(summaries)} summaries to {OUT}")


if __name__ == "__main__":
    main()
