#!/usr/bin/env python3
"""Build the small, claims-first manuscript inclusion manifest.

The full TMLR registry remains the provenance layer.  This script applies a
fixed role rule to complete canonical three-seed groups and does not select
results by score.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from selfcheck_manuscript_results import canonical_config


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "LaBraM" / "analysis" / "tmlr_registry"
OUT = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript"
SEEDS = {42, 1024, 3407}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        path.write_text("\n")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_key(row):
    return row["backbone"], row["dataset"], row["method"], row["axis"]


def homogeneous_configuration(group):
    """Require the same semantic training configuration at every seed."""
    try:
        signatures = {
            repr(sorted(canonical_config(row).items()))
            for row in group
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return len(signatures) == 1


def complete_candidate_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("candidate", "").lower() != "true":
            continue
        groups[group_key(row)].append(row)
    return {
        key: sorted(group, key=lambda row: as_int(row["seed"]) or -1)
        for key, group in groups.items()
        if {as_int(row["seed"]) for row in group} == SEEDS
        and homogeneous_configuration(group)
    }


def choose_group(groups, backbone, dataset, method, axes):
    for axis in axes:
        key = (backbone, dataset, method, axis)
        if key in groups:
            return key
    return None


def build_pair_index(pairs):
    index = {}
    for row in pairs:
        if row.get("comparison") != "aligned_vs_axis_blind":
            continue
        key = (row["left_artifact"], row["right_artifact"])
        index[key] = {
            "control_run_id": row["right_artifact"],
            "native_run_id": row["left_artifact"],
            "paired_variant": row["axis"],
            "pair_valid": row["pair_validity"],
            "parameter_match_pct": row.get("parameter_match_pct"),
        }
        reverse = (row["right_artifact"], row["left_artifact"])
        index[reverse] = {
            "control_run_id": row["left_artifact"],
            "native_run_id": row["right_artifact"],
            "paired_variant": row["axis"],
            "pair_valid": row["pair_validity"],
            "parameter_match_pct": row.get("parameter_match_pct"),
        }
    return index


def main():
    rows = read_csv(REGISTRY / "evidence_manifest.csv")
    pairs = read_csv(REGISTRY / "paired_effects.csv")
    groups = complete_candidate_groups(rows)
    pair_index = build_pair_index(pairs)
    selected = {}

    def add_group(key, *roles):
        if key is None or key not in groups:
            return
        for row in groups[key]:
            item = selected.setdefault(row["run_id"], {"row": row, "roles": set(), "selection_rules": set()})
            item["roles"].update(roles)
            item["selection_rules"].add(f"complete_three_seed:{key[2]}:{key[3]}")

    datasets = sorted({(row["backbone"], row["dataset"]) for row in rows})
    for backbone, dataset in datasets:
        if backbone == "CBraMod":
            native = choose_group(groups, backbone, dataset, "interaction_aligned", ("channel_patch", "channel", "patch"))
            probe_method = "frozen_probe"
            axisblind_method = "axis_blind"
            generic_method = "generic_bottleneck"
            lora_method = "lora"
            upper_method = "upper_k_finetune"
            dense_method = "full_finetune"
            full_native_method = "native_full_finetune"
        else:
            native = choose_group(groups, backbone, dataset, "frozen_native_channel_patch", ("channel_patch",))
            if native is None:
                native = choose_group(groups, backbone, dataset, "frozen_native_channel", ("channel",))
            if native is None:
                native = choose_group(groups, backbone, dataset, "frozen_native_patch", ("patch",))
            probe_method = "frozen_probe"
            axisblind_method = "axis_blind"
            generic_method = "frozen_native_generic"
            lora_method = "lora"
            upper_method = "upper2"
            dense_method = "dense"
            full_native_method = None

        native_axis = native[3] if native else None
        probe = choose_group(groups, backbone, dataset, probe_method, ("none",))
        axisblind = choose_group(groups, backbone, dataset, axisblind_method, ("generic_token_control",))
        generic = choose_group(groups, backbone, dataset, generic_method, ("none", "generic"))
        lora = choose_group(groups, backbone, dataset, lora_method, ("none", "qkv"))
        upper = choose_group(groups, backbone, dataset, upper_method, ("none", "upper"))
        dense = choose_group(groups, backbone, dataset, dense_method, ("none",))

        # RQ1: native adaptation versus a frozen probe.
        add_group(native, "RQ1")
        add_group(probe, "RQ1")

        # RQ2: only retain the exact native/control pair when it passes the
        # automatic parameter contract.  Pair review remains visible as an
        # audit_status and is required before final claims.
        if native and axisblind:
            native_rows = groups[native]
            axis_rows = groups[axisblind]
            pair_ok = True
            for native_row, axis_row in zip(native_rows, axis_rows):
                pair = pair_index.get((native_row["run_id"], axis_row["run_id"]))
                if pair is None or pair["pair_valid"] == "PAIR_STRUCTURALLY_INVALID":
                    pair_ok = False
                    break
            if pair_ok:
                add_group(native, "RQ2")
                add_group(axisblind, "RQ2")

        # RQ3/context: fixed controls and backbone-trainable references.
        add_group(generic, "CONTEXT")
        add_group(lora, "CONTEXT")
        add_group(upper, "RQ3")
        add_group(dense, "RQ3")
        if backbone == "CBraMod" and native_axis:
            add_group((backbone, dataset, full_native_method, native_axis), "RQ3")
        elif backbone == "LaBraM" and native_axis:
            full_key = (backbone, dataset, "native_" + native_axis, native_axis)
            add_group(full_key if full_key in groups else None, "RQ3")

        # Prespecified boundary cases remain in the compact manifest even if
        # they are not clean RQ2 cells.
        if dataset.lower() in {"seedv", "seed-v", "physionet_mi", "physionet-mi"}:
            for key in (native, probe, dense, upper):
                add_group(key, "BOUNDARY")

    output_rows = []
    row_by_id = {row["run_id"]: row for row in rows}
    for run_id, item in sorted(selected.items()):
        row = item["row"]
        roles = sorted(item["roles"])
        pair_options = []
        for pair_row in pairs:
            if pair_row.get("comparison") != "aligned_vs_axis_blind":
                continue
            if pair_row.get("left_artifact") == run_id or pair_row.get("right_artifact") == run_id:
                candidate_pair = pair_index.get((pair_row["left_artifact"], pair_row["right_artifact"]))
                if candidate_pair is not None:
                    pair_options.append(candidate_pair)
        pair = next(
            (candidate for candidate in pair_options if candidate["pair_valid"] != "PAIR_STRUCTURALLY_INVALID"),
            pair_options[0] if pair_options else None,
        )
        output_rows.append({
            "backbone": row["backbone"],
            "dataset": row["dataset"],
            "method": row["method"],
            "variant": row["axis"],
            "seed": row["seed"],
            "run_id": run_id,
            "source_artifact": row["artifact_dir"],
            "commit": "",
            "evidence_tier": "PRIMARY_CANDIDATE",
            "manuscript_role": ";".join(roles),
            "paired_control_run_id": pair["control_run_id"] if pair else "",
            "paired_variant": pair["paired_variant"] if pair else "",
            "pair_valid": pair["pair_valid"] if pair else "",
            "parameter_match_pct": pair["parameter_match_pct"] if pair else "",
            "trainability_regime": "frozen" if row["method"].startswith("frozen_") or row["method"] in {"interaction_aligned", "axis_blind", "frozen_probe"} else "trainable",
            "trainable_parameters": row["trainable_parameters"],
            "selector_metric": "validation_kappa",
            "selected_epoch": row["selected_epoch"],
            "balanced_accuracy": row["balanced_accuracy"],
            "macro_f1": row["macro_f1"],
            "kappa": row["cohen_kappa"],
            "weighted_f1": row["weighted_f1"],
            "audit_status": row.get("manual_status", "UNREVIEWED"),
            "audit_notes": row.get("review_notes", ""),
            "selection_rule": ";".join(sorted(item["selection_rules"])),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "manuscript_inclusion_manifest.csv", output_rows)

    summary = defaultdict(int)
    for row in output_rows:
        for role in row["manuscript_role"].split(";"):
            summary[(row["backbone"], row["dataset"], role)] += 1
    summary_rows = [
        {"backbone": key[0], "dataset": key[1], "role": key[2], "rows": count}
        for key, count in sorted(summary.items())
    ]
    write_csv(OUT / "manuscript_inclusion_summary.csv", summary_rows)

    md = [
        "# Manuscript inclusion manifest",
        "",
        "Generated deterministically from complete canonical three-seed groups.",
        "Selection is based on prespecified manuscript role, not score.",
        "",
        f"Selected run rows: {len(output_rows)}",
        f"Selected experimental groups: {len({(r['backbone'], r['dataset'], r['method'], r['variant']) for r in output_rows})}",
        "",
        "The rows remain `PRIMARY_CANDIDATE` and require focused audit before final manuscript use.",
        "The full 483-artifact registry remains the provenance layer.",
        "",
        "| Backbone | Dataset | Role | Rows |",
        "|---|---|---|---:|",
    ]
    md.extend(f"| {row['backbone']} | {row['dataset']} | {row['role']} | {row['rows']} |" for row in summary_rows)
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {len(output_rows)} run rows to {OUT / 'manuscript_inclusion_manifest.csv'}")


if __name__ == "__main__":
    main()
