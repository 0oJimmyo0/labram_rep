#!/usr/bin/env python3
"""Freeze an adjudicated TMLR evidence snapshot.

The command intentionally refuses to create a final snapshot while candidate
artifacts or RQ2 pair reviews remain UNREVIEWED.  The current registry is
therefore expected to fail this check until the manual review queues are
completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("balanced_accuracy", "cohen_kappa", "macro_f1", "weighted_f1")


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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message):
    raise SystemExit(f"FINALIZATION BLOCKED: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[1] / "analysis" / "tmlr_registry")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    registry = args.registry
    output = args.output_root or registry / f"frozen_{args.version}"
    manifest = read_csv(registry / "evidence_manifest.csv")
    candidates = [row for row in manifest if row.get("candidate", "").lower() == "true"]
    unreviewed = [row for row in candidates if row.get("manual_status") == "UNREVIEWED"]
    if unreviewed:
        fail(f"{len(unreviewed)} candidate artifacts still have manual_status=UNREVIEWED; review manual_review_queue.csv first")

    pairs = read_csv(registry / "paired_effects.csv")
    pair_review = read_csv(registry / "pair_review_queue.csv")
    pair_review_index = {
        (row["comparison"], row["backbone"], row["dataset"], row["axis"], row["seed"], row["left_artifact"], row["right_artifact"]): row
        for row in pair_review
    }
    artifact_status = {row["run_id"]: row.get("manual_status") for row in manifest}

    final_pairs = []
    blocked_pairs = []
    for pair in pairs:
        left_status = artifact_status.get(pair["left_artifact"])
        right_status = artifact_status.get(pair["right_artifact"])
        key = (pair["comparison"], pair["backbone"], pair["dataset"], pair["axis"], pair["seed"], pair["left_artifact"], pair["right_artifact"])
        review = pair_review_index.get(key)
        eligible = left_status == "PRIMARY" and right_status == "PRIMARY"
        if pair["comparison"] == "aligned_vs_axis_blind":
            eligible = eligible and pair.get("pair_validity") == "PAIR_ARTIFACT_UNVERIFIED"
            eligible = eligible and review is not None and review.get("pair_review_status") == "VALID"
            eligible = eligible and review.get("pair_artifacts_verified") == "True"
        if eligible:
            final_pairs.append(pair)
        elif pair["comparison"] == "aligned_vs_axis_blind":
            blocked_pairs.append(pair)

    if blocked_pairs:
        unverified = sum(
            1 for pair in blocked_pairs
            if pair.get("pair_validity") == "PAIR_ARTIFACT_UNVERIFIED"
        )
        fail(f"{len(blocked_pairs)} RQ2 pairs are not final-eligible ({unverified} remain unverified or lack pair-level VALID review)")

    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / f"evidence_manifest_{args.version}.csv"
    pair_path = output / f"paired_effects_{args.version}.csv"
    review_path = output / f"pair_review_{args.version}.csv"
    write_csv(manifest_path, manifest)
    write_csv(pair_path, final_pairs)
    write_csv(review_path, pair_review)

    contract = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(registry),
        "candidate_artifacts": len(candidates),
        "final_pairs": len(final_pairs),
        "rq2_rule": "both artifacts manually PRIMARY; pair contract valid; pair review VALID; pair artifacts verified",
        "parameter_match_tolerance_percent": 5.0,
        "metrics": list(METRICS),
    }
    contract_path = output / f"analysis_contract_{args.version}.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    review_log_path = output / f"review_log_{args.version}.csv"
    write_csv(review_log_path, candidates)

    hash_path = output / "manifest_sha256.txt"
    hashed = [manifest_path, pair_path, review_path, contract_path, review_log_path]
    hash_path.write_text("\n".join(f"{sha256(path)}  {path.name}" for path in hashed) + "\n")
    print(f"Frozen {args.version}: {len(candidates)} reviewed candidates and {len(final_pairs)} final paired rows in {output}")


if __name__ == "__main__":
    main()
