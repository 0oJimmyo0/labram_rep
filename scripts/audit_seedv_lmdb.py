#!/usr/bin/env python3
"""Audit the legacy SEED-V LMDB once before paper-grade experiments."""

import argparse
import collections
import hashlib
import json
import os
import pickle
import re
import sys

import lmdb
import numpy as np


EXPECTED_SHAPE = (62, 1, 200)
LABEL_RANGE = range(5)
EXTREME_RAW_THRESHOLD = 10000.0  # absolute value > 100 after the fixed /100 scaling


def parse_key_metadata(key):
    key = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
    match = re.match(r"(?P<subject>[^_]+)_(?P<session>[^_]+)_[^/]+\.cnt-(?P<trial>\d+)-(?P<segment>\d+)$", key)
    if match is None:
        return {"key": key}
    return {
        "key": key,
        "subject": match.group("subject"),
        "session": match.group("session"),
        "trial": int(match.group("trial")),
        "segment": int(match.group("segment")),
    }


def key_digest(keys):
    digest = hashlib.sha256()
    for key in keys:
        key = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        digest.update(len(key).to_bytes(8, "big"))
        digest.update(key)
    return digest.hexdigest()


def split_composition(keys):
    subjects = set()
    sessions = set()
    trials = set()
    windows = collections.Counter()
    unparsed = 0
    for key in keys:
        metadata = parse_key_metadata(key)
        if "subject" not in metadata:
            unparsed += 1
            continue
        subjects.add(metadata["subject"])
        sessions.add(f"{metadata['subject']}/{metadata['session']}")
        trials.add(f"{metadata['subject']}/{metadata['session']}/{metadata['trial']}")
        windows[
            f"{metadata['subject']}/{metadata['session']}/{metadata['trial']}"
        ] += 1
    return {
        "subject_ids": sorted(subjects),
        "subject_count": len(subjects),
        "subject_session_ids": sorted(sessions),
        "subject_session_count": len(sessions),
        "subject_session_trial_count": len(trials),
        "windows_per_subject_session_trial": dict(sorted(windows.items())),
        "unparsed_key_count": unparsed,
    }


def audit_split(txn, keys, split):
    labels = collections.Counter()
    shape_counts = collections.Counter()
    dtype_counts = collections.Counter()
    finite = True
    total_values = 0
    abs_values = []
    max_abs = 0.0
    for index, key in enumerate(keys):
        encoded = key.encode("utf-8") if isinstance(key, str) else key
        raw = txn.get(encoded)
        if raw is None:
            raise KeyError(f"Missing {split} key {key!r}")
        record = pickle.loads(raw)
        X = np.asarray(record["sample"])
        Y = int(np.asarray(record["label"]).reshape(-1)[0])
        shape_counts[str(tuple(X.shape))] += 1
        dtype_counts[str(X.dtype)] += 1
        labels[str(Y)] += 1
        finite = finite and bool(np.isfinite(X).all())
        total_values += X.size
        max_abs = max(max_abs, float(np.max(np.abs(X))))
        if index < 32:
            abs_values.extend(np.abs(X).reshape(-1).tolist())
    if shape_counts != {str(EXPECTED_SHAPE): len(keys)}:
        raise ValueError(f"{split} contains unexpected shapes: {dict(shape_counts)}")
    if set(labels) - {str(label) for label in LABEL_RANGE}:
        raise ValueError(f"{split} contains labels outside [0, 4]: {dict(labels)}")
    if not finite:
        raise ValueError(f"{split} contains NaN or Inf")
    abs_values = np.asarray(abs_values, dtype=np.float64)
    return {
        "sample_count": len(keys),
        "key_sha256": key_digest(keys),
        "sample_shape": list(EXPECTED_SHAPE),
        "label_counts": dict(sorted(labels.items(), key=lambda item: int(item[0]))),
        "shape_counts": dict(shape_counts),
        "dtype_counts": dict(dtype_counts),
        "finite": finite,
        "max_abs": max_abs,
        "first_32_samples_abs_percentiles": {
            str(q): float(np.percentile(abs_values, q)) for q in (50, 95, 99)
        },
        "total_values": total_values,
    }


def audit_database(txn, split_index):
    """Audit all records in one sequential LMDB cursor pass."""
    split_by_key = {}
    for split, keys in split_index.items():
        for key in keys:
            encoded = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            if encoded in split_by_key:
                raise ValueError(f"Duplicate key appears in multiple split lists: {key!r}")
            split_by_key[encoded] = split

    stats = {}
    for split in ("train", "val", "test"):
        stats[split] = {
            "labels": collections.Counter(),
            "shape_counts": collections.Counter(),
            "dtype_counts": collections.Counter(),
            "finite": True,
            "sample_count": 0,
            "total_values": 0,
            "max_abs": 0.0,
            "abs_values": [],
            "sample_max": [],
            "sample_median_abs": [],
            "sample_p99_abs": [],
            "content_hashes": set(),
            "extreme_windows": [],
        }

    found = collections.Counter()
    for key, raw in txn.cursor():
        split = split_by_key.get(bytes(key))
        if split is None:
            continue
        record = pickle.loads(raw)
        if "sample" not in record or "label" not in record:
            raise ValueError(f"SEED-V record {key!r} must contain sample and label fields")
        X = np.asarray(record["sample"])
        label_values = np.asarray(record["label"]).reshape(-1)
        if label_values.size != 1:
            raise ValueError(f"SEED-V record {key!r} must contain one scalar label")
        label = int(label_values[0])
        current = stats[split]
        current["sample_count"] += 1
        current["shape_counts"][str(tuple(X.shape))] += 1
        current["dtype_counts"][str(X.dtype)] += 1
        current["labels"][str(label)] += 1
        current["finite"] = current["finite"] and bool(np.isfinite(X).all())
        current["total_values"] += X.size
        current["max_abs"] = max(current["max_abs"], float(np.max(np.abs(X))))
        current["sample_max"].append(float(np.max(np.abs(X))))
        current["sample_median_abs"].append(float(np.median(np.abs(X))))
        current["sample_p99_abs"].append(float(np.percentile(np.abs(X), 99)))
        current["content_hashes"].add(hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest())
        max_location = tuple(int(value) for value in np.unravel_index(np.argmax(np.abs(X)), X.shape))
        max_abs = float(np.max(np.abs(X)))
        if max_abs > EXTREME_RAW_THRESHOLD:
            current["extreme_windows"].append({
                **parse_key_metadata(key),
                "label": label,
                "max_abs_raw": max_abs,
                "max_abs_after_divisor_100": max_abs / 100.0,
                "median_abs_raw": float(np.median(np.abs(X))),
                "p99_abs_raw": float(np.percentile(np.abs(X), 99)),
                "max_channel_index": max_location[0],
                "max_patch_index": max_location[1],
                "max_sample_index": max_location[2],
                "max_value_raw": float(X[max_location]),
            })
        if len(current["abs_values"]) < 32 * int(np.prod(EXPECTED_SHAPE)):
            current["abs_values"].extend(np.abs(X).reshape(-1).tolist())
        found[split] += 1

    expected_counts = {split: len(split_index[split]) for split in ("train", "val", "test")}
    if dict(found) != expected_counts:
        raise ValueError(f"LMDB records missing from split index: found={dict(found)} expected={expected_counts}")

    output = {}
    content_sets = {}
    for split, current in stats.items():
        if current["shape_counts"] != {str(EXPECTED_SHAPE): expected_counts[split]}:
            raise ValueError(f"{split} contains unexpected shapes: {dict(current['shape_counts'])}")
        if set(current["labels"]) - {str(label) for label in LABEL_RANGE}:
            raise ValueError(f"{split} contains labels outside [0, 4]: {dict(current['labels'])}")
        if not current["finite"]:
            raise ValueError(f"{split} contains NaN or Inf")
        abs_values = np.asarray(current.pop("abs_values"), dtype=np.float64)
        sample_max = np.asarray(current.pop("sample_max"), dtype=np.float64)
        sample_median_abs = np.asarray(current.pop("sample_median_abs"), dtype=np.float64)
        sample_p99_abs = np.asarray(current.pop("sample_p99_abs"), dtype=np.float64)
        content_hashes = current.pop("content_hashes")
        extreme_windows = current.pop("extreme_windows")
        content_sets[split] = content_hashes

        def percentile_summary(values):
            return {
                str(q): float(np.percentile(values, q))
                for q in (50, 90, 95, 99, 99.9, 100)
            }

        output[split] = {
            "sample_count": current["sample_count"],
            "key_sha256": key_digest(split_index[split]),
            "sample_shape": list(EXPECTED_SHAPE),
            "label_counts": dict(sorted(current["labels"].items(), key=lambda item: int(item[0]))),
            "shape_counts": dict(current["shape_counts"]),
            "dtype_counts": dict(current["dtype_counts"]),
            "finite": current["finite"],
            "max_abs": current["max_abs"],
            "first_32_samples_abs_percentiles": {
                str(q): float(np.percentile(abs_values, q)) for q in (50, 95, 99)
            },
            "total_values": current["total_values"],
            "per_sample_amplitude": {
                "sample_max_abs": percentile_summary(sample_max),
                "sample_median_abs": percentile_summary(sample_median_abs),
                "sample_p99_abs": percentile_summary(sample_p99_abs),
                "sample_max_abs_above_threshold": {
                    str(threshold): int(np.sum(sample_max > threshold))
                    for threshold in (100, 1000, 10000)
                },
                "sample_max_abs_after_divisor_100": percentile_summary(sample_max / 100.0),
            },
            "extreme_windows_after_divisor_100_gt_100": sorted(
                extreme_windows, key=lambda item: item["max_abs_raw"], reverse=True
            ),
            "unique_sample_content_hashes": len(content_hashes),
        }

    content_overlaps = {
        f"{left}-{right}": len(content_sets[left] & content_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return output, content_overlaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reported-data-path", default="",
                        help="Original dataset path to record when auditing a local copy.")
    args = parser.parse_args()

    with lmdb.open(args.data_path, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
        raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError("LMDB is missing __keys__")
        split_index = pickle.loads(raw_keys)
        result = {
            "data_path": os.path.abspath(args.reported_data_path or args.data_path),
            "audit_input_path": os.path.abspath(args.data_path),
            "expected_shape": list(EXPECTED_SHAPE),
            "label_range": [0, 4],
            "splits": {},
            "split_composition": {
                split: split_composition(split_index[split])
                for split in ("train", "val", "test")
            },
        }
        for split in ("train", "val", "test"):
            if split not in split_index:
                raise KeyError(f"LMDB is missing split {split!r}")
        result["splits"], result["sample_content_hash_overlaps"] = audit_database(txn, split_index)
        if any(result["sample_content_hash_overlaps"].values()):
            raise ValueError(
                f"Exact sample-content overlap detected: {result['sample_content_hash_overlaps']}"
            )
        split_sets = {
            split: {
                key.encode("utf-8") if isinstance(key, str) else bytes(key)
                for key in split_index[split]
            }
            for split in ("train", "val", "test")
        }
        overlaps = {
            f"{left}-{right}": len(split_sets[left] & split_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        result["split_key_overlaps"] = overlaps
        if any(overlaps.values()):
            raise ValueError(f"SEED-V split key overlap detected: {overlaps}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
