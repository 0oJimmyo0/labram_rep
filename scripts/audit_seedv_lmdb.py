#!/usr/bin/env python3
"""Audit the legacy SEED-V LMDB once before paper-grade experiments."""

import argparse
import collections
import hashlib
import json
import os
import pickle
import sys

import lmdb
import numpy as np


EXPECTED_SHAPE = (62, 1, 200)
LABEL_RANGE = range(5)


def key_digest(keys):
    digest = hashlib.sha256()
    for key in keys:
        key = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        digest.update(len(key).to_bytes(8, "big"))
        digest.update(key)
    return digest.hexdigest()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with lmdb.open(args.data_path, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
        raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError("LMDB is missing __keys__")
        split_index = pickle.loads(raw_keys)
        result = {
            "data_path": os.path.abspath(args.data_path),
            "expected_shape": list(EXPECTED_SHAPE),
            "label_range": [0, 4],
            "splits": {},
        }
        for split in ("train", "val", "test"):
            if split not in split_index:
                raise KeyError(f"LMDB is missing split {split!r}")
            result["splits"][split] = audit_split(txn, split_index[split], split)
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
