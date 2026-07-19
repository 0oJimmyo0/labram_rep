#!/usr/bin/env python3
"""Strict audit for the serialized ISRUC sequence contract."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


EXPECTED_BIPOLAR = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
EXPECTED_LABELS = {0, 1, 2, 3, 4}
RAW_TO_LABEL = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}
FILE_RE = re.compile(r"^ISRUC-group1-(?P<subject>\d+)-(?P<index>\d+)\.npy$")


def numeric_files(directory: Path):
    result = {}
    for path in directory.glob("*.npy"):
        match = FILE_RE.match(path.name)
        if match is None:
            raise AssertionError(f"Unexpected ISRUC filename: {path}")
        index = int(match.group("index"))
        if index in result:
            raise AssertionError(f"Duplicate numeric sequence index in {directory}: {index}")
        result[index] = path
    return result


def inspect_processed(root: Path, edf_root: Path):
    seq_root = root / "seq"
    label_root = root / "labels"
    if not seq_root.is_dir() or not label_root.is_dir():
        raise AssertionError(f"Expected {seq_root} and {label_root}")

    subjects = list(range(1, 101))
    discarded = {}
    total = {"train": 0, "val": 0, "test": 0}
    split_subjects = {
        "train": set(range(1, 81)),
        "val": set(range(81, 91)),
        "test": set(range(91, 101)),
    }
    split_sets = {name: set() for name in split_subjects}
    for subject in subjects:
        seq_dir = seq_root / f"ISRUC-group1-{subject}"
        label_dir = label_root / f"ISRUC-group1-{subject}"
        if not seq_dir.is_dir() or not label_dir.is_dir():
            raise AssertionError(f"Missing subject directory for subject {subject}")
        signals = numeric_files(seq_dir)
        labels = numeric_files(label_dir)
        if signals.keys() != labels.keys():
            raise AssertionError(f"Signal/label numeric indices differ for subject {subject}")
        split = "train" if subject <= 80 else "val" if subject <= 90 else "test"
        split_sets[split].add(subject)
        total[split] += len(signals)
        saved_labels = []
        for index in sorted(signals):
            signal = np.load(signals[index], mmap_mode="r")
            label = np.load(labels[index], mmap_mode="r")
            if tuple(signal.shape) != (20, 6, 6000):
                raise AssertionError(f"{signals[index]} has shape {signal.shape}")
            if tuple(label.shape) != (20,):
                raise AssertionError(f"{labels[index]} has shape {label.shape}")
            if not np.isfinite(signal).all():
                raise AssertionError(f"Non-finite signal values in {signals[index]}")
            values = set(np.asarray(label).reshape(-1).tolist())
            if not values.issubset(EXPECTED_LABELS):
                raise AssertionError(f"Unexpected remapped labels in {labels[index]}: {values}")
            saved_labels.extend(np.asarray(label).reshape(-1).tolist())
        annotation_path = edf_root / str(subject) / f"{subject}_1.txt"
        if not annotation_path.is_file():
            raise AssertionError(f"Missing first-expert annotation file: {annotation_path}")
        raw_lines = [line.strip() for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        invalid_raw = sorted(set(raw_lines) - set(RAW_TO_LABEL))
        if invalid_raw:
            raise AssertionError(f"Unexpected raw labels in {annotation_path}: {invalid_raw}")
        raw_label_count = len(raw_lines)
        saved_label_count = len(signals) * 20
        if raw_label_count < saved_label_count or raw_label_count - saved_label_count >= 20:
            raise AssertionError(
                f"Subject {subject} label truncation is invalid: raw={raw_label_count}, saved={saved_label_count}"
            )
        expected_labels = [RAW_TO_LABEL[value] for value in raw_lines[:saved_label_count]]
        if saved_labels != expected_labels:
            raise AssertionError(f"Serialized labels do not match {annotation_path} after raw mapping/truncation")
        discarded[subject] = raw_label_count - saved_label_count

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_sets[left] & split_sets[right]:
            raise AssertionError(f"Subject overlap between {left} and {right}")
    return {"subjects": subjects, "sequence_counts": total, "discarded_epochs": discarded}


def inspect_edf_channels(edf_root: Path):
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("EDF channel audit requires mne") from exc

    observed = {}
    for subject in range(1, 101):
        rec = edf_root / str(subject) / f"{subject}.rec"
        if not rec.is_file():
            raise AssertionError(f"Missing raw ISRUC recording: {rec}")
        raw = mne.io.read_raw_edf(rec, preload=False, verbose="ERROR")
        names = [str(name).strip().upper() for name in raw.ch_names]
        selected = names[2:8]
        if selected != EXPECTED_BIPOLAR:
            raise AssertionError(
                f"Subject {subject} positional columns 2:8 are {selected}, "
                f"expected {EXPECTED_BIPOLAR}"
            )
        observed[subject] = selected
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--edf-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = inspect_processed(args.data_root, args.edf_root)
    report["bipolar_channels"] = inspect_edf_channels(args.edf_root)
    report["expected_bipolar_channels"] = EXPECTED_BIPOLAR
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("ISRUC audit: PASS")


if __name__ == "__main__":
    main()
