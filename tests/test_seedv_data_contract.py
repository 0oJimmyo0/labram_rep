"""SEED-V loader contract checks using a small synthetic LMDB."""

import json
import pickle
import tempfile
from pathlib import Path
import sys

import lmdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils


def write_lmdb(root, sample, label=0):
    env = lmdb.open(str(root), map_size=32 * 1024 * 1024)
    with env.begin(write=True) as txn:
        keys = {"train": ["sample-0"], "val": ["sample-0"], "test": ["sample-0"]}
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"sample-0", pickle.dumps({"sample": sample, "label": label}))
    env.close()


def write_manifest(path, count):
    names = utils.standard_1020[:count]
    path.write_text(json.dumps({"labram_channel_names": names}), encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "seedv.lmdb"
        manifest = Path(tmp) / "channels.json"
        write_manifest(manifest, 62)
        write_lmdb(root, np.zeros((62, 1, 200), dtype=np.float32))
        dataset = utils.SEEDVLoader(
            str(root), mode="train", channel_manifest=str(manifest), expected_shape=(62, 1, 200)
        )
        assert dataset[0][0].shape == (62, 1, 200)
        assert dataset.split_metadata()["sample_count"] == 1

        bad_root = Path(tmp) / "bad.lmdb"
        write_lmdb(bad_root, np.full((62, 1, 200), np.nan, dtype=np.float32))
        try:
            utils.SEEDVLoader(
                str(bad_root), mode="train", channel_manifest=str(manifest), expected_shape=(62, 1, 200)
            )
        except ValueError as exc:
            assert "NaN or Inf" in str(exc)
        else:
            raise AssertionError("SEED-V loader accepted non-finite data")

    print("SEED-V data contract: PASS")


if __name__ == "__main__":
    main()
