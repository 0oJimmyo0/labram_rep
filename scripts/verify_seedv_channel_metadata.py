#!/usr/bin/env python3
"""Compare SEED-V channel-order artifacts with the LaBraM manifest."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xlsx_order(path):
    with zipfile.ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.itertext()) for node in shared_root.findall("x:si", NS)]
        sheet_root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        names = []
        for cell in sheet_root.findall(".//x:c", NS):
            value = cell.find("x:v", NS)
            if value is not None and cell.get("t") == "s":
                names.append(shared[int(value.text)])
        return names


def read_locs_order(path):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            names.append(fields[-1])
    return names


def normalized(names):
    return [str(name).strip().upper() for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--locs", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    xlsx_order = normalized(read_xlsx_order(args.xlsx))
    locs_order = normalized(read_locs_order(args.locs))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_order = normalized(manifest["labram_channel_names"])

    result = {
        "xlsx": {"path": str(Path(args.xlsx).resolve()), "sha256": sha256(args.xlsx), "count": len(xlsx_order)},
        "locs": {"path": str(Path(args.locs).resolve()), "sha256": sha256(args.locs), "count": len(locs_order)},
        "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": sha256(args.manifest), "count": len(manifest_order)},
        "xlsx_matches_locs": xlsx_order == locs_order,
        "xlsx_matches_manifest": xlsx_order == manifest_order,
        "locs_matches_manifest": locs_order == manifest_order,
    }
    print(json.dumps(result, indent=2))
    if not all(result[key] for key in ("xlsx_matches_locs", "xlsx_matches_manifest", "locs_matches_manifest")):
        raise SystemExit("SEED-V channel metadata mismatch")
    print("SEED-V channel metadata: PASS")


if __name__ == "__main__":
    main()
