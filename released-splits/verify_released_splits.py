"""Check gbsg_released_split_ids on all ten released GBSG repeats.

Usage (from a feat/survival-mdn checkout with PyHealth installed):
    python verify_released_splits.py --root /data/gbsg_csv \
        --released-dir /data/Survival-MDN/data/gbsg \
        [--manifest gbsg-source-row-manifest.json] [--out results.json]

Checks, per repeat: sizes, a one-to-one partition of the 2,232 source rows,
that each mapped source row reproduces the released features/event/time,
and (optionally) agreement with Aritra's data-audit manifest.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyhealth.datasets import gbsg_paper_time_transform, gbsg_released_split_ids

SPLITS = ("train", "valid", "test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--released-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("results.json"))
    a = p.parse_args()

    rows = {r["source_id"]: r for r in csv.DictReader((a.root / "gbsg.csv").open())}
    transform = gbsg_paper_time_transform(a.root)
    tmax = transform["time_divisor"]
    manifest = defaultdict(list)
    if a.manifest:
        for m in json.loads(a.manifest.read_text()):
            manifest[(m["repeat"], m["split"])].append(m["source_id"])

    # Records identical in every value (features, time, event).
    signature = defaultdict(list)
    for sid, r in rows.items():
        key = tuple(r[f"x{i}"] for i in range(7)) + (r["time"], r["event"])
        signature[key].append(sid)
    twin_of = {s: set(g) for g in signature.values() if len(g) > 1 for s in g}

    results = {"time_transform": transform, "twin_groups": sorted(
        sorted(g) for g in signature.values() if len(g) > 1), "repeats": {}}
    for repeat in range(1, 11):
        ids = gbsg_released_split_ids(a.root, a.released_dir, repeat)
        flat = [s for split in SPLITS for s in ids[split]]
        max_err = 0.0
        for split in SPLITS:
            d = np.load(a.released_dir / f"{split}_{repeat}.npz")
            x, y = d["arr_0"], d["arr_1"]
            for i, sid in enumerate(ids[split]):
                r = rows[sid]
                fx = np.array([float(r[f"x{j}"]) for j in range(7)])
                t = float(r["time"]) / tmax + 0.001
                assert float(r["event"]) == y[i, 1]
                max_err = max(max_err, np.abs(fx - x[i]).max(), abs(t - y[i, 0]))
        entry = {
            "sizes": [len(ids[s]) for s in SPLITS],
            "one_to_one": len(flat) == len(set(flat)) == len(rows),
            "max_abs_error": max_err,
        }
        if a.manifest:
            agree = differ_twin_only = differ_other = 0
            for split in SPLITS:
                for ours, theirs in zip(ids[split], manifest[(repeat, split)]):
                    if ours == theirs:
                        agree += 1
                    elif theirs in twin_of.get(ours, ()):
                        differ_twin_only += 1
                    else:
                        differ_other += 1
            entry["manifest"] = {"agree": agree, "differ_only_by_twin": differ_twin_only,
                                 "differ_other": differ_other}
        results["repeats"][repeat] = entry
        print(repeat, entry)
    a.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
