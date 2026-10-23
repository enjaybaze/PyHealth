"""Check the SUPPORT paper protocol against all ten released repeats.

Usage (PyHealth from feat/survival-mdn ab2025b or later):
    python verify_support_released_splits.py --root /data/support2 \
        --released-dir /data/Survival-MDN/data/support [--out support_results.json]

--root holds support2.csv with its first column named sno. For each repeat,
it checks that:
  * support2_released_split_ids covers the 8,873-patient cohort exactly once
  * SurvivalTimeToEventSupport2 with training-only scaling reproduces the
    released standardized features (arr_0)
  * the task's times equal the released years plus the authors' 0.001
  * death indicators match exactly
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pyhealth.datasets import (
    Support2Dataset,
    support2_paper_time_transform,
    support2_released_split_ids,
)
from pyhealth.tasks import SurvivalTimeToEventSupport2

SPLITS = ("train", "valid", "test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--released-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("support_results.json"))
    p.add_argument("--cache-dir", type=Path, default=Path("support_cache"))
    a = p.parse_args()

    time = support2_paper_time_transform()
    dataset = Support2Dataset(
        root=str(a.root), tables=["support2"], cache_dir=a.cache_dir, num_workers=1
    )
    raw = dataset.set_task(SurvivalTimeToEventSupport2(**time))
    features = {
        raw[i]["patient_id"]: raw[i]["features"].numpy().astype(float)
        for i in range(len(raw))
    }
    results = {"cohort": len(features), "time_transform": time, "repeats": {}}
    for repeat in range(1, 11):
        ids = support2_released_split_ids(a.root, a.released_dir, repeat)
        everyone = [pid for s in SPLITS for pid in ids[s]]
        train = np.stack([features[pid] for pid in ids["train"]])
        mean, scale = train.mean(0), train.std(0)
        scale[scale == 0] = 1.0
        std = dataset.set_task(
            SurvivalTimeToEventSupport2(mean.tolist(), scale.tolist(), **time)
        )
        where = {std[i]["patient_id"]: i for i in range(len(std))}
        feat_err = time_err = 0.0
        for s in SPLITS:
            rel = np.load(a.released_dir / f"{s}_{repeat}.npz")
            x = np.stack([std[where[pid]]["features"].numpy() for pid in ids[s]])
            y = np.stack([std[where[pid]]["survival"].numpy() for pid in ids[s]])
            feat_err = max(feat_err, float(np.abs(x - rel["arr_0"]).max()))
            time_err = max(
                time_err, float(np.abs(y[:, 0] - (rel["arr_1"][:, 0] + 0.001)).max())
            )
            assert np.array_equal(y[:, 1], rel["arr_1"][:, 1]), (repeat, s)
        entry = {
            "sizes": [len(ids[s]) for s in SPLITS],
            "one_to_one": len(everyone) == len(set(everyone)) == len(features),
            "max_abs_standardized_feature_error": feat_err,
            "max_abs_time_error": time_err,
        }
        results["repeats"][repeat] = entry
        print(repeat, entry)
    a.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
