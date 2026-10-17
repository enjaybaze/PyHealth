"""Load GBSG, fit training-only preprocessing and optionally train SurvivalMDN.

Download-free demonstration:
    python examples/gbsg_survival_loading.py --synthetic --output-dir /tmp/gbsg-demo

Real local benchmark, with a small Trainer/checkpoint check:
    python examples/gbsg_survival_loading.py --source-h5 /data/gbsg.h5 \
        --output-dir /tmp/gbsg-real --epochs 5

An existing prepared CSV directory can instead be supplied with --root.
Use a fresh output directory for each run. This is the alternative grouped
source-unit protocol, not the authors' released splits or paper reproduction.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyhealth.datasets import GBSGDataset, get_dataloader, prepare_gbsg_csv
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventGBSG
from pyhealth.trainer import Trainer


def write_synthetic_csv(root: Path, n: int = 40) -> None:
    """Create artificial raw records for a download-free loading example."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gbsg.csv"
    if path.exists():
        raise FileExistsError(f"Use a fresh synthetic directory: {root}")
    rng = np.random.default_rng(7)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "source_id",
                "source_group",
                "source_row",
                *[f"x{i}" for i in range(7)],
                "time",
                "event",
            ]
        )
        for i in range(n):
            x = rng.normal(size=7)
            time = 5 + np.exp(2 + 0.3 * x[0])
            writer.writerow([f"train_{i}", "train", i, *x, time, int(i % 3 != 0)])


def load_gbsg_splits(
    root: Path, output_dir: Path, seed: int = 1, split_source_ids=None
):
    """Build the actual loader/task path and persist its fitted configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GBSGDataset(root=str(root), cache_dir=output_dir / "cache", num_workers=1)
    raw_task = SurvivalTimeToEventGBSG()
    raw = [raw_task(patient)[0] for patient in dataset.iter_patients()]
    # Full source signatures, not assumed real patient identities. Grouping
    # outcomes is an explicit split-protocol choice; it does not fit a model.
    records = np.asarray([r["features"] + r["survival"] for r in raw], dtype=float)
    _, signature = np.unique(records, axis=0, return_inverse=True)
    groups = np.unique(signature)
    if len(groups) < 10:
        raise ValueError("The 80/10/10 example needs at least ten distinct records")
    pool, test = train_test_split(groups, test_size=0.1, random_state=seed)
    train, valid = train_test_split(pool, test_size=1 / 9, random_state=seed)
    memberships = {
        name: [r["patient_id"] for r, group in zip(raw, signature) if group in part]
        for name, part in zip(("train", "valid", "test"), (train, valid, test))
    }
    if split_source_ids is not None:
        memberships = {
            name: list(split_source_ids[name]) for name in ("train", "valid", "test")
        }
        supplied = [pid for ids in memberships.values() for pid in ids]
        if len(set(supplied)) != len(supplied) or set(supplied) != {
            r["patient_id"] for r in raw
        }:
            raise ValueError(
                "Split source IDs must partition all source rows without overlap"
            )
        if len(memberships["train"]) < 2 or not all(memberships.values()):
            raise ValueError(
                "Need at least two training records and nonempty held-out parts"
            )
        group_part = {}
        for name, ids in memberships.items():
            for record, group in zip(raw, signature):
                if record["patient_id"] in ids:
                    if group in group_part and group_part[group] != name:
                        raise ValueError(
                            "A source-signature group crosses split boundaries"
                        )
                    group_part[group] = name
    training_ids = set(memberships["train"])
    training = [r for r in raw if r["patient_id"] in training_ids]
    scaler = StandardScaler().fit([r["features"] for r in training])
    time_scale = float(np.median([r["survival"][0] for r in training]))
    task = SurvivalTimeToEventGBSG(scaler.mean_.tolist(), scaler.scale_.tolist())
    samples = dataset.set_task(task)
    parts = tuple(
        samples.subset(
            sorted(
                index
                for pid in memberships[name]
                for index in samples.patient_to_index[pid]
            )
        )
        for name in ("train", "valid", "test")
    )
    metadata = {
        "protocol": "grouped-source-signature-80-10-10-train-only-v1",
        "seed": seed,
        "csv_sha256": hashlib.sha256((root / "gbsg.csv").read_bytes()).hexdigest(),
        "feature_order": [f"x{i}" for i in range(7)],
        "time_unit": "months",
        "event": "1 observed; 0 censored",
        "identity": "source group/row surrogates, not original patient IDs",
        "split_source_ids": memberships,
        "counts": {name: len(part) for name, part in zip(memberships, parts)},
        "preprocessing": {
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
        },
        "model": {
            "time_scale": time_scale,
            "transform": "softplus",
            "num_components": 5,
            "hidden_dim": 32,
            "num_layers": 3,
            "batch_norm": True,
            "init": "fixed",
            "dropout": 0.0,
        },
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return parts, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Existing gbsg.csv directory")
    source.add_argument("--source-h5", type=Path, help="Local DeepSurv benchmark HDF5")
    source.add_argument(
        "--synthetic", action="store_true", help="Artificial CSV, no download"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--epochs", type=int, default=0, help="Optional small Trainer check"
    )
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    root = args.root or args.output_dir / "source"
    if args.synthetic:
        write_synthetic_csv(root)
    elif args.source_h5:
        prepare_gbsg_csv(args.source_h5, root)
    parts, metadata = load_gbsg_splits(root, args.output_dir, args.seed)
    train, valid, test = parts
    batch = next(iter(get_dataloader(train, batch_size=min(8, len(train)))))
    print("Counts:", metadata["counts"])
    print(
        "Feature/label shapes:",
        tuple(batch["features"].shape),
        tuple(batch["survival"].shape),
    )
    if args.epochs:
        torch.manual_seed(args.seed)
        model = SurvivalMDN(train, **metadata["model"])
        trainer = Trainer(
            model, device="cpu", output_path=str(args.output_dir), exp_name="smoke"
        )
        validation = get_dataloader(valid, batch_size=len(valid))
        initial = trainer.evaluate(validation)["loss"]
        batch_size = min(64, len(train))
        while len(train) % batch_size == 1:
            batch_size -= 1
        trainer.train(
            get_dataloader(train, batch_size=batch_size, shuffle=True),
            validation,
            epochs=args.epochs,
            optimizer_class=torch.optim.RMSprop,
            optimizer_params={"lr": 0.001},
            monitor="loss",
            monitor_criterion="min",
        )
        final = trainer.evaluate(validation)["loss"]
        heldout = next(iter(get_dataloader(test, batch_size=len(test))))
        grid = torch.linspace(0, max(s["survival"][0].item() for s in train), 40)
        predictions = model.predict_survival(heldout["features"], grid)
        restored = SurvivalMDN(train, **metadata["model"])
        Trainer(
            restored,
            device="cpu",
            checkpoint_path=str(args.output_dir / "smoke/best.ckpt"),
            enable_logging=False,
        )
        reloaded = restored.predict_survival(heldout["features"], grid)
        torch.testing.assert_close(predictions, reloaded, rtol=0, atol=0)
        assert torch.isfinite(predictions).all()
        assert np.isfinite(initial) and np.isfinite(final)
        np.savez(
            args.output_dir / "heldout-predictions.npz",
            source_id=heldout["source_id"],
            grid=grid.numpy(),
            survival=predictions.numpy(),
        )
        metadata["validation"] = {
            "epochs": args.epochs,
            "initial_loss": initial,
            "final_loss": final,
            "checkpoint_max_error": float((predictions - reloaded).abs().max()),
        }
        (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Validation:", metadata["validation"])


if __name__ == "__main__":
    main()
