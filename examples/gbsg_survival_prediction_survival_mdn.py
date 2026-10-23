"""Survival prediction with SurvivalMDN: train, evaluate and compare configs.

Trains SurvivalMDN (Han, Goldstein & Ranganath, "Survival Mixture Density
Networks", MLHC 2022) with the standard PyHealth Trainer, selects the best
epoch on validation negative log-likelihood, and scores held-out survival
curves with the paper's metrics: time-dependent concordance (C^td),
integrated Brier score (IBS) and integrated binomial log-likelihood (IBLL).
It then repeats the fit for each mixture size and transform you list, as a
small ablation.

Demo mode, no downloads (synthetic bimodal survival times, under a minute
on CPU):
    python examples/gbsg_survival_prediction_survival_mdn.py --synthetic

GBSG with the authors' released split 1 (download their data/gbsg folder
yourself; see the GBSGDataset docs for preparing gbsg.csv):
    python examples/gbsg_survival_prediction_survival_mdn.py \\
        --root /data/gbsg_csv --released-dir /data/Survival-MDN/data/gbsg \\
        --repeat 1 --num-components 1 5 10

Every fit uses fresh dataset subsets and its own seed, so the result of a
config does not depend on which configs ran before it.
"""

import argparse
import itertools
import logging
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    GBSGDataset,
    create_sample_dataset,
    gbsg_paper_time_transform,
    gbsg_released_split_ids,
    get_dataloader,
)
from pyhealth.metrics import survival_curve_metrics_fn
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventGBSG
from pyhealth.trainer import Trainer
from pyhealth.trainer import logger as trainer_logger


def synthetic_samples(n: int = 1500, seed: int = 0) -> list[dict]:
    """Censored times from a two-component log-normal mixture.

    A feature-dependent share of patients has an early event (around 2) and
    the rest a late one (around 10), so a single-component model fits worse
    than a mixture. About a third of patients are censored.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4))
    early = rng.uniform(size=n) < 1 / (1 + np.exp(-2.0 * x[:, 0]))
    log_t = np.where(early, np.log(2.0), np.log(10.0) + 0.4 * x[:, 1])
    event_time = np.exp(log_t + 0.25 * rng.standard_normal(n))
    censor_time = np.minimum(rng.exponential(25.0, n), 30.0)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return [
        {
            "patient_id": f"p{i}",
            "features": x[i].tolist(),
            "survival": [float(time[i]), float(event[i])],
        }
        for i in range(n)
    ]


class SyntheticSplits:
    """80/10/10 split of the synthetic cohort, rebuilt for every fit."""

    def __init__(self, seed: int = 0):
        samples = synthetic_samples(seed=seed)
        order = np.random.default_rng(seed).permutation(len(samples))
        cut = [int(0.8 * len(samples)), int(0.9 * len(samples))]
        self.parts = [[samples[i] for i in idx] for idx in np.split(order, cut)]

    def build(self, seed: int):
        return tuple(
            create_sample_dataset(
                samples=part,
                input_schema={"features": "tensor"},
                output_schema={"survival": "tensor"},
                dataset_name=f"synthetic_{name}",
            )
            for name, part in zip(("train", "valid", "test"), self.parts)
        )


class GBSGReleasedSplit:
    """One of the authors' released GBSG splits, with their time convention."""

    def __init__(self, root: Path, released_dir: Path, repeat: int, cache: Path):
        dataset = GBSGDataset(root=str(root), cache_dir=cache, num_workers=1)
        task = SurvivalTimeToEventGBSG(**gbsg_paper_time_transform(root))
        self.samples = dataset.set_task(task)
        self.ids = gbsg_released_split_ids(root, released_dir, repeat)

    def build(self, seed: int):
        # Fresh subsets: their shuffle state would otherwise carry over
        # between fits. The training shuffle follows the fit's seed.
        parts = tuple(
            self.samples.subset(
                [
                    i
                    for pid in self.ids[name]
                    for i in self.samples.patient_to_index[pid]
                ]
            )
            for name in ("train", "valid", "test")
        )
        parts[0].seed = seed
        return parts


def fit_and_score(splits, model_kwargs: dict, epochs: int, seed: int, out: Path):
    """Train one config, keep the best validation epoch, score the test set."""
    # In-memory datasets shuffle with Python's random module, streaming ones
    # with their own seed (set in build); seed everything a fit draws from.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, valid, test = splits.build(seed)
    model = SurvivalMDN(train, **model_kwargs)
    trainer = Trainer(model, device="cpu", output_path=str(out), exp_name="fit")
    batch_size = min(256, len(train))
    while len(train) % batch_size == 1:  # BatchNorm cannot train on one row
        batch_size -= 1
    valid_loader = get_dataloader(valid, batch_size=len(valid))
    trainer.train(
        get_dataloader(train, batch_size=batch_size, shuffle=True),
        valid_loader,
        epochs=epochs,
        optimizer_class=torch.optim.RMSprop,
        optimizer_params={"lr": 1e-3},
        max_grad_norm=100.0,
        monitor="loss",
        monitor_criterion="min",
        patience=10,
    )
    # PyHealth keeps one log handler per Trainer; close it so later fits
    # do not also write into this fit's log.
    for handler in list(trainer_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            trainer_logger.removeHandler(handler)
            handler.close()
    val_nll = trainer.evaluate(valid_loader)["loss"]

    heldout = next(iter(get_dataloader(test, batch_size=len(test))))
    times = heldout["survival"][:, 0].numpy()
    events = heldout["survival"][:, 1].numpy()

    def predict(grid):
        grid = torch.as_tensor(np.asarray(grid), dtype=torch.float32)
        return model.predict_survival(heldout["features"], grid).numpy()

    return val_nll, survival_curve_metrics_fn(times, events, predict)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="No downloads")
    source.add_argument("--root", type=Path, help="Folder with gbsg.csv")
    parser.add_argument(
        "--released-dir",
        type=Path,
        help="Authors' data/gbsg folder (needed with --root)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Released split 1-10")
    parser.add_argument("--num-components", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--transform", nargs="+", default=["softplus"], choices=["softplus", "exp"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where Trainer checkpoints go (default: a temp folder)",
    )
    args = parser.parse_args()
    if args.root and not args.released_dir:
        parser.error("--root needs --released-dir")

    out = args.output_dir or Path(tempfile.mkdtemp(prefix="survival_mdn_"))
    if args.synthetic:
        splits = SyntheticSplits()
    else:
        splits = GBSGReleasedSplit(
            args.root, args.released_dir, args.repeat, out / "cache"
        )

    rows = []
    for k, transform in itertools.product(args.num_components, args.transform):
        model_kwargs = {"num_components": k, "transform": transform}
        val_nll, scores = fit_and_score(
            splits, model_kwargs, args.epochs, args.seed, out / f"K{k}_{transform}"
        )
        rows.append((k, transform, val_nll, scores))

    print(f"\nCheckpoints and logs: {out}")
    print(
        f"{'K':>3} {'transform':>9} {'val NLL':>8} {'C^td':>6} {'IBS':>6} {'IBLL':>7}"
    )
    for k, transform, val_nll, s in rows:
        print(
            f"{k:>3} {transform:>9} {val_nll:>8.4f} {s['c_td']:>6.3f} "
            f"{s['ibs']:>6.3f} {s['ibll']:>7.3f}"
        )
    best = min(rows, key=lambda r: r[2])
    print(
        f"Best on validation NLL: K={best[0]}, {best[1]}. "
        "Choose configs on validation, not test, scores."
    )


if __name__ == "__main__":
    main()
