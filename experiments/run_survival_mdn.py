"""Experiment runner for the Survival MDN reproduction (T5, used in T8).

Trains SurvivalMDN on the authors' released GBSG or SUPPORT splits
(--dataset), one fit per (config, split, seed), and saves for every fit:

    <out>/<config>/split_<n>/seed_<s>/
        record.json       config, seed, split, timings, epochs, val NLL,
                          test metrics, environment and git commits
        predictions.npz   test patient IDs, times, events, grid, curves

Then writes <out>/summary.csv and <out>/summary.json: per config, mean and
SD across splits of every metric, plus mean validation NLL. Choose configs
on validation NLL only; test metrics are for reporting.

Completed fits (record.json present) are skipped, so a stopped run can be
restarted with the same command.

Example (5 epochs, 2 splits, to check timing):
    python run_survival_mdn.py --root /data/gbsg_csv \\
        --released-dir /data/Survival-MDN/data/gbsg \\
        --out runs/pilot --splits 1 2 --epochs 5

SUPPORT: --dataset support2, --root a folder with support2.csv (first
column named sno) and --released-dir the authors' data/support folder.

Ablations: pass several values, and every combination is run, e.g.
    --num-components 1 2 3 5 10 20 --transform softplus exp

Hyper-parameters not given on the command line come from --config (a JSON
file with "model" and "train" objects) or the defaults below.
"""

import argparse
import itertools
import json
import logging
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pyhealth
import torch
from pyhealth.datasets import (
    GBSGDataset,
    Support2Dataset,
    gbsg_paper_time_transform,
    gbsg_released_split_ids,
    get_dataloader,
    support2_paper_time_transform,
    support2_released_split_ids,
)
from pyhealth.metrics import survival_curve_metrics_fn
from pyhealth.metrics.survival_curve import (
    integrated_binomial_loglik_td,
    integrated_brier_score_td,
    integration_grid,
)
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventGBSG, SurvivalTimeToEventSupport2
from pyhealth.trainer import Trainer
from pyhealth.trainer import logger as trainer_logger

SPLITS = ("train", "valid", "test")

DEFAULT_MODEL = {
    "num_components": 10,
    "transform": "softplus",
    "hidden_dim": 32,
    "num_layers": 3,
    "batch_norm": True,
    "init": "fixed",
    "dropout": 0.0,
    "time_scale": 1.0,
}
# The authors' main.py defaults: 100 epochs, patience 10 on validation loss,
# RMSprop, gradient clipping at 100 (configs/train.json).
DEFAULT_TRAIN = {
    "epochs": 100,
    "patience": 10,
    "optimizer": "RMSprop",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 512,
    "grad_clip": 100.0,
}
PREDICTION_GRID_POINTS = 200


def git_commit(path: Path) -> str | None:
    """HEAD of the repo containing path, with "+dirty" if it has local edits."""
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


def config_grid(base: dict, args) -> list[tuple[str, dict, dict]]:
    """Expand ablation flags into named (model, train) configs."""
    model = {**DEFAULT_MODEL, **base.get("model", {})}
    train = {**DEFAULT_TRAIN, **base.get("train", {})}
    for key in ("epochs", "patience", "lr", "batch_size"):
        if getattr(args, key) is not None:
            train[key] = getattr(args, key)
    ks = args.num_components or [model["num_components"]]
    transforms = args.transform or [model["transform"]]
    configs = []
    for k, tf in itertools.product(ks, transforms):
        m = {**model, "num_components": k, "transform": tf}
        configs.append((f"{args.name}_K{k}_{tf}", m, dict(train)))
    return configs


class GBSGBenchmark:
    """Authors' protocol: raw features, months / cohort max + 0.002."""

    def __init__(self, root: Path, released_dir: Path, cache_dir: Path):
        self.root, self.released_dir = root, released_dir
        dataset = GBSGDataset(root=str(root), cache_dir=cache_dir, num_workers=1)
        task = SurvivalTimeToEventGBSG(**gbsg_paper_time_transform(root))
        self.samples = dataset.set_task(task)
        self.preprocessing = {"features": "raw", **gbsg_paper_time_transform(root)}

    def split(self, split: int):
        ids = gbsg_released_split_ids(self.root, self.released_dir, split)
        return ids, self.samples, self.preprocessing


class Support2Benchmark:
    """Authors' protocol: 27 features standardized on each split's training
    patients, days / 365.25 + 0.001."""

    def __init__(self, root: Path, released_dir: Path, cache_dir: Path):
        self.root, self.released_dir = root, released_dir
        self.dataset = Support2Dataset(
            root=str(root), tables=["support2"], cache_dir=cache_dir, num_workers=1
        )
        self.time = support2_paper_time_transform()
        raw = self.dataset.set_task(SurvivalTimeToEventSupport2(**self.time))
        self.raw_features = {
            raw[i]["patient_id"]: raw[i]["features"].numpy().astype(float)
            for i in range(len(raw))
        }

    def split(self, split: int):
        ids = support2_released_split_ids(self.root, self.released_dir, split)
        train = np.stack([self.raw_features[p] for p in ids["train"]])
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale == 0] = 1.0  # as sklearn's StandardScaler
        task = SurvivalTimeToEventSupport2(mean.tolist(), scale.tolist(), **self.time)
        samples = self.dataset.set_task(task)
        preprocessing = {
            "features": "standardized on this split's training patients",
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            **self.time,
        }
        return ids, samples, preprocessing


BENCHMARKS = {"gbsg": GBSGBenchmark, "support2": Support2Benchmark}


def split_parts(samples, ids: dict, seed: int):
    """Fresh train/valid/test subsets for one fit.

    The training shuffle comes from get_dataloader's seed (PyHealth #1280),
    so results do not depend on run order. Fresh subsets are kept anyway,
    so no dataset state can carry over from an earlier fit.
    """
    return tuple(
        samples.subset([i for pid in ids[name] for i in samples.patient_to_index[pid]])
        for name in SPLITS
    )


def safe_batch_size(n: int, requested: int) -> int:
    """Avoid a final batch of one, which BatchNorm cannot train on."""
    size = min(requested, n)
    while n % size == 1:
        size -= 1
    return size


def reference_compatible_widest(times, events, predict) -> dict:
    """Widest-horizon IBS/IBLL as the authors' evaluator computes them.

    Their model code evaluates curves on linspace(t_min, t_max, 1000) but
    scores them as if they were on linspace(t_min, tau, 1000), where tau is
    the widest horizon. When the last test time is censored, tau < t_max
    and the grids are misaligned (see team-notes parity/README.md). The
    paper's Table 1 widest-horizon IBS/IBLL include this effect, so these
    values are the ones to compare with Table 1. PyHealth's aligned values
    stay in test_metrics. Checked against the authors' metrics.py to 2e-15
    (parity/reference_compatible_check.py).
    """
    grid = integration_grid(times, events, 1e-8)
    surv = predict(np.linspace(times.min(), times.max(), len(grid)))
    return {
        "ibs@1e-08": float(integrated_brier_score_td(times, events, surv, grid)),
        "ibll@1e-08": float(integrated_binomial_loglik_td(times, events, surv, grid)),
    }


def close_trainer_logs() -> None:
    """PyHealth adds a file handler per Trainer and never removes it, so
    later fits would also write into earlier fits' logs."""
    for handler in list(trainer_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            trainer_logger.removeHandler(handler)
            handler.close()


def epochs_from_log(log: Path) -> dict:
    """Best epoch and epochs run, read from this fit's trainer log."""
    text = log.read_text() if log.exists() else ""
    best = re.findall(r"New best loss score \([^)]*\) at epoch-(\d+)", text)
    trained = re.findall(r"--- Train epoch-(\d+)", text)
    stopped = re.search(r"Early stopping at epoch-(\d+)", text)
    return {
        "best_epoch": int(best[-1]) if best else None,
        "epochs_run": int(trained[-1]) + 1 if trained else None,
        "early_stopped": stopped is not None,
    }


def run_one(name, model_cfg, train_cfg, split, seed, parts, run_dir, device):
    train, valid, test = parts
    # Seed every RNG the data pipeline may draw from, so a fit gives the
    # same result whether or not other fits ran earlier in this process.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SurvivalMDN(train, **model_cfg)
    # Logging must stay on: without an output path the Trainer saves no
    # best.ckpt, and load_best_model_at_last would silently keep the last
    # epoch instead of the best validation epoch.
    trainer = Trainer(
        model, device=device, output_path=str(run_dir), exp_name="trainer"
    )
    batch = safe_batch_size(len(train), train_cfg["batch_size"])
    val_loader = get_dataloader(valid, batch_size=len(valid))
    start = time.perf_counter()
    trainer.train(
        get_dataloader(train, batch_size=batch, shuffle=True, seed=seed),
        val_loader,
        epochs=train_cfg["epochs"],
        optimizer_class=getattr(torch.optim, train_cfg["optimizer"]),
        optimizer_params={"lr": train_cfg["lr"]},
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["grad_clip"],
        monitor="loss",
        monitor_criterion="min",
        patience=train_cfg["patience"],
        load_best_model_at_last=True,
    )
    train_seconds = time.perf_counter() - start
    close_trainer_logs()
    val_nll = float(trainer.evaluate(val_loader)["loss"])

    heldout = next(iter(get_dataloader(test, batch_size=len(test))))
    features = heldout["features"].to(device)
    times = heldout["survival"][:, 0].numpy()
    events = heldout["survival"][:, 1].numpy()

    def predict(grid):
        g = torch.as_tensor(np.asarray(grid), dtype=torch.float32, device=device)
        with torch.no_grad():
            return model.predict_survival(features, g).cpu().numpy()

    metrics = survival_curve_metrics_fn(times, events, predict)
    train_max = max(float(s["survival"][0]) for s in train)
    grid = np.linspace(0.0, train_max, PREDICTION_GRID_POINTS)
    np.savez_compressed(
        run_dir / "predictions.npz",
        patient_id=np.asarray(heldout["patient_id"]),
        time=times,
        event=events,
        grid=grid,
        survival=predict(grid),
    )
    order = np.argsort(times, kind="stable")
    return {
        "config": name,
        "split": split,
        "seed": seed,
        "model": model_cfg,
        "train": {**train_cfg, "batch_size_used": batch},
        "counts": {s: len(p) for s, p in zip(SPLITS, parts)},
        "val_nll": val_nll,
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "test_metrics_reference_compatible": reference_compatible_widest(
            times, events, predict
        ),
        "last_test_time_censored": bool(events[order[-1]] == 0),
        "train_seconds": train_seconds,
        "epochs_requested": train_cfg["epochs"],
        **epochs_from_log(run_dir / "trainer" / "log.txt"),
    }


def summarize(out: Path) -> list[dict]:
    """Mean and SD across splits, per config.

    Seeds are averaged within each split first, so the SD is across the
    released splits, as in the paper's tables.
    """
    rows = {}
    for path in sorted(out.glob("*/split_*/seed_*/record.json")):
        rec = json.loads(path.read_text())
        rows.setdefault(rec["config"], {}).setdefault(rec["split"], []).append(rec)
    summary = []
    for name, by_split in rows.items():
        recs = [r for split_recs in by_split.values() for r in split_recs]

        def per_split(get):
            return [float(np.mean([get(r) for r in rs])) for rs in by_split.values()]

        entry = {
            "config": name,
            "fits": len(recs),
            "splits": sorted(by_split),
            "seeds_per_split": sorted({len(rs) for rs in by_split.values()}),
            "val_nll_mean": float(np.mean(per_split(lambda r: r["val_nll"]))),
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in recs])),
        }
        for metric in recs[0]["test_metrics"]:
            vals = per_split(lambda r, m=metric: r["test_metrics"][m])
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        for metric in recs[0]["test_metrics_reference_compatible"]:
            vals = per_split(
                lambda r, m=metric: r["test_metrics_reference_compatible"][m]
            )
            entry[f"ref_{metric}_mean"] = float(np.mean(vals))
            entry[f"ref_{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        summary.append(entry)
    summary.sort(key=lambda e: e["val_nll_mean"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary:
        cols = list(summary[0])
        lines = [",".join(cols)] + [
            ",".join(
                " ".join(map(str, e[c]))
                if isinstance(e[c], list)
                else (f"{e[c]:.6g}" if isinstance(e[c], float) else str(e[c]))
                for c in cols
            )
            for e in summary
        ]
        (out / "summary.csv").write_text("\n".join(lines) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=sorted(BENCHMARKS), default="gbsg")
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Folder with gbsg.csv, or support2.csv for --dataset support2",
    )
    p.add_argument(
        "--released-dir",
        type=Path,
        required=True,
        help="Authors' data/gbsg or data/support folder with the .npz splits",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="mdn", help="Config name prefix")
    p.add_argument("--config", type=Path, help="JSON with 'model' and 'train'")
    p.add_argument("--splits", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--num-components", type=int, nargs="+")
    p.add_argument("--transform", nargs="+", choices=["softplus", "exp"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    close_trainer_logs()

    base = json.loads(args.config.read_text()) if args.config else {}
    configs = config_grid(base, args)
    args.out.mkdir(parents=True, exist_ok=True)
    env = {
        "pyhealth_commit": git_commit(Path(pyhealth.__file__).parent),
        "runner_commit": git_commit(Path(__file__).parent),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0)
        if args.device.startswith("cuda")
        else None,
        "python": platform.python_version(),
    }
    benchmark = None  # built on first use, so a finished run resumes instantly
    total = len(configs) * len(args.splits) * len(args.seeds)
    done = 0
    for split in args.splits:
        prepared = None
        for (name, model_cfg, train_cfg), seed in itertools.product(
            configs, args.seeds
        ):
            done += 1
            run_dir = args.out / name / f"split_{split}" / f"seed_{seed}"
            if (run_dir / "record.json").exists():
                print(f"[{done}/{total}] {run_dir} done, skipping")
                continue
            if benchmark is None:
                benchmark = BENCHMARKS[args.dataset](
                    args.root, args.released_dir, args.out / "cache"
                )
            if prepared is None:
                prepared = benchmark.split(split)
            ids, samples, preprocessing = prepared
            run_dir.mkdir(parents=True, exist_ok=True)
            parts = split_parts(samples, ids, seed)
            record = run_one(
                name, model_cfg, train_cfg, split, seed, parts, run_dir, args.device
            )
            record["dataset"] = args.dataset
            record["preprocessing"] = preprocessing
            record["environment"] = {**env, "device": args.device}
            (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
            m = record["test_metrics"]
            print(
                f"[{done}/{total}] {name} split {split} seed {seed}: "
                f"val NLL {record['val_nll']:.4f}  C^td {m['c_td']:.3f}  "
                f"IBS {m['ibs']:.3f}  IBLL {m['ibll']:.3f}  "
                f"({record['train_seconds']:.1f}s)"
            )
    for entry in summarize(args.out):
        print(
            f"{entry['config']}: {entry['fits']} fits, val NLL "
            f"{entry['val_nll_mean']:.4f}, C^td {entry['c_td_mean']:.3f} "
            f"± {entry['c_td_sd']:.3f}, IBS {entry['ibs_mean']:.3f} "
            f"± {entry['ibs_sd']:.3f}, IBLL {entry['ibll_mean']:.3f} "
            f"± {entry['ibll_sd']:.3f} | reference-compatible IBS "
            f"{entry['ref_ibs@1e-08_mean']:.3f}, IBLL "
            f"{entry['ref_ibll@1e-08_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
