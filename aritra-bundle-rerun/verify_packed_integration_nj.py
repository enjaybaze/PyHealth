"""Aritra's packed-label integration check, rerun against NJ's SurvivalMDN.

Reuses the bundle's data preparation (``task_dataset`` and ``synthetic`` from
``validation/mdn-deep/verify_packed_integration.py``) and repeats every check
from its ``check_case``, with ``pyhealth.models.SurvivalMDN`` from the
``feat/survival-mdn`` branch in place of the bundle's local ``native_mdn``.

Usage:
    python -I verify_packed_integration_nj.py <bundle root> <pyhealth repo> <new output dir>
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

bundle_root, pyhealth_root, output = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(pyhealth_root))
sys.path.insert(1, str(bundle_root / "validation/mdn-deep"))

from pyhealth.models import SurvivalMDN  # noqa: E402
from pyhealth.trainer import Trainer  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import verify_packed_integration as ref  # noqa: E402

ROOT = bundle_root / "validation/mdn-deep"


class StandardizedMDN(SurvivalMDN):
    """NJ's model plus the bundle's training-fitted feature standardisation."""

    def __init__(self, dataset, meta):
        prep = meta["preprocessor"]
        super().__init__(
            dataset,
            num_components=5,
            hidden_dim=32,
            num_layers=3,
            time_scale=float(prep["time_scale"]),
        )
        self.register_buffer("mean", torch.tensor(prep["mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(prep["scale"], dtype=torch.float32))

    def mixture_params(self, features):
        features = features.to(self.device).float()
        return super().mixture_params((features - self.mean) / self.scale)


def check_case(name):
    started = time.perf_counter()
    if name == "synthetic":
        data, meta, _ = ref.synthetic()
    else:
        folder = ROOT / "data-audit/prepared" / name / "seed1"
        meta = json.loads((folder / "metadata.json").read_text())
        data = {}
        for split in ["train", "valid", "test"]:
            with np.load(folder / f"{split}.npz") as values:
                data[split] = dict(values)
    ids = {s: set(a["source_id"].tolist()) for s, a in data.items()}
    assert not ids["train"] & ids["valid"] and not ids["train"] & ids["test"]
    assert not ids["valid"] & ids["test"]
    datasets = {s: ref.task_dataset(a, meta["features"], name) for s, a in data.items()}
    torch.manual_seed(6250)
    model = StandardizedMDN(datasets["train"], meta)
    assert model.mode is None and model.label_keys == ["survival"]
    batch_size = 32 if name == "synthetic" else 128
    loaders = {
        s: DataLoader(
            Subset(ds, list(range(len(ds)))),
            batch_size=batch_size if s == "train" else len(ds),
            shuffle=s == "train",
        )
        for s, ds in datasets.items()
    }
    run = output / name
    run.mkdir()
    trainer = Trainer(model, device="cpu", enable_logging=False)
    assert type(trainer) is Trainer
    trainer.exp_path = str(run)
    before = trainer.evaluate(loaders["valid"])["loss"]
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    history, train_losses, grad_checks = [], [], []

    def record(module, args, result):
        loss = result["loss"]
        assert loss.ndim == 0 and torch.isfinite(loss)
        (train_losses if module.training else history).append(float(loss.detach()))

    def check_grad(g):
        assert torch.isfinite(g).all()
        grad_checks.append(True)

    hook = model.register_forward_hook(record)
    ghooks = [p.register_hook(check_grad) for p in model.parameters() if p.requires_grad]
    trainer.train(
        loaders["train"],
        val_dataloader=loaders["valid"],
        epochs=5,
        optimizer_class=torch.optim.RMSprop,
        optimizer_params={"lr": 0.001},
        max_grad_norm=100,
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
    )
    hook.remove()
    for h in ghooks:
        h.remove()
    assert len(history) == 5 and len(train_losses) == 5 * len(loaders["train"])
    assert any(not torch.equal(initial[k], p) for k, p in model.named_parameters())
    after = trainer.evaluate(loaders["valid"])["loss"]
    assert np.isclose(after, min(history), atol=1e-6, rtol=0)
    assert after < before, (name, before, after)

    model.eval()
    valid = next(iter(loaders["valid"]))
    with torch.no_grad():
        x, labels = valid["x"], valid["survival"]
        params = model.mixture_params(x)
        log_p = model.log_density(params, labels[:, 0])
        log_s = model.log_survival(params, labels[:, 0])
        per_patient = -torch.where(labels[:, 1].bool(), log_p, log_s)
        direct = float(per_patient.double().mean())
        k = max(2, len(labels) // 3)
        weighted = sum(
            float(model(x=x[i : i + k], survival=labels[i : i + k])["loss"])
            * len(x[i : i + k])
            for i in range(0, len(x), k)
        ) / len(x)
    assert np.isclose(after, direct, atol=2e-6, rtol=1e-6), (after, direct)
    assert np.isclose(after, weighted, atol=2e-6, rtol=1e-6), (after, weighted)

    test = next(iter(loaders["test"]))
    # The bundle's grid starts at 0; NJ's model requires positive times, so the
    # first point is moved just above zero.
    hi = float(np.quantile(data["train"]["time"], 0.9))
    grid = torch.linspace(0, hi, 51)
    grid[0] = hi * 1e-6
    preds = model.predict_survival(test["x"], grid)
    assert torch.isfinite(preds).all() and ((preds >= 0) & (preds <= 1)).all()
    assert (torch.diff(preds, dim=1) <= 1e-6).all()
    restored = StandardizedMDN(datasets["train"], meta)
    reload = Trainer(
        restored,
        checkpoint_path=str(run / "best.ckpt"),
        device="cpu",
        enable_logging=False,
    )
    restored_loss = reload.evaluate(loaders["valid"])["loss"]
    restored.eval()
    reload_error = float((restored.predict_survival(test["x"], grid) - preds).abs().max())
    assert reload_error == 0 and restored_loss == after
    result = {
        "case": name,
        "status": "passed",
        "split_counts": {s: len(ds) for s, ds in datasets.items()},
        "validation_loss_before": before,
        "best_validation_loss": after,
        "validation_history": history,
        "selected_epoch": int(np.argmin(history)) + 1,
        "validation_mean_abs_error": abs(after - direct),
        "weighted_chunk_mean_abs_error": abs(after - weighted),
        "checkpoint_reload_max_error": reload_error,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    logging.getLogger("pyhealth.trainer").setLevel(logging.WARNING)
    results = [check_case(n) for n in ["synthetic", "gbsg", "support_train_vocab"]]
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(
            f"{r['case']:20s} {r['split_counts']} "
            f"val NLL {r['validation_loss_before']:.6f} -> {r['best_validation_loss']:.6f} "
            f"(epoch {r['selected_epoch']}), mean err {r['validation_mean_abs_error']:.1e}, "
            f"reload err {r['checkpoint_reload_max_error']}, {r['seconds']}s"
        )
    print("ALL THREE CHECKS PASSED WITH NJ'S SurvivalMDN")


if __name__ == "__main__":
    main()
