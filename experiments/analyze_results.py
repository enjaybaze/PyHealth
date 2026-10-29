"""Tables and figures from the 300-fit course run (T8).

Reads the run records (record.json files) produced by run_course_plan.py and
writes, under --out:

  selection.csv          per split: the MDN config chosen by validation NLL,
                         and each baseline's config chosen by validation IBS
  main_table.csv/.md     test C^td, IBS, IBLL (mean, SD, SE over 10 splits)
                         for Cox, DeepSurv and MDN, with the paper's Table 1
  paired.csv             per-split differences MDN - DeepSurv and MDN - Cox
  ablation_k.csv         K = 1, 2, 3, 5, 10, 20 (softplus), all other settings
                         at the defaults
  ablation_transform.csv K=10 exp vs softplus, paired by split
  tuning_check.csv       which MDN tuning candidate wins how often, for
                         Aritra's "is the search range limiting us" check
  ablation_k.png         validation NLL and test C^td against K
  summary.md             everything above in one readable page

Model selection uses validation scores only. Test scores are reported, never
used to choose. IBS and IBLL are compared with the paper using the
reference-compatible values (the authors' widest-horizon convention);
everything else uses PyHealth's aligned values.

    python analyze_results.py --runs ~/runs/course --out ~/runs/course-analysis
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DATASETS = ("gbsg", "support2")
NAMES = {"gbsg": "GBSG", "support2": "SUPPORT"}
MDN_TUNING = ("mdn_k10_softplus", "mdn_slowlr", "mdn_wide")
K_CONFIGS = {
    1: "mdn_k1_softplus",
    2: "mdn_k2_softplus",
    3: "mdn_k3_softplus",
    5: "mdn_k5_softplus",
    10: "mdn_k10_softplus",
    20: "mdn_k20_softplus",
}
METRICS = ("c_td", "ibs", "ibll")
# Han et al. (2022) Table 1 at the widest horizon, as in our proposal.
# The paper reports standard errors of at most 0.010.
PAPER = {
    ("gbsg", "cox"): (0.645, 0.177, -0.523),
    ("gbsg", "deepsurv"): (0.663, 0.172, -0.509),
    ("gbsg", "mdn"): (0.668, 0.172, -0.504),
    ("support2", "cox"): (0.596, 0.194, -0.568),
    ("support2", "deepsurv"): (0.609, 0.190, -0.559),
    ("support2", "mdn"): (0.628, 0.190, -0.559),
}
PAPER_SE = 0.010
HIGHER_IS_BETTER = {"c_td": True, "ibs": False, "ibll": True, "val_nll": False}


def load(runs: Path) -> dict:
    """{(dataset, plan_config, split): record} for every fit outside _preflight."""
    out = {}
    for path in runs.glob("*/*/**/record.json"):
        rel = path.relative_to(runs).parts
        if rel[0] not in DATASETS:
            continue
        rec = json.loads(path.read_text())
        key = (rel[0], rel[1], int(rec["split"]))
        if key in out:
            raise ValueError(f"two records for {key}: keep one run per fit")
        out[key] = rec
    return out


def test_scores(rec: dict) -> dict:
    m, ref = rec["test_metrics"], rec["test_metrics_reference_compatible"]
    return {
        "c_td": m["c_td"],
        "ibs": m["ibs"],
        "ibll": m["ibll"],
        "ibs_ref": ref["ibs@1e-08"],
        "ibll_ref": ref["ibll@1e-08"],
        "dcal_p": rec["d_calibration"]["p_value"],
    }


def validation_score(rec: dict) -> float:
    """MDN: validation NLL. Baselines: their recorded selection value (IBS)."""
    if "val_nll" in rec:
        return rec["val_nll"]
    if rec.get("selection_metric") != "validation_ibs":
        raise ValueError(
            f"unexpected baseline selection metric {rec.get('selection_metric')}"
        )
    return rec["selection_value"]


def select(records: dict, dataset: str, configs, split: int) -> tuple[str, dict]:
    candidates = {c: records[(dataset, c, split)] for c in configs}
    best = min(candidates, key=lambda c: (validation_score(candidates[c]), c))
    return best, candidates[best]


def describe(values) -> dict:
    v = np.asarray(values, float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {
        "mean": float(v.mean()),
        "sd": sd,
        "se": sd / math.sqrt(len(v)),
        "n": len(v),
    }


def paired(a, b, metric: str) -> dict:
    """a - b per split, oriented so a positive 'better_splits' count favors a."""
    d = np.asarray(a, float) - np.asarray(b, float)
    better = d > 0 if HIGHER_IS_BETTER[metric] else d < 0
    t = stats.ttest_rel(a, b) if np.any(d != 0) else None
    return {
        **describe(d),
        "better_splits": int(better.sum()),
        "p_paired_t": float(t.pvalue) if t is not None else 1.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace("-0.", "−0.") if x < 0 else f"{x:.{d}f}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--runs", type=Path, required=True, help="run_course_plan.py --runs-root"
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    runs, out = a.runs.expanduser().resolve(), a.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load(runs)
    splits = sorted({k[2] for k in records})
    baseline_configs = {
        (d, m): sorted(
            {c for (dd, c, _) in records if dd == d and c.startswith(m + "_")}
        )
        for d in DATASETS
        for m in ("cox", "deepsurv")
    }
    if len(records) != 300 or splits != list(range(1, 11)):
        raise ValueError(f"expected 300 fits over splits 1-10, found {len(records)}")

    # 1. Per-split selection on validation only.
    chosen = {}
    selection_rows = []
    for d in DATASETS:
        for s in splits:
            row = {"dataset": d, "split": s}
            for model, cands in (
                ("mdn", MDN_TUNING),
                ("cox", baseline_configs[(d, "cox")]),
                ("deepsurv", baseline_configs[(d, "deepsurv")]),
            ):
                cfg, rec = select(records, d, cands, s)
                chosen[(d, model, s)] = rec
                row[f"{model}_config"] = cfg
                row[f"{model}_validation"] = round(validation_score(rec), 6)
            selection_rows.append(row)
    write_csv(out / "selection.csv", selection_rows)

    # 2. Main table with the paper's numbers.
    main_rows, md = [], []
    md.append(
        "| Dataset | Model | C^td | IBS (aligned) | IBLL (aligned) | IBS (ref.) "
        "| IBLL (ref.) | Paper C^td / IBS / IBLL | D-cal p > 0.05 |"
    )
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for d in DATASETS:
        for model in ("cox", "deepsurv", "mdn"):
            scores = [test_scores(chosen[(d, model, s)]) for s in splits]
            row = {"dataset": d, "model": model}
            for key in ("c_td", "ibs", "ibll", "ibs_ref", "ibll_ref"):
                desc = describe([x[key] for x in scores])
                row.update(
                    {
                        f"{key}_mean": round(desc["mean"], 4),
                        f"{key}_sd": round(desc["sd"], 4),
                        f"{key}_se": round(desc["se"], 4),
                    }
                )
            paper = PAPER[(d, model)]
            row.update(
                {"paper_c_td": paper[0], "paper_ibs": paper[1], "paper_ibll": paper[2]}
            )
            ours = (row["c_td_mean"], row["ibs_ref_mean"], row["ibll_ref_mean"])
            row["within_2_paper_se"] = all(
                abs(o - q) <= 2 * PAPER_SE for o, q in zip(ours, paper)
            )
            row["dcal_pass_splits"] = sum(x["dcal_p"] > 0.05 for x in scores)
            main_rows.append(row)
            md.append(
                f"| {NAMES[d]} | {model} | {fmt(row['c_td_mean'])} ± {fmt(row['c_td_sd'])} "
                f"| {fmt(row['ibs_mean'])} ± {fmt(row['ibs_sd'])} "
                f"| {fmt(row['ibll_mean'])} ± {fmt(row['ibll_sd'])} "
                f"| {fmt(row['ibs_ref_mean'])} | {fmt(row['ibll_ref_mean'])} "
                f"| {fmt(paper[0])} / {fmt(paper[1])} / {fmt(paper[2])} "
                f"| {row['dcal_pass_splits']}/10 |"
            )
    write_csv(out / "main_table.csv", main_rows)
    (out / "main_table.md").write_text("\n".join(md) + "\n")

    # 3. Paired differences.
    paired_rows = []
    for d in DATASETS:
        for other in ("deepsurv", "cox"):
            for metric in METRICS:
                a_ = [test_scores(chosen[(d, "mdn", s)])[metric] for s in splits]
                b_ = [test_scores(chosen[(d, other, s)])[metric] for s in splits]
                r = paired(a_, b_, metric)
                paired_rows.append(
                    {
                        "dataset": d,
                        "comparison": f"mdn - {other}",
                        "metric": metric,
                        **{
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in r.items()
                        },
                    }
                )
    write_csv(out / "paired.csv", paired_rows)

    # 4. Mixture-size ablation (fixed default settings, softplus).
    k_rows = []
    for d in DATASETS:
        base = {s: records[(d, K_CONFIGS[10], s)] for s in splits}
        for k, cfg in K_CONFIGS.items():
            recs = [records[(d, cfg, s)] for s in splits]
            row = {"dataset": d, "K": k}
            vals = {
                "val_nll": [r["val_nll"] for r in recs],
                **{m: [test_scores(r)[m] for r in recs] for m in METRICS},
            }
            for m, v in vals.items():
                desc = describe(v)
                row[f"{m}_mean"], row[f"{m}_se"] = (
                    round(desc["mean"], 4),
                    round(desc["se"], 4),
                )
            if k != 10:
                ref = [base[s]["val_nll"] for s in splits]
                r = paired(vals["val_nll"], ref, "val_nll")
                row["val_nll_vs_K10_mean_diff"] = round(r["mean"], 4)
                row["val_nll_vs_K10_p"] = round(r["p_paired_t"], 4)
                r = paired(
                    vals["c_td"], [test_scores(base[s])["c_td"] for s in splits], "c_td"
                )
                row["c_td_vs_K10_mean_diff"] = round(r["mean"], 4)
                row["c_td_vs_K10_p"] = round(r["p_paired_t"], 4)
            else:
                row.update(
                    {
                        "val_nll_vs_K10_mean_diff": 0.0,
                        "val_nll_vs_K10_p": 1.0,
                        "c_td_vs_K10_mean_diff": 0.0,
                        "c_td_vs_K10_p": 1.0,
                    }
                )
            k_rows.append(row)
    write_csv(out / "ablation_k.csv", k_rows)

    # 5. Transform ablation, paired by split.
    tf_rows = []
    for d in DATASETS:
        for m in ("val_nll", *METRICS):
            get = (
                (lambda r: r["val_nll"])
                if m == "val_nll"
                else (lambda r, m=m: test_scores(r)[m])
            )
            sp = [get(records[(d, "mdn_k10_softplus", s)]) for s in splits]
            ex = [get(records[(d, "mdn_k10_exp", s)]) for s in splits]
            r = paired(sp, ex, m)
            tf_rows.append(
                {
                    "dataset": d,
                    "metric": m,
                    "softplus_mean": round(np.mean(sp), 4),
                    "exp_mean": round(np.mean(ex), 4),
                    "softplus_minus_exp": round(r["mean"], 4),
                    "softplus_better_splits": r["better_splits"],
                    "p_paired_t": round(r["p_paired_t"], 4),
                }
            )
    write_csv(out / "ablation_transform.csv", tf_rows)

    # 6. Tuning range check.
    tune_rows = []
    for d in DATASETS:
        wins = defaultdict(int)
        for s in splits:
            wins[
                next(
                    r["mdn_config"]
                    for r in selection_rows
                    if r["dataset"] == d and r["split"] == s
                )
            ] += 1
        for cfg in MDN_TUNING:
            v = [records[(d, cfg, s)]["val_nll"] for s in splits]
            tune_rows.append(
                {
                    "dataset": d,
                    "model": "mdn",
                    "config": cfg,
                    "wins": wins[cfg],
                    "val_mean": round(float(np.mean(v)), 4),
                }
            )
        for model in ("cox", "deepsurv"):
            wins = defaultdict(int)
            for r in selection_rows:
                if r["dataset"] == d:
                    wins[r[f"{model}_config"]] += 1
            for cfg in baseline_configs[(d, model)]:
                v = [validation_score(records[(d, cfg, s)]) for s in splits]
                tune_rows.append(
                    {
                        "dataset": d,
                        "model": model,
                        "config": cfg,
                        "wins": wins[cfg],
                        "val_mean": round(float(np.mean(v)), 4),
                    }
                )
    write_csv(out / "tuning_check.csv", tune_rows)

    plot_k(k_rows, out / "ablation_k.png")
    summary(out, main_rows, paired_rows, k_rows, tf_rows, tune_rows, md)
    print(f"Wrote results to {out}")


def plot_k(k_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, mark = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    ks = sorted(K_CONFIGS)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), constrained_layout=True)
    for col, d in enumerate(DATASETS):
        rows = {r["K"]: r for r in k_rows if r["dataset"] == d}
        for row_i, (metric, label) in enumerate(
            (
                ("val_nll", "Validation NLL (lower is better)"),
                ("c_td", "Test C^td (higher is better)"),
            )
        ):
            ax = axes[row_i, col]
            x = np.arange(len(ks))
            y = [rows[k][f"{metric}_mean"] for k in ks]
            e = [rows[k][f"{metric}_se"] for k in ks]
            ax.errorbar(
                x,
                y,
                yerr=e,
                color=mark,
                linewidth=2,
                marker="o",
                markersize=6,
                capsize=3,
                elinewidth=1.2,
            )
            ax.set_xticks(x, [str(k) for k in ks], color=muted)
            ax.tick_params(colors=muted, labelsize=9)
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.set_title(f"{NAMES[d]}: {label}", fontsize=10, color=ink, loc="left")
            if row_i == 1:
                ax.set_xlabel("Mixture components K", color=muted, fontsize=9)
    fig.suptitle(
        "Mixture-size ablation: mean ± standard error over 10 released splits",
        fontsize=11,
        color=ink,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=200, facecolor="white")


def summary(out, main_rows, paired_rows, k_rows, tf_rows, tune_rows, md) -> None:
    lines = [
        "# Course run results (300 fits)",
        "",
        "Selection on validation only: MDN by validation NLL among its three tuning "
        "candidates, baselines by validation IBS among their three configs, per split. "
        'Test means ± SD over the 10 released splits. "ref." = the authors\' '
        "widest-horizon convention, used for the paper comparison.",
        "",
        *md,
        "",
        "## MDN minus baselines (test, paired by split)",
        "",
        "| Dataset | Comparison | Metric | Mean diff | SD | Splits MDN better | Paired t p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in paired_rows:
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['comparison']} | {r['metric']} "
            f"| {fmt(r['mean'], 4)} | {fmt(r['sd'], 4)} | {r['better_splits']}/10 "
            f"| {fmt(r['p_paired_t'], 3)} |"
        )
    lines += [
        "",
        "## Mixture size K (softplus, default settings)",
        "",
        "| Dataset | K | Val NLL | C^td | IBS | IBLL | Δ val NLL vs K=10 (p) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in k_rows:
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['K']} | {fmt(r['val_nll_mean'], 4)} "
            f"| {fmt(r['c_td_mean'])} | {fmt(r['ibs_mean'])} | {fmt(r['ibll_mean'])} "
            f"| {fmt(r['val_nll_vs_K10_mean_diff'], 4)} ({fmt(r['val_nll_vs_K10_p'], 3)}) |"
        )
    lines += [
        "",
        "## Transform (K=10): softplus minus exp, paired by split",
        "",
        "| Dataset | Metric | Softplus | Exp | Diff | Splits softplus better | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in tf_rows:
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['metric']} | {fmt(r['softplus_mean'], 4)} "
            f"| {fmt(r['exp_mean'], 4)} | {fmt(r['softplus_minus_exp'], 4)} "
            f"| {r['softplus_better_splits']}/10 | {fmt(r['p_paired_t'], 3)} |"
        )
    lines += [
        "",
        "## Tuning candidates: wins by split (validation only)",
        "",
        "| Dataset | Model | Config | Wins | Mean validation score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in tune_rows:
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['model']} | {r['config']} | {r['wins']}/10 "
            f"| {fmt(r['val_mean'], 4)} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
