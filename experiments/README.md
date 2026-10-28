# Experiment runner (T5, used in T8)

> **Superseded for T8 (Oct 27).** Aritra's PR #1, merged into
> `feat/survival-mdn` as `d473380`, moved this runner into the PR as
> `examples/survival_benchmark/run_survival_mdn.py` and added D-calibration,
> exact observed-time probabilities and versioned run folders. On split 1
> it reproduces this runner's numbers exactly (GBSG val NLL 0.3557,
> SUPPORT 0.5447). Run the final experiments with `run_mdn_plan.py`, which
> runs the plan in `course-run-plan/` through that runner. This file
> and `run_survival_mdn.py` stay as the record of the pilots below.

## Final MDN runs: `run_mdn_plan.py`

`course-run-plan/` is Aritra's proposed plan, from his Oct 26 handoff
(approval status: see `course-run-plan/budget-proposal.json`): 300
fits in total, of which NJ owns the 180 MDN fits. `run_mdn_plan.py` swaps
the plan's placeholder paths for real ones and runs the fits in two stages:

1. **Tuning (60 fits):** K=10 softplus, the slower learning rate and the
   wider network, on 10 splits × 2 datasets.
2. **Ablations (120 fits):** K = 1, 2, 3, 5, 20, and K=10 with the exp
   transform.

The script stops at the first failure and shows that fit's log. Rerunning
the same command reuses every completed fit.

```bash
conda activate survmdn
cd ~/survmdn-notes/experiments
python run_mdn_plan.py --pyhealth ~/PyHealth \
    --gbsg-root ~/survmdn-data/csv --support-root ~/survmdn-data/support2 \
    --released-root ~/survmdn-data/Survival-MDN --runs-root ~/runs/course \
    --stage tuning
```

Checked Oct 27 with 2 fits in the sandbox: GBSG split 1 gave val NLL
0.3557, matching the pilot, and a rerun reused both fits.

`run_survival_mdn.py` trains SurvivalMDN on the authors' 10 released GBSG
or SUPPORT splits (`--dataset gbsg|support2`) and records everything T8
needs. It lives here, not in the PyHealth
PR. It imports PyHealth from `feat/survival-mdn` (198659d or later).

## What one fit does

1. Load the split from the released `.npz` files and apply the paper
   protocol:
   - **GBSG:** `gbsg_released_split_ids`; raw features; times are
     `months / 87.36 + 0.002`.
   - **SUPPORT:** `support2_released_split_ids`; all 27 features are
     standardized on that split's training patients; times are
     `days / 365.25 + 0.001`. The fitted means and scales are saved in
     each record.
2. Train with the PyHealth `Trainer` using the authors' defaults:
   - RMSprop, learning rate 1e-3
   - batch size 512, gradient clipping at 100
   - up to 100 epochs, early stopping after 10 epochs without a better
     validation NLL
   - the best validation epoch is reloaded at the end
3. Score the test set with `survival_curve_metrics_fn` (C^td, IBS and IBLL
   at the 1e-8, 0.2 and 0.4 horizons).
4. Also compute the **reference-compatible** widest-horizon IBS and IBLL
   (see below).
5. Save the outputs:

   ```
   <out>/<config>/split_<n>/seed_<s>/
       record.json       config, counts, val NLL, metrics, best epoch,
                         epochs run, seconds, environment, git commits
       predictions.npz   test source IDs, times, events, a 200-point grid,
                         survival curves
       trainer/          PyHealth log, best.ckpt, last.ckpt
   <out>/summary.json, summary.csv
   ```

`summary.*` reports, per config, the mean and SD across splits of every
metric. Seeds are averaged within a split first. Rows are sorted by mean
validation NLL. **Pick configs by validation NLL only. Test metrics are for
reporting.**

Completed fits are skipped, so a stopped run resumes with the same command.

## Reproducibility

The same fit gives identical results whether it runs alone or after other
fits. The runner passes each fit's seed to `get_dataloader` for the
training shuffle and builds fresh subsets per fit. It also closes PyHealth's
per-`Trainer` log handlers after each fit, since PyHealth never removes
them and later fits would otherwise write into earlier fits' logs.

Before the branch merged upstream PyHealth 2.1.0 (#1280, "seeded per-loader
shuffles"), we found the same shuffle problems independently: litdata
subsets carried an epoch counter between fits and ignored our seed. The
runner worked around them then. Since the merge it uses the upstream API.

Checked after the merge: split 1, K=10 gives validation NLL 0.3557 both
alone and after a K=1 fit, the same as the Oct 22 pilot.

## Reference-compatible widest-horizon IBS/IBLL (t5-10)

The authors' evaluator misaligns its grid at the widest horizon when the
last test time is censored (`parity/README.md`). The last test time is
censored in **every** released test split for GBSG (10/10) and SUPPORT
(10/10). For METABRIC it is 6/10. So the paper's Table 1 widest-horizon
IBS and IBLL always include this effect.

The runner reports two sets of numbers:

- `test_metrics` holds PyHealth's aligned values.
- `test_metrics_reference_compatible` reproduces the authors' computation,
  for comparison with Table 1.

`parity/reference_compatible_check.py` checks the second set against the
authors' `metrics.py`: they agree to within 2e-15 on all six synthetic
cases. On real GBSG the gap is small. Across the 10 splits, mean IBS is
0.1756 (aligned) vs 0.1752 (reference-compatible). The largest single-split
gap is 0.0017 IBS and 0.0043 IBLL, on split 4.

## Pilot, Oct 22 (`pilot-2026-10-22/`)

Default config (K=10, softplus), seed 1, CPU in the sandbox, untuned:

| Split | Val NLL | C^td | IBS | IBS ref | IBLL | IBLL ref | Best / run epochs | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3557 | 0.656 | 0.1790 | 0.1788 | -0.5307 | -0.5303 | 28 / 39 | 14.2 |
| 2 | 0.3988 | 0.648 | 0.1707 | 0.1706 | -0.5006 | -0.5003 | 16 / 27 | 10.2 |
| 3 | 0.3130 | 0.700 | 0.1687 | 0.1688 | -0.5022 | -0.5023 | 16 / 27 | 9.7 |
| 4 | 0.3688 | 0.633 | 0.1817 | 0.1800 | -0.5361 | -0.5318 | 15 / 26 | 10.5 |
| 5 | 0.3535 | 0.655 | 0.1638 | 0.1633 | -0.4858 | -0.4846 | 12 / 23 | 8.7 |
| 6 | 0.3986 | 0.639 | 0.1710 | 0.1706 | -0.5052 | -0.5043 | 9 / 20 | 6.9 |
| 7 | 0.4054 | 0.660 | 0.1820 | 0.1814 | -0.5290 | -0.5277 | 10 / 21 | 8.1 |
| 8 | 0.3427 | 0.681 | 0.1748 | 0.1749 | -0.5189 | -0.5190 | 27 / 38 | 14.0 |
| 9 | 0.3628 | 0.648 | 0.1931 | 0.1931 | -0.5630 | -0.5630 | 9 / 20 | 8.0 |
| 10 | 0.3811 | 0.672 | 0.1707 | 0.1706 | -0.5053 | -0.5051 | 17 / 28 | 11.0 |
| **Mean ± SD** | 0.3680 | 0.659 ± 0.020 | 0.176 ± 0.009 | 0.175 ± 0.009 | -0.518 ± 0.022 | -0.517 ± 0.022 | | 10.1 |

These are untuned smoke numbers, not results. Every fit stopped early,
after 20–39 epochs. The 0.2 and 0.4 horizons gave identical scores on all
10 splits, and matched the widest horizon on 6 of them. GBSG's censoring
curve mostly stays above 0.4, so all three horizons often land on the same
time.

## Timing for the budget (t5-07)

On GBSG, a fit takes about 10 s on CPU. Building the dataset cache adds
about 2 minutes once per output folder. Rough GBSG fit counts:

| Block | Fits | Time at ~10 s |
| --- | --- | --- |
| Main run (1 config × 10 splits) | 10 | ~2 min |
| A1: K ∈ {1,2,3,5,10,20} | 60 | ~10 min |
| A2: exp vs softplus | 10 new | ~2 min |
| Tuning, e.g. 30 configs × 10 splits | 300 | ~50 min |

SUPPORT pilot, Oct 22 (default config, splits 1–3, CPU in the sandbox,
untuned): about 26 s per fit, with early stopping at 21–25 epochs. C^td
was 0.620 ± 0.007, IBS 0.189 ± 0.002 and IBLL −0.559 ± 0.008. The
reference-compatible IBS matched the aligned IBS to 4 decimal places.
Rough SUPPORT time at about 26 s per fit: main runs (10 fits) ~5 min,
ablations (80 fits) ~35 min, 300-fit tuning ~2.2 h. NJ's VM CPU ran about
1.6× slower than the sandbox on GBSG, so expect about 40 s per fit there.

On NJ's VM (Oct 22), SUPPORT took about 38 s per fit and matched the
sandbox to 3 decimal places (split 3 val NLL 0.4807 vs 0.4803). The input
CSVs are byte-identical (sha256 `3275ece4…` with `sno` added; the original
is `79621945…`), so the gap comes from the two CPUs' floating-point
arithmetic, not the data. GBSG matched exactly. Run every reported
experiment on one machine.

## Commands

```bash
conda activate survmdn
cd ~/PyHealth && git pull --ff-only origin feat/survival-mdn
cd ~ && git clone -b team-notes https://github.com/enjaybaze/PyHealth.git survmdn-notes
cd ~/survmdn-notes/experiments

# main config on all 10 splits
python run_survival_mdn.py --root ~/survmdn-data/csv \
    --released-dir ~/survmdn-data/Survival-MDN/data/gbsg --out ~/runs/main

# SUPPORT: support2.csv with its first column named sno
python run_survival_mdn.py --dataset support2 --root ~/survmdn-data/support2 \
    --released-dir ~/survmdn-data/Survival-MDN/data/support --out ~/runs/support-main

# mixture-size and transform ablations
python run_survival_mdn.py --root ~/survmdn-data/csv \
    --released-dir ~/survmdn-data/Survival-MDN/data/gbsg --out ~/runs/ablation \
    --num-components 1 2 3 5 10 20 --transform softplus exp
```

`--root` is a folder containing `gbsg.csv`. Make one with
`prepare_gbsg_csv(<gbsg.h5>, <folder>)`, or let
`examples/gbsg_survival_loading.py --source-h5` write it. Use `--config` (a
JSON file with `"model"` and `"train"` objects) for tuning trials, and
`--seeds 1 2 3` for more than one seed per split.
