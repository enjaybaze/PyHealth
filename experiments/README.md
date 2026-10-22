# Experiment runner (T5, used in T8)

`run_survival_mdn.py` trains SurvivalMDN on the authors' 10 released GBSG
splits and records everything T8 needs. It lives here, not in the PyHealth
PR. It imports PyHealth from `feat/survival-mdn` (198659d or later).

## What one fit does

1. Load the split from the released `.npz` files with
   `gbsg_released_split_ids`. Use the paper protocol: raw features, and
   `months / 87.36 + 0.002` from `gbsg_paper_time_transform`.
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
fits. Two fixes make that true:

- **Fresh subsets per fit.** PyHealth's subsets are litdata streaming
  datasets. Their shuffle order depends on an epoch counter that advances
  every time the subset is iterated, and on a seed that defaults to 42.
  Reusing subsets made a fit depend on the fits before it, and every seed
  shuffled the same way. Each fit now builds new subsets, and the training
  shuffle uses the fit's seed.
- **Separate trainer logs.** PyHealth adds a log file handler per
  `Trainer` and never removes it, so later fits wrote into earlier fits'
  logs. The runner closes those handlers after each fit.

Checked: split 1, K=10 gives validation NLL 0.3557 both alone and after a
K=1 fit, and seed 2 gives a different fit (0.3879).

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

SUPPORT is not wired in yet. It needs a loader on Aritra's T4 side, and it
has about 5 times as many patients, so expect slower fits. Time a pilot
once it is available.

## Commands

```bash
conda activate survmdn
cd ~/PyHealth && git pull --ff-only origin feat/survival-mdn
cd ~ && git clone -b team-notes https://github.com/enjaybaze/PyHealth.git survmdn-notes
cd ~/survmdn-notes/experiments

# main config on all 10 splits
python run_survival_mdn.py --root ~/survmdn-data/csv \
    --released-dir ~/survmdn-data/Survival-MDN/data/gbsg --out ~/runs/main

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
