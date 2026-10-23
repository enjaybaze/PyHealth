# Draft PR description

Title: **Add Survival MDN: survival mixture density model, survival-curve metrics, and GBSG/SUPPORT2 time-to-event tasks**

Open from `enjaybaze:feat/survival-mdn` into `sunlabuiuc:master` as a
**draft**, with "Allow edits by maintainers" ticked. Everything below the
line is the PR body.

---

## Summary

This PR adds a full time-to-event pipeline to PyHealth, reproducing
**Survival Mixture Density Networks** (Han, Goldstein & Ranganath, MLHC
2022; [PMLR](https://proceedings.mlr.press/v182/han22a.html),
[arXiv](https://arxiv.org/abs/2208.10759)). It has four parts:

- **Model:** `SurvivalMDN`, a neural network that outputs a full survival
  curve S(t | x) for each patient.
- **Metrics:** C^td, IBS and IBLL, the paper's evaluation metrics.
- **Data:** a GBSG dataset loader, plus time-to-event tasks for GBSG and
  SUPPORT2.
- **Paper splits:** helpers that load the authors' ten released splits
  with their exact preprocessing.

PyHealth has no survival mode today. Existing SUPPORT2 tasks predict
fixed-horizon probabilities, not censored times. This PR keeps the
existing `Trainer` unchanged. The label is a packed `survival: [time,
event]` tensor, so the model runs with `mode=None` and trains on its own
censored likelihood.

**Contribution type:** model + metrics + datasets/tasks.

**Contributors:** NJ Njoku (@enjaybaze) and Aritra Saha (@arittrosaha),
Georgia Tech CSE 6250.

## What's included

| Area | Files | What it adds |
| --- | --- | --- |
| Model | `pyhealth/models/survival_mdn.py` | `SurvivalMDN`: a Gaussian mixture over a transformed time (softplus or exp), log-domain censored negative log-likelihood, `predict_survival(features, t_grid)`. Architecture follows the authors' reference implementation (bias-free Linear → BatchNorm → PReLU blocks, same head initialization and clamps). `time_scale` is a buffer, so checkpoints carry it. |
| Metrics | `pyhealth/metrics/survival_curve.py` | IPCW time-dependent concordance (Antolini C^td), integrated Brier score and integrated binomial log-likelihood at censoring-quantile horizons, `survival_curve_metrics_fn`. |
| GBSG data | `pyhealth/datasets/gbsg.py`, `configs/gbsg.yaml` | `prepare_gbsg_csv` (DeepSurv HDF5 → CSV with provenance), `GBSGDataset`, `gbsg_released_split_ids`, `gbsg_paper_time_transform`. |
| SUPPORT2 data | `pyhealth/datasets/support2.py` | `support2_released_split_ids`, `support2_paper_time_transform`. Builds on the existing `Support2Dataset`. |
| Tasks | `pyhealth/tasks/survival_time_to_event.py` | `SurvivalTimeToEventGBSG` (7 features) and `SurvivalTimeToEventSupport2` (8,873-patient benchmark cohort, 27 features). Both emit `survival: [time, event]` and take optional training-only scaling and a time divisor/offset. |
| Examples | `examples/gbsg_survival_prediction_survival_mdn.py`, `examples/gbsg_survival_loading.py` | Train, evaluate and compare mixture sizes and transforms. `--synthetic` needs no downloads and runs in seconds. Loading and checkpoint demo. |
| Docs | `docs/api/...` | Pages for the model, metrics, GBSG dataset and both tasks; SUPPORT2 page extended. |
| Tests | `tests/core/test_survival_mdn.py`, `test_survival_curve_metrics.py`, `test_gbsg.py`, `test_support2_survival.py`, `test_survival_mdn_example.py`, `survival_synthetic.py` | 60 tests. They check the math against closed-form densities, verify p(t) = −dS/dt with autograd, recover a known distribution, use hand-computed metric fixtures, and cover released-split mapping, checkpoint reload and the example end to end. No downloads; seconds to run. |

## How it was checked

- **Model vs. the authors' code.** With the same weights, our mixture
  parameters and survival curves match the authors' model to float
  precision, in both train and eval mode.
- **Metrics vs. the authors' evaluator.** Agreement is within 2e-15 at
  every horizon, with one exception. When the last test time is censored,
  the authors' evaluator pairs its widest-horizon curves with a shifted
  time grid. PyHealth keeps the grids aligned, so its values differ there.
  The docs explain this.
- **Released splits.** All 10 GBSG and all 10 SUPPORT splits map
  one-to-one onto the source records. With the paper's preprocessing, the
  tasks reproduce the authors' arrays to float32 precision: SUPPORT
  standardized features within 7.6e-6, times within 4e-7.
- **CI.** `tools/check_pr_rules.py` passes, and the new tests pass under
  `python -m unittest discover -t tests -s tests/core`.

## Data and licensing

The authors' released split files come from their repository, which has
no license. This PR only *reads* files the user downloads; none are
committed. GBSG comes from the
[DeepSurv benchmark](https://github.com/jaredleekatzman/DeepSurv). SUPPORT2
comes from [Vanderbilt Biostatistics](https://hbiostat.org/data/repo/supportdesc).

## Notes for reviewers

- The model lives at `pyhealth/models/survival_mdn.py`, and the metrics in
  a new `pyhealth/metrics/survival_curve.py` with distinct function names.
  That avoids overlapping with PR #948, which adds `pyhealth/metrics/survival.py`.
- This branch includes a merge of current `master` (PyHealth 2.1.0).
  The only conflicts were adjacent import lines.
- Experiment results for the paper comparison will follow in a later
  update. This draft is for review of the API and implementation.
