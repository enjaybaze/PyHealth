# Draft upstream PR description

Title: **Add GBSG/SUPPORT2 time-to-event tasks, released-split helpers and
survival-curve metrics for SurvivalMDN**

Open from `arittrosaha:aritra/survival-pipeline-integration` into
`sunlabuiuc:master`, with "Allow edits by maintainers" ticked. Reviewed in
[PR #2](https://github.com/enjaybaze/PyHealth/pull/2), reviewed by both contributors at
[`230889c`](https://github.com/arittrosaha/PyHealth/commit/230889ca322267fa158141923f38a291bcf2822d). The local
branch has since been rebased onto upstream `ea26a4b` and is awaiting publication
and CI on the rebased version. Everything below the line is the proposed PR body.

---

**Contributors:** Aritra Saha (@arittrosaha, asaha98@gatech.edu) and
Ugochukwu (NJ) Njoku (@enjaybaze), Georgia Tech CSE 6250

**Contribution type:** Dataset + Task, with survival-curve metrics and examples

**Paper:** Han, Goldstein & Ranganath, *Survival Mixture Density Networks*,
MLHC 2022 ([PMLR](https://proceedings.mlr.press/v182/han22a.html),
[arXiv](https://arxiv.org/abs/2208.10759))

**Scope agreed in:** [sunlabuiuc/PyHealth#1357](https://github.com/sunlabuiuc/PyHealth/issues/1357)

## Description

This PR builds a complete survival workflow around the `SurvivalMDN` model
merged in #1279. It adds a GBSG dataset and time-to-event tasks for GBSG and
the existing SUPPORT2 dataset, plus metrics for predicted survival curves. **The upstream model, its tests and the `Trainer` are unchanged.**

- **GBSG dataset:** `GBSGDataset`, plus `prepare_gbsg_csv`, which converts the
  public DeepSurv HDF5 file to CSV and records its provenance.
- **Time-to-event tasks:** `SurvivalTimeToEventGBSG` (7 features) and
  `SurvivalTimeToEventSupport2` (the paper's 8,873-patient complete-case cohort,
  27 features, built on the existing `Support2Dataset`). Both output `features`,
  `duration` and `event`, so they work with `SurvivalMDN` and the standard
  `Trainer` as they are. Optional arguments apply feature scaling fitted on
  training patients only, and the paper's time conventions.
- **Released splits:** `gbsg_released_split_ids` and
  `support2_released_split_ids` map the authors' ten released
  train/validation/test splits onto dataset records, so users can run the
  paper's exact protocol. `gbsg_paper_time_transform` and
  `support2_paper_time_transform` return the matching time conventions.
- **Survival-curve metrics** (`pyhealth.metrics.survival_curve`):
  time-dependent concordance (Antolini C^td), integrated Brier score and
  integrated binomial log-likelihood with IPCW weighting at censoring-quantile
  horizons, D-calibration (Haider et al.), and `survival_curve_metrics_fn`.
  These complement `pyhealth.metrics.survival`, added in #1368, which provides
  Harrell's C-index and single-horizon Brier/AUC scores. Our module adds
  IPCW curve comparisons, integrated scores and D-calibration with the
  Survival MDN evaluator's conventions. The APIs have distinct names and
  weighting/tie conventions; substituting one for the other changes scores.
- **Examples:** an end-to-end pipeline (`examples/survival_pipeline.py`), a
  loading and checkpoint-reload demo, and an optional benchmark that compares
  `SurvivalMDN` with Cox and DeepSurv on the released splits.
  `--standardize-gbsg` switches GBSG to training-only feature standardization.
  Cox and DeepSurv need two optional packages (`scikit-survival`, `pycox`),
  listed in the example's own `requirements.txt`. The library itself needs no
  new dependencies.

## Files to review

| Files | What to look at |
| --- | --- |
| `pyhealth/datasets/gbsg.py`, `pyhealth/datasets/configs/gbsg.yaml` | GBSG dataset, HDF5 conversion, released-split mapping |
| `pyhealth/datasets/support2.py` | Added SUPPORT2 released-split mapping and time convention (existing code untouched) |
| `pyhealth/tasks/survival_time_to_event.py` | The two tasks |
| `pyhealth/metrics/survival_curve.py` | C^td, IBS, IBLL, D-calibration |
| `examples/survival_pipeline.py`, `examples/survival_pipeline.md` | Start here: dataset → task → `SurvivalMDN` → `Trainer` → metrics |
| `examples/gbsg_survival_loading.py` | Loading and checkpoint reload |
| `examples/survival_benchmark/` | Optional MDN/Cox/DeepSurv benchmark and calibration replay |
| `tests/core/test_gbsg.py`, `test_gbsg_benchmark_preprocessing.py`, `test_support2_survival.py`, `test_survival_curve_metrics.py`, `test_survival_pipeline.py`, `test_survival_benchmark.py` | Tests |
| `docs/api/...`, `docs/survival_pipeline.rst` | API pages and a workflow guide |

All tests use synthetic data, download nothing and train for at most one
epoch. Benchmark tests skip when the optional packages are missing.

## How it was checked

- **Published-head CI:** at `230889c`, the full core suite passed (1,834 passed,
  105 skipped, 450 subtests), along with both NLP tests and contribution rules.
  [Python/NLP checks](https://github.com/enjaybaze/PyHealth/actions/runs/41726846075)
  and [contribution rules](https://github.com/enjaybaze/PyHealth/actions/runs/41726846093)
  both passed. These checks cover the published `230889c` head, before the
  local rebase onto `ea26a4b`. Update the head and CI links after publishing
  and validating the rebased branch; CI on that revision is still pending.
- **Local upstream compatibility:** At local head `f9767e6`, 2,185 core tests and 676 subtests passed, with 127 skips and no test failures. Both five-epoch native pilots, targeted API/workflow documentation, package build, lint and contribution checks passed. Local CPU Python 3.12.14/Torch 2.7.1 differs from CI. Python waited in process-tracker finalization after the core summary and was interrupted; a clean supported CI run on the rebased head is still required.
- **Metrics vs. the paper's evaluator:** with the same inputs, our C^td, IBS
  and IBLL match the authors' released `metrics.py` to within 2e-15. The one
  exception is described under "Notes for reviewers".
- **Released splits:** all 10 GBSG and all 10 SUPPORT splits map one-to-one onto
  the source records. With the paper's preprocessing, the tasks reproduce the
  authors' arrays to float32 precision.
- **Upstream model:** five-epoch pilots on GBSG and SUPPORT split 1, on CPU and
  CUDA, trained with finite losses, falling validation loss, and exact
  checkpoint reload.

## Results

### Ablation with PyHealth's SurvivalMDN (aligned-grid evaluation)

Does training-only feature standardization change GBSG performance with the
upstream model? Both conditions ran on all ten released splits, with the model
and training settings fixed (10 components, 3 × 32 hidden layers, softplus,
RMSprop at 1e-3, at most 100 epochs, patience 10, seed 1). Checkpoints were
chosen by validation NLL only, and time labels were identical in both
conditions. The primary comparison is test IBS at the widest horizon using
aligned prediction/scoring grids, separately from the original reproduction
results below that use the reference-compatible convention.

| Metric (test unless noted) | Raw | Standardized | Standardized − raw (SD) | Standardized better |
| --- | --- | --- | --- | --- |
| IBS, widest horizon (primary; lower is better) | 0.1741 | 0.1677 | −0.0064 (0.0051) | 9/10 splits |
| C^td (higher is better) | 0.6641 | 0.6730 | +0.0089 (0.0103) | 7/10 |
| IBLL (higher is better) | −0.5147 | −0.5003 | +0.0145 (0.0116) | 9/10 |
| Validation NLL (lower is better) | 0.3668 | 0.3631 | −0.0037 (0.0255) | 6/10 |

Means over the ten released splits; SDs are across the ten paired
differences. Splits reuse patients, so these are descriptive differences, not
independent tests. Training-only standardization improved test IBS and IBLL in
9 of 10 splits, with a smaller and less consistent gain in concordance.
Validation NLL barely changed. Standardized fits stopped earlier (best epoch
6–14 vs 10–28), and no fit reached the 100-epoch cap. The 20 fits took 13.4
minutes in a CPU-only run with 4 threads. All 20 records were checked independently,
and every best checkpoint reloads to the saved predictions within 1e-6.

D-calibration p-values by split (raw / standardized):

| Split | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw | 0.735 | 0.991 | 0.856 | 0.315 | 0.204 | 0.968 | 0.727 | 0.981 | 0.531 | 0.864 |
| Standardized | 0.393 | 0.979 | 0.980 | **0.035** | 0.901 | 0.526 | 0.885 | 0.854 | 0.292 | 0.595 |

Calibration is not rejected (p > 0.05) in 10/10 raw and 9/10 standardized
splits; standardized split 4 is the exception. These counts alone don't show
that standardization worsens calibration.

### Paper reproduction (our own implementation, not the upstream model)

The paper reproduction for our course project used our own Survival MDN
implementation, written before #1279 was merged. It is **not** the model in
this PR. Its code is preserved at tag
[`course-run-2026-10-29`](https://github.com/enjaybaze/PyHealth/tree/course-run-2026-10-29)
in our fork, and the analysis is on the
[`team-notes` branch](https://github.com/enjaybaze/PyHealth/tree/team-notes/experiments/results-2026-10-29).
It ran 300 fits on the authors' ten released splits, all on one CPU
environment. Each model's settings were chosen per split on validation data
only.

| Dataset | Model | C^td (± SD) | IBS | IBLL | Paper C^td / IBS / IBLL |
| --- | --- | --- | --- | --- | --- |
| GBSG | Cox | 0.658 ± 0.019 | 0.174 | −0.514 | 0.645 / 0.177 / −0.523 |
| GBSG | DeepSurv | 0.675 ± 0.016 | 0.167 | −0.497 | 0.663 / 0.172 / −0.509 |
| GBSG | Survival MDN | 0.663 ± 0.020 | 0.173 | −0.512 | 0.668 / 0.172 / −0.504 |
| SUPPORT | Cox | 0.596 ± 0.009 | 0.194 | −0.569 | 0.596 / 0.194 / −0.568 |
| SUPPORT | DeepSurv | 0.604 ± 0.008 | 0.191 | −0.563 | 0.609 / 0.190 / −0.559 |
| SUPPORT | Survival MDN | 0.626 ± 0.010 | 0.189 | −0.559 | 0.628 / 0.190 / −0.559 |

Means over the ten released test splits. IBS and IBLL are **reference-compatible**:
they use the authors' widest-horizon evaluator convention, separately from
the aligned-grid ablation above.

- Survival MDN matches the paper within two published standard errors on all
  six of its comparisons; 17 of the 18 comparisons pass overall. The exception
  is SUPPORT DeepSurv's IBLL, just outside (−0.5632 vs −0.559).
- On SUPPORT, MDN beats DeepSurv on C^td in all 10 splits (+0.022 on average).
  On GBSG, our DeepSurv is ahead of MDN in 8 of 10 splits, even with default
  settings; the paper reports the opposite. Our DeepSurv standardizes its
  inputs. In the ablation above, training-only standardization improved the
  upstream SurvivalMDN on GBSG and brought its performance close to our earlier
  DeepSurv results. This supports preprocessing as a possible contributor to
  the gap, although the comparison across experiments doesn't show how much it
  explains.
- On raw GBSG inputs, the upstream SurvivalMDN (C^td 0.664, IBS 0.174) is close
  to our own implementation with its default settings (0.662, 0.175). Their
  aggregate performance is similar, but individual predictions differ, so the
  two are not equivalent implementations.
- Ablations: one mixture component fits clearly worse. Under our settings,
  three components give most of the likelihood gain. The exponential time
  transform fits SUPPORT's validation data better than softplus in all 10
  splits, but the test results are mixed.

## Timeline and overlap with #1279

We developed our implementation independently. #1279 was merged while our work
was in progress, and we found it on Oct 29. Every date below is from our
fork's commit history or our correspondence.

| Date (2026) | Event |
| --- | --- |
| Oct 16 | Our Survival MDN model and survival-curve metrics committed to our fork; GBSG dataset and task added |
| Oct 18 | Released GBSG split support |
| Oct 22 | SUPPORT2 task and released-split mapping |
| **Oct 23** | **#1279 (`SurvivalMDN`, Neil Hajela) merged upstream** |
| Oct 26–27 | D-calibration, input checks and the Cox/DeepSurv/MDN benchmark |
| Oct 29 | 300-fit reproduction run on our own model; overlap with #1279 found while preparing this PR |
| Oct 29–30 | Course staff and maintainers contacted; Dataset + Task scope approved; this branch rebuilt on upstream `master` around the merged model |
| Oct 30–31 | Optional GBSG standardization added; CI passes at `e32a85b` |
| Nov 3 | Twenty upstream-model ablation outputs independently verified; slow tests trimmed and rerun by both contributors |
| Nov 4 | Local rebase onto upstream `ea26a4b`; compatibility checks and metrics API documentation updated |

**What we dropped:** our own `pyhealth/models/survival_mdn.py`. It used the same
file and class name as #1279, so it stays in our fork for the course
reproduction only.

**Who built what:**

| Component | Built by |
| --- | --- |
| GBSG dataset, HDF5 conversion and GBSG task | Aritra |
| GBSG released-split mapping and time convention | NJ |
| SUPPORT2 task, released-split mapping and time convention | NJ |
| Survival-curve metrics (C^td, IBS, IBLL, `survival_curve_metrics_fn`) | NJ |
| D-calibration and input validation | Aritra |
| Cox/DeepSurv/MDN benchmark and calibration replay | Aritra |
| Integration with the upstream model (duration/event outputs), pipeline example and workflow docs | Aritra |
| GBSG standardization option and ablation design | Aritra |
| Our original model and the 300-fit reproduction run and analysis | NJ (run on NJ's VM; baselines and verification by Aritra) |

## Notes for reviewers

- **Time conventions.** Following the paper, GBSG times are
  `months / cohort maximum + 0.002` and SUPPORT times are
  `days / 365.25 + 0.001`. The GBSG maximum uses the full cohort, as the
  authors' released files do. The task docstrings and
  `docs/survival_pipeline.rst` say this explicitly. By default, both tasks keep
  the source units.
- **Widest-horizon IBS/IBLL.** When the last test time is censored, which
  happens on every released GBSG and SUPPORT test split, the authors' evaluator
  pairs its widest-horizon curves with a shifted time grid. PyHealth's metrics
  keep the grids aligned. The benchmark reports both, labeled separately. In our
  GBSG runs, the difference was at most 0.0017 IBS on any split.
- **Data and licensing.** No data is committed. The helpers only read files the
  user downloads: GBSG from the
  [DeepSurv repository](https://github.com/jaredleekatzman/DeepSurv), SUPPORT2
  from [Vanderbilt Biostatistics](https://hbiostat.org/data/repo/supportdesc),
  and the released splits from the
  [authors' repository](https://github.com/XintianHan/Survival-MDN).
- **Docs placement.** `docs/index.rst` lists the workflow guide under Quick
  Navigation. We are retaining that placement for review and can move the
  page if maintainers prefer tutorials/examples.


