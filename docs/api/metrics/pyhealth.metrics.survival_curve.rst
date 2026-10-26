pyhealth.metrics.survival_curve
===================================

Time-dependent metrics for models that predict a full survival curve
``S(t | x)`` per patient from right-censored data, following the evaluation
protocol of Han et al., "Survival Mixture Density Networks" (MLHC 2022):
inverse-probability-of-censoring weighted time-dependent concordance
(Antolini's C^td), integrated Brier score and integrated binomial
log-likelihood, each truncated at horizons set by the censoring distribution.

.. code-block:: python

    from pyhealth.metrics import survival_curve_metrics_fn

    # predict(grid) returns survival probabilities of shape [N, len(grid)]
    def predict(grid):
        return model.predict_survival(features, torch.as_tensor(grid)).numpy()

    scores = survival_curve_metrics_fn(times, events, predict)
    # {'c_td': ..., 'ibs': ..., 'ibll': ..., 'c_td@0.2': ..., ...}

The evaluator chooses the time grids, so every model is scored identically.
These metrics complement risk-score metrics such as Harrell's C-index.

.. currentmodule:: pyhealth.metrics.survival_curve

.. autofunction:: survival_curve_metrics_fn

.. autofunction:: concordance_td

.. autofunction:: integrated_brier_score_td

.. autofunction:: integrated_binomial_loglik_td

.. autofunction:: censoring_survival

.. autofunction:: horizon

.. autofunction:: integration_grid

.. autofunction:: d_calibration

D-calibration receives each patient's survival probability at their own
observed time, preserving patient order. It returns censored fractional bin
counts, the chi-square statistic and its p-value. It follows Haider et al.,
Appendix B.5, Algorithm 2, including its 1e-5 floor and upper-inclusive bins.
Save these conventions with results. A large p-value is not evidence of
individual or clinical calibration.

Metric inputs must be finite probabilities in [0, 1], nonincreasing within
1e-6 along finite, nonnegative and strictly increasing time grids.
