"""Malformed-input regressions against the real survival model and metrics."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics.survival_curve import (
    concordance_td,
    integrated_binomial_loglik_td,
    integrated_brier_score_td,
)
from pyhealth.models import SurvivalMDN
from tests.core.survival_synthetic import make_survival_samples


class TestSurvivalInputValidation(unittest.TestCase):
    def test_model_requires_one_label_per_patient(self):
        samples = create_sample_dataset(
            samples=make_survival_samples(n=8, seed=2),
            input_schema={"features": "tensor"},
            output_schema={"survival": "tensor"},
            dataset_name="survival_input_validation",
        )
        batch = next(iter(get_dataloader(samples, batch_size=4)))
        model = SurvivalMDN(samples, num_components=2)
        with self.assertRaisesRegex(ValueError, "one.*label.*patient"):
            model(features=batch["features"], survival=batch["survival"][:1])
        repeated = batch["survival"][:1].repeat(4, 1)
        self.assertTrue(
            torch.isfinite(model(features=batch["features"], survival=repeated)["loss"])
        )

    def test_metrics_reject_invalid_curves(self):
        times = np.array([1.0, 2.0, 3.0])
        events = np.array([1, 1, 0])
        for curve in (
            [np.nan] * 3,
            [np.inf] * 3,
            [1.5] * 3,
            [-0.1] * 3,
            [0.9, 0.5, 0.7],
        ):
            for score in (
                concordance_td,
                integrated_brier_score_td,
                integrated_binomial_loglik_td,
            ):
                with (
                    self.subTest(curve=curve, metric=score.__name__),
                    self.assertRaises(ValueError),
                ):
                    score(times, events, np.tile(curve, (3, 1)), times)

    def test_metrics_reject_nonfinite_grid(self):
        for score in (
            concordance_td,
            integrated_brier_score_td,
            integrated_binomial_loglik_td,
        ):
            with self.subTest(metric=score.__name__), self.assertRaises(ValueError):
                score(
                    [1.0, 2.0, 3.0],
                    [1, 1, 0],
                    np.full((3, 3), 0.5),
                    [1.0, 2.0, np.nan],
                )

    def test_roundoff_tolerance_for_monotonicity(self):
        surv = np.tile([0.9, 0.7, 0.7 + 1e-8], (3, 1))
        self.assertTrue(
            np.isfinite(
                integrated_brier_score_td(
                    [1.0, 2.0, 3.0], [1, 1, 0], surv, [1.0, 2.0, 3.0]
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
