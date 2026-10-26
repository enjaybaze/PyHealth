"""Independent fixed-count fixtures for the public D-calibration function."""

import unittest

import numpy as np
from scipy.stats import chi2

from pyhealth.metrics import d_calibration


class TestDCalibration(unittest.TestCase):
    def test_uncensored_uniform_deciles(self):
        result = d_calibration(np.arange(0.05, 1.0, 0.1), np.ones(10))
        np.testing.assert_allclose(result["bin_counts"], np.ones(10))
        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_paper_censored_example(self):
        result = d_calibration([0.25], [0])
        np.testing.assert_allclose(
            result["bin_counts"], [0.4, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_mixed_hand_computed_counts_and_statistic(self):
        result = d_calibration([0.25, 0.25, 1.0], [0, 1, 1], num_bins=4)
        np.testing.assert_allclose(result["bin_counts"], [2, 0, 0, 1])
        self.assertAlmostEqual(result["statistic"], 11 / 3)
        self.assertAlmostEqual(result["p_value"], chi2.sf(11 / 3, 3))

    def test_algorithm_two_boundary_convention(self):
        result = d_calibration([0.0, 0.25, 0.5, 0.75, 1.0], [1] * 5, num_bins=4)
        np.testing.assert_allclose(result["bin_counts"], [2, 1, 1, 1])

    def test_censoring_at_one_is_uniform(self):
        result = d_calibration([1.0], [0], num_bins=4)
        np.testing.assert_allclose(result["bin_counts"], [0.25] * 4)

    def test_extreme_predictions_use_declared_floor(self):
        result = d_calibration([0.0, 1e-15, 1.0], [0, 0, 1], num_bins=4)
        np.testing.assert_allclose(result["bin_counts"], [2, 0, 0, 1])
        self.assertEqual(result["probability_floor"], 1e-5)

    def test_every_patient_contributes_one(self):
        for p in np.linspace(0, 1, 101):
            for event in (0, 1):
                result = d_calibration([p], [event])
                self.assertAlmostEqual(sum(result["bin_counts"]), 1.0)
                self.assertTrue(np.all(np.asarray(result["bin_counts"]) >= 0))

    def test_rejects_invalid_input(self):
        for p, e in [
            ([], []),
            ([0.5], []),
            ([[0.5]], [[1]]),
            ([np.nan], [1]),
            ([1.1], [0]),
            ([-0.1], [1]),
            ([0.5], [2]),
        ]:
            with self.subTest(probability=p, events=e), self.assertRaises(ValueError):
                d_calibration(p, e)
        for bins in [1, 0, 2.5, True]:
            with self.assertRaises(ValueError):
                d_calibration([0.5], [1], bins)


if __name__ == "__main__":
    unittest.main()
