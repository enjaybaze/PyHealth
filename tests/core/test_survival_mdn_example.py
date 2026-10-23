"""The SurvivalMDN example runs end to end on its synthetic data."""

import math
import tempfile
import unittest
from pathlib import Path

from examples.gbsg_survival_prediction_survival_mdn import (
    SyntheticSplits,
    fit_and_score,
    synthetic_samples,
)


class TestSurvivalMDNExample(unittest.TestCase):
    def test_synthetic_data(self):
        samples = synthetic_samples(n=200)
        self.assertEqual(len(samples), 200)
        self.assertTrue(all(s["survival"][0] > 0 for s in samples))
        censored = sum(s["survival"][1] == 0 for s in samples) / len(samples)
        self.assertTrue(0.05 < censored < 0.7)

    def test_fit_and_score_is_finite_and_repeatable(self):
        splits = SyntheticSplits()
        with tempfile.TemporaryDirectory() as tmp:
            runs = [
                fit_and_score(splits, {"num_components": 2}, 2, 1, Path(tmp) / str(i))
                for i in range(2)
            ]
        (nll, scores), (nll_again, scores_again) = runs
        self.assertTrue(math.isfinite(nll))
        self.assertEqual(nll, nll_again)
        for key in ("c_td", "ibs", "ibll"):
            self.assertTrue(math.isfinite(scores[key]))
            self.assertEqual(scores[key], scores_again[key])


if __name__ == "__main__":
    unittest.main()
