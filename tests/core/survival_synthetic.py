"""Synthetic right-censored survival data for SurvivalMDN tests.

Shared fixture for the model and data tracks (T1). Samples follow the T1
contract: ``{"patient_id", "features": [D floats], "survival": [time, event]}``
with ``event = 1`` for an observed event and ``0`` for right censoring.

Event times are log-normal with a covariate-dependent location, so the true
survival curve is known in closed form and model fits can be checked against
it. Censoring times are exponential and independent of the event times, with
a scale chosen so the sample's censoring rate is close to the requested one.
Everything is generated locally; nothing is downloaded.
"""

from collections.abc import Sequence
from math import erf, sqrt

import numpy as np

# True model: log T = BETA0 + x @ BETA + SIGMA * eps, eps ~ N(0, 1).
BETA0 = 2.0
BETA = (0.6, -0.4, 0.0)
SIGMA = 0.5


def _location(features: np.ndarray) -> np.ndarray:
    """Returns the log-time location ``BETA0 + x @ beta`` for each row."""
    beta = np.zeros(features.shape[1])
    k = min(len(BETA), features.shape[1])
    beta[:k] = BETA[:k]
    return BETA0 + features @ beta


def make_survival_samples(
    n: int = 200,
    n_features: int = 3,
    censoring_rate: float = 0.4,
    seed: int = 0,
) -> list[dict]:
    """Generates censored survival samples in the T1 contract shape.

    Args:
        n: number of patients.
        n_features: covariates per patient, drawn from N(0, 1).
        censoring_rate: target fraction of censored patients, in [0, 1).
            ``0`` gives no censoring.
        seed: random seed; the same seed gives the same samples.

    Returns:
        A list of ``n`` sample dictionaries.

    Examples:
        >>> samples = make_survival_samples(n=4, seed=1)
        >>> sorted(samples[0])
        ['features', 'patient_id', 'survival']
        >>> time, event = samples[0]["survival"]
        >>> time > 0 and event in (0.0, 1.0)
        True
    """
    if not 0 <= censoring_rate < 1:
        raise ValueError("censoring_rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features))
    event_times = np.exp(_location(x) + SIGMA * rng.standard_normal(n))

    if censoring_rate == 0:
        times, events = event_times, np.ones(n)
    else:
        unit = rng.exponential(1.0, size=n)
        # Pick the exponential scale whose censoring rate is closest to target.
        scales = np.geomspace(0.01, 100.0, 400) * np.median(event_times)
        rates = [(unit * s < event_times).mean() for s in scales]
        scale = scales[int(np.argmin(np.abs(np.array(rates) - censoring_rate)))]
        censor_times = unit * scale
        events = (event_times <= censor_times).astype(float)
        times = np.minimum(event_times, censor_times)

    return [
        {
            "patient_id": f"p{i:04d}",
            "features": [float(v) for v in x[i]],
            "survival": [float(times[i]), float(events[i])],
        }
        for i in range(n)
    ]


def true_survival(
    features: Sequence[Sequence[float]], t_grid: Sequence[float]
) -> np.ndarray:
    """True survival probabilities ``S(t | x)`` under the generating model.

    Args:
        features: covariates, shape ``[B, D]``.
        t_grid: positive times, shape ``[T]``.

    Returns:
        Array of shape ``[B, T]``.

    Examples:
        >>> s = true_survival([[0.0, 0.0, 0.0]], [np.exp(BETA0)])
        >>> round(float(s[0, 0]), 3)
        0.5
    """
    x = np.asarray(features, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    z = (np.log(t)[None, :] - _location(x)[:, None]) / SIGMA
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    return 1.0 - cdf
