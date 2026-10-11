"""Parity check: PyHealth survival_curve metrics vs the authors' metrics.py.

Usage: python -I parity_check.py <path to authors' repo> <path to pyhealth repo>
The authors' code is imported only to compute reference numbers here; none of
it is copied into PyHealth.
"""

import importlib.util
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
ref_path, pyhealth_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, pyhealth_path)
from pyhealth.metrics.survival_curve import (  # noqa: E402
    censoring_survival,
    horizon,
    survival_curve_metrics_fn,
)

spec = importlib.util.spec_from_file_location("ref_metrics", f"{ref_path}/metrics.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def make_case(n, seed, last_censored):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    mu = 2.0 + 0.6 * x[:, 0] - 0.4 * x[:, 1]
    t_event = np.exp(mu + 0.5 * rng.standard_normal(n))
    t_cens = rng.exponential(12.0, n)
    times = np.round(np.minimum(t_event, t_cens), 1) + 0.1  # creates ties
    events = (t_event <= t_cens).astype(float)
    order = np.argsort(times, kind="stable")
    times, events, mu = times[order], events[order], mu[order]
    events[-1] = 0.0 if last_censored else 1.0
    # A deliberately imperfect model: noisy log-normal curves.
    mu_hat = mu + 0.3 * rng.standard_normal(n)

    def predict(grid):
        from scipy.stats import norm

        z = (np.log(np.asarray(grid))[None, :] - mu_hat[:, None]) / 0.6
        return norm.sf(z)

    return times, events, predict


def reference(times, events, predict):
    t = torch.tensor(times, dtype=torch.float64)
    lab = torch.tensor(events, dtype=torch.float64)
    eval_t = np.unique(times)
    s_eval = predict(eval_t)
    out = {}
    for eps, key in [(1e-8, "1e-08"), (0.2, "0.2"), (0.4, "0.4")]:
        m = ref.IPWCIndexMeter(eps=eps)
        m.add(
            {
                "t": t,
                "eval_t": torch.tensor(eval_t),
                "cum_hazard_seqs": torch.tensor(-np.log(s_eval).T),
            },
            lab,
        )
        out[f"c_td@{key}"] = m.value()[0]
    g_t = censoring_survival(times, events, times)
    for eps, key in [(0, "1e-08"), (0.2, "0.2"), (0.4, "0.4")]:
        # Grid the authors' model code uses for this horizon.
        t_max = times[-1] if eps == 0 else times[g_t > eps].max()
        grid = np.linspace(times[0], t_max, ref.NUM_INT_STEPS)
        seqs = torch.tensor(predict(grid).T)
        for name, cls in [("ibs", ref.BrierScoreMeter), ("ibll", ref.BinomialLogLikelihoodMeter)]:
            m = cls(eps=eps)
            m.add({"t": t, "survival_seqs" if eps == 0 else f"survival_seqs_{eps}": seqs}, lab)
            out[f"{name}@{key}"] = float(m.value())
    return out


for last_censored in (False, True):
    for seed in (0, 1, 2):
        times, events, predict = make_case(400, seed, last_censored)
        ours = survival_curve_metrics_fn(times, events, predict)
        theirs = reference(times, events, predict)
        worst = max(abs(ours[k] - theirs[k]) for k in theirs)
        widest_gap = max(abs(ours[k] - theirs[k]) for k in theirs if "1e-08" in k)
        narrow_gap = max(abs(ours[k] - theirs[k]) for k in theirs if "1e-08" not in k)
        print(
            f"last_censored={last_censored} seed={seed} "
            f"censored={1 - events.mean():.2f} "
            f"tau_widest={horizon(times, events, 1e-8):.1f} t_max={times[-1]:.1f} | "
            f"max|diff| widest={widest_gap:.2e} narrower={narrow_gap:.2e}"
        )
        if seed == 0:
            for k in sorted(theirs):
                print(f"    {k:12s} ours={ours[k]: .6f} ref={theirs[k]: .6f}")
