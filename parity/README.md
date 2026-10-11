# Metric parity check (T5)

Compares `pyhealth/metrics/survival_curve.py` (commit `77ee223` on
`feat/survival-mdn`) with the authors' evaluator `metrics.py` in
[XintianHan/Survival-MDN](https://github.com/XintianHan/Survival-MDN)
(commit in `reference_commit.txt`). The authors' code is only imported here
to compute reference numbers; none of it is in the PR. It has no license
file, so it is not redistributed on this branch either.

## Run

```bash
git clone https://github.com/XintianHan/Survival-MDN.git /tmp/ref
pip install lifelines scipy
python -I parity_check.py /tmp/ref /path/to/PyHealth
```

## Result (Oct 10, 2026), see `parity_output.txt`

Six synthetic test sets of 400 patients, about 47-51% censored, with tied
times and an imperfect log-normal model:

| Metric and horizon | Max abs difference |
| --- | --- |
| C^td at 1e-8, 0.2, 0.4 | < 2e-15 |
| IBS and IBLL at 0.2 and 0.4 | < 2e-15 |
| IBS and IBLL at the widest horizon, last observed time an event | < 2e-15 |
| IBS and IBLL at the widest horizon, last observed time censored | up to 0.13 |

## The one difference: a grid misalignment in the reference

For the widest horizon (`eps = 0`), the reference model code evaluates the
survival curves on `linspace(t_min, t_max, 1000)` where `t_max` is the
largest observed time, but `BrierScoreMeter` and
`BinomialLogLikelihoodMeter` score them as if they were on
`linspace(t_min, tau, 1000)` with `tau = max{t : G(t) > 0}`. When the
largest observed time is censored, `G` is 0 there, so `tau < t_max` and
each curve value is paired with an earlier time than the one it was
computed at. The 0.2 and 0.4 horizons build both grids the same way and are
unaffected, as is C^td.

PyHealth keeps the grids aligned. **Implication for the reproduction:** the
paper's widest-horizon IBS and IBLL (proposal Table 1) may include this
effect if the last test time in a split is censored. Before comparing our
numbers with Table 1, check how often that happens in the released splits,
and consider also reporting a reference-compatible value computed in the
experiment scripts (not in PyHealth). To decide with Aritra in T5.
