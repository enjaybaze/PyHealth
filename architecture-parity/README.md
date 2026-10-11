# Architecture parity with the authors' model (T1)

`architecture_parity.py` checks that `pyhealth.models.SurvivalMDN` (commit
`99a8bb7`) computes the same function as the authors' released model
(`MukundMDNHelper` + `SurvivalMDN_Dist` in
[XintianHan/Survival-MDN](https://github.com/XintianHan/Survival-MDN) at
`71c61f6`, the commit in `../parity/reference_commit.txt`). It builds the
authors' model, randomises every weight and BatchNorm statistic, copies them
into ours layer by layer, and compares outputs on 256 random patients in
training mode (batch statistics) and evaluation mode (running statistics).

The authors' code is only imported at run time; it has no license and is
not in the PR or on this branch.

## Result (Oct 11, 2026), see `results.json`

Four configurations: (D, hidden, K) = (7, 32, 5), (27, 64, 10), (7, 8, 1),
(9, 128, 20), each in train and eval mode.

| Output | Max abs difference |
| --- | --- |
| Mixture weights, means, scales | < 1e-6 (float32 rounding) |
| Survival S(t) | < 1e-6 |
| Log density, where the reference's epsilon is negligible | < 1e-4 |

The layer structure maps one to one (9 encoder modules plus 3 heads), so
the architecture, clamps and parameter transforms match exactly. The only
differences are the reference's deliberate epsilons: it adds 1e-6 inside its
inverse softplus and inside its log-sum-exp. Where a time lies far in the
tails of every component, that epsilon floors the reference density, so it
reads higher than the exact value. The script checks that the reference is
higher at every such point and compares the rest within 1e-4.

## Rerun

```bash
git clone https://github.com/XintianHan/Survival-MDN.git /tmp/ref
pip install lifelines
python -I architecture_parity.py /tmp/ref /path/to/PyHealth results.json
```
