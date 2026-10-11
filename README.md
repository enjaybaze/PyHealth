# Survival MDN team notes (not part of the PyHealth PR)

This branch holds working records for the CSE 6250 Survival MDN project
(NJ and Aritra). It shares no history with PyHealth and is never merged.
Code lives on `feat/survival-mdn`.

## Pinned environment (T1, recorded Oct 10, 2026)

| Item | Value |
| --- | --- |
| PyHealth base commit | `eb55f9c` (Oct 8, 2026, "Add BaseModel.set_pos_weight...") |
| Python | 3.13.16 (CI also uses 3.13) |
| PyTorch | 2.14.1 (CUDA 13.0 wheel from PyPI) |
| Key packages | numpy 2.2.6, pandas 2.3.3, polars 1.35.2, scikit-learn 1.7.2, litdata 0.2.67, ruff 0.17.0 |
| Full lock | `requirements-lock.txt` on this branch (133 packages) |

### Recreate it

```bash
git clone https://github.com/enjaybaze/PyHealth.git && cd PyHealth
git checkout feat/survival-mdn
uv venv -p 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[lint]" pytest
# optional, to match exactly:
git show origin/team-notes:requirements-lock.txt > /tmp/lock.txt
uv pip install -r /tmp/lock.txt
```

## Baseline test results (Oct 10, 2026)

Focused baseline on the pinned environment, in a sandbox without general
internet access:

```bash
python -m unittest tests.core.test_model_mode tests.core.test_logistic_regression \
  tests.core.test_mlp tests.core.test_sample_dataset tests.core.test_processor_schemas \
  tests.core.test_pos_weight tests.core.test_survival_mdn
```

- 43 tests run, 37 pass.
- The 6 errors are all blocked downloads (`OSError: Tunnel connection failed:
  403`), in `test_processor_schemas` (3) and MICRON's DDI download in
  `test_model_mode` (3). They are environment-related, not code failures.
- `tests/core/test_survival_mdn.py`: 7/7 pass.

The full `tests/core` suite (`make test` in CI) did not complete in the
sandbox; rerun it on a machine with internet access before opening the PR.

## PR checker status

`python tools/check_pr_rules.py --base eb55f9c --head HEAD` on
`feat/survival-mdn` at `6f5a2c8`: passes except the examples requirement
(no file under `examples/` yet; planned for T6).
