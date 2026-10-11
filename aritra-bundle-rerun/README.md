# Aritra's integration check, rerun against NJ's SurvivalMDN (T1)

Aritra's bundle (`SurvivalMDN_Validation_Handoff.zip`, Oct 11, 2026) checks
packed `survival: [time, event]` labels, `mode=None` and the standard PyHealth
`Trainer` using his local `native_mdn.py`. T1 required repeating it against
NJ's implementation. `verify_packed_integration_nj.py` reuses the bundle's own
data preparation and repeats every assertion from its `check_case` with
`pyhealth.models.SurvivalMDN` from `feat/survival-mdn` (commit `77ee223`).

Same settings as the bundle: K = 5, width 32, 3 hidden layers, softplus,
RMSprop at 0.001, 5 epochs, training batches of 32 (synthetic) or 128 (real),
whole validation split in one batch, training-fitted feature standardisation
and time scale from the bundle metadata. Environment: the pinned team
environment (Python 3.13, PyHealth `eb55f9c` + branch, torch 2.14.1, CPU).

## Result: all three cases pass (Oct 11, 2026)

| Case | Train / valid / test | Val NLL before -> best | Aritra's native MDN best | Val mean error | Reload error |
| --- | --- | --- | --- | --- | --- |
| Synthetic | 96 / 31 / 33 | 2.488 -> 2.164 | 2.362 | 1.9e-9 | 0 |
| GBSG seed 1 | 1786 / 223 / 223 | 3.287 -> 2.992 | 2.999 | 2.4e-7 | 0 |
| SUPPORT seed 1 (train-only vocab) | 5370 / 1776 / 1727 | 6.789 -> 4.506 | 4.520 | 5.8e-7 | 0 |

Every assertion passed: packed-label order and event polarity, disjoint
splits, finite losses and gradients, parameter updates, validation loss
decreasing, Trainer validation loss equal to the direct per-patient mean and
to patient-weighted uneven chunks, bounded non-increasing curves, best
checkpoint restored exactly by a fresh `Trainer(checkpoint_path=...)`.

NLLs are in each dataset's original time units, so the two implementations'
numbers are comparable within a dataset (not across datasets). The
architectures differ (NJ: Linear/ReLU, no BatchNorm; Aritra: the authors'
bias-free Linear/BatchNorm/PReLU), so close but not identical values are
expected. As in the bundle, the best epoch was always epoch 5, so rollback to
an earlier best epoch is not exercised here (NJ's own unit test covers
checkpoint selection on synthetic data).

One adaptation: the bundle's curve grid starts at t = 0; NJ's model requires
positive times, so the first grid point is moved to 1e-6 x the grid maximum.

## Rerun

```bash
unzip SurvivalMDN_Validation_Handoff.zip -d /tmp/bundle
python -I verify_packed_integration_nj.py /tmp/bundle/SurvivalMDN_Validation_Handoff \
    /path/to/PyHealth /tmp/rerun-nj
```

The output directory must not already exist.
