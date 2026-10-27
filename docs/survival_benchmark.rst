Survival benchmarks
==================

The portable examples under ``examples/survival_benchmark/`` train SurvivalMDN,
classical Cox and neural CoxPH/DeepSurv with native GBSG and SUPPORT survival tasks.
They demonstrate shared curve metrics, censoring-aware D-calibration, checkpoint
restoration and reproducible fit records. No course-specific notes branch is
required. Run these commands from a PyHealth source checkout.

Download-free example
---------------------

.. code-block:: bash

   python -m examples.survival_benchmark.run_survival_mdn \
       --dataset synthetic --splits 1 --epochs 3 --device cpu --out /tmp/mdn-demo

Cox/DeepSurv are optional examples, not additional core PyHealth dependencies:

.. code-block:: bash

   pip install -r examples/survival_benchmark/requirements.txt
   python -m examples.survival_benchmark.run_baselines \
       --dataset synthetic --splits 1 --epochs 3 --out /tmp/baseline-demo
   python -m examples.survival_benchmark.calibrate_saved \
       /tmp/baseline-demo/cox/split_1/seed_1/predictions.npz \
       --out /tmp/calibration.json

Real inputs and conventions
---------------------------

Real runs use user-supplied public CSVs and the paper authors' released NPZ split
arrays. These files and the authors' source code are not distributed here.
The native task emits ``survival: [time, event]``, where event 1 is observed and
0 is censored. Every patient has one feature row and one matching outcome row.

GBSG baselines fit feature means/scales on training patients only, while MDN
keeps the raw reference covariates. SUPPORT already standardizes using the
selected training split, so its adapters apply no extra standardization.
Time transforms and released membership are shared across the models.

Select baseline settings on validation IBS and MDN settings on validation NLL.
Keep test scores for reporting. Pycox's tied-time approximation, baseline step
curve tails and separately named reference-compatible paper scores are explicit
implementation conventions, not exact original DeepSurv parity claims.

Saved outputs retain code/input and artifact fingerprints. Changed settings,
inputs or missing saved outputs cannot silently reuse an old fit. D-calibration
uses exact probabilities at observed times; it can be recalculated from saved
predictions or regenerated from a trusted checkpoint without retraining.

Full instructions and API usage
------------------------------

The source checkout's ``examples/survival_benchmark/README.md`` documents real
GBSG/SUPPORT commands, optional dependencies, transforms, selection rules,
metrics, checkpoint formats and tests. The metric API is
:doc:`api/metrics/pyhealth.metrics.survival_curve`.

These examples implement the pipeline from `Survival Mixture Density Networks
<https://proceedings.mlr.press/v182/han22a.html>`_ and use `D-calibration from
Haider et al. <https://jmlr.org/papers/v21/18-772.html>`_. Synthetic runs are
interface demonstrations, not tuned paper-comparison results.
