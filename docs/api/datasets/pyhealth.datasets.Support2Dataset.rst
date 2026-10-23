pyhealth.datasets.Support2Dataset
==================================

Overview
--------

The SUPPORT2 (Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments) dataset contains data on seriously ill hospitalized adults. It includes patient demographics, diagnoses, clinical measurements, and outcomes such as survival and hospital mortality.

The dataset is commonly used for mortality prediction, length of stay prediction, and other clinical outcome prediction tasks.

The Vanderbilt ``support2.csv`` leaves its first column, the patient
number, unnamed. ``Support2Dataset`` reads it as ``sno``, so name it first,
for example with ``sed -i '1s/^/sno,/' support2.csv``.

.. autoclass:: pyhealth.datasets.Support2Dataset
    :members:
    :undoc-members:
    :show-inheritance:

Time-to-event benchmark and the authors' released splits
--------------------------------------------------------

:class:`~pyhealth.tasks.SurvivalTimeToEventSupport2` builds the DeepSurv
SUPPORT benchmark used by Survival MDN: the 8,873 patients with all eight
numeric covariates observed, 27 features (eight numeric covariates and 19
drop-first one-hot columns for six categorical ones), and time to death.

The Survival MDN authors released ten train/validation/test splits of this
cohort as ``{train,valid,test}_{1..10}.npz`` in their repository's
``data/support`` folder. That repository has no license, so download the
files yourself; PyHealth only reads them. Their conventions:

* all 27 features are standardized with the mean and standard deviation of
  each split's training patients;
* times are ``d.time / 365.25`` (years), plus 0.001 added by their loader.

``support2_released_split_ids`` maps each released row back to its patient
number by features, death indicator and time, and checks that every cohort
patient is used exactly once. All ten repeats map one-to-one, and with
training-only scaling the task reproduces the released standardized
features to float32 precision.

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from pyhealth.datasets import (
        Support2Dataset, support2_paper_time_transform, support2_released_split_ids,
    )
    from pyhealth.tasks import SurvivalTimeToEventSupport2

    dataset = Support2Dataset(root="/data/support2", tables=["support2"])
    ids = support2_released_split_ids("/data/support2", "/data/Survival-MDN/data/support", 1)
    raw = dataset.set_task(SurvivalTimeToEventSupport2())
    index = {raw[i]["patient_id"]: i for i in range(len(raw))}
    scaler = StandardScaler().fit([raw[index[p]]["features"].tolist() for p in ids["train"]])
    samples = dataset.set_task(SurvivalTimeToEventSupport2(
        scaler.mean_.tolist(), scaler.scale_.tolist(), **support2_paper_time_transform()
    ))

.. autofunction:: pyhealth.datasets.support2_released_split_ids

.. autofunction:: pyhealth.datasets.support2_paper_time_transform

