GBSG benchmark
==============

This loader uses the DeepSurv Rotterdam/GBSG benchmark: 2,232 source records
and seven ordered covariates. It is different from the stand-alone 686-person
GBSG2 dataset. The endpoint is recurrence-free survival, with event 1 for an
observed event and 0 for censoring. Source times are in months.

Preparation and provenance
-------------------------

Obtain the local ``gbsg_cancer_train_test.h5`` release from the
`DeepSurv benchmark <https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/gbsg>`_.
Install ``pyhealth[survival-data]`` to enable HDF5 preparation. The loader
itself reads CSV and requires no HDF5 dependency or network access.

.. code-block:: python

    from pyhealth.datasets import GBSGDataset, prepare_gbsg_csv

    prepare_gbsg_csv("/data/gbsg_cancer_train_test.h5", "/data/gbsg_csv")
    dataset = GBSGDataset(root="/data/gbsg_csv", cache_dir="/data/cache")
    samples = dataset.set_task()

The explicit preparation step writes ``gbsg.csv`` and ``gbsg-source.json``
with source/CSV hashes, source-group counts, feature order, units and event
semantics. It preserves all rows and assigns IDs from original source group
and row position, such as ``train_42``. These are surrogate record IDs;
original patient IDs are unavailable. Equal values do not prove duplicate
patients and are retained. Duplicate surrogate IDs are errors.

CSV columns are ``source_id``, ``source_group``, ``source_row``, ``x0`` through
``x6``, ``time`` and ``event``. Features follow the HDF5's original column
order; positional names avoid assigning unsupported clinical descriptions.
The loader rejects missing/nonfinite values, invalid outcomes and ambiguous
IDs. It does not impute, add time offsets, standardize, or assign new splits.

Preprocessing and examples
-------------------------

The default task emits raw features and a packed ``survival: [time, event]``
tensor. Fit feature means and scales on training records only and pass them
to ``SurvivalTimeToEventGBSG``. Keep label and prediction times in source
months. Supply the training-derived ``time_scale`` to SurvivalMDN once;
persist it with the model configuration because weights alone do not save it.

``examples/gbsg_survival_loading.py`` demonstrates loading, grouping identical
full records within splits, training-only standardization, saved split IDs
and preprocessing, and optional training/checkpoint restoration with NJ's
SurvivalMDN and the standard Trainer. ``--synthetic`` runs without downloads;
``--source-h5`` uses a real local source; ``--root`` reads a prepared CSV.
``--epochs 5`` enables the small training check. Validation uses one full batch.

This example uses an alternative grouped 80/10/10 protocol. The HDF5's
original groups and this example's groups are not the MDN authors' released
split membership. Paper reproduction requires those released splits and their
separate normalization/offset conventions; do not combine their scores with
source-unit experiment results as if the protocols were identical.

.. autoclass:: pyhealth.datasets.GBSGDataset
    :members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.prepare_gbsg_csv
