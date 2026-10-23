GBSG benchmark
==============

This loader uses the DeepSurv Rotterdam/GBSG benchmark: 2,232 source records
and seven ordered covariates. It is different from the stand-alone 686-person
GBSG2 dataset. The endpoint is recurrence-free survival, with event 1 for an
observed event and 0 for censoring. Source times are in months.

Preparation and provenance
--------------------------

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
IDs. The loader itself does not impute, add time offsets, standardize, or
assign new splits; those choices belong to the task and the split protocol.

Preprocessing and examples
--------------------------

The default task emits raw features and a packed ``survival: [time, event]``
tensor. Fit feature means and scales on training records only and pass them
to ``SurvivalTimeToEventGBSG``. Keep label and prediction times in source
months. Supply the training-derived ``time_scale`` to SurvivalMDN; it is
saved in the model's checkpoint, and the example also records it in
``run.json``.

``examples/gbsg_survival_loading.py`` demonstrates loading, grouping identical
full records within splits, training-only standardization, saved split IDs
and preprocessing, and optional training/checkpoint restoration with
SurvivalMDN and the standard Trainer. ``--synthetic`` runs without downloads;
``--source-h5`` uses a real local source; ``--root`` reads a prepared CSV.
``--epochs 5`` enables the small training check. Validation uses one full batch.

By default this example uses an alternative grouped 80/10/10 protocol. The
HDF5's original groups and this example's groups are not the MDN authors'
released split membership; do not combine scores from the two protocols as
if they were the same.

Paper protocol (authors' released splits)
-----------------------------------------

The Survival MDN authors released ten train/validation/test splits of these
same 2,232 records as ``{train,valid,test}_{1..10}.npz`` (1,785 / 223 / 224
rows each), in their repository's ``data/gbsg`` folder. That repository has
no license, so download the files yourself; PyHealth only reads them.

Their conventions differ from the default task:

* features are used raw, with no standardization;
* each time is divided by the largest time in the full cohort (87.36 months),
  and 0.001 is added in the released files plus another 0.001 by their data
  loader, so models see ``months / 87.36 + 0.002``;
* SurvivalMDN's ``time_scale`` is left at 1.0.

``gbsg_released_split_ids`` maps each released row back to a source ID by its
features, event and converted time (the files carry no IDs), and checks that
every source record is used exactly once. All ten released repeats map
one-to-one. Two pairs of records are identical in every value; each pair is
assigned in source order, which cannot change any split's data.
``gbsg_paper_time_transform`` returns the task arguments for the time
conversion.

.. code-block:: python

    from pyhealth.datasets import gbsg_paper_time_transform, gbsg_released_split_ids
    from pyhealth.tasks import SurvivalTimeToEventGBSG

    ids = gbsg_released_split_ids("/data/gbsg_csv", "/data/Survival-MDN/data/gbsg", 1)
    task = SurvivalTimeToEventGBSG(**gbsg_paper_time_transform("/data/gbsg_csv"))
    samples = dataset.set_task(task)

The example runs this protocol with ``--released-dir`` and ``--repeat`` and
reports test C^td, IBS and IBLL with ``survival_curve_metrics_fn``. The
divisor uses every record, including test records, exactly as the authors
did; use it only to reproduce their results.

.. autoclass:: pyhealth.datasets.GBSGDataset
    :members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.prepare_gbsg_csv

.. autofunction:: pyhealth.datasets.gbsg_released_split_ids

.. autofunction:: pyhealth.datasets.gbsg_paper_time_transform
