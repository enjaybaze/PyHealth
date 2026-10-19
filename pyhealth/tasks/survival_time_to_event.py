"""Packed time-to-event labels for the Rotterdam/GBSG benchmark."""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np

from pyhealth.data import Patient

from .base_task import BaseTask

GBSG_FEATURES = tuple(f"x{i}" for i in range(7))
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _gbsg_values(record: Mapping, source_id: str) -> tuple[list[float], float, float]:
    """Parse one source record without imputation or label repair."""
    values = []
    for field in (*GBSG_FEATURES, "time", "event"):
        try:
            value = float(record[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"GBSG record {source_id}: missing/non-numeric {field}"
            ) from exc
        if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
            raise ValueError(f"GBSG record {source_id}: {field} must be finite float32")
        values.append(value)
    features, time, event = values[:7], values[7], values[8]
    if time <= 0 or np.float32(time) <= 0:
        raise ValueError(f"GBSG record {source_id}: time must be strictly positive")
    if event not in (0, 1):
        raise ValueError(
            f"GBSG record {source_id}: event must be 1 observed or 0 censored"
        )
    return features, time, event


class SurvivalTimeToEventGBSG(BaseTask):
    """Extract seven ordered features and a packed GBSG survival label.

    By default, times remain in source months with no normalization or offset.
    The benchmark endpoint is recurrence-free survival. The task emits
    ``features: [7]`` and ``survival: [time, event]``, with 1 for an observed
    event and 0 for censoring. Tensor labels select ``mode=None`` in
    SurvivalMDN, allowing the standard Trainer to use the model's loss.

    Args:
        feature_mean: Optional seven means fitted on training records only.
        feature_scale: Matching seven positive training standard deviations.
            Supply both arguments or neither. Constant features should use
            a scale of 1, as in sklearn's StandardScaler. Without these
            arguments, features are returned in their source representation.
        time_divisor: Positive divisor applied to every time. Default 1.0
            (source months).
        time_offset: Non-negative value added after dividing. Default 0.0.
            The emitted time is ``time / time_divisor + time_offset``. The
            Survival MDN authors' protocol uses the largest time in the full
            2,232-record cohort as the divisor and an offset of 0.002; see
            :func:`~pyhealth.datasets.gbsg_paper_time_transform`.

    Examples:
        >>> from pyhealth.tasks import SurvivalTimeToEventGBSG
        >>> task = SurvivalTimeToEventGBSG()
        >>> task.output_schema
        {'survival': 'tensor'}
        >>> paper = SurvivalTimeToEventGBSG(time_divisor=87.36, time_offset=0.002)
        >>> paper.time_offset
        0.002
    """

    task_name = "SurvivalTimeToEventGBSG"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"survival": "tensor"}

    def __init__(
        self,
        feature_mean: Sequence[float] | None = None,
        feature_scale: Sequence[float] | None = None,
        time_divisor: float = 1.0,
        time_offset: float = 0.0,
    ) -> None:
        super().__init__()
        time_divisor, time_offset = float(time_divisor), float(time_offset)
        if not math.isfinite(time_divisor) or time_divisor <= 0:
            raise ValueError("time_divisor must be finite and positive")
        if not math.isfinite(time_offset) or time_offset < 0:
            raise ValueError("time_offset must be finite and non-negative")
        self.time_divisor, self.time_offset = time_divisor, time_offset
        if (feature_mean is None) != (feature_scale is None):
            raise ValueError("Supply both feature_mean and feature_scale, or neither")
        self.feature_mean = None if feature_mean is None else list(feature_mean)
        self.feature_scale = None if feature_scale is None else list(feature_scale)
        if self.feature_mean is not None:
            mean = np.asarray(self.feature_mean, dtype=float)
            scale = np.asarray(self.feature_scale, dtype=float)
            if (
                mean.shape != (7,)
                or scale.shape != (7,)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError(
                    "Expected seven finite means and seven positive scales"
                )
            self.feature_mean = mean.tolist()
            self.feature_scale = scale.tolist()

    def __call__(self, patient: Patient) -> list[dict]:
        """Return one sample, rejecting missing or multiple source records.

        Args:
            patient: Patient containing exactly one ``gbsg`` event. Its ID
                is a surrogate source-row identifier, not an original ID.

        Returns:
            One sample with the source ID, features and packed survival label.

        Raises:
            ValueError: For invalid features/labels or ambiguous source rows.
        """
        events = patient.get_events("gbsg")
        if len(events) != 1:
            raise ValueError(
                f"GBSG record {patient.patient_id}: expected exactly one gbsg event, "
                f"got {len(events)}"
            )
        features, time, event = _gbsg_values(events[0].attr_dict, patient.patient_id)
        if self.feature_mean is not None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                features = (
                    (np.asarray(features) - self.feature_mean) / self.feature_scale
                ).tolist()
            if not all(math.isfinite(v) and abs(v) <= _FLOAT32_MAX for v in features):
                raise ValueError("Standardized GBSG features must be finite float32")
        time = time / self.time_divisor + self.time_offset
        if not (time > 0 and np.float32(time) > 0):
            raise ValueError(
                f"GBSG record {patient.patient_id}: converted time must be positive"
            )
        return [
            {
                "patient_id": patient.patient_id,
                "source_id": patient.patient_id,
                "features": features,
                "survival": [time, event],
            }
        ]
