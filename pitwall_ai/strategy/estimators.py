"""Lap-time estimators behind one small interface.

The simulator only needs ``predict(compounds, tyre_ages, lap_numbers) -> seconds`` so it
does not care whether the numbers come from editable assumptions, an empirical fit to
data, or a trained machine-learning model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from pitwall_ai.config import DRY_COMPOUNDS
from pitwall_ai.data.demo_data import DEMO_ASSUMPTIONS


class LapTimeEstimator(Protocol):
    """Anything that can estimate lap times for a batch of laps."""

    name: str

    def predict(
        self, compounds: Sequence[str], tyre_ages: np.ndarray, lap_numbers: np.ndarray
    ) -> np.ndarray: ...


class ParametricEstimator:
    """Transparent, editable model::

        lap = base_pace + compound_offset + degradation * (tyre_age - 1) - fuel_gain * (lap - 1)

    ``tyre_age`` is 1 on the first lap of a fresh set (FastF1's ``TyreLife`` convention).
    """

    def __init__(
        self,
        base_pace: float,
        compound_offset: dict[str, float],
        degradation_per_lap: dict[str, float],
        fuel_gain_per_lap: float,
        name: str = "Manual assumptions",
    ) -> None:
        if base_pace <= 0:
            raise ValueError("base_pace must be positive.")
        if set(compound_offset) != set(degradation_per_lap):
            raise ValueError("compound_offset and degradation_per_lap need the same compounds.")
        self.base_pace = base_pace
        self.compound_offset = dict(compound_offset)
        self.degradation_per_lap = dict(degradation_per_lap)
        self.fuel_gain_per_lap = fuel_gain_per_lap
        self.name = name

    @property
    def compounds(self) -> tuple[str, ...]:
        return tuple(self.compound_offset)

    @classmethod
    def demo_defaults(cls, base_pace: float = 90.0) -> ParametricEstimator:
        """Defaults taken from the synthetic demo-data model (clearly demonstration only)."""
        return cls(
            base_pace,
            DEMO_ASSUMPTIONS["compound_offset"],
            DEMO_ASSUMPTIONS["degradation_per_lap"],
            DEMO_ASSUMPTIONS["fuel_gain_per_lap"]["all"],
            name="Demonstration assumptions (synthetic)",
        )

    def predict(self, compounds, tyre_ages, lap_numbers) -> np.ndarray:
        ages = np.asarray(tyre_ages, dtype=float)
        laps = np.asarray(lap_numbers, dtype=float)
        try:
            offset = np.array([self.compound_offset[c] for c in compounds])
            degradation = np.array([self.degradation_per_lap[c] for c in compounds])
        except KeyError as exc:
            raise ValueError(f"No assumptions for compound {exc}.") from exc
        return (
            self.base_pace + offset + degradation * (ages - 1) - self.fuel_gain_per_lap * (laps - 1)
        )


class EmpiricalEstimator:
    """Least-squares fit of lap time (relative to the event median) on tyre age and lap.

    For each compound: ``relative_time = a_c + b_c * tyre_age`` plus a shared
    ``c * lap_number`` term. Absolute pace is then ``base_pace + relative_time``, where
    ``base_pace`` is supplied by the user (e.g. the event median). This is a simple summary
    of the data, not a causal model of tyre wear.
    """

    def __init__(
        self,
        base_pace: float,
        intercepts: dict[str, float],
        slopes: dict[str, float],
        lap_slope: float,
        name: str = "Empirical fit to dataset",
    ) -> None:
        self.base_pace = base_pace
        self.intercepts = intercepts
        self.slopes = slopes
        self.lap_slope = lap_slope
        self.name = name

    @property
    def compounds(self) -> tuple[str, ...]:
        return tuple(self.intercepts)

    @classmethod
    def fit(cls, laps: pd.DataFrame, base_pace: float, min_rows: int = 30) -> EmpiricalEstimator:
        """Fit from a cleaned lap table. Raises ``ValueError`` if data are insufficient."""
        needed = {"Compound", "TyreLife", "LapNumber", "LapTimeSeconds", "EventId"}
        if not needed <= set(laps.columns):
            raise ValueError(f"Need columns: {', '.join(sorted(needed))}")
        data = laps[laps["Compound"].isin(DRY_COMPOUNDS)].copy()
        if len(data) < min_rows:
            raise ValueError(f"Only {len(data)} usable laps; at least {min_rows} are needed.")
        data["relative"] = data["LapTimeSeconds"] - data.groupby("EventId")[
            "LapTimeSeconds"
        ].transform("median")
        compounds = [c for c in DRY_COMPOUNDS if (data["Compound"] == c).sum() >= 10]
        if len(compounds) < 1:
            raise ValueError("No compound has at least 10 laps.")
        data = data[data["Compound"].isin(compounds)]
        columns = [(data["Compound"] == c).astype(float) for c in compounds]
        columns += [(data["Compound"] == c).astype(float) * data["TyreLife"] for c in compounds]
        columns.append(data["LapNumber"].astype(float))
        design = np.column_stack(columns)
        coef, *_ = np.linalg.lstsq(design, data["relative"].to_numpy(), rcond=None)
        k = len(compounds)
        return cls(
            base_pace,
            dict(zip(compounds, coef[:k], strict=True)),
            dict(zip(compounds, coef[k : 2 * k], strict=True)),
            float(coef[-1]),
        )

    def predict(self, compounds, tyre_ages, lap_numbers) -> np.ndarray:
        ages = np.asarray(tyre_ages, dtype=float)
        laps = np.asarray(lap_numbers, dtype=float)
        try:
            a = np.array([self.intercepts[c] for c in compounds])
            b = np.array([self.slopes[c] for c in compounds])
        except KeyError as exc:
            raise ValueError(f"Compound {exc} was not present in the fitted data.") from exc
        return self.base_pace + a + b * ages + self.lap_slope * laps


class ModelLapEstimator:
    """Wrap a trained :class:`~pitwall_ai.ml.prediction.LapTimePredictor`.

    The model is evaluated once on a (compound x lap x tyre age) grid for a chosen driver
    and event, then looked up. This keeps optimisation fast even for a random forest.
    """

    def __init__(
        self,
        predictor,
        driver: str,
        event_name: str,
        compounds: Sequence[str] = DRY_COMPOUNDS,
        max_lap: int = 80,
        max_age: int = 60,
    ):
        self.name = f"Trained model: {predictor.model_name} ({driver}, {event_name})"
        self._compounds = {c: i for i, c in enumerate(compounds)}
        self.max_lap, self.max_age = max_lap, max_age
        laps = np.arange(1, max_lap + 1)
        ages = np.arange(0, max_age + 1)
        grid = pd.DataFrame(
            [(c, a, lap) for c in compounds for lap in laps for a in ages],
            columns=["Compound", "TyreLife", "LapNumber"],
        )
        grid["Driver"], grid["EventName"] = driver, event_name
        values = predictor.predict(grid)
        self._table = values.reshape(len(compounds), len(laps), len(ages))

    @property
    def compounds(self) -> tuple[str, ...]:
        return tuple(self._compounds)

    def predict(self, compounds, tyre_ages, lap_numbers) -> np.ndarray:
        ages = np.asarray(tyre_ages, dtype=int)
        laps = np.asarray(lap_numbers, dtype=int)
        if ages.max() > self.max_age or laps.max() > self.max_lap or laps.min() < 1:
            raise ValueError(
                f"Lap/tyre-age outside the precomputed range (laps<={self.max_lap}, "
                f"age<={self.max_age})."
            )
        idx = np.array([self._compounds[c] for c in compounds])
        return self._table[idx, laps - 1, ages]
