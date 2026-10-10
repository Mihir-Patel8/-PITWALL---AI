"""Deterministic and Monte Carlo race-strategy simulation.

A *strategy* is a sequence of stints. Each stint has a tyre compound and a number of laps.
A pit stop happens between consecutive stints and costs ``pit_loss_seconds`` exactly once.

Total estimated remaining time::

    total = sum(estimated lap times) + sum(pit-stop losses)

The simulator knows nothing about how lap times are estimated (see ``estimators.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitwall_ai.config import DEFAULT_PIT_LOSS_SECONDS, RANDOM_SEED
from pitwall_ai.strategy.estimators import LapTimeEstimator
from pitwall_ai.strategy.scenarios import Scenario


@dataclass(frozen=True)
class Stint:
    """One run on a single set of tyres."""

    compound: str
    laps: int


@dataclass(frozen=True)
class Strategy:
    """Ordered stints; pit stops occur between them."""

    stints: tuple[Stint, ...]
    name: str = ""

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def total_laps(self) -> int:
        return sum(s.laps for s in self.stints)

    @property
    def tyre_sequence(self) -> str:
        return " > ".join(s.compound for s in self.stints)

    def pit_laps(self, start_lap: int = 1) -> list[int]:
        """Race lap numbers at the end of which the car pits."""
        laps, cumulative = [], 0
        for stint in self.stints[:-1]:
            cumulative += stint.laps
            laps.append(start_lap + cumulative - 1)
        return laps

    def label(self) -> str:
        return self.name or " | ".join(f"{s.compound} x{s.laps}" for s in self.stints)

    def validate(
        self,
        total_laps: int,
        *,
        allowed_compounds: tuple[str, ...] | None = None,
        min_stint: int = 1,
        require_two_compounds: bool = False,
    ) -> None:
        """Raise ``ValueError`` if the strategy is not a legal race plan for ``total_laps``."""
        if not self.stints:
            raise ValueError("A strategy needs at least one stint.")
        for stint in self.stints:
            if not isinstance(stint.laps, int | np.integer) or stint.laps < min_stint:
                raise ValueError(f"Every stint must be a whole number of laps >= {min_stint}.")
            if allowed_compounds is not None and stint.compound not in allowed_compounds:
                raise ValueError(f"Compound '{stint.compound}' is not available.")
        if self.total_laps != total_laps:
            raise ValueError(
                f"Stint lengths add up to {self.total_laps} laps but {total_laps} remain."
            )
        if require_two_compounds and len({s.compound for s in self.stints}) < 2:
            raise ValueError("At least two different compounds must be used.")


@dataclass(frozen=True)
class SimulationConfig:
    """Race situation and assumptions shared by all candidate strategies."""

    total_laps: int
    pit_loss_seconds: float = DEFAULT_PIT_LOSS_SECONDS
    start_lap: int = 1  # race lap number of the first remaining lap
    start_tyre_age: int = 1  # age (FastF1 TyreLife) of the current tyres on that lap
    scenario: Scenario = field(default_factory=Scenario)

    def validate(self) -> None:
        if self.total_laps < 1:
            raise ValueError("total_laps must be at least 1.")
        if self.pit_loss_seconds < 0:
            raise ValueError("pit_loss_seconds must not be negative.")
        if self.start_lap < 1 or self.start_tyre_age < 1:
            raise ValueError("start_lap and start_tyre_age must be >= 1.")
        self.scenario.validate()


@dataclass
class StrategyResult:
    """Deterministic simulation output for one strategy."""

    strategy: Strategy
    lap_times: np.ndarray  # estimated seconds for every remaining lap
    tyre_ages: np.ndarray
    pit_laps: list[int]
    pit_costs: list[float]

    @property
    def driving_time(self) -> float:
        return float(self.lap_times.sum())

    @property
    def pit_cost(self) -> float:
        return float(sum(self.pit_costs))

    @property
    def total_time(self) -> float:
        return self.driving_time + self.pit_cost


def tyre_profile(strategy: Strategy, start_tyre_age: int = 1) -> tuple[list[str], np.ndarray]:
    """Compound and tyre age for every lap. Age resets to 1 after each stop."""
    compounds: list[str] = []
    ages: list[int] = []
    for index, stint in enumerate(strategy.stints):
        first_age = start_tyre_age if index == 0 else 1
        compounds.extend([stint.compound] * stint.laps)
        ages.extend(range(first_age, first_age + stint.laps))
    return compounds, np.array(ages)


def _components(strategy: Strategy, estimator: LapTimeEstimator, config: SimulationConfig):
    """Split lap times into (fresh-tyre pace, tyre-age part, scenario multiplier)."""
    compounds, ages = tyre_profile(strategy, config.start_tyre_age)
    race_laps = np.arange(config.start_lap, config.start_lap + strategy.total_laps)
    actual = np.asarray(estimator.predict(compounds, ages, race_laps), dtype=float)
    fresh = np.asarray(estimator.predict(compounds, np.ones_like(ages), race_laps), dtype=float)
    scenario = config.scenario
    fresh = fresh + scenario.pace_offset_seconds
    degradation = (actual - fresh + scenario.pace_offset_seconds) * scenario.degradation_scale
    factor = scenario.lap_factor(race_laps)
    return compounds, ages, fresh, degradation, factor


def simulate_strategy(
    strategy: Strategy, estimator: LapTimeEstimator, config: SimulationConfig
) -> StrategyResult:
    """Estimate remaining race time for one strategy (deterministic)."""
    config.validate()
    strategy.validate(config.total_laps)
    _, ages, fresh, degradation, factor = _components(strategy, estimator, config)
    pit_laps = strategy.pit_laps(config.start_lap)
    pit_costs = [config.scenario.pit_loss(config.pit_loss_seconds, lap) for lap in pit_laps]
    return StrategyResult(strategy, factor * (fresh + degradation), ages, pit_laps, pit_costs)


def monte_carlo(
    strategies: list[Strategy],
    estimator: LapTimeEstimator,
    config: SimulationConfig,
    *,
    n_runs: int = 500,
    degradation_sigma: float = 0.20,
    compound_pace_sigma: float = 0.15,
    pit_loss_sigma: float = 1.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Explore how rankings change when the *assumptions* are uncertain.

    Each run draws, from fixed-seed normal/lognormal distributions:

    * a multiplier on each compound's degradation (lognormal, ``degradation_sigma``),
    * an extra pace offset per compound in seconds (normal, ``compound_pace_sigma``),
    * the loss of each pit stop in seconds (normal, ``pit_loss_sigma``).

    All strategies share the same random draws (common random numbers), so differences
    between them are not caused by sampling luck. The output percentiles describe the
    spread of *simulated outcomes under these invented uncertainty distributions*. They are
    NOT statistical confidence intervals for real race results.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if not strategies:
        raise ValueError("No strategies supplied.")
    names = [s.label() for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError("Strategy names must be unique.")
    config.validate()

    parts = []
    for strategy in strategies:
        strategy.validate(config.total_laps)
        compounds, _, fresh, degradation, factor = _components(strategy, estimator, config)
        pit_after = strategy.pit_laps(config.start_lap)
        base_pit = np.array(
            [config.scenario.pit_loss(config.pit_loss_seconds, lap) for lap in pit_after]
        )
        parts.append((compounds, fresh, degradation, factor, base_pit))

    all_compounds = sorted({c for p in parts for c in p[0]})
    index = {c: i for i, c in enumerate(all_compounds)}
    rng = np.random.default_rng(seed)
    deg_mult = rng.lognormal(0.0, degradation_sigma, size=(n_runs, len(all_compounds)))
    pace_off = rng.normal(0.0, compound_pace_sigma, size=(n_runs, len(all_compounds)))
    max_stops = max(s.n_stops for s in strategies)
    pit_noise = rng.normal(0.0, pit_loss_sigma, size=(n_runs, max(max_stops, 1)))

    totals = np.zeros((n_runs, len(strategies)))
    for column, (compounds, fresh, degradation, factor, base_pit) in enumerate(parts):
        comp_idx = np.array([index[c] for c in compounds])
        lap_part = factor * (fresh + pace_off[:, comp_idx] + deg_mult[:, comp_idx] * degradation)
        pit_part = (base_pit + pit_noise[:, : len(base_pit)]).clip(min=0).sum(axis=1)
        totals[:, column] = lap_part.sum(axis=1) + pit_part

    winners = np.bincount(totals.argmin(axis=1), minlength=len(strategies))
    return pd.DataFrame(
        {
            "Strategy": names,
            "MeanTime": totals.mean(axis=0),
            "StdTime": totals.std(axis=0),
            "P10": np.percentile(totals, 10, axis=0),
            "P90": np.percentile(totals, 90, axis=0),
            "FractionOfRunsBest": winners / n_runs,
        }
    )
