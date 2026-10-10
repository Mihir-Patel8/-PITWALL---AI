"""Enumerate valid candidate strategies and rank them by estimated remaining time."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pandas as pd

from pitwall_ai.strategy.estimators import LapTimeEstimator
from pitwall_ai.strategy.simulator import SimulationConfig, Stint, Strategy, simulate_strategy

MAX_CANDIDATES = 20_000


def _compositions(
    total: int, parts: int, low: int, high: int, step: int
) -> Iterator[tuple[int, ...]]:
    """Stint lengths (length ``parts``) that sum to ``total``, each within [low, high].

    All but the last stint are drawn from ``range(low, high + 1, step)``; the last stint takes
    the remainder and must also be within bounds.
    """
    if parts == 1:
        if low <= total <= high:
            yield (total,)
        return
    for first in range(low, min(high, total - low * (parts - 1)) + 1, step):
        for rest in _compositions(total - first, parts - 1, low, high, step):
            yield (first, *rest)


def generate_candidates(
    total_laps: int,
    start_compound: str,
    available_compounds: tuple[str, ...],
    *,
    min_stops: int = 1,
    max_stops: int = 2,
    min_stint: int = 5,
    max_stint: int | None = None,
    step: int = 1,
    require_two_compounds: bool = True,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Strategy]:
    """Build every valid strategy that satisfies the constraints.

    The first stint always uses ``start_compound``. Raises ``ValueError`` for impossible
    parameter combinations or when the search space exceeds ``max_candidates`` (increase
    ``step`` or tighten stint limits).
    """
    if total_laps < 1:
        raise ValueError("total_laps must be at least 1.")
    if start_compound not in available_compounds:
        raise ValueError("start_compound must be one of the available compounds.")
    if not 0 <= min_stops <= max_stops:
        raise ValueError("Need 0 <= min_stops <= max_stops.")
    if min_stint < 1 or step < 1:
        raise ValueError("min_stint and step must be >= 1.")
    max_stint = max_stint or total_laps
    if max_stint < min_stint:
        raise ValueError("max_stint must be >= min_stint.")

    candidates: list[Strategy] = []
    for stops in range(min_stops, max_stops + 1):
        n_stints = stops + 1
        if n_stints * min_stint > total_laps:
            continue  # cannot fit that many stints
        sequences = [
            (start_compound, *rest)
            for rest in itertools.product(available_compounds, repeat=stops)
            if not require_two_compounds or len({start_compound, *rest}) >= 2
        ]
        for lengths in _compositions(total_laps, n_stints, min_stint, max_stint, step):
            for sequence in sequences:
                stints = tuple(Stint(c, n) for c, n in zip(sequence, lengths, strict=True))
                candidates.append(Strategy(stints, name=_name(stints)))
                if len(candidates) > max_candidates:
                    raise ValueError(
                        f"More than {max_candidates} candidates; increase the pit-lap step or "
                        "narrow the stint-length limits."
                    )
    if not candidates:
        raise ValueError(
            "No valid strategies for these constraints (stints too long/short, or the "
            "two-compound rule cannot be met with 0 stops)."
        )
    return candidates


def _name(stints: tuple[Stint, ...]) -> str:
    stops = len(stints) - 1
    detail = " > ".join(f"{s.compound[0]}{s.laps}" for s in stints)
    return f"{stops}-stop: {detail}"


def rank_strategies(
    strategies: list[Strategy], estimator: LapTimeEstimator, config: SimulationConfig
) -> pd.DataFrame:
    """Simulate every strategy and return a table sorted by estimated total time."""
    if not strategies:
        raise ValueError("No strategies to rank.")
    rows = []
    for strategy in strategies:
        result = simulate_strategy(strategy, estimator, config)
        rows.append(
            {
                "Strategy": strategy.label(),
                "Stops": strategy.n_stops,
                "TyreSequence": strategy.tyre_sequence,
                "PitLaps": ", ".join(map(str, result.pit_laps)) or "-",
                "DrivingTime": result.driving_time,
                "PitCost": result.pit_cost,
                "TotalTime": result.total_time,
            }
        )
    table = pd.DataFrame(rows).sort_values("TotalTime", kind="stable").reset_index(drop=True)
    table["DeltaToBest"] = table["TotalTime"] - table["TotalTime"].iloc[0]
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def best_by_stop_count(ranking: pd.DataFrame) -> pd.DataFrame:
    """Best candidate for each number of stops (e.g. best one-stop vs best two-stop)."""
    return (
        ranking.sort_values("TotalTime")
        .groupby("Stops", as_index=False)
        .first()
        .sort_values("Stops")
    )


def explain_choice(ranking: pd.DataFrame) -> str:
    """Plain-English reason for the top-ranked candidate."""
    best = ranking.iloc[0]
    text = (
        f"Under the chosen assumptions, **{best['Strategy']}** has the lowest estimated remaining "
        f"time ({best['TotalTime']:.1f} s = {best['DrivingTime']:.1f} s on track + "
        f"{best['PitCost']:.1f} s in the pits)."
    )
    if len(ranking) > 1:
        other = ranking.iloc[1]
        text += (
            f" The runner-up, {other['Strategy']}, is {other['DeltaToBest']:.2f} s slower "
            f"(pit cost {other['PitCost']:.1f} s vs {best['PitCost']:.1f} s)."
        )
    other_stops = ranking[ranking["Stops"] != best["Stops"]]
    if not other_stops.empty:
        alt = other_stops.iloc[0]
        trade = (
            "fewer stops save pit time but tyres run longer"
            if best["Stops"] < alt["Stops"]
            else "extra stops cost pit time but keep tyres fresher"
        )
        text += (
            f" The best {int(alt['Stops'])}-stop alternative is {alt['DeltaToBest']:.2f} s slower: "
            f"{trade}."
        )
    return text + (
        " This is the best result under the model and assumptions, not necessarily the "
        "optimal real-world decision (traffic, safety cars, weather and risk are not modelled)."
    )
