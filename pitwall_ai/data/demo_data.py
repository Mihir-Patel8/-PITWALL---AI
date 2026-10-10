"""Deterministic SYNTHETIC lap data for demonstration, testing and offline use.

This module is deliberately independent of the FastF1 pipeline. Nothing here is real
racing data: circuits, drivers and lap times are all invented from the simple physical
model below, so every row is labelled ``DataSource == "synthetic"``.

Lap-time model (seconds)::

    lap = circuit_base + driver_offset + compound_offset
          + degradation[compound] * (tyre_age - 1) - fuel_gain * (lap_number - 1)
          + noise

The same constants are exposed as :data:`DEMO_ASSUMPTIONS` so the strategy simulator can
use them as its default, clearly labelled, demonstration assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall_ai.config import DATA_SOURCE_SYNTHETIC, RANDOM_SEED

DEMO_LABEL = "DEMONSTRATION DATA (synthetic - not real Formula 1 data)"

#: Invented circuits: (name, base lap seconds, race laps).
DEMO_CIRCUITS: tuple[tuple[str, float, int], ...] = (
    ("Demo Circuit Alpha", 88.0, 56),
    ("Demo Circuit Bravo", 94.5, 52),
    ("Demo Circuit Charlie", 81.2, 62),
    ("Demo Circuit Delta", 101.0, 48),
    ("Demo Circuit Echo", 90.3, 58),
    ("Demo Circuit Foxtrot", 85.7, 60),
)

#: Transparent physical assumptions behind the synthetic data (and the default simulator).
DEMO_ASSUMPTIONS: dict[str, dict[str, float]] = {
    "compound_offset": {"SOFT": -0.8, "MEDIUM": 0.0, "HARD": 0.6},
    "degradation_per_lap": {"SOFT": 0.090, "MEDIUM": 0.055, "HARD": 0.035},
    "fuel_gain_per_lap": {"all": 0.030},
}

_DRIVER_CODES = [f"DR{i:02d}" for i in range(1, 11)]  # invented codes, not real drivers
_NOISE_STD = 0.25
_PIT_LANE_EXTRA = 18.0


MAX_DEMO_EVENTS = 2 * len(DEMO_CIRCUITS)  # every circuit appears in two synthetic "editions"


def generate_demo_laps(
    n_events: int = MAX_DEMO_EVENTS,
    n_drivers: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create a reproducible synthetic lap table.

    Circuits repeat in a second synthetic "edition" (like real seasons do), so a
    held-out event usually has a circuit that also appears in the training events.
    The result contains pit-in/out laps and a few slow "safety-car style" laps on
    purpose, so that the cleaning rules have something realistic to remove.

    Raises:
        ValueError: for non-positive or too-large ``n_events`` / ``n_drivers``.
    """
    if not 1 <= n_events <= MAX_DEMO_EVENTS:
        raise ValueError(f"n_events must be between 1 and {MAX_DEMO_EVENTS}")
    if not 1 <= n_drivers <= len(_DRIVER_CODES):
        raise ValueError(f"n_drivers must be between 1 and {len(_DRIVER_CODES)}")

    rng = np.random.default_rng(seed)
    compound_offset = DEMO_ASSUMPTIONS["compound_offset"]
    degradation = DEMO_ASSUMPTIONS["degradation_per_lap"]
    fuel_gain = DEMO_ASSUMPTIONS["fuel_gain_per_lap"]["all"]
    compounds = ["SOFT", "MEDIUM", "HARD"]
    driver_offsets = {code: float(rng.normal(0.0, 0.35)) for code in _DRIVER_CODES[:n_drivers]}

    rows: list[dict[str, object]] = []
    for event_index in range(1, n_events + 1):
        circuit, circuit_base, race_laps = DEMO_CIRCUITS[(event_index - 1) % len(DEMO_CIRCUITS)]
        base = circuit_base + float(rng.normal(0.0, 0.3))  # small edition-to-edition variation
        safety_car_laps = set(rng.integers(5, race_laps - 5, size=3).tolist())
        for driver, offset in driver_offsets.items():
            n_stops = int(rng.choice([1, 2], p=[0.6, 0.4]))
            pit_laps = sorted(
                rng.choice(np.arange(12, race_laps - 10), size=n_stops, replace=False).tolist()
            )
            boundaries = [0, *pit_laps, race_laps]
            stint_compounds = rng.choice(compounds, size=n_stops + 1).tolist()
            if len(set(stint_compounds)) == 1:  # keep the "two different compounds" habit
                stint_compounds[-1] = "HARD" if stint_compounds[0] != "HARD" else "MEDIUM"

            for stint_index, (start, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:], strict=True), 1
            ):
                compound = stint_compounds[stint_index - 1]
                for lap in range(start + 1, end + 1):
                    age = lap - start
                    lap_time = (
                        base
                        + offset
                        + compound_offset[compound]
                        + degradation[compound] * (age - 1)
                        - fuel_gain * (lap - 1)
                        + rng.normal(0.0, _NOISE_STD)
                    )
                    is_pit_in = lap == end and end != race_laps
                    is_pit_out = lap == start + 1 and start != 0
                    status = "1"
                    if lap in safety_car_laps:
                        status = "4"
                        lap_time *= 1.35
                    if is_pit_in or is_pit_out:
                        lap_time += _PIT_LANE_EXTRA / 2
                    rows.append(
                        {
                            "Season": 0,  # 0 = not applicable (synthetic)
                            "RoundNumber": event_index,
                            "EventName": circuit,
                            "EventId": f"demo_{event_index:02d}",
                            "SessionType": "R",
                            "Driver": driver,
                            "LapNumber": lap,
                            "LapTimeSeconds": round(float(lap_time), 3),
                            "Compound": compound,
                            "TyreLife": float(age),
                            "Stint": stint_index,
                            "IsPitLap": bool(is_pit_in or is_pit_out),
                            "TrackStatus": status,
                            "DataSource": DATA_SOURCE_SYNTHETIC,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.attrs["label"] = DEMO_LABEL
    return frame


def is_synthetic(df: pd.DataFrame) -> bool:
    """True if every row of ``df`` is flagged as synthetic demonstration data."""
    if "DataSource" not in df.columns or df.empty:
        return False
    return bool((df["DataSource"] == DATA_SOURCE_SYNTHETIC).all())
