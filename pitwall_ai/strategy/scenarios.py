"""Hypothetical what-if scenarios applied on top of a lap-time estimator.

IMPORTANT: none of these are *predictions*. PITWALL AI has no training data about weather or
Safety Car timing, so these are user-configurable hypotheticals with documented effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """Adjustments applied by the simulator.

    Attributes:
        degradation_scale: multiplies the tyre-age part of the lap time (1.0 = unchanged;
            1.3 = tyres wear 30 % faster, e.g. a hot or abrasive day).
        pace_offset_seconds: constant seconds added to every lap (e.g. hypothetical damp track).
        safety_car_start_lap: first race lap under a hypothetical Safety Car (``None`` = none).
        safety_car_laps: number of consecutive neutralised laps.
        safety_car_lap_time_factor: lap-time multiplier during the Safety Car (> 1).
        safety_car_pit_loss_factor: multiplier on pit-stop loss when the stop is made on a
            Safety Car lap (< 1, as the field is slow so the stop costs less).
    """

    name: str = "No scenario"
    degradation_scale: float = 1.0
    pace_offset_seconds: float = 0.0
    safety_car_start_lap: int | None = None
    safety_car_laps: int = 0
    safety_car_lap_time_factor: float = 1.4
    safety_car_pit_loss_factor: float = 0.5

    def validate(self) -> None:
        if self.degradation_scale < 0:
            raise ValueError("degradation_scale must be >= 0.")
        if self.safety_car_start_lap is not None and (
            self.safety_car_start_lap < 1 or self.safety_car_laps < 1
        ):
            raise ValueError("Safety Car start lap and length must be >= 1.")
        if self.safety_car_lap_time_factor < 1:
            raise ValueError("safety_car_lap_time_factor must be >= 1.")
        if not 0 <= self.safety_car_pit_loss_factor <= 1:
            raise ValueError("safety_car_pit_loss_factor must be between 0 and 1.")

    @property
    def has_safety_car(self) -> bool:
        return self.safety_car_start_lap is not None and self.safety_car_laps > 0

    def is_safety_car_lap(self, race_lap: int) -> bool:
        if not self.has_safety_car:
            return False
        start = self.safety_car_start_lap
        return start <= race_lap < start + self.safety_car_laps

    def lap_factor(self, race_laps: np.ndarray) -> np.ndarray:
        """Per-lap multiplier (1.0 except on Safety Car laps)."""
        return np.array(
            [
                self.safety_car_lap_time_factor if self.is_safety_car_lap(int(lap)) else 1.0
                for lap in race_laps
            ]
        )

    def pit_loss(self, base_loss: float, pit_after_lap: int) -> float:
        """Pit-stop loss for a stop made at the end of ``pit_after_lap``."""
        if self.is_safety_car_lap(pit_after_lap):
            return base_loss * self.safety_car_pit_loss_factor
        return base_loss

    def describe(self) -> list[str]:
        lines = []
        if self.degradation_scale != 1.0:
            lines.append(f"Tyre degradation scaled by x{self.degradation_scale:.2f}")
        if self.pace_offset_seconds:
            lines.append(f"All laps {self.pace_offset_seconds:+.2f} s")
        if self.has_safety_car:
            end = self.safety_car_start_lap + self.safety_car_laps - 1
            lines.append(
                f"Hypothetical Safety Car laps {self.safety_car_start_lap}-{end}: lap times "
                f"x{self.safety_car_lap_time_factor:.2f}, "
                f"pit loss x{self.safety_car_pit_loss_factor:.2f}"
            )
        return lines or ["No scenario adjustments"]
