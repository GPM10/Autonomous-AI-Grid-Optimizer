"""Simple forecasters for solar and demand profiles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Mapping, Sequence


Observation = Mapping[str, float]


@dataclass
class RollingStats:
    window: int

    def mean(self, series: Sequence[float]) -> float:
        if not series:
            return 0.0
        length = min(len(series), self.window)
        return sum(series[-length:]) / length


class SolarForecaster:
    """Forecast future solar production using rolling mean + diurnal bias."""

    def __init__(self, window: int = 6):
        self.window = window
        self.stats = RollingStats(window=window)

    def __call__(self, history: Deque[Observation], current_hour: float) -> float:
        if not history:
            return 0.0
        solar_series = [obs["solar"] for obs in history]
        base = self.stats.mean(solar_series)
        # Encourage diurnal rise between 8-16h, otherwise decay.
        if 8 <= current_hour <= 16:
            bias = 0.15
        elif 5 <= current_hour < 8 or 16 < current_hour <= 19:
            bias = 0.05
        else:
            bias = -0.2
        forecast = max(base * (1 + bias), 0.0)
        return forecast


class DemandForecaster:
    """Forecast demand using rolling mean plus mild evening peak weighting."""

    def __init__(self, window: int = 12):
        self.window = window
        self.stats = RollingStats(window=window)

    def __call__(self, history: Deque[Observation], current_hour: float) -> float:
        if not history:
            return 0.0
        demand_series = [obs["demand"] for obs in history]
        base = self.stats.mean(demand_series)
        if 17 <= current_hour <= 21:
            bias = 0.1
        elif 0 <= current_hour <= 5:
            bias = -0.07
        else:
            bias = 0.0
        return max(base * (1 + bias), 0.3)
