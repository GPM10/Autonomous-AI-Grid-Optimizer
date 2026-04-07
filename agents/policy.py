"""Hybrid controller that mixes heuristics with forecast signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

Observation = Mapping[str, float]


@dataclass
class HybridController:
    """Simple strategy that balances cost, carbon, and battery health."""

    charge_threshold: float = 0.3
    discharge_threshold: float = 0.6

    def decide(
        self,
        observation: Observation,
        solar_forecast: float,
        demand_forecast: float,
        carbon_index: str,
    ) -> tuple[int, str]:
        battery = observation["battery"]
        price = observation["price"]
        carbon = observation["carbon"]
        net = solar_forecast - demand_forecast

        # Encourage imports only when carbon index is "low" or cost is cheap
        low_carbon = carbon_index in {"low", "very low"}
        cheap_energy = price < 0.1

        if net > 0.5 and battery < 0.9 * 10:
            return 1, "Charge surplus solar into battery"
        if net < -0.5 and battery > 0.2 * 10:
            return 2, "Discharge battery to cover deficit"
        if (low_carbon or cheap_energy) and battery < self.charge_threshold * 10:
            return 3, "Import from grid while carbon/cost favorable"
        if battery > self.discharge_threshold * 10 and carbon > 180:
            return 2, "Emit stored energy to avoid dirty imports"
        return 0, "Hold state; forecasts balanced"
