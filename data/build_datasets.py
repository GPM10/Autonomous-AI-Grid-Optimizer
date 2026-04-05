"""Utility to synthesize multi-day solar/load/carbon datasets without external deps."""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List


@dataclass
class ScenarioConfig:
    days: int
    demand_base: float
    solar_scale: float
    price_base: float
    carbon_base: float
    seed: int


def _daily_profile() -> List[float]:
    hours = range(24)
    solar_curve = [max(math.sin((h - 6) / 24 * math.pi), 0.0) for h in hours]
    demand_curve = [0.6 + 0.4 * math.cos((h - 18) / 24 * 2 * math.pi) for h in hours]
    return solar_curve, demand_curve


def _noise(seed: int, count: int, stddev: float) -> Iterable[float]:
    rng = random.Random(seed)
    for _ in range(count):
        yield 1.0 + rng.uniform(-stddev, stddev)


def build_dataset(cfg: ScenarioConfig) -> List[dict]:
    solar_curve, demand_curve = _daily_profile()
    base_time = datetime(2024, 1, 1)
    rows: List[dict] = []
    for day in range(cfg.days):
        start = base_time + timedelta(days=day)
        solar_noise = list(_noise(cfg.seed + day * 3, 24, 0.05))
        demand_noise = list(_noise(cfg.seed + day * 5, 24, 0.03))
        price_noise = list(_noise(cfg.seed + day * 7, 24, 0.02))
        carbon_noise = list(_noise(cfg.seed + day * 11, 24, 0.05))
        for hour in range(24):
            solar = cfg.solar_scale * solar_curve[hour] * solar_noise[hour]
            demand = cfg.demand_base * demand_curve[hour] * demand_noise[hour]
            price = cfg.price_base * (0.8 + 0.4 * demand_curve[hour]) * price_noise[hour]
            carbon = cfg.carbon_base * (1.2 - 0.5 * solar_curve[hour]) * carbon_noise[hour]
            rows.append(
                {
                    "time": (start + timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S"),
                    "solar": f"{max(solar, 0):.3f}",
                    "demand": f"{max(demand, 0.3):.3f}",
                    "price": f"{max(price, 0.05):.4f}",
                    "carbon_intensity": f"{max(carbon, 50):.2f}",
                }
            )
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate train/eval CSVs for the microgrid env.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--eval-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = ScenarioConfig(args.train_days, 4.0, 11.0, 0.11, 220.0, args.seed)
    eval_cfg = ScenarioConfig(args.eval_days, 4.4, 10.5, 0.115, 210.0, args.seed + 101)

    write_csv(build_dataset(train_cfg), out_dir / "train_data.csv")
    write_csv(build_dataset(eval_cfg), out_dir / "eval_data.csv")
    print(f"Wrote datasets to {out_dir}")


if __name__ == "__main__":
    main()
