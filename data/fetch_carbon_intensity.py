"""Fetch carbon intensity data from the NESO API and store/merge it."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.carbon_intensity import (  # noqa: E402
    default_window,
    fetch_national_intensity,
    fetch_regional_intensity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download National Grid carbon intensity data and optionally merge with an existing dataset."
    )
    parser.add_argument("--start", help="Start datetime (ISO8601). Defaults to now rounded to the hour.")
    parser.add_argument("--end", help="End datetime (ISO8601).")
    parser.add_argument("--hours", type=int, default=36, help="If --end not supplied, fetch this many hours after start.")
    parser.add_argument("--region-id", type=int, help="Filter regional data by region id (1-17).")
    parser.add_argument("--postcode", help="Filter regional data by outward postcode (e.g. SW1).")
    parser.add_argument("--local-csv", help="Use an existing NESO CSV download instead of calling the API.")
    parser.add_argument("--output", default="data/carbon_intensity_live.csv", help="CSV path for the downloaded data.")
    parser.add_argument(
        "--merge-dataset",
        help="Optional dataset (e.g., data/train_data.csv) whose carbon_intensity column will be updated.",
    )
    parser.add_argument(
        "--merge-column",
        default="forecast",
        help="Column to use when merging into an existing dataset (forecast|actual). Defaults to forecast.",
    )
    return parser.parse_args()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _flatten_national(entries: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for entry in entries:
        intensity = entry.get("intensity", {})
        rows.append(
            {
                "from": entry.get("from"),
                "to": entry.get("to"),
                "forecast": intensity.get("forecast"),
                "actual": intensity.get("actual"),
                "index": intensity.get("index"),
                "region_id": "",
                "region_name": "GB",
                "dnoregion": "National",
            }
        )
    return rows


def _flatten_regional(entries: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for entry in entries:
        if isinstance(entry, str):
            continue
        for region in entry.get("regions", []):
            if isinstance(region, str):
                continue
            intensity = region.get("intensity", {})
            rows.append(
                {
                    "from": entry.get("from"),
                    "to": entry.get("to"),
                    "forecast": intensity.get("forecast"),
                    "actual": intensity.get("actual"),
                    "index": intensity.get("index"),
                    "region_id": region.get("regionid"),
                    "region_name": region.get("shortname"),
                    "dnoregion": region.get("dnoregion"),
                    "postcode": region.get("postcode"),
                }
            )
    return rows


def _load_local_csv(csv_path: Path, start: Optional[datetime], end: Optional[datetime]) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Local CSV '{csv_path}' not found.")
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("Local CSV must contain a 'datetime' column.")
    df["from"] = pd.to_datetime(df["datetime"], utc=True)
    if start:
        df = df[df["from"] >= start]
    if end:
        df = df[df["from"] <= end]
    if df.empty:
        return []
    value_cols = [c for c in df.columns if c not in {"datetime", "from"}]
    if not value_cols:
        raise ValueError("Local CSV must contain at least one regional column.")
    df = df.melt(id_vars=["from"], value_vars=value_cols, var_name="region_name", value_name="forecast")
    df = df.dropna(subset=["forecast"])
    df = df.rename(columns={"from": "from_ts"})
    step = pd.Series(pd.unique(df["from_ts"])).sort_values().diff().dropna().mode()
    if not step.empty:
        delta = step.iloc[0].to_pytimedelta()
    else:
        delta = timedelta(minutes=30)
    df["to_ts"] = df["from_ts"] + delta
    rows = [
        {
            "from": row.from_ts.isoformat(),
            "to": row.to_ts.isoformat(),
            "forecast": float(row.forecast),
            "actual": None,
            "index": None,
            "region_id": None,
            "region_name": row.region_name,
            "dnoregion": row.region_name,
        }
        for row in df.itertuples(index=False)
    ]
    return rows


def _write_csv(rows: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _merge_with_dataset(dataset_path: Path, rows: List[Dict], column: str) -> None:
    if not rows:
        return
    column = column if column in rows[0] else "forecast"
    dataset = pd.read_csv(dataset_path)
    if "time" not in dataset.columns:
        raise ValueError(f"Dataset {dataset_path} must contain a 'time' column.")
    dataset["time"] = pd.to_datetime(dataset["time"])

    df = pd.DataFrame(rows)
    df["from"] = pd.to_datetime(df["from"])
    df = df.sort_values("from")

    merged = pd.merge_asof(
        dataset.sort_values("time"),
        df[["from", column]].rename(columns={"from": "ci_time", column: "ci_value"}),
        left_on="time",
        right_on="ci_time",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    merged["carbon_intensity"] = merged["ci_value"].combine_first(merged.get("carbon_intensity"))
    merged = merged.drop(columns=["ci_time", "ci_value"])
    merged.to_csv(dataset_path, index=False)


def main():
    args = parse_args()
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    if not args.local_csv:
        start = start or default_window()["start"]
        end = end or (start + timedelta(hours=args.hours))

    if args.region_id and args.postcode:
        raise ValueError("Specify only one of --region-id or --postcode.")

    if args.local_csv:
        rows = _load_local_csv(Path(args.local_csv), start, end)
    elif args.region_id or args.postcode:
        entries = fetch_regional_intensity(start, end, region_id=args.region_id, postcode=args.postcode)
        rows = _flatten_regional(entries)
    else:
        entries = fetch_national_intensity(start, end)
        rows = _flatten_national(entries)

    if not rows:
        raise SystemExit("No data returned from the Carbon Intensity API.")

    output_path = Path(args.output)
    _write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")

    if args.merge_dataset:
        dataset_path = Path(args.merge_dataset)
        _merge_with_dataset(dataset_path, rows, args.merge_column)
        print(f"Merged {output_path} into {dataset_path}")


if __name__ == "__main__":
    main()
