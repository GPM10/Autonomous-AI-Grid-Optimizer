"""Extract slices from the OPSD SQLite time-series package."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd


TABLE_TEMPLATE = "time_series_{resolution:02d}min_singleindex"


def available_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    query = f"PRAGMA table_info({table});"
    cur = conn.execute(query)
    return [row[1] for row in cur.fetchall()]


def build_sql(country: str, table: str, start: str, end: str, columns: List[str]) -> str:
    select_cols = ",\n            ".join(columns)
    sql = f"""
        SELECT
            utc_timestamp AS time,
            {select_cols}
        FROM {table}
        WHERE utc_timestamp BETWEEN :start AND :end
        ORDER BY utc_timestamp
    """
    return sql


def extract_opsd_slice(
    db_path: str,
    country: str,
    start: str,
    end: str,
    resolution: int = 60,
    constant_carbon: float = 150.0,
) -> pd.DataFrame:
    table = TABLE_TEMPLATE.format(resolution=resolution)
    conn = sqlite3.connect(db_path)

    cols = available_columns(conn, table)
    mappings = {
        "demand": f"{country}_load_actual_entsoe_transparency",
        "solar": f"{country}_solar_generation_actual",
        "price": f"{country}_price_day_ahead",
        "wind_onshore": f"{country}_wind_onshore_generation_actual",
    }

    select_cols = []
    for alias, column in mappings.items():
        if column in cols:
            select_cols.append(f'"{column}" AS {alias}')
        elif alias in {"price", "wind_onshore"}:
            select_cols.append(f"NULL AS {alias}")
        else:
            conn.close()
            raise SystemExit(f"Column '{column}' not found in table {table}. Check country code or resolution.")

    sql = build_sql(country, table, start, end, select_cols)
    df = pd.read_sql_query(sql, conn, params={"start": start, "end": end})
    conn.close()

    if df.empty:
        raise SystemExit("Query returned no rows. Adjust time window or country.")

    df["carbon_intensity"] = constant_carbon
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OPSD SQLite slices into CSV format.")
    parser.add_argument("--db-path", default="data/time_series.sqlite", help="Path to OPSD SQLite file")
    parser.add_argument("--country", required=True, help="Country/prefix as in OPSD columns (e.g., GB_GBN, FR)")
    parser.add_argument("--start", required=True, help="Start timestamp (ISO8601, e.g., 2020-01-01T00:00:00Z)")
    parser.add_argument("--end", required=True, help="End timestamp (ISO8601)")
    parser.add_argument("--resolution", type=int, choices=(15, 30, 60), default=60, help="Resolution in minutes")
    parser.add_argument("--output", default="data/opsd_export.csv", help="Output CSV path")
    parser.add_argument("--constant-carbon", type=float, default=150.0, help="Fallback carbon intensity value to assign")
    args = parser.parse_args()

    df = extract_opsd_slice(
        db_path=args.db_path,
        country=args.country,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
        constant_carbon=args.constant_carbon,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Exported {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
