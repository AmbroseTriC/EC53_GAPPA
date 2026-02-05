#!/usr/bin/env python3
"""
Clean fuel price data to province-level averages (2019-01 to 2025-01).
Outputs: Cleaning/energy_price_cleaned.xlsx
"""

from __future__ import annotations

import pandas as pd

INPUT_XLSX = "EnePrice-18100001/18100001.xlsx"
INPUT_SHEET = "18100001"
OUTPUT_XLSX = "Cleaning/energy_price_cleaned.xlsx"

TARGET_REGIONS = ["Ontario", "Manitoba", "Saskatchewan", "Alberta", "Yukon"]
TARGET_FUELS = [
    "Household heating fuel",
    "Regular unleaded gasoline at self service filling stations",
]


def map_region(geo: str) -> str | None:
    if not isinstance(geo, str):
        return None
    for region in TARGET_REGIONS:
        if region in geo:
            return region
    return None


def main() -> int:
    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], format="%Y-%m", errors="coerce")

    start = pd.Timestamp("2019-01-01")
    end = pd.Timestamp("2025-01-01")
    df = df[(df["REF_DATE"] >= start) & (df["REF_DATE"] <= end)]

    df = df[df["Type of fuel"].isin(TARGET_FUELS)]

    df["Region"] = df["GEO"].apply(map_region)
    df = df[df["Region"].notna()]

    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df[df["VALUE"].notna()]

    result = (
        df.groupby(["REF_DATE", "Region", "Type of fuel", "UOM"], as_index=False)[
            "VALUE"
        ]
        .mean()
        .sort_values(["REF_DATE", "Region", "Type of fuel"])
    )

    result.to_excel(OUTPUT_XLSX, index=False)
    print(f"Wrote {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
