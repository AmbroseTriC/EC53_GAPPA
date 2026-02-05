#!/usr/bin/env python3
"""
Clean expenditure data (2019-2025) for selected regions and categories.
Outputs: Cleaning/Expenditure/expenditure_cleaned.xlsx
"""

from __future__ import annotations

import pandas as pd

INPUT_XLSX = "Expenditure-11100223/11100223.xlsx"
OUTPUT_XLSX = "Cleaning/Expenditure/expenditure_cleaned.xlsx"

TARGET_REGIONS = ["Canada", "Ontario", "Manitoba", "Saskatchewan", "Alberta", "Yukon"]
TARGET_CATEGORIES = [
    "Natural gas for principal accommodation",
    "Gas and other fuels (all vehicles and tools)",
]

QUINTILE_MAP = {
    "Lowest quintile": "1st quintile",
    "Second quintile": "2nd quintile",
    "Third quintile": "3rd quintile",
    "Fourth quintile": "4th quintile",
    "Highest quintile": "5th quintile",
    "All quintiles": "All quintiles",
}


def main() -> int:
    df = pd.read_excel(INPUT_XLSX, sheet_name=0)

    df["REF_DATE"] = pd.to_numeric(df["REF_DATE"], errors="coerce")
    df = df[(df["REF_DATE"] >= 2019) & (df["REF_DATE"] <= 2025)]

    df = df[df["GEO"].isin(TARGET_REGIONS)]
    df = df[df["Household expenditures, summary-level categories"].isin(TARGET_CATEGORIES)]

    df = df[df["Before-tax household income quintile"].isin(QUINTILE_MAP.keys())]
    df["Income_Quintile"] = df["Before-tax household income quintile"].map(QUINTILE_MAP)

    out = df[
        [
            "REF_DATE",
            "GEO",
            "Income_Quintile",
            "Household expenditures, summary-level categories",
            "UOM",
            "VALUE",
        ]
    ].copy()

    out.to_excel(OUTPUT_XLSX, index=False)
    print(f"Wrote {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
