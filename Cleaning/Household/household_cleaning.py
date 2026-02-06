#!/usr/bin/env python3
"""
Clean household counts (2019-2024) for selected regions and quintiles.
Outputs: Cleaning/Household/household_cleaned.xlsx
"""

from __future__ import annotations

import pandas as pd

INPUT_XLSX = "Household-36100101/36100101.xlsx"
OUTPUT_XLSX = "Cleaning/Household/household_cleaned.xlsx"

TARGET_SOCIO = [
    "Ontario, households",
    "Manitoba, households",
    "Saskatchewan, households",
    "Alberta, households",
    "Yukon, households",
    "All households",
]

REGION_MAP = {
    "Ontario, households": "Ontario",
    "Manitoba, households": "Manitoba",
    "Saskatchewan, households": "Saskatchewan",
    "Alberta, households": "Alberta",
    "Yukon, households": "Yukon",
    "All households": "Canada",
}

QUINTILE_MAP = {
    "Lowest quintile": "1st quintile",
    "Second quintile": "2nd quintile",
    "Third quintile": "3rd quintile",
    "Fourth quintile": "4th quintile",
    "Highest quintile": "5th quintile",
    "All quintiles": "All quintiles",
}


def main() -> int:
    df = pd.read_excel(
        INPUT_XLSX,
        sheet_name=0,
        usecols=[
            "REF_DATE",
            "GEO",
            "Quintile",
            "Socio-demographic characteristics",
            "UOM",
            "VALUE",
        ],
    )

    df["REF_DATE"] = pd.to_numeric(df["REF_DATE"], errors="coerce")
    df = df[(df["REF_DATE"] >= 2019) & (df["REF_DATE"] <= 2024)]

    df = df[df["Socio-demographic characteristics"].isin(TARGET_SOCIO)]
    df = df[df["Quintile"].isin(QUINTILE_MAP.keys())]

    df["Region"] = df["Socio-demographic characteristics"].map(REGION_MAP)
    df["Income_Quintile"] = df["Quintile"].map(QUINTILE_MAP)

    out = df[
        [
            "REF_DATE",
            "Region",
            "Income_Quintile",
            "UOM",
            "VALUE",
        ]
    ].copy()

    out.to_excel(OUTPUT_XLSX, index=False)
    print(f"Wrote {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
