#!/usr/bin/env python3
"""
Compute fuel-consumption tax burden metrics from cleaned workbooks.
Outputs: Calculation/tax_burden_analysis.xlsx
"""

from __future__ import annotations

import pandas as pd

ENERGY_PATH = "Cleaning/Energy Price/energy_price_cleaned.xlsx"
EXPENDITURE_PATH = "Cleaning/Expenditure/expenditure_cleaned.xlsx"
HOUSEHOLD_PATH = "Cleaning/Household/household_cleaned.xlsx"
TAX_PATH = "Crawling/Tax Rate Items.xlsx"
OUTPUT_PATH = "Calculation/tax_burden_analysis.xlsx"

TARGET_REGIONS = ["Ontario", "Manitoba", "Saskatchewan", "Alberta", "Yukon"]
TARGET_YEARS = [2019, 2021, 2023]
TARGET_QUINTILES = [
    "1st quintile",
    "2nd quintile",
    "3rd quintile",
    "4th quintile",
    "5th quintile",
    "All quintiles",
]
FUEL_KEYS = ["HHF", "Gasoline"]

EXP_CAT_TO_FUEL = {
    "Natural gas for principal accommodation": "HHF",
    "Gas and other fuels (all vehicles and tools)": "Gasoline",
}
ENERGY_FUEL_TO_KEY = {
    "Household heating fuel": "HHF",
    "Regular unleaded gasoline at self service filling stations": "Gasoline",
}
TAX_FUEL_TO_KEY = {
    "Marketable natural gas": "HHF",
    "Gasoline": "Gasoline",
}


def _append_missing(flags: list[str], present: bool, label: str) -> None:
    if not present:
        flags.append(label)


def _build_missing_string(row: pd.Series) -> str:
    flags: list[str] = []
    _append_missing(flags, pd.notna(row["PerHouseholdExpenditure_CAD"]), "missing_expenditure")
    _append_missing(flags, pd.notna(row["Households_Number"]), "missing_households")
    _append_missing(flags, pd.notna(row["RetailPrice_Cents_per_L"]), "missing_energy_price")
    _append_missing(flags, pd.notna(row["TaxRate_CAD_per_L"]), "missing_tax_rate")
    return ";".join(flags)


def _collapse_missing(series: pd.Series) -> str:
    tokens: set[str] = set()
    for val in series.dropna():
        if not val:
            continue
        tokens.update(x for x in str(val).split(";") if x)
    if not tokens:
        return ""
    return ";".join(sorted(tokens))


def _date_overlap_days(
    start: pd.Timestamp,
    end: pd.Timestamp | pd.NaT,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
) -> int:
    actual_end = end if pd.notna(end) else year_end
    overlap_start = max(start, year_start)
    overlap_end = min(actual_end, year_end)
    if overlap_end < overlap_start:
        return 0
    return int((overlap_end - overlap_start).days + 1)


def _annual_tax_rate(df_rates: pd.DataFrame) -> pd.DataFrame:
    rates = df_rates.copy()
    rates["Period_Start"] = pd.to_datetime(rates["Period_Start"], errors="coerce")
    rates["Period_End"] = pd.to_datetime(rates["Period_End"], errors="coerce")

    # Remove exact PIT duplicates.
    rates = rates.drop_duplicates(
        subset=["Period_Start", "Period_End", "Fuel_Type", "Province", "Rate"]
    )

    # If an open-ended period exists with same start/province/fuel/rate, keep open-ended only.
    open_keys = set(
        tuple(x)
        for x in rates[rates["Period_End"].isna()][
            ["Period_Start", "Fuel_Type", "Province", "Rate"]
        ].itertuples(index=False, name=None)
    )
    mask_drop = (
        rates["Period_End"].notna()
        & rates.apply(
            lambda r: (r["Period_Start"], r["Fuel_Type"], r["Province"], r["Rate"])
            in open_keys,
            axis=1,
        )
    )
    rates = rates[~mask_drop].copy()

    rows: list[dict[str, object]] = []
    for (province, fuel_type), grp in rates.groupby(["Province", "Fuel_Type"]):
        for year in TARGET_YEARS:
            year_start = pd.Timestamp(year=year, month=1, day=1)
            year_end = pd.Timestamp(year=year, month=12, day=31)
            days_in_year = int((year_end - year_start).days + 1)

            weighted_sum = 0.0
            for _, row in grp.iterrows():
                if pd.isna(row["Period_Start"]):
                    continue
                days = _date_overlap_days(
                    row["Period_Start"], row["Period_End"], year_start, year_end
                )
                if days > 0:
                    weighted_sum += float(row["Rate"]) * days

            annual_rate = weighted_sum / days_in_year
            rows.append(
                {
                    "Year": year,
                    "Region": province,
                    "Fuel_Key": TAX_FUEL_TO_KEY[fuel_type],
                    "TaxRate_CAD_per_L": annual_rate,
                }
            )

    return pd.DataFrame(rows)


def load_energy() -> pd.DataFrame:
    energy = pd.read_excel(ENERGY_PATH)
    energy = energy[energy["Region"].isin(TARGET_REGIONS)].copy()
    energy["Year"] = pd.to_datetime(energy["REF_DATE"], errors="coerce").dt.year
    energy = energy[energy["Year"].isin(TARGET_YEARS)].copy()
    energy = energy[energy["Type of fuel"].isin(ENERGY_FUEL_TO_KEY.keys())].copy()
    energy["Fuel_Key"] = energy["Type of fuel"].map(ENERGY_FUEL_TO_KEY)
    energy["VALUE"] = pd.to_numeric(energy["VALUE"], errors="coerce")
    out = (
        energy.groupby(["Year", "Region", "Fuel_Key"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"VALUE": "RetailPrice_Cents_per_L"})
    )
    return out


def load_expenditure() -> pd.DataFrame:
    exp = pd.read_excel(EXPENDITURE_PATH)
    exp = exp[exp["GEO"].isin(TARGET_REGIONS)].copy()
    exp["REF_DATE"] = pd.to_numeric(exp["REF_DATE"], errors="coerce")
    exp = exp[exp["REF_DATE"].isin(TARGET_YEARS)].copy()
    exp = exp[exp["Income_Quintile"].isin(TARGET_QUINTILES)].copy()
    exp = exp[
        exp["Household expenditures, summary-level categories"].isin(EXP_CAT_TO_FUEL.keys())
    ].copy()
    exp["Fuel_Key"] = exp["Household expenditures, summary-level categories"].map(
        EXP_CAT_TO_FUEL
    )
    exp["VALUE"] = pd.to_numeric(exp["VALUE"], errors="coerce")
    out = exp.rename(
        columns={
            "REF_DATE": "Year",
            "GEO": "Region",
            "VALUE": "PerHouseholdExpenditure_CAD",
        }
    )[
        ["Year", "Region", "Income_Quintile", "Fuel_Key", "PerHouseholdExpenditure_CAD"]
    ]
    return out


def load_households() -> pd.DataFrame:
    hh = pd.read_excel(HOUSEHOLD_PATH)
    hh = hh[hh["Region"].isin(TARGET_REGIONS)].copy()
    hh["REF_DATE"] = pd.to_numeric(hh["REF_DATE"], errors="coerce")
    hh = hh[hh["REF_DATE"].isin(TARGET_YEARS)].copy()
    hh = hh[hh["Income_Quintile"].isin(TARGET_QUINTILES)].copy()
    hh["VALUE"] = pd.to_numeric(hh["VALUE"], errors="coerce")
    out = hh.rename(
        columns={"REF_DATE": "Year", "VALUE": "Households_Number"}
    )[["Year", "Region", "Income_Quintile", "Households_Number"]]
    return out


def load_tax_rates() -> pd.DataFrame:
    rates = pd.read_excel(TAX_PATH, sheet_name="Rates")
    rates = rates[rates["Province"].isin(TARGET_REGIONS)].copy()
    rates = rates[rates["Fuel_Type"].isin(TAX_FUEL_TO_KEY.keys())].copy()
    rates["Rate"] = pd.to_numeric(rates["Rate"], errors="coerce")
    rates = rates[rates["Rate"].notna()].copy()
    return _annual_tax_rate(rates)


def build_fuel_level() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [TARGET_YEARS, TARGET_REGIONS, TARGET_QUINTILES, FUEL_KEYS],
        names=["Year", "Region", "Income_Quintile", "Fuel_Key"],
    )
    base = idx.to_frame(index=False)

    fuel = base.copy()
    fuel = fuel.merge(load_expenditure(), how="left")
    fuel = fuel.merge(load_households(), how="left")
    fuel = fuel.merge(load_energy(), on=["Year", "Region", "Fuel_Key"], how="left")
    fuel = fuel.merge(load_tax_rates(), on=["Year", "Region", "Fuel_Key"], how="left")

    fuel["TotalExpenditure_CAD"] = (
        fuel["PerHouseholdExpenditure_CAD"] * fuel["Households_Number"]
    )
    fuel["RetailPrice_CAD_per_L"] = fuel["RetailPrice_Cents_per_L"] / 100.0
    fuel["Volume_L"] = fuel["TotalExpenditure_CAD"] / fuel["RetailPrice_CAD_per_L"]
    fuel["DirectTax_CAD"] = fuel["Volume_L"] * fuel["TaxRate_CAD_per_L"]
    fuel["EffectiveTax_CAD"] = fuel["DirectTax_CAD"] * 2.0

    fuel["Missing_Inputs"] = fuel.apply(_build_missing_string, axis=1)
    fuel["Is_Complete"] = fuel["Missing_Inputs"].eq("")

    return fuel[
        [
            "Year",
            "Region",
            "Income_Quintile",
            "Fuel_Key",
            "PerHouseholdExpenditure_CAD",
            "Households_Number",
            "TotalExpenditure_CAD",
            "RetailPrice_Cents_per_L",
            "RetailPrice_CAD_per_L",
            "TaxRate_CAD_per_L",
            "Volume_L",
            "DirectTax_CAD",
            "EffectiveTax_CAD",
            "Is_Complete",
            "Missing_Inputs",
        ]
    ].sort_values(["Year", "Region", "Income_Quintile", "Fuel_Key"])


def build_summary(fuel_level: pd.DataFrame) -> pd.DataFrame:
    grp = fuel_level.groupby(["Year", "Region", "Income_Quintile"], as_index=False)
    summary = grp.agg(
        EffectiveTax_Total_CAD=("EffectiveTax_CAD", lambda s: s.sum(min_count=2)),
        TargetExpenditure_Total_CAD=("TotalExpenditure_CAD", lambda s: s.sum(min_count=2)),
        Missing_Inputs=("Missing_Inputs", _collapse_missing),
    )
    summary["EffectiveTax_Pct_of_TargetExpenditure"] = (
        summary["EffectiveTax_Total_CAD"] / summary["TargetExpenditure_Total_CAD"] * 100.0
    )
    summary["Is_Complete"] = summary["Missing_Inputs"].eq("")
    return summary[
        [
            "Year",
            "Region",
            "Income_Quintile",
            "EffectiveTax_Total_CAD",
            "TargetExpenditure_Total_CAD",
            "EffectiveTax_Pct_of_TargetExpenditure",
            "Is_Complete",
            "Missing_Inputs",
        ]
    ].sort_values(["Year", "Region", "Income_Quintile"])


def main() -> int:
    fuel_level = build_fuel_level()
    summary = build_summary(fuel_level)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        fuel_level.to_excel(writer, sheet_name="Fuel_Level", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
