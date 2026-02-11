#!/usr/bin/env python3
"""
compute_ggppa_progressivity_v4.py

Purpose
-------
Replicates a reduced-form incidence model for Canada's federal fuel charge (GGPPA Part I)
and the associated Climate Action Incentive / Canada Carbon Rebate (CCR/CAI), by
province-income quintile-policy year.

Outputs
-------
1) ggppa_progressivity_results_v4.csv
   Quintile-by-province-by-year incidence measures (direct, gross w/ indirect wedge, net
   under CCR scenarios), plus effective tax rates.

2) ggppa_kakwani_indices_v4.csv
   Kakwani indices for selected measures, using total expenditure as the ranking variable.

3) ggppa_finance_crosswalk_v4.csv
   Diagnostics showing how indirect wedges were calibrated to Finance Canada anchors.

Data inputs (expected in the working directory)
----------------------------------------------
- 11100223.xlsx   (Statistics Canada Table 11-10-0223-01 extract)
- 18100001.xlsx   (Statistics Canada Table 18-10-0001-01 extract)
- 36100101.xlsx   (Statistics Canada Table 36-10-0101-01 extract)
- Tax Rate Items_v2.xlsx (provided extract of federal fuel-charge rate items)

Notes
-----
- Expenditure years in the provided extract are 2019, 2021, 2023; intermediate years are
  linearly interpolated. 2024-2025 are held at 2023 levels absent additional data.

- Indirect effects are represented via a multiplicative wedge kappa_{r,t} calibrated to
  Finance Canada "average cost impact per household" anchors in 2021, 2022, and 2024.
  Anchor-year wedges are enforced exactly; non-anchor years are predicted from a
  province-level OLS fit in statutory carbon-price space.

- Natural gas quantities are approximated from expenditures using province-specific GJ/$
  factors (reduced form). The conversion 1 GJ = 26.853 m^3 is used to convert statutory
  $/m^3 rates to $/GJ.

- Alberta is treated as not receiving CAI in 2019; Alberta 2020 CAI amounts reflect the
  15-month adjustment described in Finance Canada materials.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


PROVINCES = ["Ontario", "Manitoba", "Saskatchewan", "Alberta"]
QUINTILES = [
    "Lowest quintile",
    "Second quintile",
    "Third quintile",
    "Fourth quintile",
    "Highest quintile",
]
YEARS_FULL = list(range(2019, 2026))

# Statutory carbon price schedule used in kappa calibration.
CARBON_PRICE = {2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80, 2025: 0}

# Fuel types and CPI series labels used in 18-10-0001-01 extract
GAS_TYPE = "Regular unleaded gasoline at self service filling stations"
OIL_TYPE = "Household heating fuel"

# Expenditure categories used in 11-10-0223-01 extract
CAT_MAP = {
    "gas_exp": "Gas and other fuels (all vehicles and tools)",
    "ng_exp": "Natural gas for principal accommodation",
    "oil_exp": "Other fuel for principal accommodation",
    "total_exp": "Total expenditure",
}

# Reduced-form natural gas conversion (GJ per dollar of expenditure), by province
GJ_PER_DOLLAR = {
    "Ontario": 0.0720192545,
    "Manitoba": 0.1386298932,
    "Saskatchewan": 0.0878257722,
    "Alberta": 0.0858479021,
}

M3_PER_GJ = 26.853  # CRA Schedule 2 conversion


# CCR schedules (base amounts) by policy year and province: {first, second, child}
CCR_SCHEDULE = {
    2019: {
        "Ontario": {"first": 154, "second": 77, "child": 38},
        "Manitoba": {"first": 170, "second": 85, "child": 42},
        "Saskatchewan": {"first": 305, "second": 152, "child": 76},
        "Alberta": {"first": 0, "second": 0, "child": 0},
    },
    2020: {
        "Ontario": {"first": 224, "second": 112, "child": 56},
        "Manitoba": {"first": 243, "second": 121, "child": 61},
        "Saskatchewan": {"first": 405, "second": 202, "child": 101},
        "Alberta": {"first": 444, "second": 222, "child": 111},  # 15 months
    },
    2021: {
        "Ontario": {"first": 300, "second": 150, "child": 75},
        "Manitoba": {"first": 360, "second": 180, "child": 90},
        "Saskatchewan": {"first": 500, "second": 250, "child": 125},
        "Alberta": {"first": 490, "second": 245, "child": 123},
    },
    2022: {
        "Ontario": {"first": 373, "second": 186, "child": 93},
        "Manitoba": {"first": 416, "second": 208, "child": 104},
        "Saskatchewan": {"first": 550, "second": 275, "child": 138},
        "Alberta": {"first": 539, "second": 270, "child": 135},
    },
    2023: {
        "Ontario": {"first": 488, "second": 244, "child": 122},
        "Manitoba": {"first": 528, "second": 264, "child": 132},
        "Saskatchewan": {"first": 680, "second": 340, "child": 170},
        "Alberta": {"first": 772, "second": 386, "child": 193},
    },
    2024: {
        "Ontario": {"first": 560, "second": 280, "child": 140},
        "Manitoba": {"first": 600, "second": 300, "child": 150},
        "Saskatchewan": {"first": 752, "second": 376, "child": 188},
        "Alberta": {"first": 900, "second": 450, "child": 225},
    },
    2025: {p: {"first": 0, "second": 0, "child": 0} for p in PROVINCES},
}

RURAL_TOPUP = {2019: 0.10, 2020: 0.10, 2021: 0.10, 2022: 0.10, 2023: 0.10, 2024: 0.20, 2025: 0.0}

# Finance Canada cost-impact anchors for indirect wedge calibration (dollars per household)
FINANCE_COST_IMPACT = {
    2021: {"Ontario": 439, "Manitoba": 462, "Saskatchewan": 720, "Alberta": 598},
    2022: {"Ontario": 578, "Manitoba": 559, "Saskatchewan": 734, "Alberta": 700},
    2024: {"Ontario": 869, "Manitoba": 828, "Saskatchewan": 1156, "Alberta": 1056},
}


def city_to_province(city_geo: str) -> str | None:
    if not isinstance(city_geo, str):
        return None
    for p in PROVINCES:
        if p in city_geo:
            return p
    return None


def weighted_gini(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumw = np.cumsum(w)
    cumxw = np.cumsum(x * w)
    if cumxw[-1] == 0:
        return 0.0
    cumw_norm = cumw / cumw[-1]
    cumxw_norm = cumxw / cumxw[-1]
    B = np.trapz(cumxw_norm, cumw_norm)
    return 1 - 2 * B


def concentration_coefficient(tax: np.ndarray, income: np.ndarray, w: np.ndarray) -> float:
    tax = np.asarray(tax, dtype=float)
    income = np.asarray(income, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(income)
    tax = tax[order]
    w = w[order]
    cumw = np.cumsum(w)
    cumt = np.cumsum(tax * w)
    if cumt[-1] == 0:
        return 0.0
    cumw_norm = cumw / cumw[-1]
    cumt_norm = cumt / cumt[-1]
    B = np.trapz(cumt_norm, cumw_norm)
    return 1 - 2 * B


def kakwani_index(tax: np.ndarray, income: np.ndarray, w: np.ndarray) -> tuple[float | float("nan"), float, float]:
    G = weighted_gini(income, w)
    total = float(np.sum(tax * w))
    if total <= 0 or abs(total) < 1e-9:
        return (float("nan"), G, float("nan"))
    C = concentration_coefficient(tax, income, w)
    return (C - G, G, C)


def load_expenditures(path: str) -> pd.DataFrame:
    exp = pd.read_excel(path, sheet_name="11100223")
    exp = exp[exp["REF_DATE"].between(2019, 2023)]
    exp = exp[(exp["GEO"].isin(PROVINCES)) & (exp["Before-tax household income quintile"].isin(QUINTILES))]
    exp = exp[exp["Household expenditures, summary-level categories"].isin(CAT_MAP.values())].copy()

    exp = exp[["REF_DATE", "GEO", "Before-tax household income quintile", "Household expenditures, summary-level categories", "VALUE"]]
    exp.rename(
        columns={
            "REF_DATE": "Year",
            "GEO": "Province",
            "Before-tax household income quintile": "Quintile",
            "VALUE": "Expenditure",
        },
        inplace=True,
    )
    piv = exp.pivot_table(
        index=["Province", "Quintile", "Year"],
        columns="Household expenditures, summary-level categories",
        values="Expenditure",
        aggfunc="mean",
    ).reset_index()
    piv.rename(columns={v: k for k, v in CAT_MAP.items()}, inplace=True)

    full_index = pd.MultiIndex.from_product([PROVINCES, QUINTILES, YEARS_FULL], names=["Province", "Quintile", "Year"])
    bal = piv.set_index(["Province", "Quintile", "Year"]).reindex(full_index).reset_index()
    for col in ["gas_exp", "ng_exp", "oil_exp", "total_exp"]:
        bal[col] = bal.groupby(["Province", "Quintile"])[col].transform(lambda s: s.astype(float).interpolate(limit_direction="both"))
    return bal


def load_prices(path: str) -> pd.DataFrame:
    price = pd.read_excel(path, sheet_name="18100001")
    price["date"] = pd.to_datetime(price["REF_DATE"])
    price["Year"] = price["date"].dt.year
    price = price[price["Year"].between(2019, 2025)]
    price = price[price["Type of fuel"].isin([GAS_TYPE, OIL_TYPE])].copy()
    price["Province"] = price["GEO"].apply(city_to_province)
    price = price[price["Province"].isin(PROVINCES)]
    price["Value"] = price["VALUE"].astype(float)

    annual = price.groupby(["Province", "Year", "Type of fuel"])["Value"].mean().reset_index()
    piv = annual.pivot_table(index=["Province", "Year"], columns="Type of fuel", values="Value").reset_index()
    piv.rename(columns={GAS_TYPE: "gasoline_cents_per_litre", OIL_TYPE: "heatingfuel_cents_per_litre"}, inplace=True)

    piv["gasoline_price"] = piv["gasoline_cents_per_litre"] / 100.0
    piv["heating_oil_price"] = piv["heatingfuel_cents_per_litre"] / 100.0

    # Fill missing heating-oil prices (not observed in AB in the provided extract) with the across-province mean for that year
    avg_oil_by_year = piv.groupby("Year")["heating_oil_price"].mean()
    mask = piv["heating_oil_price"].isna()
    piv.loc[mask, "heating_oil_price"] = piv.loc[mask, "Year"].map(avg_oil_by_year)

    return piv[["Province", "Year", "gasoline_price", "heating_oil_price"]]


def load_households(path: str) -> pd.DataFrame:
    hh = pd.read_excel(path, sheet_name="36100101")
    hh = hh[hh["REF_DATE"].between(2019, 2024)]
    hh = hh[hh["Quintile"].isin(QUINTILES)].copy()
    prov_char_map = {p: f"{p}, households" for p in PROVINCES}
    hh = hh[hh["Socio-demographic characteristics"].isin(prov_char_map.values())]
    hh = hh[["REF_DATE", "Quintile", "Socio-demographic characteristics", "VALUE"]]
    hh.rename(columns={"REF_DATE": "Year", "Socio-demographic characteristics": "Char", "VALUE": "Households"}, inplace=True)
    inv = {v: k for k, v in prov_char_map.items()}
    hh["Province"] = hh["Char"].map(inv)
    hh = hh[["Province", "Quintile", "Year", "Households"]]

    full_index = pd.MultiIndex.from_product([PROVINCES, QUINTILES, YEARS_FULL], names=["Province", "Quintile", "Year"])
    bal = hh.set_index(["Province", "Quintile", "Year"]).reindex(full_index).reset_index()
    bal["Households"] = bal.groupby(["Province", "Quintile"])["Households"].transform(lambda s: s.astype(float).interpolate(limit_direction="both"))
    return bal


def build_rates(path: str) -> pd.DataFrame:
    fuel_cols = ["Gasoline", "Light fuel oil", "Marketable natural gas"]
    rates = pd.read_excel(path, sheet_name="Rates")
    rates = rates[rates["Fuel_Type"].isin(fuel_cols)]
    rates = rates[rates["Province"].isin(PROVINCES)].copy()
    rates["Rate"] = pd.to_numeric(rates["Rate"], errors="coerce")
    rates["Period_Start"] = pd.to_datetime(rates["Period_Start"], errors="coerce")
    rates["Year"] = rates["Period_Start"].dt.year
    rates = rates[rates["Year"].between(min(YEARS_FULL), max(YEARS_FULL))]

    piv = (
        rates.pivot_table(
            index=["Year", "Province"],
            columns="Fuel_Type",
            values="Rate",
            aggfunc="mean",
        )
        .reset_index()
    )
    for col in fuel_cols:
        if col not in piv.columns:
            piv[col] = float("nan")

    full_index = pd.MultiIndex.from_product(
        [PROVINCES, YEARS_FULL], names=["Province", "Year"]
    )
    bal = (
        piv.set_index(["Province", "Year"])[fuel_cols]
        .reindex(full_index)
        .reset_index()
    )

    # Alberta: no fuel charge modeled in 2019 (legacy model rule).
    mask_ab_2019 = (bal["Province"] == "Alberta") & (bal["Year"] == 2019)
    for col in fuel_cols:
        bal.loc[mask_ab_2019, col] = 0.0

    return bal[["Province", "Year", *fuel_cols]]


def build_ccr() -> pd.DataFrame:
    rows: list[dict] = []
    for y in YEARS_FULL:
        for prov in PROVINCES:
            sch = CCR_SCHEDULE.get(y, {}).get(prov, {"first": 0, "second": 0, "child": 0})
            first, second, child = sch["first"], sch["second"], sch["child"]

            base_single = first
            base_couple = first + second
            base_family4 = first + second + 2 * child

            top = RURAL_TOPUP.get(y, 0.0)
            rows.append(
                {
                    "Province": prov,
                    "Year": y,
                    "CCR_single_base": base_single,
                    "CCR_couple_base": base_couple,
                    "CCR_family4_base": base_family4,
                    "CCR_single_rural": base_single * (1 + top),
                    "CCR_couple_rural": base_couple * (1 + top),
                    "CCR_family4_rural": base_family4 * (1 + top),
                }
            )
    return pd.DataFrame(rows)


def calibrate_kappa(df: pd.DataFrame) -> pd.DataFrame:
    # Household-weighted provincial average direct tax by year
    direct_avg = (
        df.groupby(["Province", "Year"])
        .apply(lambda g: np.average(g["direct_tax"], weights=g["Households"]))
        .reset_index(name="direct_tax_avg")
    )
    direct_lookup = {
        (str(r["Province"]), int(r["Year"])): float(r["direct_tax_avg"])
        for _, r in direct_avg.iterrows()
    }

    anchor_years = sorted(int(y) for y in FINANCE_COST_IMPACT.keys())
    anchor: dict[str, dict[int, float]] = {}
    for y, prov_dict in FINANCE_COST_IMPACT.items():
        if y not in CARBON_PRICE:
            raise ValueError(f"Anchor year {y} exists in FINANCE_COST_IMPACT but is missing from CARBON_PRICE.")
        for prov, cost in prov_dict.items():
            key = (prov, y)
            if key not in direct_lookup:
                raise ValueError(f"Missing direct tax average for anchor calibration at Province={prov}, Year={y}.")
            direct = direct_lookup[key]
            anchor.setdefault(prov, {})[y] = cost / direct - 1

    rows: list[dict] = []
    for prov in PROVINCES:
        if prov not in anchor:
            raise ValueError(f"No Finance anchors available for province {prov}.")
        for y in anchor_years:
            if y not in anchor[prov]:
                raise ValueError(f"Missing Finance anchor for Province={prov}, Year={y}.")

        X = np.array([[1.0, float(CARBON_PRICE[y])] for y in anchor_years], dtype=float)
        yvals = np.array([anchor[prov][y] for y in anchor_years], dtype=float)
        alpha, beta = np.linalg.lstsq(X, yvals, rcond=None)[0]

        for y in YEARS_FULL:
            P = CARBON_PRICE[y]
            if P == 0:
                k = 0.0
            elif y in anchor[prov]:
                k = anchor[prov][y]
            else:
                k = max(0.0, alpha + beta * P)
            rows.append({"Province": prov, "Year": y, "kappa": k})
    return pd.DataFrame(rows)


def main() -> None:
    exp = load_expenditures("11100223.xlsx")
    prices = load_prices("18100001.xlsx")
    hh = load_households("36100101.xlsx")
    rates = build_rates("Tax Rate Items_v2.xlsx")
    ccr = build_ccr()

    df = exp.merge(prices, on=["Province", "Year"], how="left").merge(hh, on=["Province", "Quintile", "Year"], how="left").merge(rates, on=["Province", "Year"], how="left")

    # Statutory carbon price schedule (used for transparency and kappa calibration)
    df["carbon_price_per_tCO2e"] = df["Year"].map(CARBON_PRICE).astype(float)

    # Quantities
    df["gasoline_litres"] = df["gas_exp"] / df["gasoline_price"]
    df["oil_litres"] = df["oil_exp"] / df["heating_oil_price"]
    df["ng_GJ_per_dollar"] = df["Province"].map(GJ_PER_DOLLAR).astype(float)
    df["ng_GJ"] = df["ng_exp"] * df["ng_GJ_per_dollar"]
    df["ng_rate_per_GJ"] = df["Marketable natural gas"] * M3_PER_GJ
    df["ng_m3"] = df["ng_GJ"] * M3_PER_GJ

    # Direct taxes
    df["tax_gasoline"] = df["gasoline_litres"] * df["Gasoline"]
    df["tax_oil"] = df["oil_litres"] * df["Light fuel oil"]
    df["tax_ng"] = df["ng_GJ"] * df["ng_rate_per_GJ"]
    df["direct_tax"] = df[["tax_gasoline", "tax_oil", "tax_ng"]].sum(axis=1)

    # Indirect wedge
    kappa = calibrate_kappa(df)
    df = df.merge(kappa, on=["Province", "Year"], how="left")
    df["gross_cost"] = df["direct_tax"] * (1 + df["kappa"])
    df["indirect_cost"] = df["gross_cost"] - df["direct_tax"]

    # CCR + net incidence
    df = df.merge(ccr, on=["Province", "Year"], how="left")
    for typ in ["single", "couple", "family4"]:
        df[f"net_{typ}_base"] = df["gross_cost"] - df[f"CCR_{typ}_base"]
        df[f"net_{typ}_rural"] = df["gross_cost"] - df[f"CCR_{typ}_rural"]

    # Effective tax rates
    df["etr_direct_tax"] = df["direct_tax"] / df["total_exp"]
    df["etr_gross_cost"] = df["gross_cost"] / df["total_exp"]
    for typ in ["single", "couple", "family4"]:
        df[f"etr_net_{typ}_base"] = df[f"net_{typ}_base"] / df["total_exp"]
        df[f"etr_net_{typ}_rural"] = df[f"net_{typ}_rural"] / df["total_exp"]

    # Output main results
    out_cols = [
        "Province",
        "Year",
        "carbon_price_per_tCO2e",
        "Quintile",
        "Households",
        "gas_exp",
        "ng_exp",
        "oil_exp",
        "total_exp",
        "gasoline_price",
        "heating_oil_price",
        "gasoline_litres",
        "oil_litres",
        "ng_GJ_per_dollar",
        "ng_GJ",
        "ng_m3",
        "Gasoline",
        "Light fuel oil",
        "Marketable natural gas",
        "ng_rate_per_GJ",
        "tax_gasoline",
        "tax_oil",
        "tax_ng",
        "direct_tax",
        "kappa",
        "indirect_cost",
        "gross_cost",
        "CCR_single_base",
        "CCR_couple_base",
        "CCR_family4_base",
        "CCR_single_rural",
        "CCR_couple_rural",
        "CCR_family4_rural",
        "net_single_base",
        "net_couple_base",
        "net_family4_base",
        "net_single_rural",
        "net_couple_rural",
        "net_family4_rural",
        "etr_direct_tax",
        "etr_gross_cost",
        "etr_net_single_base",
        "etr_net_couple_base",
        "etr_net_family4_base",
        "etr_net_single_rural",
        "etr_net_couple_rural",
        "etr_net_family4_rural",
    ]
    df[out_cols].to_csv("ggppa_progressivity_results_v4.csv", index=False)

    # Kakwani indices (selected measures)
    measures = ["direct_tax", "gross_cost", "net_single_base"]
    kak_rows = []
    for (prov, year), sub in df.groupby(["Province", "Year"]):
        income = sub["total_exp"].values
        w = sub["Households"].values
        for m in measures:
            K, G, C = kakwani_index(sub[m].values, income, w)
            kak_rows.append(
                {
                    "Province": prov,
                    "Year": year,
                    "Measure": m,
                    "Kakwani": K,
                    "Gini_expenditure": G,
                    "Concentration": C,
                    "Total_net_revenue": float(np.sum(sub[m].values * w)),
                }
            )
    pd.DataFrame(kak_rows).to_csv("ggppa_kakwani_indices_v4.csv", index=False)

    # Calibration diagnostic
    direct_avg = (
        df.groupby(["Province", "Year"])
        .apply(lambda g: np.average(g["direct_tax"], weights=g["Households"]))
        .reset_index(name="Direct_model_avg")
    )
    gross_avg = (
        df.groupby(["Province", "Year"])
        .apply(lambda g: np.average(g["gross_cost"], weights=g["Households"]))
        .reset_index(name="Gross_model_avg")
    )
    diag = direct_avg.merge(gross_avg, on=["Province", "Year"], how="left")
    diag["Finance_cost_impact"] = diag.apply(lambda r: FINANCE_COST_IMPACT.get(int(r["Year"]), {}).get(r["Province"], np.nan), axis=1)
    anchor_years = sorted(int(y) for y in FINANCE_COST_IMPACT.keys())
    diag = diag[diag["Year"].isin(anchor_years)].copy()
    diag["Gross_minus_finance"] = diag["Gross_model_avg"] - diag["Finance_cost_impact"]
    diag.to_csv("ggppa_finance_crosswalk_v4.csv", index=False)


if __name__ == "__main__":
    main()
