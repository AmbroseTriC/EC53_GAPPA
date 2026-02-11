#!/usr/bin/env python3
"""Replicate v4 progressivity figures from ggppa_progressivity_results_v4.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROVINCE_ORDER = ["Alberta", "Manitoba", "Ontario", "Saskatchewan"]
QUINTILE_ORDER = [
    "Lowest quintile",
    "Second quintile",
    "Third quintile",
    "Fourth quintile",
    "Highest quintile",
]
QUINTILE_TICKS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
YEAR_ORDER = list(range(2019, 2026))

PROVINCE_COLORS = {
    "Alberta": "#1f77b4",
    "Manitoba": "#ff7f0e",
    "Ontario": "#2ca02c",
    "Saskatchewan": "#d62728",
}

REQUIRED_COLUMNS = [
    "Province",
    "Year",
    "Quintile",
    "Households",
    "gross_cost",
    "etr_gross_cost",
    "etr_net_family4_base",
]

FIGURE_FACE_COLOR = "#ffffff"
AXES_FACE_COLOR = "#ffffff"
LINEWIDTH = 1.6
MARKERSIZE = 7


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / "../v4/ggppa_progressivity_results_v4.csv").resolve()
    parser = argparse.ArgumentParser(description="Replicate three v4 figures from model output CSV.")
    parser.add_argument("--input-csv", type=Path, default=default_input, help="Path to ggppa_progressivity_results_v4.csv.")
    parser.add_argument("--output-dir", type=Path, default=script_dir, help="Directory to save figures.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_coverage(df: pd.DataFrame) -> None:
    provinces_present = set(df["Province"].unique().tolist())
    missing_provinces = sorted(set(PROVINCE_ORDER) - provinces_present)
    if missing_provinces:
        raise ValueError(f"Missing required provinces: {missing_provinces}")

    quintiles_present = set(df["Quintile"].unique().tolist())
    missing_quintiles = [q for q in QUINTILE_ORDER if q not in quintiles_present]
    if missing_quintiles:
        raise ValueError(f"Missing required quintiles: {missing_quintiles}")

    years_present = set(int(y) for y in df["Year"].unique().tolist())
    missing_years = [y for y in YEAR_ORDER if y not in years_present]
    if missing_years:
        raise ValueError(f"Missing required years for trend figure: {missing_years}")

    # Figure 1 and 2 must have complete province x quintile coverage for 2024.
    sub_2024 = df[df["Year"] == 2024].copy()
    if sub_2024.empty:
        raise ValueError("Missing year 2024 required for figure 1 and figure 2.")

    counts_2024 = sub_2024.groupby(["Province", "Quintile"]).size()
    missing_pairs: list[tuple[str, str]] = []
    duplicate_pairs: list[tuple[str, str, int]] = []
    for province in PROVINCE_ORDER:
        for quintile in QUINTILE_ORDER:
            key = (province, quintile)
            n = int(counts_2024.get(key, 0))
            if n == 0:
                missing_pairs.append(key)
            elif n > 1:
                duplicate_pairs.append((province, quintile, n))
    if missing_pairs:
        raise ValueError(f"Missing 2024 province-quintile rows: {missing_pairs}")
    if duplicate_pairs:
        raise ValueError(f"Duplicate 2024 province-quintile rows found: {duplicate_pairs}")

    # Figure 3 needs at least one observation for each province-year pair.
    counts_py = df.groupby(["Province", "Year"]).size()
    missing_py: list[tuple[str, int]] = []
    for province in PROVINCE_ORDER:
        for year in YEAR_ORDER:
            if int(counts_py.get((province, year), 0)) == 0:
                missing_py.append((province, year))
    if missing_py:
        raise ValueError(f"Missing province-year rows for trend figure: {missing_py}")

    # Weighted averages require strictly positive denominator.
    bad_weights = (
        df.groupby(["Province", "Year"], as_index=False)["Households"]
        .sum()
        .query("Households <= 0")
    )
    if not bad_weights.empty:
        bad = list(zip(bad_weights["Province"], bad_weights["Year"], bad_weights["Households"]))
        raise ValueError(f"Non-positive province-year household totals: {bad}")


def setup_axes(fig_size: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    ax.set_facecolor(AXES_FACE_COLOR)
    return fig, ax


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, facecolor="#ffffff")
    fig.savefig(pdf_path, facecolor="#ffffff")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_etr_gross_2024(df: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    sub = df[df["Year"] == 2024].copy()
    fig, ax = setup_axes((11, 6))
    x = np.arange(len(QUINTILE_ORDER))
    for province in PROVINCE_ORDER:
        vals = (
            sub[sub["Province"] == province]
            .set_index("Quintile")
            .reindex(QUINTILE_ORDER)["etr_gross_cost"]
            .to_numpy(dtype=float)
        )
        ax.plot(
            x,
            vals,
            label=province,
            color=PROVINCE_COLORS[province],
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
    ax.set_xticks(x, QUINTILE_TICKS)
    ax.set_xlabel("Before-tax income quintile")
    ax.set_ylabel("ETR (gross cost / total expenditure)")
    ax.legend(loc="upper right", frameon=False)
    return save_figure(fig, output_dir, "fig_etr_gross_2024", dpi)


def plot_etr_net_family4_base_2024(df: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    sub = df[df["Year"] == 2024].copy()
    fig, ax = setup_axes((11, 6))
    x = np.arange(len(QUINTILE_ORDER))
    for province in PROVINCE_ORDER:
        vals = (
            sub[sub["Province"] == province]
            .set_index("Quintile")
            .reindex(QUINTILE_ORDER)["etr_net_family4_base"]
            .to_numpy(dtype=float)
        )
        ax.plot(
            x,
            vals,
            label=province,
            color=PROVINCE_COLORS[province],
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
    ax.set_xticks(x, QUINTILE_TICKS)
    ax.set_xlabel("Before-tax income quintile")
    ax.set_ylabel("ETR (net cost, family of 4, base CCR) / total expenditure")
    ax.legend(loc="upper left", frameon=False)
    return save_figure(fig, output_dir, "fig_etr_net_family4_base_2024", dpi)


def plot_gross_cost_avg_trend(df: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    tmp = df.copy()
    tmp["weighted_gross"] = tmp["gross_cost"] * tmp["Households"]
    trend = (
        tmp.groupby(["Province", "Year"], as_index=False)[["weighted_gross", "Households"]]
        .sum()
        .assign(gross_avg=lambda d: d["weighted_gross"] / d["Households"])
    )

    fig, ax = setup_axes((11, 6))
    x = np.array(YEAR_ORDER, dtype=int)
    for province in PROVINCE_ORDER:
        vals = (
            trend[trend["Province"] == province]
            .set_index("Year")
            .reindex(YEAR_ORDER)["gross_avg"]
            .to_numpy(dtype=float)
        )
        ax.plot(
            x,
            vals,
            label=province,
            color=PROVINCE_COLORS[province],
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
    ax.set_xticks(YEAR_ORDER)
    ax.set_xlabel("Year")
    ax.set_ylabel("Household-weighted provincial average gross cost (CAD/year)")
    ax.legend(loc="upper left", frameon=False)
    return save_figure(fig, output_dir, "fig_gross_cost_avg_trend", dpi)


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    validate_columns(df)
    validate_coverage(df)

    written: list[Path] = []
    written.extend(plot_etr_gross_2024(df, output_dir, args.dpi))
    written.extend(plot_etr_net_family4_base_2024(df, output_dir, args.dpi))
    written.extend(plot_gross_cost_avg_trend(df, output_dir, args.dpi))

    print("Completed figure export:")
    for p in written:
        print(f"- {p}")


if __name__ == "__main__":
    main()
