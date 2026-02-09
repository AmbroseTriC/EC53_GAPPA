#!/usr/bin/env python3
"""
Single-version GGPPA Schedule 2 rates scraper (v2).

Reads one fixed URL and writes a rates-only workbook.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


SOURCE_URL = "https://laws-lois.justice.gc.ca/eng/acts/g-11.55/20250315/P1TT3xt3.html"
OUTPUT_XLSX = "Crawling/Tax Rate Items_v2.xlsx"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

TARGET_FUELS = {
    "gasoline": "Gasoline",
    "light fuel oil": "Light fuel oil",
    "marketable natural gas": "Marketable natural gas",
}


def fetch_html(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def parse_version_date(url: str) -> str:
    match = re.search(r"/(\d{8})/P1TT3xt3\.html$", url)
    if not match:
        raise ValueError(f"Unable to parse version date from URL: {url}")
    return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")


def clean_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("Â", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\([a-zA-Z]\)\s*", "", text)
    text = re.sub(r"^[a-zA-Z]\)\s*", "", text)
    if text.startswith("[") and text.endswith("]"):
        return ""
    return text


def normalize_fuel_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def find_schedule_heading(soup: BeautifulSoup, label: str) -> Optional[Tag]:
    for h2 in soup.find_all("h2"):
        if label in h2.get_text(" ", strip=True):
            return h2
    return None


def get_schedule_boundaries(soup: BeautifulSoup) -> Dict[str, str]:
    boundaries: Dict[str, str] = {}
    for h2 in soup.find_all("h2", class_="scheduleLabel"):
        text = h2.get_text(" ", strip=True)
        if "SCHEDULE 1" in text:
            boundaries["SCHEDULE 1"] = h2.get("id")
        elif "SCHEDULE 2" in text:
            boundaries["SCHEDULE 2"] = h2.get("id")
        elif "SCHEDULE 3" in text:
            boundaries["SCHEDULE 3"] = h2.get("id")
    return boundaries


def iter_until(elem: Tag, stop_ids: Iterable[str]) -> Iterable[Tag]:
    for node in elem.find_all_next():
        if isinstance(node, Tag) and node.get("id") in stop_ids:
            break
        yield node


def parse_period_from_caption(caption: str) -> Tuple[str, str]:
    caption = caption.strip()
    bounded = re.search(
        r"period beginning on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+and ending on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        caption,
        re.IGNORECASE,
    )
    if bounded:
        start = datetime.strptime(bounded.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
        end = datetime.strptime(bounded.group(2), "%B %d, %Y").strftime("%Y-%m-%d")
        return start, end

    after = re.search(
        r"applicable after\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        caption,
        re.IGNORECASE,
    )
    if after:
        start_dt = datetime.strptime(after.group(1), "%B %d, %Y") + timedelta(days=1)
        return start_dt.strftime("%Y-%m-%d"), ""

    year_only = re.search(r"applicable in\s+(\d{4})", caption, re.IGNORECASE)
    if year_only:
        year = int(year_only.group(1))
        return f"{year}-01-01", f"{year}-12-31"

    return "", ""


def is_relevant_caption(caption: str) -> bool:
    c = caption.lower()
    return (
        "period beginning on" in c and "ending on" in c
    ) or "applicable after" in c or "applicable in" in c


def parse_period_from_text(text: str) -> Tuple[str, str]:
    return parse_period_from_caption(text)


def expand_table(table: Tag) -> List[List[str]]:
    rows = table.find_all("tr")
    grid: List[List[str]] = []
    rowspans: Dict[int, Tuple[int, str]] = {}

    for row in rows:
        cells = row.find_all(["th", "td"])
        current: List[str] = []
        col_idx = 0

        while col_idx in rowspans:
            span_count, span_text = rowspans[col_idx]
            current.append(span_text)
            if span_count <= 1:
                del rowspans[col_idx]
            else:
                rowspans[col_idx] = (span_count - 1, span_text)
            col_idx += 1

        for cell in cells:
            while col_idx in rowspans:
                span_count, span_text = rowspans[col_idx]
                current.append(span_text)
                if span_count <= 1:
                    del rowspans[col_idx]
                else:
                    rowspans[col_idx] = (span_count - 1, span_text)
                col_idx += 1

            text = cell.get_text(" ", strip=True)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            for _ in range(colspan):
                current.append(text)

            if rowspan > 1:
                for i in range(colspan):
                    rowspans[col_idx + i] = (rowspan - 1, text)
            col_idx += colspan

        while col_idx in rowspans:
            span_count, span_text = rowspans[col_idx]
            current.append(span_text)
            if span_count <= 1:
                del rowspans[col_idx]
            else:
                rowspans[col_idx] = (span_count - 1, span_text)
            col_idx += 1

        grid.append(current)

    return grid


def locate_header_row(rows: List[List[str]]) -> Optional[int]:
    for idx, row in enumerate(rows[:10]):
        values = [c.strip() for c in row]
        if (
            "Item" in values
            and "Type" in values
            and "Unit" in values
            and "Listed Province" in values
            and "Rate" in values
        ):
            return idx
    return None


def locate_multiperiod_header_row(rows: List[List[str]]) -> Optional[int]:
    for idx, row in enumerate(rows[:12]):
        values = [c.strip() for c in row]
        if "Type" not in values or "Unit" not in values:
            continue
        if any("Rates of charge applicable" in c for c in values):
            return idx
    return None


def extract_rates(soup: BeautifulSoup, version_date: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    boundaries = get_schedule_boundaries(soup)

    schedule2 = find_schedule_heading(soup, "SCHEDULE 2")
    if not schedule2:
        return results

    stop_ids = {boundaries.get("SCHEDULE 3")} if boundaries.get("SCHEDULE 3") else set()

    for node in iter_until(schedule2, stop_ids):
        if node.name != "figure":
            continue

        figcaption = node.find("figcaption")
        if not figcaption:
            continue
        caption = figcaption.get_text(" ", strip=True)
        if not is_relevant_caption(caption):
            continue

        period_start, period_end = parse_period_from_caption(caption)

        table = node.find("table")
        if not table:
            continue
        expanded = expand_table(table)
        header_idx = locate_header_row(expanded)
        if header_idx is not None:
            header = expanded[header_idx]

            def col_idx(name: str) -> Optional[int]:
                for i, cell in enumerate(header):
                    if cell.strip() == name:
                        return i
                return None

            idx_type = col_idx("Type")
            idx_unit = col_idx("Unit")
            idx_province = col_idx("Listed Province")
            idx_rate = col_idx("Rate")
            if None in (idx_type, idx_unit, idx_province, idx_rate):
                continue

            max_idx = max(idx_type, idx_unit, idx_province, idx_rate)
            for row in expanded[header_idx + 1 :]:
                if len(row) <= max_idx:
                    continue

                fuel_raw = clean_text(row[idx_type])
                unit = clean_text(row[idx_unit])
                province = clean_text(row[idx_province])
                rate_raw = clean_text(row[idx_rate])

                if not fuel_raw and not rate_raw:
                    continue
                if fuel_raw.lower() == "type" and rate_raw.lower() == "rate":
                    continue

                normalized = normalize_fuel_name(fuel_raw)
                if normalized not in TARGET_FUELS:
                    continue

                rate_value = pd.to_numeric(rate_raw, errors="coerce")
                if pd.isna(rate_value):
                    continue

                results.append(
                    {
                        "Version_Date": version_date,
                        "Period_Start": period_start,
                        "Period_End": period_end,
                        "Fuel_Type": TARGET_FUELS[normalized],
                        "Unit": unit,
                        "Province": province,
                        "Rate": float(rate_value),
                    }
                )
            continue

        # Table 5-style layout: rate periods are separate columns and no province column.
        header_idx = locate_multiperiod_header_row(expanded)
        if header_idx is None:
            continue

        header = expanded[header_idx]

        def col_idx(name: str) -> Optional[int]:
            for i, cell in enumerate(header):
                if cell.strip() == name:
                    return i
            return None

        idx_type = col_idx("Type")
        idx_unit = col_idx("Unit")
        if None in (idx_type, idx_unit):
            continue

        period_cols: List[Tuple[int, str, str]] = []
        for idx, cell in enumerate(header):
            if "Rates of charge applicable" not in cell:
                continue
            col_start, col_end = parse_period_from_text(cell)
            if not col_start and not col_end:
                continue
            period_cols.append((idx, col_start, col_end))

        if not period_cols:
            continue

        max_idx = max([idx_type, idx_unit] + [x[0] for x in period_cols])
        for row in expanded[header_idx + 1 :]:
            if len(row) <= max_idx:
                continue
            fuel_raw = clean_text(row[idx_type])
            unit = clean_text(row[idx_unit])
            if not fuel_raw:
                continue
            if fuel_raw.lower() == "type":
                continue

            normalized = normalize_fuel_name(fuel_raw)
            if normalized not in TARGET_FUELS:
                continue

            for period_col_idx, col_start, col_end in period_cols:
                rate_raw = clean_text(row[period_col_idx])
                rate_value = pd.to_numeric(rate_raw, errors="coerce")
                if pd.isna(rate_value):
                    continue
                results.append(
                    {
                        "Version_Date": version_date,
                        "Period_Start": col_start,
                        "Period_End": col_end,
                        "Fuel_Type": TARGET_FUELS[normalized],
                        "Unit": unit,
                        "Province": "",
                        "Rate": float(rate_value),
                    }
                )

    return results


def expand_blank_province_rows(
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    canonical_provinces = sorted(
        {
            str(r.get("Province", "")).strip()
            for r in rows
            if str(r.get("Province", "")).strip()
        }
    )

    if not canonical_provinces:
        print("Warning: no province-specific rows found; leaving blank Province rows as-is.")
        return rows

    expanded: List[Dict[str, object]] = []
    for row in rows:
        province = str(row.get("Province", "")).strip()
        if province:
            expanded.append(row)
            continue
        for p in canonical_provinces:
            clone = dict(row)
            clone["Province"] = p
            expanded.append(clone)
    return expanded


def main() -> int:
    html = fetch_html(SOURCE_URL)
    soup = BeautifulSoup(html, "html.parser")
    version_date = parse_version_date(SOURCE_URL)
    rates = extract_rates(soup, version_date)

    rates_expanded = expand_blank_province_rows(rates)

    rates_df = pd.DataFrame(
        rates_expanded,
        columns=[
            "Version_Date",
            "Period_Start",
            "Period_End",
            "Fuel_Type",
            "Unit",
            "Province",
            "Rate",
        ],
    )
    rates_df = rates_df.drop_duplicates()
    rates_df = rates_df.sort_values(
        by=["Period_Start", "Fuel_Type", "Province"], na_position="last"
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        rates_df.to_excel(writer, sheet_name="Rates", index=False)

    print(f"Wrote {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
