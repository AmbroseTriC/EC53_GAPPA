#!/usr/bin/env python3
"""
Greenhouse Gas Pollution Pricing Act (PIT Index) scraper.

Outputs:
  - Crawling/ggppp_history.xlsx with sheets:
      * Jurisdiction: Version_Date, Act_Citation, Covered_Provinces
      * Rates: Version_Date, Period_Start, Period_End, Fuel_Type, Unit, Province, Rate
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


PIT_INDEX_URL = "https://laws-lois.justice.gc.ca/eng/acts/g-11.55/PITIndex.html"
OUTPUT_XLSX = "Crawling/ggppp_history.xlsx"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class VersionLink:
    version_date: str
    url: str


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_version_links(html: str) -> List[VersionLink]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[VersionLink] = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        if "P1TT3" not in href:
            continue
        m = re.search(r"From\s+(\d{4}-\d{2}-\d{2})", text)
        if not m:
            continue
        version_date = m.group(1)
        if href.startswith("http"):
            url = href
        else:
            url = f"https://laws-lois.justice.gc.ca/eng/acts/g-11.55/{href.lstrip('/')}"
        links.append(VersionLink(version_date=version_date, url=url))
    return links


def clean_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("Â", "")
    text = re.sub(r"\s+", " ", text).strip()
    # Drop leading list markers like "(a)" or "a)"
    text = re.sub(r"^\([a-zA-Z]\)\s*", "", text)
    text = re.sub(r"^[a-zA-Z]\)\s*", "", text)
    # Remove bracketed repeal notes and similar references
    if text.startswith("[") and text.endswith("]"):
        return ""
    return text


def extract_act_citation(soup: BeautifulSoup) -> str:
    # Breadcrumb contains: "S.C. 2018, c. 12, s. 186 - Table of Contents"
    for li in soup.select("ol.breadcrumb li"):
        txt = li.get_text(" ", strip=True)
        if "S.C." in txt and "c." in txt:
            return txt.replace(" - Table of Contents", "").replace("\xa0", " ").strip()
    # Fallback: search any string containing S.C.
    for txt in soup.stripped_strings:
        if "S.C." in txt and "c." in txt:
            return txt.replace(" - Table of Contents", "").replace("\xa0", " ").strip()
    return ""


def find_schedule_heading(
    soup: BeautifulSoup, schedule_label: str
) -> Optional[Tag]:
    for h2 in soup.find_all("h2"):
        if schedule_label in h2.get_text(" ", strip=True):
            return h2
    return None


def iter_until(elem: Tag, stop_ids: Iterable[str]) -> Iterable[Tag]:
    for el in elem.find_all_next():
        if isinstance(el, Tag) and el.get("id") in stop_ids:
            break
        yield el


def get_schedule_boundaries(soup: BeautifulSoup) -> Dict[str, str]:
    ids: Dict[str, str] = {}
    for h2 in soup.find_all("h2", class_="scheduleLabel"):
        text = h2.get_text(" ", strip=True)
        if "SCHEDULE 1" in text:
            ids["SCHEDULE 1"] = h2.get("id")
        elif "SCHEDULE 2" in text:
            ids["SCHEDULE 2"] = h2.get("id")
        elif "SCHEDULE 3" in text:
            ids["SCHEDULE 3"] = h2.get("id")
    return ids


def extract_provinces_from_table(table: Tag) -> List[str]:
    rows = table.find_all("tr")
    if not rows:
        return []
    # Find header row to locate province column
    header_cells = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    province_idx = None
    for i, h in enumerate(header_cells):
        if "Province" in h:
            province_idx = i
            break
        if "Name of Province" in h:
            province_idx = i
            break
    if province_idx is None:
        return []
    provinces: List[str] = []
    for r in rows[1:]:
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])]
        if len(cells) <= province_idx:
            continue
        val = clean_text(cells[province_idx])
        if not val:
            continue
        provinces.append(val)
    # Preserve order but unique
    seen = set()
    ordered = []
    for p in provinces:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def extract_part1_provinces(soup: BeautifulSoup) -> List[str]:
    boundaries = get_schedule_boundaries(soup)
    sched1 = find_schedule_heading(soup, "SCHEDULE 1")
    if not sched1:
        return []

    stop_ids = {boundaries.get("SCHEDULE 2")} if boundaries.get("SCHEDULE 2") else set()

    # Try to find Part 1 heading first
    part1_heading = None
    for h2 in sched1.find_all_next("h2"):
        txt = h2.get_text(" ", strip=True)
        if "PART 1" in txt and "Part 1 of the Act" in txt:
            part1_heading = h2
            break
        if h2.get("id") in stop_ids:
            break

    def tables_in_section(start: Tag, stop: Optional[Tag]) -> List[Tag]:
        tables: List[Tag] = []
        for el in start.find_all_next():
            if stop is not None and el is stop:
                break
            if el.get("id") in stop_ids:
                break
            if el.name == "table":
                tables.append(el)
        return tables

    # If Part 1 heading exists, use first provinces table between Part 1 and Part 2/Schedule 2.
    if part1_heading is not None:
        part2_heading = None
        for h2 in part1_heading.find_all_next("h2"):
            txt = h2.get_text(" ", strip=True)
            if "PART 2" in txt and "Part 2 of the Act" in txt:
                part2_heading = h2
                break
            if h2.get("id") in stop_ids:
                break
        tables = tables_in_section(part1_heading, part2_heading)
        for table in tables:
            provinces = extract_provinces_from_table(table)
            if provinces:
                return provinces

    # Fallback: first "List of Provinces" table under Schedule 1
    for el in iter_until(sched1, stop_ids):
        if el.name == "table":
            provinces = extract_provinces_from_table(el)
            if provinces:
                return provinces
    return []


def parse_period_from_caption(caption: str) -> Tuple[str, str]:
    caption = caption.strip()
    # period beginning on April 1, 2019 and ending on March 31, 2020
    m = re.search(
        r"period beginning on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+and ending on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        caption,
        re.IGNORECASE,
    )
    if m:
        start = datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
        end = datetime.strptime(m.group(2), "%B %d, %Y").strftime("%Y-%m-%d")
        return start, end
    if re.search(r"applicable in\s+2018", caption, re.IGNORECASE):
        return "2018-01-01", "2018-12-31"
    m = re.search(
        r"applicable after\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        caption,
        re.IGNORECASE,
    )
    if m:
        start_dt = datetime.strptime(m.group(1), "%B %d, %Y") + timedelta(days=1)
        return start_dt.strftime("%Y-%m-%d"), ""
    return "", ""


def locate_multiperiod_header_row(rows: List[List[str]]) -> Optional[int]:
    for i, row in enumerate(rows[:12]):
        row_norm = [c.strip() for c in row]
        if "Type" not in row_norm or "Unit" not in row_norm:
            continue
        if any("Rates of charge applicable" in c for c in row_norm):
            return i
    return None


def expand_table(table: Tag) -> List[List[str]]:
    rows = table.find_all("tr")
    grid: List[List[str]] = []
    rowspans: Dict[int, Tuple[int, str]] = {}
    for r in rows:
        cells = r.find_all(["th", "td"])
        row: List[str] = []
        col_idx = 0
        while col_idx in rowspans:
            span_count, span_text = rowspans[col_idx]
            row.append(span_text)
            if span_count <= 1:
                del rowspans[col_idx]
            else:
                rowspans[col_idx] = (span_count - 1, span_text)
            col_idx += 1

        for cell in cells:
            while col_idx in rowspans:
                span_count, span_text = rowspans[col_idx]
                row.append(span_text)
                if span_count <= 1:
                    del rowspans[col_idx]
                else:
                    rowspans[col_idx] = (span_count - 1, span_text)
                col_idx += 1
            text = cell.get_text(" ", strip=True)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for _ in range(colspan):
                row.append(text)
            if rowspan > 1:
                for i in range(colspan):
                    rowspans[col_idx + i] = (rowspan - 1, text)
            col_idx += colspan

        # fill in any trailing rowspans
        while col_idx in rowspans:
            span_count, span_text = rowspans[col_idx]
            row.append(span_text)
            if span_count <= 1:
                del rowspans[col_idx]
            else:
                rowspans[col_idx] = (span_count - 1, span_text)
            col_idx += 1
        grid.append(row)
    return grid


def locate_header_row(rows: List[List[str]]) -> Optional[int]:
    for i, row in enumerate(rows[:10]):
        row_norm = [c.strip() for c in row]
        if (
            "Item" in row_norm
            and "Type" in row_norm
            and "Unit" in row_norm
            and "Listed Province" in row_norm
            and "Rate" in row_norm
        ):
            return i
    return None


def extract_rates(soup: BeautifulSoup, version_date: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    boundaries = get_schedule_boundaries(soup)
    sched2 = find_schedule_heading(soup, "SCHEDULE 2")
    if not sched2:
        return results

    stop_ids = {boundaries.get("SCHEDULE 3")} if boundaries.get("SCHEDULE 3") else set()

    for el in iter_until(sched2, stop_ids):
        if el.name != "figure":
            continue
        caption = ""
        if el.find("figcaption"):
            caption = el.find("figcaption").get_text(" ", strip=True)
        table = el.find("table")
        if not table:
            continue
        period_start, period_end = parse_period_from_caption(caption)

        expanded = expand_table(table)
        header_idx = locate_header_row(expanded)
        if header_idx is not None:
            header = expanded[header_idx]

            # map column indexes
            def col_idx(col_name: str) -> Optional[int]:
                for i, h in enumerate(header):
                    if h.strip() == col_name:
                        return i
                return None

            idx_type = col_idx("Type")
            idx_unit = col_idx("Unit")
            idx_prov = col_idx("Listed Province")
            idx_rate = col_idx("Rate")
            if None in (idx_type, idx_unit, idx_prov, idx_rate):
                continue

            for row in expanded[header_idx + 1 :]:
                if len(row) <= max(idx_type, idx_unit, idx_prov, idx_rate):
                    continue
                fuel = clean_text(row[idx_type])
                unit = clean_text(row[idx_unit])
                prov = clean_text(row[idx_prov])
                rate = clean_text(row[idx_rate])
                if not fuel and not rate:
                    continue
                # skip header-like rows
                if fuel.lower() == "type" and rate.lower() == "rate":
                    continue
                rate_value = pd.to_numeric(rate, errors="coerce")
                if pd.isna(rate_value):
                    continue
                results.append(
                    {
                        "Version_Date": version_date,
                        "Period_Start": period_start,
                        "Period_End": period_end,
                        "Fuel_Type": fuel,
                        "Unit": unit,
                        "Province": prov,
                        "Rate": float(rate_value),
                    }
                )
            continue

        # Table 5 layout: no "Listed Province", and rate periods are separate columns.
        header_idx = locate_multiperiod_header_row(expanded)
        if header_idx is None:
            continue
        header = expanded[header_idx]

        def col_idx(col_name: str) -> Optional[int]:
            for i, h in enumerate(header):
                if h.strip() == col_name:
                    return i
            return None

        idx_type = col_idx("Type")
        idx_unit = col_idx("Unit")
        if None in (idx_type, idx_unit):
            continue

        period_cols: List[Tuple[int, str, str]] = []
        for idx, h in enumerate(header):
            if "Rates of charge applicable" not in h:
                continue
            col_start, col_end = parse_period_from_caption(h)
            if not col_start and not col_end:
                continue
            period_cols.append((idx, col_start, col_end))

        if not period_cols:
            continue

        max_idx = max([idx_type, idx_unit] + [x[0] for x in period_cols])
        for row in expanded[header_idx + 1 :]:
            if len(row) <= max_idx:
                continue
            fuel = clean_text(row[idx_type])
            unit = clean_text(row[idx_unit])
            if not fuel or fuel.lower() == "type":
                continue

            for col_idx_rate, col_start, col_end in period_cols:
                rate = clean_text(row[col_idx_rate])
                rate_value = pd.to_numeric(rate, errors="coerce")
                if pd.isna(rate_value):
                    continue
                results.append(
                    {
                        "Version_Date": version_date,
                        "Period_Start": col_start,
                        "Period_End": col_end,
                        "Fuel_Type": fuel,
                        "Unit": unit,
                        "Province": "",
                        "Rate": float(rate_value),
                    }
                )
    return results


def scrape() -> Tuple[pd.DataFrame, pd.DataFrame]:
    index_html = fetch_html(PIT_INDEX_URL)
    version_links = parse_version_links(index_html)
    if not version_links:
        raise RuntimeError("No PIT versions found.")

    jurisdiction_rows: List[Dict[str, str]] = []
    rates_rows: List[Dict[str, str]] = []

    for v in version_links:
        try:
            html = fetch_html(v.url)
        except Exception as exc:  # pragma: no cover - network error
            print(f"Warning: failed to fetch {v.url}: {exc}", file=sys.stderr)
            continue
        soup = BeautifulSoup(html, "html.parser")
        act_citation = extract_act_citation(soup)
        provinces = extract_part1_provinces(soup)
        jurisdiction_rows.append(
            {
                "Version_Date": v.version_date,
                "Act_Citation": act_citation,
                "Covered_Provinces": "; ".join(provinces),
            }
        )

        rates = extract_rates(soup, v.version_date)
        rates_rows.extend(rates)

    jurisdiction_df = pd.DataFrame(
        jurisdiction_rows,
        columns=["Version_Date", "Act_Citation", "Covered_Provinces"],
    )
    rates_df = pd.DataFrame(
        rates_rows,
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
    return jurisdiction_df, rates_df


def main() -> int:
    jurisdiction_df, rates_df = scrape()
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        jurisdiction_df.to_excel(writer, sheet_name="Jurisdiction", index=False)
        rates_df.to_excel(writer, sheet_name="Rates", index=False)
    print(f"Wrote {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
