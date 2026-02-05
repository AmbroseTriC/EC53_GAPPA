# Greenhouse Gas Pollution Pricing Act PIT Scraper

This scraper collects historical versions of the Greenhouse Gas Pollution Pricing Act
from the Point-in-Time (PIT) index and compiles jurisdiction and fuel charge rates into
an Excel workbook.

## Data Source
- Point-in-Time Index: `https://laws-lois.justice.gc.ca/eng/acts/g-11.55/PITIndex.html`

## Requirements
- Python 3.9+
- `requests`
- `beautifulsoup4`
- `pandas`
- `openpyxl`

## How to Run
From the `Crawling` folder:

```bash
python Crawl.py
```

## Output
The script writes: `Crawling/ggppp_history.xlsx`

### Sheet: Jurisdiction
Columns:
- `Version_Date`
- `Act_Citation`
- `Covered_Provinces`

### Sheet: Rates
Columns:
- `Version_Date`
- `Period_Start`
- `Period_End`
- `Fuel_Type`
- `Unit`
- `Province`
- `Rate`

## Period Parsing Notes
- Captions like “period beginning on April 1, 2019 and ending on March 31, 2020”
  map to `Period_Start=2019-04-01`, `Period_End=2020-03-31`.
- Captions like “Rates of charge applicable in 2018” map to
  `Period_Start=2018-01-01`, `Period_End=2018-12-31`.
- Captions like “Rates of charge applicable after March 31, 2023” map to
  `Period_Start=2023-04-01`, `Period_End=""` (open-ended).
