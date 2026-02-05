# Cleaned Energy Price Workbook

This workbook contains province/territory-level monthly average fuel prices
derived from Statistics Canada table 18-10-0001 (input file
`EnePrice-18100001/18100001.xlsx`).

## Output File
- `Cleaning/energy_price_cleaned.xlsx`

## Coverage
- Date range: January 2019 through January 2025 (inclusive)
- Regions: Ontario, Manitoba, Saskatchewan, Alberta, Yukon
- Fuels:
  - Household heating fuel
  - Regular unleaded gasoline at self service filling stations

## Processing Summary
- City-level rows are mapped to province/territory using the `GEO` field.
- Monthly values are averaged across cities within each province/territory.
- Units are preserved (no unit conversion).

## Columns
- `REF_DATE`: Month (YYYY-MM)
- `Region`: Province/territory
- `Type of fuel`: Fuel category
- `UOM`: Unit of measurement
- `VALUE`: Province/territory monthly average value

## How It’s Produced
Run:

```bash
python Cleaning/energy_price_cleaning.py
```
